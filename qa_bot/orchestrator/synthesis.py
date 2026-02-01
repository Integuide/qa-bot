"""Synthesis agent that generates final QA reports."""

import re
from datetime import datetime

from qa_bot.ai.base import AIProvider
from qa_bot.orchestrator.shared_state import SharedFlowState
from qa_bot.orchestrator.flow import FlowStatus


class SynthesisAgent:
    """
    Generates final QA report after exploration completes.

    - Collects all issues from all workers
    - Deduplicates similar issues
    - Categorizes by severity and type
    - Includes action summaries for reproduction steps
    - Writes detailed markdown report
    """

    def __init__(self, ai_provider: AIProvider):
        self.ai = ai_provider

    def _summarize_action(self, action: dict) -> dict:
        """
        Create a human-readable summary of an action.

        Extracts the key action type and target from the code.
        """
        code = action.get("code", "")
        success = action.get("success", True)
        url = action.get("url", "")

        # Try to extract a description from the code
        description = self._extract_action_description(code)

        return {
            "description": description,
            "success": success,
            "url": url,
            "code": code[:100] if code else ""  # Keep short code snippet for context
        }

    def _extract_action_description(self, code: str) -> str:
        """
        Extract a human-readable description from action code.

        Parses common Playwright patterns to create descriptions like:
        - "Click 'Login' button"
        - "Fill email field with test@example.com"
        - "Navigate to /products"
        """
        if not code:
            return "Unknown action"

        code = code.strip()

        # qa.done() - completion
        if match := re.search(r'qa\.done\(["\'](.+?)["\']\)', code):
            return f"Completed: {match.group(1)[:60]}"

        # qa.block() - blocking
        if match := re.search(r'qa\.block\(["\'](.+?)["\']\)', code):
            return f"Blocked: {match.group(1)[:60]}"

        # qa.report_issue() - issue reported
        if match := re.search(r'qa\.report_issue\(["\'](.+?)["\']', code):
            return f"Reported issue: {match.group(1)[:50]}"

        # qa.add_flow() - new flow
        if match := re.search(r'qa\.add_flow\(["\'](.+?)["\']', code):
            return f"Created flow: {match.group(1)}"

        # page.goto() - navigation
        if match := re.search(r'page\.goto\(["\'](.+?)["\']\)', code):
            return f"Navigate to {match.group(1)[:50]}"

        # Click actions
        if "click" in code.lower():
            # get_by_role("button", name="X").click()
            if match := re.search(r'get_by_role\(["\'](\w+)["\'],\s*name=["\'](.+?)["\']\)\.click', code):
                return f"Click '{match.group(2)}' {match.group(1)}"
            # get_by_text("X").click()
            if match := re.search(r'get_by_text\(["\'](.+?)["\']\)\.click', code):
                return f"Click text '{match.group(1)}'"
            # get_by_label("X").click()
            if match := re.search(r'get_by_label\(["\'](.+?)["\']\)\.click', code):
                return f"Click '{match.group(1)}' field"
            # qa.click_ref()
            if match := re.search(r'qa\.click_ref\((\d+)\)', code):
                return f"Click element ref {match.group(1)}"
            return "Click action"

        # Fill actions
        if "fill" in code.lower():
            # get_by_label("X").fill("Y")
            if match := re.search(r'get_by_label\(["\'](.+?)["\']\)\.fill\(["\'](.+?)["\']\)', code):
                value = match.group(2)
                # Mask sensitive values
                sensitive_patterns = [
                    'password', 'pwd', 'secret', 'token', 'key', 'auth',
                    'credential', 'pin', 'api', 'private', 'ssn', 'cvv'
                ]
                if any(s in match.group(1).lower() for s in sensitive_patterns):
                    value = "***"
                elif len(value) > 20:
                    value = value[:17] + "..."
                return f"Fill '{match.group(1)}' with '{value}'"
            # get_by_placeholder("X").fill("Y")
            if match := re.search(r'get_by_placeholder\(["\'](.+?)["\']\)\.fill\(["\'](.+?)["\']\)', code):
                value = match.group(2)
                if len(value) > 20:
                    value = value[:17] + "..."
                return f"Fill '{match.group(1)}' field with '{value}'"
            # qa.fill_ref()
            if match := re.search(r'qa\.fill_ref\((\d+),\s*["\'](.+?)["\']\)', code):
                value = match.group(2)
                if len(value) > 20:
                    value = value[:17] + "..."
                return f"Fill element ref {match.group(1)} with '{value}'"
            return "Fill form field"

        # Resize actions
        if match := re.search(r'qa\.resize\((\d+),\s*(\d+)\)', code):
            return f"Resize viewport to {match.group(1)}x{match.group(2)}"

        # Keyboard actions
        if match := re.search(r'page\.keyboard\.press\(["\'](.+?)["\']\)', code):
            return f"Press {match.group(1)} key"

        # Wait actions
        if "wait" in code.lower():
            return "Wait for page update"

        # Default: first line of code, truncated
        first_line = code.split('\n')[0][:60]
        return first_line if first_line else "Unknown action"

    async def generate_report(self, shared_state: SharedFlowState) -> str:
        """
        Generate final QA synthesis report.

        Args:
            shared_state: The shared state containing all exploration data

        Returns:
            Markdown-formatted report
        """
        # Calculate duration
        duration_seconds = (datetime.now() - shared_state.start_time).total_seconds()
        if duration_seconds < 60:
            duration = f"{int(duration_seconds)} seconds"
        elif duration_seconds < 3600:
            duration = f"{int(duration_seconds / 60)} minutes"
        else:
            hours = int(duration_seconds / 3600)
            minutes = int((duration_seconds % 3600) / 60)
            duration = f"{hours}h {minutes}m"

        # Get all flows for building issue-flow mapping
        all_flows = await shared_state.get_all_flows()

        # Build issue lookup: (description, url) -> flow_name for efficient mapping
        issue_to_flow: dict[tuple[str, str], str] = {}
        for flow in all_flows:
            for flow_issue in flow.issues:
                key = (flow_issue.get("description", ""), flow_issue.get("url", ""))
                if key not in issue_to_flow:  # Keep first match
                    issue_to_flow[key] = flow.flow_name

        # Get all issues and enrich with flow context
        all_issues = await shared_state.get_all_issues()
        issues = []
        for issue in all_issues:
            issue_dict = {
                "description": issue.description,
                "severity": issue.severity,
                "url": issue.url,
                "context": issue.action_context
            }
            # Look up which flow this issue belongs to
            flow_name = issue_to_flow.get((issue.description, issue.url))
            if flow_name:
                issue_dict["flow_name"] = flow_name
            issues.append(issue_dict)

        # Get completed flows with rich action summaries
        completed_flows = []
        for flow in all_flows:
            if flow.status == FlowStatus.COMPLETED:
                # Create action summary for reproduction steps
                action_summary = [
                    self._summarize_action(action)
                    for action in flow.actions
                ]

                completed_flows.append({
                    "flow_name": flow.flow_name,
                    "completion_reason": flow.completion_reason or "Completed",
                    "action_count": len(flow.actions),
                    "issue_count": len(flow.issues),
                    "urls_visited": flow.urls_visited,
                    "action_summary": action_summary
                })

        # Call AI to generate report
        try:
            result = await self.ai.generate_synthesis_report(
                target_url=shared_state.target_url,
                duration=duration,
                flows_tested=len(completed_flows),
                issues=issues,
                completed_flows=completed_flows,
            )

            # Track token usage by type
            input_tokens = result.get("input_tokens", 0)
            output_tokens = result.get("output_tokens", 0)
            cache_read_tokens = result.get("cache_read_tokens", 0)
            cache_creation_tokens = result.get("cache_creation_tokens", 0)
            if input_tokens or output_tokens or cache_read_tokens or cache_creation_tokens:
                await shared_state.add_token_usage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cache_read_tokens=cache_read_tokens,
                    cache_creation_tokens=cache_creation_tokens,
                )

            report = result.get("report", "")

            # Log the synthesis call
            if shared_state.chat_logger:
                context_summary = (
                    f"Target URL: {shared_state.target_url}\n"
                    f"Duration: {duration}\n"
                    f"Flows tested: {len(completed_flows)}\n"
                    f"Issues found: {len(issues)}"
                )
                shared_state.chat_logger.log_synthesis_call(
                    context=context_summary,
                    report=report,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens
                )

            return report
        except Exception as e:
            # Fallback to simple report if AI fails
            return self._generate_fallback_report(
                shared_state.target_url,
                duration,
                issues,
                completed_flows,
            )

    def _generate_fallback_report(
        self,
        target_url: str,
        duration: str,
        issues: list[dict],
        completed_flows: list[dict],
    ) -> str:
        """Generate a simple report without AI."""
        lines = [
            "# QA Test Report",
            "",
            f"**Target URL:** {target_url}",
            f"**Duration:** {duration}",
            f"**Flows Tested:** {len(completed_flows)}",
            f"**Issues Found:** {len(issues)}",
            "",
        ]

        # Issues by severity
        critical = [i for i in issues if i.get("severity") == "critical"]
        major = [i for i in issues if i.get("severity") == "major"]
        minor = [i for i in issues if i.get("severity") == "minor"]
        cosmetic = [i for i in issues if i.get("severity") == "cosmetic"]

        if critical:
            lines.append("## Critical Issues")
            for issue in critical:
                lines.append(f"- {issue['description']}")
                if issue.get("url"):
                    lines.append(f"  URL: {issue['url']}")
            lines.append("")

        if major:
            lines.append("## Major Issues")
            for issue in major:
                lines.append(f"- {issue['description']}")
                if issue.get("url"):
                    lines.append(f"  URL: {issue['url']}")
            lines.append("")

        if minor:
            lines.append("## Minor Issues")
            for issue in minor:
                lines.append(f"- {issue['description']}")
            lines.append("")

        if cosmetic:
            lines.append("## Cosmetic Issues")
            for issue in cosmetic:
                lines.append(f"- {issue['description']}")
            lines.append("")

        lines.append("## Flows Tested")
        for flow in completed_flows:
            lines.append(f"- {flow['flow_name']}: {flow['completion_reason']}")
        lines.append("")

        if not issues:
            lines.append("**No issues found during testing.**")

        return "\n".join(lines)
