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

        Handles the current JSON action format with action_type, ref, text, etc.
        """
        success = action.get("success", True)
        url = action.get("page_url", action.get("url", ""))

        description = self._describe_action(action)

        return {
            "description": description,
            "success": success,
            "url": url,
        }

    def _describe_action(self, action: dict) -> str:
        """
        Create a human-readable description from an action dict.

        Handles the structured action format:
        {"action_type": "left_click", "ref": "ref_5", "reasoning": "...", ...}
        """
        action_type = action.get("action_type", "")
        reasoning = action.get("reasoning", "")
        ref = action.get("ref", "")
        element = action.get("element", "")

        if not action_type:
            # Fallback for legacy code-based actions
            code = action.get("code", "")
            return self._extract_action_description_from_code(code) if code else "Unknown action"

        # Stringify ref in case it's an integer
        if ref and not isinstance(ref, str):
            ref = str(ref)

        # Click actions
        if action_type in ("left_click", "right_click", "double_click", "triple_click"):
            click_name = action_type.replace("_", " ").title()
            target = element or ref or "element"
            return f"{click_name} {target}"

        # Hover
        if action_type == "hover":
            target = element or ref or "element"
            return f"Hover over {target}"

        # Type text
        if action_type == "type":
            text = action.get("text", "")
            target = element or ref or "field"
            # Mask sensitive fields (passwords, tokens, etc.)
            sensitive_keywords = [
                'password', 'pwd', 'secret', 'token', 'key', 'auth',
                'credential', 'pin', 'api', 'private', 'ssn', 'cvv'
            ]
            target_lower = (element or ref or "").lower()
            if any(s in target_lower for s in sensitive_keywords):
                text = "***"
            elif len(text) > 30:
                text = text[:27] + "..."
            return f"Type '{text}' into {target}"

        # Form input
        if action_type == "form_input":
            value = action.get("value", "")
            target = element or ref or "field"
            return f"Set {target} to '{value}'"

        # Scroll
        if action_type == "scroll":
            direction = action.get("scroll_direction", "down")
            amount = action.get("scroll_amount", 3)
            return f"Scroll {direction} ({amount} ticks)"

        # Scroll to element
        if action_type == "scroll_to":
            target = element or ref or "element"
            return f"Scroll to {target}"

        # Key press
        if action_type == "key":
            key = action.get("key", "")
            modifiers = action.get("modifiers", [])
            if modifiers:
                return f"Press {'+'.join(modifiers)}+{key}"
            return f"Press {key} key"

        # Navigate
        if action_type == "navigate":
            target_url = action.get("target_url", action.get("url", ""))
            if target_url in ("back", "forward"):
                return f"Navigate {target_url}"
            return f"Navigate to {target_url[:50]}" if target_url else "Navigate"

        # Wait
        if action_type == "wait":
            duration = action.get("duration", "")
            return f"Wait {duration}s" if duration else "Wait for page update"

        # Resize
        if action_type == "resize":
            width = action.get("width", "?")
            height = action.get("height", "?")
            return f"Resize viewport to {width}x{height}"

        # Screenshot/Zoom
        if action_type == "screenshot":
            return "Take screenshot"
        if action_type == "zoom":
            return "Zoom into region for inspection"

        # Done
        if action_type == "done":
            reason = action.get("reason", "")
            return f"Completed: {reason[:60]}" if reason else "Flow completed"

        # Block
        if action_type == "block":
            reason = action.get("reason", "")
            return f"Blocked: {reason[:60]}" if reason else "Blocked"

        # Report issue
        if action_type == "report_issue":
            desc = action.get("issue_description", "")
            return f"Reported issue: {desc[:50]}" if desc else "Reported issue"

        # Add flow
        if action_type == "add_flow":
            flow_name = action.get("flow_name", "")
            return f"Created flow: {flow_name}" if flow_name else "Created new flow"

        # Request data
        if action_type == "request_data":
            name = action.get("request_name", "")
            return f"Requested data: {name}" if name else "Requested user data"

        # Close popup
        if action_type == "close_popup":
            return "Close popup window"

        # Fallback: use reasoning or action_type
        if reasoning:
            return reasoning[:60]
        return action_type.replace("_", " ").title()

    def _extract_action_description_from_code(self, code: str) -> str:
        """
        Legacy fallback: extract description from Playwright-style action code.
        """
        if not code:
            return "Unknown action"

        code = code.strip()

        if match := re.search(r'qa\.done\(["\'](.+?)["\']\)', code):
            return f"Completed: {match.group(1)[:60]}"
        if match := re.search(r'qa\.block\(["\'](.+?)["\']\)', code):
            return f"Blocked: {match.group(1)[:60]}"
        if match := re.search(r'qa\.report_issue\(["\'](.+?)["\']', code):
            return f"Reported issue: {match.group(1)[:50]}"
        if match := re.search(r'qa\.add_flow\(["\'](.+?)["\']', code):
            return f"Created flow: {match.group(1)}"
        if match := re.search(r'page\.goto\(["\'](.+?)["\']\)', code):
            return f"Navigate to {match.group(1)[:50]}"

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
