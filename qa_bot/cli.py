#!/usr/bin/env python3
"""CLI entry point for QA Bot - runs exploration without web server.

This provides a command-line interface to the same core exploration logic
used by the web UI. Results can be output as markdown or JSON.

Usage:
    python -m qa_bot.cli https://example.com
    python -m qa_bot.cli https://example.com --goal "Test checkout flow" --max-duration 10
    python -m qa_bot.cli https://example.com --output json --output-file results.json
    python -m qa_bot.cli https://example.com --log-level full  # Show all activity

    # With credentials from file
    python -m qa_bot.cli https://example.com --credentials .env.test

    # With inline credentials
    python -m qa_bot.cli https://example.com --credential "USERNAME=user" --credential "PASSWORD=pass"
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from qa_bot.config import LOG_LEVEL, MAX_CONCURRENT_API_CALLS, DEFAULT_MODEL
from qa_bot.utils.secrets import is_sensitive_field_label

# Configure logging with level from environment
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

from qa_bot.ai import MissingAPIKeyError, create_provider
from qa_bot.orchestrator.coordinator import FlowExplorationOrchestrator
from qa_bot.orchestrator.shared_state import SMOKE_GOAL


LogLevel = Literal["quiet", "summary", "full"]


# The GitHub Actions runner runs any output line that starts with "::"
# (after leading whitespace) as a workflow command: ::add-mask::,
# ::stop-commands::, a fake ::error::. Issue descriptions, verdict titles,
# flow names and errors are model output from site-influenced text.
_LINE_BREAK = re.compile(r"\r\n|\r|\n")


def _neutralise_command_line(line: str) -> str:
    stripped = line.lstrip()
    if stripped.startswith("::"):
        return line[: len(line) - len(stripped)] + ": :" + stripped[2:]
    return line


def log_safe(text) -> str:
    """Model- or site-influenced text made safe to embed in ONE CI log line:
    CR/LF flattened to a space (as entrypoint.sh flattens its copy of the
    verdict titles) so it cannot start a line of its own, and a leading
    ``::`` broken up in case the text opens the line."""
    return _neutralise_command_line(" ".join(_LINE_BREAK.split(str(text or ""))))


def ci_log_lines(text: str) -> str:
    """A formatted log entry with every physical line's leading ``::``
    broken up — the backstop for fields not flattened by log_safe (flow
    names, errors, element labels, reasons)."""
    return "\n".join(_neutralise_command_line(line) for line in _LINE_BREAK.split(text))


def parse_credentials_file(file_path: str) -> dict[str, str]:
    """
    Parse credentials from a file in env format (KEY=value).

    Lines starting with # are ignored.
    Empty lines are ignored.
    """
    credentials = {}
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Credentials file not found: {file_path}")

    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            # Parse KEY=value
            if '=' not in line:
                print(f"Warning: Skipping invalid line {line_num} in {file_path}: missing '='", file=sys.stderr)
                continue
            key, _, value = line.partition('=')
            key = key.strip()
            value = value.strip()
            # Remove quotes if present
            if (value.startswith('"') and value.endswith('"')) or \
               (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]
            credentials[key] = value

    return credentials


def parse_inline_credential(credential_str: str) -> tuple[str, str]:
    """
    Parse a single inline credential in KEY=value format.

    Returns (key, value) tuple.
    """
    if '=' not in credential_str:
        raise ValueError(f"Invalid credential format: {credential_str}. Expected KEY=value")
    key, _, value = credential_str.partition('=')
    return key.strip(), value.strip()


def _issue_attribution(event: dict) -> str:
    """" (worker-2, flow: Checkout, turn 12, source: console)" for an issue
    event — the pointer from a CLI-printed finding to the worker transcript
    turn that produced it. Empty when the event carries no attribution."""
    data = event.get("data", {})
    parts = []
    if event.get("worker_id"):
        parts.append(event["worker_id"])
    if data.get("flow_name"):
        parts.append(f"flow: {data['flow_name']}")
    if data.get("turn") is not None:
        parts.append(f"turn {data['turn']}")
    if data.get("source"):
        parts.append(f"source: {data['source']}")
    return f" ({', '.join(parts)})" if parts else ""


def _issue_label(data: dict, severity: str) -> str:
    """"[severity]" for a new finding. A dedup repeat (``is_new`` False: the
    same text on the same page filed again) is not counted anywhere and
    carries the severity the AI re-filed — printed as "[CRITICAL]" right
    after "[Re-check] NOT REPRODUCED (demoted ...)" it read as a live
    critical — so it is labelled as a repeat, with no bare severity tag."""
    if data.get("is_new") is False:
        return f"(repeat, not counted — filed again as {severity.lower()})"
    return f"[{severity}]"


def _recheck_log_line(event_type: str, data: dict) -> str | None:
    """The CI-log line for an independent re-check event (summary and full)."""
    title = log_safe(data.get("title") or "critical finding")
    if event_type == "recheck_scheduled":
        if data.get("linked"):
            return f"[Re-check] Critical joins a re-check already scheduled: {title}"
        return f"[Re-check] Re-checking critical: {title}"
    if event_type == "recheck_skipped":
        return f"[Re-check] Not re-checked ({log_safe(data.get('reason', ''))}): {title}"
    if event_type == "recheck_started":
        return f"[Re-check] Verifier {data.get('worker_id', '')} started: {title}"
    if event_type == "recheck_result":
        outcome = (data.get("outcome") or "inconclusive").upper().replace("_", " ")
        effect = {
            "NOT REPRODUCED": "demoted to major, tagged UNCONFIRMED",
            "REPRODUCED": "stays critical",
        }.get(outcome, "stays critical")
        reason = log_safe(data.get("reason") or "")[:200]
        return f"[Re-check] {outcome} ({effect}): {title}" + (f" — {reason}" if reason else "")
    return None


def _apply_recheck_result(issues: list[dict], data: dict) -> None:
    """Update the CLI's copies of the issues a re-check settled.

    The result's ``issues`` list (and the gate's raw-count fallback) is
    built from ``issue`` events, which fire before the re-check ends; match
    each settled issue on (original description, url) and take its final
    severity/description, so a not-reproduced critical stops counting.

    An entry below the settled issue's original severity that does not
    already carry its final text is a separate, earlier filing of the same
    text (``SharedFlowState.add_issue`` keeps a critical that raises a
    major one as its own issue); the re-check never covered it.
    """
    for settled in data.get("issues") or []:
        texts = {settled.get("original_description"), settled.get("description")}
        original_severity = (settled.get("recheck") or {}).get("original_severity")
        for entry in issues:
            if entry.get("url") != settled.get("url") or entry.get("description") not in texts:
                continue
            if (original_severity and entry.get("severity") != original_severity
                    and entry.get("description") != settled.get("description")):
                continue
            entry["severity"] = settled.get("severity", entry.get("severity"))
            entry["description"] = settled.get("description", entry.get("description"))
            entry["recheck"] = settled.get("recheck")


def format_event_for_log(event: dict, log_level: LogLevel) -> str | None:
    """
    Format an event for CLI logging based on log level.

    Args:
        event: The event dict from the orchestrator
        log_level: One of 'quiet', 'summary', or 'full'

    Returns:
        Formatted string to print, or None to skip
    """
    event_type = event.get("type", "")
    data = event.get("data", {})

    if log_level == "quiet":
        return None

    recheck_line = _recheck_log_line(event_type, data)
    if recheck_line:
        if log_level == "summary":
            return None if event_type == "recheck_started" else recheck_line
        return f"[{datetime.now().strftime('%H:%M:%S')}] {recheck_line}"

    # Summary level: flow lifecycle and issues only
    if log_level == "summary":
        if event_type == "flow_started":
            flow_name = data.get("flow_name", "Unknown")
            return f"[Flow Started] {flow_name}"
        elif event_type == "flow_completed":
            flow_name = data.get("flow_name", "Unknown")
            reason = data.get("reason", "")
            return f"[Flow Completed] {flow_name}" + (f" - {reason}" if reason else "")
        elif event_type == "flow_failed":
            flow_name = data.get("flow_name", "Unknown")
            error = data.get("error", "Unknown error")
            return f"[Flow Failed] {flow_name}: {error}"
        elif event_type == "flow_retrying":
            flow_name = data.get("flow_name", "Unknown")
            error = data.get("error", "Unknown error")
            return f"[Flow Retrying] {flow_name}: {error}"
        elif event_type == "flow_interrupted":
            flow_name = data.get("flow_name", "Unknown")
            reason = data.get("completion_reason", "")
            return f"[Flow Interrupted] {flow_name}" + (f" - {reason}" if reason else "")
        elif event_type == "issue":
            severity = data.get("severity", "unknown")
            # Issue events carry "description", not "title" — every issue
            # used to log as "Unknown issue" at summary level. The `or` chain
            # also covers a present-but-null/empty description (slicing None
            # would raise inside the formatter).
            title = log_safe(data.get("title") or (data.get("description") or "Unknown issue"))[:100]
            return f"[Issue] {_issue_label(data, severity)} {title}{_issue_attribution(event)}"
        elif event_type == "exploration_complete":
            flows = data.get("flows_explored", 0)
            issues = data.get("issues_found", 0)
            duration = data.get("duration_seconds", 0)
            return f"[Complete] {flows} flows explored, {issues} issues found ({duration:.1f}s)"
        elif event_type == "ai_error":
            error = data.get("error", "Unknown error")
            return f"[AI Error] {error}"
        return None

    # Full level: all events (similar to web UI activity log)
    if log_level == "full":
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Lifecycle events
        if event_type == "exploration_started":
            url = data.get("target_url", "")
            return f"[{timestamp}] Exploration started: {url}"
        elif event_type == "exploration_complete":
            flows = data.get("flows_explored", 0)
            issues = data.get("issues_found", 0)
            duration = data.get("duration_seconds", 0)
            return f"[{timestamp}] Exploration complete: {flows} flows, {issues} issues ({duration:.1f}s)"
        elif event_type == "stopping":
            return f"[{timestamp}] Stopping exploration..."

        # Flow events
        elif event_type == "flow_started":
            flow_name = data.get("flow_name", "Unknown")
            worker_id = data.get("worker_id", "")
            return f"[{timestamp}] [Flow] Started: {flow_name} (worker: {worker_id})"
        elif event_type == "flow_completed":
            flow_name = data.get("flow_name", "Unknown")
            reason = data.get("reason", "")
            return f"[{timestamp}] [Flow] Completed: {flow_name}" + (f" - {reason}" if reason else "")
        elif event_type == "flow_failed":
            flow_name = data.get("flow_name", "Unknown")
            error = data.get("error", "Unknown error")
            return f"[{timestamp}] [Flow] Failed: {flow_name}: {error}"
        elif event_type == "flow_retrying":
            flow_name = data.get("flow_name", "Unknown")
            error = data.get("error", "Unknown error")
            return f"[{timestamp}] [Flow] Retrying: {flow_name}: {error}"
        elif event_type == "flow_interrupted":
            flow_name = data.get("flow_name", "Unknown")
            reason = data.get("completion_reason", "")
            return f"[{timestamp}] [Flow] Interrupted: {flow_name}" + (f" - {reason}" if reason else "")
        elif event_type == "flow_skipped_by_supervisor":
            flow_name = data.get("flow_name", "Unknown")
            reason = data.get("reason", "")
            return f"[{timestamp}] [Flow] Skipped: {flow_name}" + (f" - {reason}" if reason else "")

        # Worker events
        elif event_type == "worker_started":
            worker_id = data.get("worker_id", "")
            return f"[{timestamp}] [Worker] Started: {worker_id}"
        elif event_type == "worker_stopped":
            worker_id = data.get("worker_id", "")
            return f"[{timestamp}] [Worker] Stopped: {worker_id}"
        elif event_type == "worker_error":
            worker_id = data.get("worker_id", "")
            error = data.get("error", "Unknown error")
            return f"[{timestamp}] [Worker] Error ({worker_id}): {error}"

        # Action events
        elif event_type == "action":
            action_type = data.get("action_type", "unknown")
            description = data.get("description", "")
            worker_id = data.get("worker_id", "")
            if action_type in ("left_click", "right_click", "double_click", "triple_click"):
                ref = data.get("ref", "")
                coordinate = data.get("coordinate")
                # Fall back to whatever target info exists — a missing
                # `element` label must not hide the ref/coordinate the model
                # actually targeted (log forensics for coordinate clicks)
                element = data.get("element") or ref or (
                    f"coordinate {tuple(coordinate)}" if coordinate else "unknown element"
                )
                click_name = action_type.replace("_", " ").title()
                suffix = f" ({ref})" if ref and element != ref else ""
                return f"[{timestamp}] [Action] {click_name}: {element}{suffix}"
            elif action_type == "hover":
                ref = data.get("ref", "")
                coordinate = data.get("coordinate")
                element = data.get("element") or ref or (
                    f"coordinate {tuple(coordinate)}" if coordinate else "unknown element"
                )
                return f"[{timestamp}] [Action] Hover: {element}"
            elif action_type == "type":
                element = data.get("element", "unknown element")
                text = data.get("text", "")[:30]  # Truncate long text
                # CI runs this at --log-level full, so a value typed into a
                # password-looking field must not reach workflow stdout
                # (the worker masks known credential values at the source;
                # this catches passwords the AI invented or was told inline)
                if is_sensitive_field_label(element):
                    text = "***"
                return f"[{timestamp}] [Action] Type: '{text}' into {element}"
            elif action_type == "form_input":
                element = data.get("element", "unknown element")
                value = data.get("value", "")
                if is_sensitive_field_label(element):
                    value = "***"
                return f"[{timestamp}] [Action] Form input: {value} into {element}"
            elif action_type == "scroll":
                direction = data.get("scroll_direction", "down")
                return f"[{timestamp}] [Action] Scroll {direction}"
            elif action_type == "scroll_to":
                element = data.get("element", "unknown element")
                return f"[{timestamp}] [Action] Scroll to: {element}"
            elif action_type == "find_text":
                needle = data.get("text", "")
                return f"[{timestamp}] [Action] Find text: '{needle[:40]}'"
            elif action_type == "navigate":
                # The worker's action event carries the destination as
                # target_url (worker.py); "url" is kept as a fallback
                url = data.get("target_url") or data.get("url", "")
                return f"[{timestamp}] [Action] Navigate: {url}"
            elif action_type == "wait":
                duration = data.get("duration", 0)
                return f"[{timestamp}] [Action] Wait {duration}s"
            elif action_type == "key":
                key = data.get("key", "")
                modifiers = data.get("modifiers", [])
                key_str = "+".join(modifiers + [key]) if modifiers else key
                return f"[{timestamp}] [Action] Press key: {key_str}"
            elif action_type == "left_click_drag":
                start = data.get("start_coordinate", [])
                end = data.get("coordinate", [])
                return f"[{timestamp}] [Action] Drag: {start} → {end}"
            elif action_type == "screenshot":
                full_page = data.get("full_page", False)
                return f"[{timestamp}] [Action] Screenshot" + (" (full page)" if full_page else "")
            elif action_type == "zoom":
                region = data.get("region", [])
                return f"[{timestamp}] [Action] Zoom: {region}"
            elif action_type == "resize":
                width = data.get("width", 0)
                height = data.get("height", 0)
                return f"[{timestamp}] [Action] Resize: {width}x{height}"
            elif action_type == "add_flow":
                flow_name = data.get("flow_name", "")
                return f"[{timestamp}] [Action] Add flow: {flow_name}"
            elif action_type == "done":
                reason = data.get("reason", "")
                return f"[{timestamp}] [Action] Done: {reason}"
            elif action_type == "block":
                reason = data.get("reason", "")
                return f"[{timestamp}] [Action] Blocked: {reason}"
            else:
                return f"[{timestamp}] [Action] {action_type}: {description}"

        # Issue events
        elif event_type == "issue":
            severity = data.get("severity", "unknown")
            description = log_safe(data.get("description") or "")[:100]  # Truncate; tolerate null
            # Issue events carry "description", not "title" — every issue
            # used to log a literal "Unknown issue" headline
            title = log_safe(data.get("title")) or description or "Unknown issue"
            detail = description if data.get("title") else ""
            return (
                f"[{timestamp}] [Issue] {_issue_label(data, severity.upper())} {title}{_issue_attribution(event)}"
                + (f"\n           {detail}" if detail else "")
            )

        # Checkpoint events
        elif event_type == "checkpoint_created":
            flow_name = data.get("flow_name", "Unknown")
            return f"[{timestamp}] [Checkpoint] Created for: {flow_name}"
        elif event_type == "checkpoint_claimed":
            flow_name = data.get("flow_name", "Unknown")
            return f"[{timestamp}] [Checkpoint] Claimed: {flow_name}"

        # Supervisor events
        elif event_type == "supervisor_started":
            return f"[{timestamp}] [Supervisor] Started"
        elif event_type == "supervisor_stopped":
            return f"[{timestamp}] [Supervisor] Stopped"
        elif event_type == "supervisor_action":
            action = data.get("action", "unknown")
            target = data.get("target", "")
            return f"[{timestamp}] [Supervisor] Action: {action}" + (f" -> {target}" if target else "")
        elif event_type == "supervisor_reviewing_flows":
            count = data.get("pending_count", 0)
            return f"[{timestamp}] [Supervisor] Reviewing {count} pending flows"
        elif event_type == "supervisor_reviewing_block":
            worker_id = data.get("worker_id", "")
            return f"[{timestamp}] [Supervisor] Reviewing blocked worker: {worker_id}"

        # Approval events
        elif event_type == "approval_request":
            action_desc = data.get("action_description", "Unknown action")
            return f"[{timestamp}] [Approval] Action requires approval: {action_desc}"
        elif event_type == "flow_blocked_for_approval":
            reason = data.get("reason", "")
            return f"[{timestamp}] [Approval] Flow blocked for approval: {reason}"

        # AI thinking events (condensed for CLI)
        elif event_type == "ai_thinking_start":
            worker_id = data.get("worker_id", "")
            return f"[{timestamp}] [AI] Thinking... ({worker_id})"
        elif event_type == "ai_thinking_complete":
            # Skip - we already showed start
            return None
        elif event_type == "ai_thinking_delta":
            # Skip streaming deltas in CLI (too verbose)
            return None

        # AI error events
        elif event_type == "ai_error":
            error = data.get("error", "Unknown error")
            return f"[{timestamp}] [AI Error] {error}"

        # Screenshot events
        elif event_type == "screenshot":
            url = data.get("url", "")[:60]
            index = data.get("index", 0)
            return f"[{timestamp}] [Screenshot] #{index} captured: {url}"

        # Synthesis events
        elif event_type == "synthesis_started":
            return f"[{timestamp}] [Synthesis] Generating report..."
        elif event_type == "synthesis_complete":
            return f"[{timestamp}] [Synthesis] Report complete"
        elif event_type == "synthesis_error":
            error = data.get("error", "Unknown error")
            return f"[{timestamp}] [Synthesis] Error: {error}"

        # Progress events
        elif event_type == "progress":
            active = data.get("active_workers", 0)
            pending = data.get("pending_flows", 0)
            completed = data.get("completed_flows", 0)
            tokens = data.get("tokens_used", 0)
            return f"[{timestamp}] [Progress] Workers: {active} active | Flows: {completed} done, {pending} pending | Tokens: {tokens:,}"

        # Error events
        elif event_type == "error":
            message = data.get("message", "Unknown error")
            fatal = data.get("fatal", False)
            prefix = "FATAL" if fatal else "Error"
            return f"[{timestamp}] [{prefix}] {message}"

        # Unknown events - show in debug
        else:
            return f"[{timestamp}] [{event_type}] {json.dumps(data)[:100]}"

    return None


async def run_exploration(
    url: str,
    goal: str,
    max_agents: int,
    max_duration: int,
    api_key: Optional[str],
    model: str,
    headless: bool = True,
    log_level: LogLevel = "quiet",
    credentials: Optional[dict[str, str]] = None,
    max_cost_usd: float = 5.0,
    skip_permissions: bool = False,
    known_issues: str = "",
    smoke: bool = False,
    previous_report: str = "",
    openrouter_api_key: Optional[str] = None,
    recheck_criticals: Optional[bool] = None,
) -> dict:
    """
    Run QA exploration and return results.

    Args:
        url: Target URL to test
        goal: Testing goal/focus
        max_agents: Maximum parallel agents
        max_duration: Maximum duration in minutes
        api_key: Anthropic API key (needed for Claude models)
        model: Model id — a Claude id, or an OpenRouter "vendor/slug" id
            such as openai/gpt-6-luna (see qa_bot.ai.create_provider)
        headless: Run browser in headless mode
        log_level: One of 'quiet', 'summary', or 'full'
        credentials: Optional dict of credentials (e.g., {"USERNAME": "user", "PASSWORD": "pass"})
        max_cost_usd: Maximum cost in USD (default: $5.00)
        skip_permissions: If True, auto-approve all irreversible actions (dangerous!)
        known_issues: Operator-provided known-issues/environment-caveats text;
            workers skip re-investigating matches and the report lists them as
            one-line "observed again" notes instead of findings
        smoke: Smoke mode — preflight probe, then one turn-capped worker with
            the fixed SMOKE_GOAL (``goal`` is ignored), no flow forking, and
            a report tagged "SMOKE TEST ONLY — not a regression test"
        previous_report: The previous run's report text for the same target;
            synthesis labels each finding NEW / recurring against it and
            lists previous findings not seen again. Never demotes anything.
        openrouter_api_key: OpenRouter API key (needed for "vendor/slug" models)
        recheck_criticals: Independent re-check of CRITICAL findings; None =
            default (``RECHECK_CRITICALS`` env, else on — the CLI is
            non-interactive). A not-reproduced critical arrives as major.

    Returns:
        dict with report, issues, and metadata. ``completed`` is False when the
        run ended without an exploration_complete event (startup failure or a
        fatal crash with nothing salvaged); ``error`` carries the fatal error
        message when one occurred (including salvaged runs).
        ``charged_cost_usd`` is what the provider's API reported charging
        (OpenRouter's usage.cost); None for Claude, which reports no charge.

    Raises:
        MissingAPIKeyError: the chosen model's provider has no key.
    """
    ai_provider = create_provider(
        model,
        anthropic_api_key=api_key,
        openrouter_api_key=openrouter_api_key,
        max_concurrent_calls=MAX_CONCURRENT_API_CALLS,
    )
    orchestrator = FlowExplorationOrchestrator(
        ai_provider=ai_provider,
        max_agents=max_agents,
        headless=headless,
        credentials=credentials,
        skip_permissions=skip_permissions,
        interactive=False  # CLI has no human to respond to blocks
    )

    events = []
    report = None
    verdict = None
    issues = []
    flows_explored = 0
    duration_seconds = 0
    estimated_cost_usd = 0
    token_breakdown = {}
    exploration_completed = False
    fatal_error = None
    recheck_summary = None

    if smoke:
        goal = SMOKE_GOAL  # the orchestrator forces this too; keep the result honest

    if log_level != "quiet":
        print(f"Starting {'SMOKE TEST' if smoke else 'exploration'} of {url}", file=sys.stderr)
        print(f"Goal: {log_safe(goal)}", file=sys.stderr)
        limit_info = f"Max agents: {max_agents}, Max duration: {max_duration} min, Max cost: ${max_cost_usd:.2f}"
        print(limit_info, file=sys.stderr)
        print("-" * 70, file=sys.stderr)

    async for event in orchestrator.run_exploration(
        target_url=url,
        goal=goal,
        max_agents=max_agents,
        max_duration_minutes=max_duration,
        max_cost_usd=max_cost_usd,
        known_issues=known_issues,
        smoke=smoke,
        previous_report=previous_report,
        recheck_criticals=recheck_criticals,
    ):
        events.append(event)
        event_type = event.get("type", "")

        # Capture synthesis report and its curated verdict (None when
        # synthesis fell back to a deterministic/non-AI report)
        if event_type == "synthesis_complete":
            report = event.get("data", {}).get("report", "")
            verdict = event.get("data", {}).get("verdict")

        # Capture issues. A dedup repeat (is_new False: the same description
        # on the same page, filed again) is not a new finding: counting it
        # double-counted repeats in issues_found and the raw-count gate, and
        # it carries the severity the AI re-filed, so a repeat arriving after
        # its re-check demoted the original re-counted it as critical.
        if event_type == "issue":
            data = event.get("data", {})
            if data.get("is_new") is not False:
                issues.append(data)
        elif event_type == "recheck_result":
            _apply_recheck_result(issues, event.get("data", {}))

        # Capture final summary
        if event_type == "exploration_complete":
            exploration_completed = True
            data = event.get("data", {})
            flows_explored = data.get("flows_explored", 0)
            duration_seconds = data.get("duration_seconds", 0)
            final_progress = data.get("final_progress") or {}
            estimated_cost_usd = final_progress.get("cost_usd", 0)
            token_breakdown = final_progress.get("token_breakdown", {})
            recheck_summary = data.get("recheck_summary")

        # Capture fatal errors. The orchestrator yields these instead of
        # raising (so SSE/CLI consumers get a structured event), which means
        # the generator can end "normally" after a total failure — track it
        # here so the caller can fail loudly instead of reporting success.
        if event_type == "error" and event.get("data", {}).get("fatal"):
            fatal_error = event.get("data", {}).get("message") or "Unknown fatal error"

        # Log event based on log level
        log_line = format_event_for_log(event, log_level)
        if log_line:
            print(ci_log_lines(log_line), file=sys.stderr)

    if log_level != "quiet":
        print("-" * 70, file=sys.stderr)

    charged_cost_usd = getattr(ai_provider, "charged_cost_usd", None)

    return {
        "report": report or "No report generated",
        "issues": issues,
        "issues_found": len(issues),
        # Curated verdict from the synthesis report. The exit code below and
        # the action's gate use curated_critical_count when it is non-null
        # (the report is the source of truth) and fall back to the RAW
        # "issues" list when it is null (no AI verdict: fallback / NOT
        # TESTED report). Both numbers stay visible side by side.
        "curated_critical_count": verdict["critical_count"] if verdict else None,
        "curated_critical_titles": (
            [c["title"] for c in verdict["critical"]] if verdict else None
        ),
        "curated_verdict": verdict,
        # Run-over-run labels from the verdict (None when no previous report
        # was supplied, or synthesis produced no verdict).
        "new_findings_count": verdict.get("new") if verdict else None,
        "recurring_findings_count": verdict.get("recurring") if verdict else None,
        # Independent re-check of criticals: per-issue outcome counts
        # ({"reproduced", "not_reproduced", "inconclusive", "not_rechecked"}),
        # None when no critical was a candidate. `issues` above already
        # carries each settled issue's final severity.
        "recheck_summary": recheck_summary,
        "flows_explored": flows_explored,
        "duration_seconds": round(duration_seconds, 2),
        "estimated_cost_usd": round(estimated_cost_usd, 4),
        "charged_cost_usd": (
            round(charged_cost_usd, 4)
            if isinstance(charged_cost_usd, (int, float)) else None
        ),
        "tokens": token_breakdown,
        "completed": exploration_completed,
        "error": fatal_error,
        "target_url": url,
        "goal": goal,
        "mode": "smoke" if smoke else "full",
        "timestamp": datetime.now().isoformat()
    }


def curated_excess_warning(curated: Optional[int], still_critical: int) -> Optional[str]:
    """A ``::warning::`` line (a GitHub Actions annotation) when the curated
    verdict lists more criticals than the run has issues still critical
    after the severity guard and the independent re-check, else None.

    Synthesis may legitimately merge majors into one critical, but a
    finding the re-check did not reproduce, re-titled so the verdict
    backstop could not match it, looks exactly the same — and fails the
    gate. Numbers only: titles are model output and stay out of a
    workflow-command line.
    """
    if curated is None or curated <= still_critical:
        return None
    return (
        f"::warning::The curated report lists {curated} critical finding(s) but only "
        f"{still_critical} worker finding(s) are still critical after the severity "
        "guard and the independent re-check; check that no Critical Issues entry "
        "is a finding the re-check did not reproduce (tagged UNCONFIRMED) under a new title"
    )


def load_previous_report(value: str) -> str:
    """Resolve ``--previous-report``: a path to a report file, or inline text.

    The GitHub Action passes the previous run's report.md inline (the mono
    wrapper runs the CLI in a container that can't see the runner's files);
    a local user passes a path. A value naming an existing file is read,
    anything else is the report itself. Empty in → empty out.
    """
    if not value or not value.strip():
        return ""
    candidate = value.strip()
    if "\n" not in candidate and os.path.isfile(candidate):
        with open(candidate, encoding="utf-8") as f:
            return f.read()
    if "\n" not in candidate and (candidate.endswith(".md") or "/" in candidate) and len(candidate) < 512:
        # Looks like a path that does not exist: a typo or wrong cwd. Do not
        # feed the path string to synthesis as if it were the previous report.
        print(
            f"Warning: --previous-report '{candidate}' is not an existing file; "
            "ignoring it (pass the report text inline or a valid path).",
            file=sys.stderr,
        )
        return ""
    return value


# Per-mode defaults for the limit flags (None on the argparse side so an
# explicit value always wins over the mode default).
LIMIT_DEFAULTS = {
    "full": {"max_agents": 3, "max_duration": 30, "max_cost": 5.0},
    "smoke": {"max_agents": 1, "max_duration": 5, "max_cost": 0.75},
}


def resolve_limits(args) -> tuple[int, int, float]:
    """Return (max_agents, max_duration, max_cost) for the parsed args.

    Each limit is the explicit flag value when given, else the default for
    the run mode: 3 agents / 30 min / $5.00 for a full run, 1 / 5 / $0.75
    for ``--smoke``.
    """
    defaults = LIMIT_DEFAULTS["smoke" if getattr(args, "smoke", False) else "full"]
    return (
        args.max_agents if args.max_agents is not None else defaults["max_agents"],
        args.max_duration if args.max_duration is not None else defaults["max_duration"],
        args.max_cost if args.max_cost is not None else defaults["max_cost"],
    )


def main():
    parser = argparse.ArgumentParser(
        description="QA Bot - AI-powered website testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage (quiet, just outputs report)
    python -m qa_bot.cli https://example.com

    # With custom goal
    python -m qa_bot.cli https://example.com --goal "Test the checkout flow"

    # Quick exploration with fewer agents
    python -m qa_bot.cli https://example.com --max-agents 2 --max-duration 5

    # JSON output to file
    python -m qa_bot.cli https://example.com --output json --output-file results.json

    # Show flow-level progress (summary)
    python -m qa_bot.cli https://example.com --log-level summary

    # Show full activity log (like web UI)
    python -m qa_bot.cli https://example.com --log-level full

    # Shorthand for summary logging
    python -m qa_bot.cli https://example.com -v

    # Shorthand for full logging
    python -m qa_bot.cli https://example.com -vv

    # Smoke test only: one worker opens each top-level nav link once
    # (defaults: 1 agent, 5 min, $0.75; report tagged SMOKE TEST ONLY)
    python -m qa_bot.cli https://example.com --smoke
        """
    )

    parser.add_argument(
        "url",
        help="Target URL to test"
    )
    parser.add_argument(
        "--goal", "-g",
        default="Explore user flows and find bugs, broken elements, or usability problems.",
        help="Testing goal/focus (default: general exploration)"
    )
    parser.add_argument(
        "--known-issues",
        default="",
        help=(
            "Known issues / environment caveats the team has already "
            "acknowledged (free text, e.g. one per line). Workers won't "
            "re-investigate matching observations and the report lists them "
            "as one-line 'observed again' notes instead of new findings."
        )
    )
    parser.add_argument(
        "--previous-report",
        default="",
        help=(
            "The previous run's report for the same target — a path to its "
            "report.md, or the markdown itself. The report labels each finding "
            "NEW or 'recurring (seen in previous run)' and lists previous "
            "findings not observed this run. Labels only: severity is never "
            "changed and nothing is suppressed (use --known-issues for that)."
        )
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Smoke test only: preflight probe, then a single turn-capped "
            "worker loads the target and opens each top-level navigation "
            "link once (no sign-up/login/forms, no flow forking). --goal is "
            "ignored; the report is tagged 'SMOKE TEST ONLY — not a "
            "regression test'. Defaults become 1 agent, 5 min, $0.75 "
            "unless overridden."
        )
    )
    parser.add_argument(
        "--no-recheck-criticals",
        dest="recheck_criticals",
        action="store_false",
        default=None,
        help=(
            "Disable the independent re-check of CRITICAL findings. By "
            "default each new critical gets a short verification flow in a "
            "fresh browser; one it does not reproduce is demoted to major "
            "and tagged UNCONFIRMED (also: RECHECK_CRITICALS=false)."
        )
    )
    parser.add_argument(
        "--max-agents", "-a",
        type=int,
        default=None,
        help="Maximum parallel agents (default: 3; 1 with --smoke)"
    )
    parser.add_argument(
        "--max-cost",
        type=float,
        default=None,
        help="Maximum cost in USD (default: $5.00; $0.75 with --smoke)"
    )
    parser.add_argument(
        "--max-duration", "-d",
        type=int,
        default=None,
        help="Maximum duration in minutes (default: 30; 5 with --smoke)"
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)"
    )
    parser.add_argument(
        "--openrouter-api-key",
        default=None,
        help=(
            "OpenRouter API key, needed only for an OpenRouter model id "
            "(or set OPENROUTER_API_KEY env var)"
        )
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            f"Model to use (default: {DEFAULT_MODEL}). A 'vendor/slug' id "
            "such as openai/gpt-6-luna runs on OpenRouter and needs "
            "--openrouter-api-key; screenshots and prompts then go to "
            "OpenRouter and the model vendor"
        )
    )
    parser.add_argument(
        "--testmail-api-key",
        default=None,
        help="Testmail.app API key for email verification testing (or set TESTMAIL_API_KEY env var)"
    )
    parser.add_argument(
        "--testmail-namespace",
        default=None,
        help="Testmail.app namespace (or set TESTMAIL_NAMESPACE env var)"
    )
    parser.add_argument(
        "--output", "-o",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format (default: markdown)"
    )
    parser.add_argument(
        "--output-file", "-f",
        help="Write output to file instead of stdout"
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Run browser in headed mode (visible, for debugging)"
    )
    parser.add_argument(
        "--log-level", "-l",
        choices=["quiet", "summary", "full"],
        default="quiet",
        help="Log verbosity: quiet (default), summary (flow events), full (all activity)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Increase log verbosity (-v for summary, -vv for full)"
    )
    parser.add_argument(
        "--credentials", "-c",
        help="Path to credentials file (env format: KEY=value per line)"
    )
    parser.add_argument(
        "--credential",
        action="append",
        dest="inline_credentials",
        metavar="KEY=VALUE",
        help="Inline credential (can be specified multiple times)"
    )
    parser.add_argument(
        "--dangerously-skip-permissions",
        action="store_true",
        help="Auto-approve all irreversible actions (payments, deletions, etc.). Use with caution."
    )

    args = parser.parse_args()

    # Determine log level from --log-level or -v flags
    if args.log_level != "quiet":
        log_level = args.log_level
    elif args.verbose >= 2:
        log_level = "full"
    elif args.verbose == 1:
        log_level = "summary"
    else:
        log_level = "quiet"

    # API keys from args or environment; only the chosen model's provider
    # needs one (create_provider raises MissingAPIKeyError otherwise).
    api_key = args.api_key or os.getenv("ANTHROPIC_API_KEY")
    openrouter_api_key = args.openrouter_api_key or os.getenv("OPENROUTER_API_KEY")

    # Parse credentials
    credentials = {}
    if args.credentials:
        try:
            credentials.update(parse_credentials_file(args.credentials))
            if log_level != "quiet":
                print(f"Loaded {len(credentials)} credentials from {args.credentials}", file=sys.stderr)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error parsing credentials file: {e}", file=sys.stderr)
            sys.exit(1)

    if args.inline_credentials:
        for cred_str in args.inline_credentials:
            try:
                key, value = parse_inline_credential(cred_str)
                credentials[key] = value
            except ValueError as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)
        if log_level != "quiet":
            print(f"Added {len(args.inline_credentials)} inline credentials", file=sys.stderr)

    # Set Testmail.app credentials if provided (coordinator reads from env)
    testmail_key = args.testmail_api_key or os.getenv("TESTMAIL_API_KEY")
    testmail_namespace = args.testmail_namespace or os.getenv("TESTMAIL_NAMESPACE")
    if testmail_key and testmail_namespace:
        os.environ["TESTMAIL_API_KEY"] = testmail_key
        os.environ["TESTMAIL_NAMESPACE"] = testmail_namespace
        if log_level != "quiet":
            print("Email testing enabled (Testmail.app)", file=sys.stderr)
    elif testmail_key or testmail_namespace:
        print("Warning: Both --testmail-api-key and --testmail-namespace are required for email testing", file=sys.stderr)

    max_agents, max_duration, max_cost = resolve_limits(args)
    try:
        previous_report = load_previous_report(args.previous_report)
    except OSError as e:
        print(f"Error reading previous report: {e}", file=sys.stderr)
        sys.exit(1)
    if previous_report and log_level != "quiet":
        print(
            f"Previous report loaded ({len(previous_report)} chars) — findings "
            "will be labelled NEW / recurring",
            file=sys.stderr,
        )
    if args.smoke and args.goal != parser.get_default("goal") and log_level != "quiet":
        print("Note: --smoke uses a fixed smoke goal; --goal is ignored", file=sys.stderr)

    # Run exploration
    try:
        result = asyncio.run(run_exploration(
            url=args.url,
            goal=args.goal,
            known_issues=args.known_issues,
            max_agents=max_agents,
            max_duration=max_duration,
            api_key=api_key,
            model=args.model,
            headless=not args.headed,
            log_level=log_level,
            credentials=credentials if credentials else None,
            max_cost_usd=max_cost,
            skip_permissions=args.dangerously_skip_permissions,
            smoke=args.smoke,
            previous_report=previous_report,
            openrouter_api_key=openrouter_api_key,
            recheck_criticals=args.recheck_criticals,
        ))
    except MissingAPIKeyError as e:
        flag = "--openrouter-api-key" if e.env_var == "OPENROUTER_API_KEY" else "--api-key"
        print(f"Error: {e}", file=sys.stderr)
        print(f"Set it via {flag} or the {e.env_var} environment variable", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nExploration cancelled", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Error during exploration: {e}", file=sys.stderr)
        sys.exit(1)

    # A run that never produced exploration_complete failed outright
    # (Playwright launch failure, fatal crash with nothing salvaged, ...).
    # Fail loudly and do NOT write a result file: CI consumers (e.g. the
    # GitHub Action entrypoint) treat the result file's existence as success,
    # so writing one here would turn a total failure into a green run with
    # an empty report.
    if not result.get("completed"):
        message = result.get("error") or "exploration ended before completing"
        print(f"Error: QA exploration failed: {message}", file=sys.stderr)
        sys.exit(2)

    # Format output
    if args.output == "json":
        output = json.dumps(result, indent=2, default=str)
    else:
        output = result["report"]

    # Write output
    if args.output_file:
        with open(args.output_file, "w") as f:
            f.write(output)
        if log_level != "quiet":
            print(f"Output written to {args.output_file}", file=sys.stderr)
    else:
        print(output)

    # Exit 1 on critical issues. Gate on the CURATED count from the synthesis
    # verdict when present (the report is the source of truth — a raw worker
    # critical that synthesis downgraded must not fail the run), else on the
    # RAW worker-reported count (synthesis fell back to a non-AI report).
    # Mirrors action/entrypoint.sh so CLI and action agree.
    raw_critical_count = sum(
        1 for i in result["issues"]
        if i.get("severity", "").lower() == "critical"
    )
    curated = result.get("curated_critical_count")
    gate_count = curated if curated is not None else raw_critical_count
    if curated is not None and curated < raw_critical_count and log_level != "quiet":
        # The curated verdict is AI-written from site-influenced text; a
        # downgrade must be visible wherever the exit code is read.
        print(
            f"Warning: synthesis downgraded {raw_critical_count - curated} of "
            f"{raw_critical_count} raw critical finding(s); review the report's "
            "Likely False Positives / Known Issues sections before trusting a pass.",
            file=sys.stderr,
        )
    warning = curated_excess_warning(curated, raw_critical_count)
    if warning and log_level != "quiet":
        print(warning, file=sys.stderr)
    if gate_count > 0:
        if log_level != "quiet":
            if curated is not None:
                titles = result.get("curated_critical_titles") or []
                titles_note = f": {' | '.join(log_safe(t) for t in titles)}" if titles else ""
                print(
                    f"Found {curated} critical issue(s) after curation{titles_note} "
                    f"({raw_critical_count} raw worker finding(s))",
                    file=sys.stderr,
                )
            else:
                print(
                    f"Found {raw_critical_count} critical issue(s) "
                    f"(raw worker findings; synthesis produced no curated verdict)",
                    file=sys.stderr,
                )
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
