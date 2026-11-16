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

Exit codes (the table is in --help): 0 tested and no critical finding gates,
1 critical findings, 2 no verdict (aborted, nothing tested, or a
configuration or run error), 130 interrupted.
"""

import argparse
import asyncio
import json
import logging
import os
import re
import stat
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from qa_bot import __version__
from qa_bot.config import (
    DEFAULT_CLAUDE_MODEL,
    DEFAULT_OPENROUTER_MODEL,
    LOG_LEVEL,
    MAX_CONCURRENT_API_CALLS,
    MAX_COST_CAP_USD,
    ModelChoice,
    is_openrouter_model,
    resolve_model,
)
from qa_bot.utils.secrets import is_sensitive_field_label

# Configure logging with level from environment
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

from qa_bot.ai import MissingAPIKeyError, create_provider
from qa_bot.gate import EXIT_NO_VERDICT, EXIT_PASS, NO_REPORT, evaluate_gate
from qa_bot.orchestrator.coordinator import FlowExplorationOrchestrator
from qa_bot.orchestrator.shared_state import SMOKE_GOAL
from qa_bot.run_inputs import (
    MAX_AGENTS_CAP,
    check_max_agents,
    check_max_cost,
    check_max_duration,
    normalize_target_url,
)


LogLevel = Literal["quiet", "summary", "full"]

# The HTTP clients log one INFO line per request ("HTTP Request: POST …
# 200 OK"): httpx for OpenRouter and the preflight probe, httpx2 inside the
# Anthropic SDK. That is one stderr line per AI call, hundreds in a full run.
_HTTP_REQUEST_LOGGERS = ("httpx", "httpx2")


def quiet_http_request_logs(log_level: LogLevel) -> None:
    """Keep the per-request HTTP lines out of a ``quiet`` or ``summary`` run.

    ``--log-level full`` keeps them (the Action runs at full, and people
    debug from that log), and so does ``LOG_LEVEL=DEBUG``. Keyed on the
    CLI's own flag, not on whether LOG_LEVEL is set: .env.example sets
    LOG_LEVEL=INFO, so "set explicitly" is true for most installs. Only
    these two loggers change. Warnings and errors still print, and the
    ``Model:`` line is a print, not a log record, so no level can hide it.
    """
    if log_level == "full" or LOG_LEVEL == "DEBUG":
        return
    for name in _HTTP_REQUEST_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


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
                element = data.get("element") or data.get("ref") or "unknown element"
                # One log line, cut to 30 chars
                text = log_safe(data.get("text", ""))[:30]
                # CI runs this at --log-level full, so a value typed into a
                # password-looking field must not reach workflow stdout
                # (the worker masks known credential values at the source;
                # this catches passwords the AI invented or was told inline)
                if is_sensitive_field_label(element):
                    text = "***"
                return f"[{timestamp}] [Action] Type: '{text}' into {element}"
            elif action_type == "form_input":
                # A model may give a ref and no element label, so fall back
                # to the ref like the click lines
                element = data.get("element") or data.get("ref") or "unknown element"
                value = data.get("value", "")
                if not isinstance(value, (bool, int, float)):
                    # A select option, a date, free text or a list of options:
                    # one log line, cut like typed text (a checkbox's True or
                    # a number as it is)
                    value = log_safe(str(value))[:30]
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
    testmail_api_key: Optional[str] = None,
    testmail_namespace: Optional[str] = None,
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
        testmail_api_key, testmail_namespace: Testmail.app credentials for
            the run's EmailService (experimental: nothing reads email yet).

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
    log_format_failed = False

    if smoke:
        goal = SMOKE_GOAL  # the orchestrator forces this too; keep the result honest

    if log_level != "quiet":
        print(f"QA Bot {__version__}: starting {'SMOKE TEST' if smoke else 'exploration'} of {url}", file=sys.stderr)
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
        testmail_api_key=testmail_api_key,
        testmail_namespace=testmail_namespace,
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

        # Log event based on log level. The log is a view of the run, not
        # part of it: an event the formatter cannot handle (a field of an
        # unexpected type, a new event shape) must cost one log line, not
        # the run. Unguarded, the exception left this loop and discarded
        # everything the run had found and been charged for.
        try:
            log_line = format_event_for_log(event, log_level)
            if log_line:
                print(ci_log_lines(log_line), file=sys.stderr)
        except Exception as e:
            if not log_format_failed:
                log_format_failed = True
                print(
                    f"Warning: could not format a '{log_safe(event_type)}' event for the log "
                    f"({type(e).__name__}: {log_safe(e)}); the run continues, and further "
                    "log formatting failures are not reported",
                    file=sys.stderr,
                )

    if log_level != "quiet":
        print("-" * 70, file=sys.stderr)

    charged_cost_usd = getattr(ai_provider, "charged_cost_usd", None)

    return {
        "report": report or NO_REPORT,
        "issues": issues,
        "issues_found": len(issues),
        # Curated verdict from the synthesis report. The gate
        # (qa_bot.gate.evaluate_gate, and the action's own) uses
        # curated_critical_count when it is non-null (the report is the
        # source of truth) and falls back to the RAW "issues" list when it
        # is null (no AI verdict: fallback / NOT TESTED report). Both
        # numbers stay visible side by side.
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
        # The model that ran — it decides where screenshots, page text and
        # credentials went (summary.json carries the same field).
        "model": model,
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


def read_known_issues_file(path: str) -> str:
    """The entries of a ``--known-issues-file``, in the documented
    ``.qa-known-issues.md`` format: one issue per line, with ``#`` comment
    lines and blank lines dropped (what the workflows' ``grep -v`` pair
    does before they pass the rest inline).

    A file has its own flag, where ``--previous-report`` takes a path or
    text in one, on purpose. ``--known-issues`` is what the Action passes
    its ``known-issues`` input to, and a one-line value there can come from
    less trusted text than the workflow file. If such a value could name a
    file, it would be read inside the Action's container and sent to the
    model as a known issue, and from there into the report.
    """
    with open(path, encoding="utf-8") as f:
        return "\n".join(
            line for line in f.read().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )


def is_a_file_name_not_an_issue(known_issues: str) -> bool:
    """True when the ``--known-issues`` text is unmistakably a file name:
    one word ending in .md or .txt (``.qa-known-issues.md``). The text is
    still sent as it is; ``main`` warns that ``--known-issues-file`` was
    probably meant. A one-line known issue often contains a ``/`` ("The
    /pricing page shows the old plan name"), so a slash alone says nothing.
    """
    candidate = known_issues.strip()
    return (
        bool(candidate)
        and not any(c.isspace() for c in candidate)
        and candidate.endswith((".md", ".txt"))
    )


# Per-mode defaults for the limit flags (None on the argparse side so an
# explicit value always wins over the mode default).
LIMIT_DEFAULTS = {
    "full": {"max_agents": 3, "max_duration": 30, "max_cost": 5.0},
    "smoke": {"max_agents": 1, "max_duration": 5, "max_cost": 0.75},
}


def resolve_limits(args) -> tuple[int, int, float]:
    """Return (max_agents, max_duration, max_cost) for the parsed args: the
    values the run will use.

    Each limit is the explicit flag value when given, else the default for
    the run mode: 3 agents / 30 min / $5.00 for a full run, 1 / 5 / $0.75
    for ``--smoke``. They are checked by the rule the web API uses
    (qa_bot.run_inputs): zero, a negative number or NaN raises ValueError
    (it used to start a run that could test nothing), and a value above the
    cap is lowered to it with a note on stderr, where the coordinator used
    to lower it silently after the banner had printed the larger number.
    """
    defaults = LIMIT_DEFAULTS["smoke" if getattr(args, "smoke", False) else "full"]
    asked_agents = args.max_agents if args.max_agents is not None else defaults["max_agents"]
    asked_duration = args.max_duration if args.max_duration is not None else defaults["max_duration"]
    asked_cost = args.max_cost if args.max_cost is not None else defaults["max_cost"]

    max_agents = check_max_agents(asked_agents, "--max-agents")
    max_duration = check_max_duration(asked_duration, "--max-duration")
    max_cost = check_max_cost(asked_cost, "--max-cost")
    if max_agents != asked_agents:
        print(
            f"Note: --max-agents {asked_agents} is above the maximum of "
            f"{MAX_AGENTS_CAP}; this run uses {max_agents}",
            file=sys.stderr,
        )
    if max_cost != asked_cost:
        print(
            f"Note: --max-cost {asked_cost:g} is above the maximum of "
            f"${max_cost:.2f}; this run uses ${max_cost:.2f}",
            file=sys.stderr,
        )
    return max_agents, max_duration, max_cost


def prepare_output_file(path: str) -> Optional[str]:
    """Get ``--output-file`` ready before the run is paid for. Returns what
    is wrong with the path, or None when the result can be written there.

    An earlier run's result at the path is removed: a run that then fails
    writes no result file, and a wrapper that looks for the file would read
    the previous run's (possibly green) result as this one's. Nothing is
    created: an empty file would read as a result too.

    Only a regular file is removed. A device, a FIFO or a symlink
    (``/dev/stdout``) is written in place and only has to be writable. So
    is a regular file that cannot be removed but can be written (one in a
    read-only directory, a Docker single-file bind mount): it is emptied,
    which is at least not the earlier result.
    """
    if os.path.isdir(path):
        return "it is a directory"
    if os.path.exists(path):
        if os.path.islink(path) or not os.path.isfile(path):
            return None if os.access(path, os.W_OK) else "it is not writable"
        try:
            os.unlink(path)
        except OSError as e:
            try:
                os.truncate(path, 0)
            except OSError:
                return f"an earlier result is there and could not be removed or emptied ({e.strerror})"
        return None
    directory = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(directory):
        return f"the directory {directory} does not exist"
    if not os.access(directory, os.W_OK):
        return f"the directory {directory} is not writable"
    return None


class InputFileError(Exception):
    """A file the run was told to read cannot be used. ``str()`` is the
    line for stderr."""


def read_input_files(args) -> tuple[dict[str, str], str, str]:
    """Everything a run reads from a file the caller named: (the
    ``--credentials`` file's credentials, the ``--known-issues-file``
    entries, the previous report). Raises InputFileError for the first one
    that cannot be used.

    One function so that ``main`` can read them all before it prepares
    ``--output-file``, which removes what is at that path.
    """
    credentials = {}
    if args.credentials:
        try:
            credentials = parse_credentials_file(args.credentials)
        except FileNotFoundError as e:
            raise InputFileError(f"Error: {e}") from e
        except Exception as e:
            raise InputFileError(f"Error parsing credentials file: {e}") from e
    # UnicodeDecodeError too: a binary file named by mistake was a traceback
    # and exit 1, the "critical findings" code.
    known_issues_entries = ""
    if args.known_issues_file:
        try:
            known_issues_entries = read_known_issues_file(args.known_issues_file)
        except (OSError, UnicodeDecodeError) as e:
            raise InputFileError(f"Error reading known issues: {e}") from e
    try:
        previous_report = load_previous_report(args.previous_report)
    except (OSError, UnicodeDecodeError) as e:
        raise InputFileError(f"Error reading previous report: {e}") from e
    return credentials, known_issues_entries, previous_report


def resolve_cli_model(args) -> ModelChoice:
    """The model a CLI run uses, and why.

    ``--model`` wins. Otherwise a key passed by flag says which provider the
    caller means: with exactly one of ``--api-key`` / ``--openrouter-api-key``
    that provider's default runs, whatever other key the environment or
    .env holds — ``--api-key sk-ant-…`` must not run GPT-6 Luna on an
    ambient OpenRouter key and send page data to OpenAI. Everything else
    (no key flag, or both) is resolve_model's rule over all the keys at
    hand. The action passes its keys by environment, never by flag, so it
    always takes that last path.
    """
    if not (args.model or "").strip():
        if args.api_key and not args.openrouter_api_key:
            return ModelChoice(DEFAULT_CLAUDE_MODEL, "default for the Anthropic key passed with --api-key")
        if args.openrouter_api_key and not args.api_key:
            return ModelChoice(
                DEFAULT_OPENROUTER_MODEL, "default for the OpenRouter key passed with --openrouter-api-key",
            )
    return resolve_model(
        args.model,
        bool(args.openrouter_api_key or os.getenv("OPENROUTER_API_KEY")),
        bool(args.api_key or os.getenv("ANTHROPIC_API_KEY")),
    )


def model_notice(choice: ModelChoice) -> str:
    """The stderr line(s) naming the model a run uses and why, printed at
    every log level, quiet included.

    An OPENROUTER_API_KEY exported for some other tool, or found in a .env,
    switches a run without flags to GPT-6 Luna. So an OpenRouter model also
    gets a line saying where the data goes. Without it, a run could send
    screenshots, page text and test credentials to OpenRouter and OpenAI
    without anyone having chosen that.
    """
    notice = f"Model: {choice.model} ({choice.reason})"
    if is_openrouter_model(choice.model):
        vendor = choice.model.split("/", 1)[0]
        notice += (
            "\nNote: screenshots, page text and test credentials go to OpenRouter "
            f"and the model vendor ({vendor}), not Anthropic; pass --model "
            f"{DEFAULT_CLAUDE_MODEL} to keep them on Anthropic"
        )
    return notice


def format_markdown_output(result: dict) -> str:
    """The ``--output markdown`` text: the report, then a footer naming the
    model, because the synthesis report doesn't name it.

    Only the CLI's markdown output gets the footer. The JSON ``report``
    field stays the bare report, because the action adds its own "Generated
    by QA Bot using <model>" footer to it.
    """
    model = result.get("model")
    if not model:
        return result["report"]
    provider = "OpenRouter" if is_openrouter_model(model) else "Anthropic"
    return f"{result['report'].rstrip()}\n\n---\n*Model: {model} via {provider}*"


def main():
    parser = argparse.ArgumentParser(
        # Named as it is run. argparse's default is the file name, so usage
        # and error lines read "cli.py", which is not a command (and
        # "__main__.py" under `python -m qa_bot`). An installed copy
        # (`pip install`, the public repo's pyproject.toml) runs as `qa-bot`.
        prog="qa-bot" if Path(sys.argv[0]).name == "qa-bot" else "python -m qa_bot.cli",
        description="QA Bot - AI-powered website testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage (the report on stdout; stderr carries the Model line,
    # warnings and errors)
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

Exit codes:
    0    the site was tested and no critical finding gates
    1    critical findings (the report's curated count when it has a
         verdict, else the raw worker count)
    2    no verdict: the run aborted (rejected key, no credit, a crash);
         nothing was tested (unreachable target, no user flow completed,
         no report; --allow-zero-flows accepts a run with no flow tested);
         or a configuration or usage error (missing key, bad credentials
         file, unwritable --output-file, a URL or limit that cannot work)
    130  interrupted (Ctrl-C)
    The result JSON's "gate" key carries the same verdict and its reasons.
    Gate a deploy on it with:  python -m qa_bot.cli URL && deploy
        """
    )

    parser.add_argument(
        "url",
        help="Target URL to test (https:// is added when no scheme is given)"
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
            "acknowledged (free text, e.g. one per line; always text, never "
            "a path: see --known-issues-file). Workers won't re-investigate "
            "matching observations and the report lists them as one-line "
            "'observed again' notes instead of new findings."
        )
    )
    parser.add_argument(
        "--known-issues-file",
        metavar="PATH",
        help=(
            "Read known issues from a file such as .qa-known-issues.md: one "
            "per line; blank lines and lines starting with # are skipped. "
            "Its entries are added to any --known-issues text."
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
        help=f"Maximum parallel agents (default: 3; 1 with --smoke; at most {MAX_AGENTS_CAP})"
    )
    parser.add_argument(
        "--max-cost",
        type=float,
        default=None,
        help=(
            "Maximum cost in USD (default: $5.00; $0.75 with --smoke; at most "
            f"${MAX_COST_CAP_USD:.0f})"
        )
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
        help=(
            "Anthropic API key for Claude models (or set ANTHROPIC_API_KEY "
            "env var). Passed without --openrouter-api-key and --model, the "
            f"run uses {DEFAULT_CLAUDE_MODEL} even when an OpenRouter key is "
            "in the environment"
        )
    )
    parser.add_argument(
        "--openrouter-api-key",
        default=None,
        help=(
            "OpenRouter API key for 'vendor/slug' models (or set "
            "OPENROUTER_API_KEY env var). With it and no --model, the run "
            f"uses {DEFAULT_OPENROUTER_MODEL}"
        )
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            f"Model to use (default: {DEFAULT_OPENROUTER_MODEL} when an "
            f"OpenRouter key is set, else {DEFAULT_CLAUDE_MODEL}; a key "
            "passed by flag alone picks its own provider's default). A "
            "'vendor/slug' id runs on OpenRouter and needs "
            "--openrouter-api-key; screenshots, page text and test "
            "credentials then go to OpenRouter and the model vendor"
        )
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"qa-bot {__version__}",
        help=(
            "Print the version and exit. A release from the public repo "
            "(Integuide/qa-bot) carries its tag's number; a monorepo checkout "
            "is 0.0.0.dev0"
        ),
    )
    parser.add_argument(
        "--print-model",
        action="store_true",
        help=(
            "Print the model this run would use and why (two lines), then "
            "exit without running — the GitHub Action resolves its default "
            "model this way"
        )
    )
    parser.add_argument(
        "--testmail-api-key",
        default=None,
        help=(
            "Testmail.app API key (or set TESTMAIL_API_KEY env var). "
            "Experimental: accepted, but the bot cannot read verification "
            "emails yet"
        )
    )
    parser.add_argument(
        "--testmail-namespace",
        default=None,
        help="Testmail.app namespace (or set TESTMAIL_NAMESPACE env var). Experimental, as above"
    )
    parser.add_argument(
        "--output", "-o",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format (default: markdown)"
    )
    parser.add_argument(
        "--output-file", "-f",
        help=(
            "Write output to file instead of stdout. A result an earlier run "
            "left at that path is removed when the run starts, so a run that "
            "fails never sits next to a previous run's result"
        )
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
        help=(
            "Log verbosity on stderr: quiet (default: the Model line, "
            "warnings, errors and a failing Gate line), summary (flow "
            "events), full (all activity, including one line per HTTP request)"
        )
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
        "--allow-zero-flows",
        action="store_true",
        help=(
            "Exit 0 even when no user flow was tested (a report-only run; the "
            "Action's fail-on-zero-flows: 'false'). By default a run that "
            "tested nothing exits 2, so an unreachable or mistyped target can "
            "never pass a deploy gate."
        )
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
    quiet_http_request_logs(log_level)

    # API keys from args or environment; only the chosen model's provider
    # needs one (create_provider raises MissingAPIKeyError otherwise).
    api_key = args.api_key or os.getenv("ANTHROPIC_API_KEY")
    openrouter_api_key = args.openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
    # --model wins; then a key passed alone by flag picks its provider's
    # default; otherwise GPT-6 Luna when an OpenRouter key is set, else
    # Claude Sonnet 5.5 (resolve_cli_model / qa_bot.config.resolve_model — the
    # rule the action and the web UI share). AI_MODEL is deliberately not
    # read: it is the web UI server's override, and qa-bot/.env sets it for
    # local UI runs.
    model_choice = resolve_cli_model(args)
    if args.print_model:
        print(model_choice.model)
        print(model_choice.reason)
        sys.exit(0)

    # Everything below that can refuse the run does it here, before any
    # browser or AI spend, and exits 2 (no verdict).
    #
    # The input files are read first, because preparing --output-file
    # removes what is at that path and an input may be at the same one:
    # `--previous-report report.md --output-file report.md` is the local
    # run-over-run loop. A file that cannot be read is reported only after
    # the output file is prepared: an earlier run's result must not outlive
    # that failure, or any of the ones that follow.
    input_error = None
    credentials, known_issues_entries, previous_report = {}, "", ""
    try:
        credentials, known_issues_entries, previous_report = read_input_files(args)
    except InputFileError as e:
        input_error = e
    # A result replaces an earlier one with the earlier one's permissions
    # (never looser: the umask still applies), so a file the caller made
    # private (`mktemp`) is not swapped for a world-readable one.
    output_mode = 0o666
    if args.output_file:
        if os.path.isfile(args.output_file):
            output_mode = stat.S_IMODE(os.stat(args.output_file).st_mode)
        problem = prepare_output_file(args.output_file)
        if problem:
            print(f"Error: cannot write --output-file {args.output_file}: {problem}", file=sys.stderr)
            sys.exit(EXIT_NO_VERDICT)
    if input_error:
        print(input_error, file=sys.stderr)
        sys.exit(EXIT_NO_VERDICT)

    # The URL and the limits by the web API's rule (qa_bot.run_inputs). The
    # URL is checked after --print-model on purpose: the Action resolves its
    # model with `cli URL --print-model` first, and a bad URL must not show
    # up there as "could not resolve the model".
    try:
        url = normalize_target_url(args.url)
        max_agents, max_duration, max_cost = resolve_limits(args)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(EXIT_NO_VERDICT)
    if url != args.url.strip():
        # Every log level: the run is about to test something the caller
        # did not literally type.
        print(f"Note: no scheme in '{log_safe(args.url.strip())}'; testing {log_safe(url)}", file=sys.stderr)

    if args.credentials and log_level != "quiet":
        print(f"Loaded {len(credentials)} credentials from {args.credentials}", file=sys.stderr)
    if args.inline_credentials:
        for cred_str in args.inline_credentials:
            try:
                key, value = parse_inline_credential(cred_str)
                credentials[key] = value
            except ValueError as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(EXIT_NO_VERDICT)
        if log_level != "quiet":
            print(f"Added {len(args.inline_credentials)} inline credentials", file=sys.stderr)

    # Testmail.app credentials, passed to the run below. (They used to be
    # written to os.environ here, after qa_bot.config had already read it,
    # so the two flags never reached the coordinator.) The feature is not
    # active: no email-reading action exists yet, so the line must not say
    # "enabled" while entrypoint.sh, the docs and the READMEs say otherwise.
    testmail_key = args.testmail_api_key or os.getenv("TESTMAIL_API_KEY")
    testmail_namespace = args.testmail_namespace or os.getenv("TESTMAIL_NAMESPACE")
    if testmail_key and testmail_namespace:
        if log_level != "quiet":
            print(
                "Email testing: Testmail.app credentials set (experimental: "
                "email-reading actions are not active yet, so email "
                "verification steps are reported as untestable)",
                file=sys.stderr,
            )
    elif testmail_key or testmail_namespace:
        print(
            "Warning: Testmail.app needs both --testmail-api-key and "
            "--testmail-namespace; the one given is ignored",
            file=sys.stderr,
        )
        testmail_key = testmail_namespace = None

    # --known-issues is text and is passed on unchanged (what the Action
    # sends); a file's entries come from --known-issues-file and go first.
    known_issues = args.known_issues
    if args.known_issues_file:
        known_issues = "\n".join(part for part in (known_issues_entries, known_issues) if part.strip())
        if log_level != "quiet":
            print(
                f"Known issues: {len(known_issues_entries.splitlines())} entr(y/ies) from "
                f"{log_safe(args.known_issues_file)}",
                file=sys.stderr,
            )
    if is_a_file_name_not_an_issue(args.known_issues):
        # Every log level: it used to go to the workers as the known issue
        # with nothing said, and nothing was suppressed.
        name = log_safe(args.known_issues.strip())
        print(
            f"Warning: --known-issues takes the known issues as text, and '{name}' "
            "looks like a file name. It is sent as text and no file is read; "
            f"use --known-issues-file {name} to read a file.",
            file=sys.stderr,
        )
    if previous_report and log_level != "quiet":
        print(
            f"Previous report loaded ({len(previous_report)} chars) — findings "
            "will be labelled NEW / recurring",
            file=sys.stderr,
        )
    if args.smoke and args.goal != parser.get_default("goal") and log_level != "quiet":
        print("Note: --smoke uses a fixed smoke goal; --goal is ignored", file=sys.stderr)
    # Printed at every log level, quiet (the default) included, because the
    # model decides where screenshots, page text and credentials go, and an
    # ambient OPENROUTER_API_KEY can change it. It goes to stderr so that
    # --output json on stdout stays parseable. --print-model exited above,
    # so its stdout is still only its two lines.
    print(model_notice(model_choice), file=sys.stderr)

    # Run exploration
    try:
        result = asyncio.run(run_exploration(
            url=url,
            goal=args.goal,
            known_issues=known_issues,
            max_agents=max_agents,
            max_duration=max_duration,
            api_key=api_key,
            model=model_choice.model,
            headless=not args.headed,
            log_level=log_level,
            credentials=credentials if credentials else None,
            max_cost_usd=max_cost,
            skip_permissions=args.dangerously_skip_permissions,
            smoke=args.smoke,
            previous_report=previous_report,
            openrouter_api_key=openrouter_api_key,
            recheck_criticals=args.recheck_criticals,
            testmail_api_key=testmail_key,
            testmail_namespace=testmail_namespace,
        ))
    except MissingAPIKeyError as e:
        if not api_key and not openrouter_api_key and not args.model:
            # No key at all and no model asked for: either key works.
            print(
                "Error: no API key is set. Set OPENROUTER_API_KEY / "
                f"--openrouter-api-key (runs {DEFAULT_OPENROUTER_MODEL}) or "
                f"ANTHROPIC_API_KEY / --api-key (runs {DEFAULT_CLAUDE_MODEL})",
                file=sys.stderr,
            )
        else:
            flag = "--openrouter-api-key" if e.env_var == "OPENROUTER_API_KEY" else "--api-key"
            print(f"Error: {e}", file=sys.stderr)
            print(f"Set it via {flag} or the {e.env_var} environment variable", file=sys.stderr)
        sys.exit(EXIT_NO_VERDICT)
    except KeyboardInterrupt:
        print("\nExploration cancelled", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        # Not 1: exit 1 means "critical findings" and nothing else.
        print(f"Error during exploration: {e}", file=sys.stderr)
        sys.exit(EXIT_NO_VERDICT)

    # The verdict: exit code, final stderr line and the result's `gate` key
    # all come from qa_bot.gate.evaluate_gate.
    gate = evaluate_gate(result, fail_on_zero_flows=not args.allow_zero_flows)
    gate_line = (
        f"Gate: {gate.status.replace('_', ' ').upper()} (exit {gate.exit_code}): "
        f"{log_safe(gate.message)}"
    )

    # A run that never produced exploration_complete failed outright
    # (Playwright launch failure, fatal crash with nothing salvaged, ...).
    # Fail loudly and do NOT write a result file: CI consumers (e.g. the
    # GitHub Action entrypoint) treat the result file's existence as a run
    # that produced a report, so writing one here would post an empty one.
    # A salvaged run (completed, with `error` set) does write it, and exits 2
    # below. (`completed: false` is the gate's "aborted" too, so the exit
    # code here is the gate's.)
    if result.get("completed") is False:
        print(gate_line, file=sys.stderr)
        sys.exit(gate.exit_code)

    result["gate"] = gate.as_dict()

    # Format output
    if args.output == "json":
        output = json.dumps(result, indent=2, default=str)
    else:
        output = format_markdown_output(result)

    # Write output
    if args.output_file:
        try:
            with open(args.output_file, "w", opener=lambda p, flags: os.open(p, flags, output_mode)) as f:
                f.write(output)
        except OSError as e:
            # The run is paid for: put its result on stdout rather than lose it.
            print(output)
            print(
                f"Error: could not write --output-file {args.output_file} ({e}); "
                "the result was printed to stdout instead",
                file=sys.stderr,
            )
            sys.exit(EXIT_NO_VERDICT)
        if log_level != "quiet":
            print(f"Output written to {args.output_file}", file=sys.stderr)
    else:
        print(output)

    curated, raw_critical_count = gate.curated_critical_count, gate.raw_critical_count
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

    # A failing verdict is said at every log level, quiet included: a
    # non-zero exit with nothing on stderr leaves a script's author guessing
    # which of "critical findings" and "nothing was tested" happened. The
    # result file above is written either way, so the report and the PR
    # comment survive a salvaged abort (state-machines.md GATE-1).
    if gate.exit_code != EXIT_PASS or log_level != "quiet":
        print(gate_line, file=sys.stderr)
    sys.exit(gate.exit_code)


if __name__ == "__main__":
    main()
