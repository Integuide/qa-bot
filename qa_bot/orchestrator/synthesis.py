"""Synthesis agent that generates final QA reports."""

import json
import logging
import re
from datetime import datetime
from typing import Optional

from qa_bot.ai.base import AIProvider
from qa_bot.ai.prompts import format_issue_trace
from qa_bot.orchestrator.shared_state import SharedFlowState
from qa_bot.orchestrator.flow import FlowStatus
from qa_bot.utils.secrets import is_sensitive_field_label

logger = logging.getLogger(__name__)


# The synthesis prompt asks the model to end its output with a fenced
# ```qa-verdict block holding a small JSON object (the curated critical list,
# major count, known-issues-observed-again count). The CI gate (and the CLI
# exit code) fail on this CURATED verdict when it parses, and fall back to
# the raw worker counts only when it is absent.
# Tolerate the block appearing anywhere (a model that adds a trailing
# sign-off still gets parsed) — the LAST block wins.
_VERDICT_BLOCK = re.compile(
    r"[ \t]*```[ \t]*qa-verdict[ \t]*\r?\n(?P<body>.*?)\r?\n[ \t]*```[ \t]*",
    re.DOTALL | re.IGNORECASE,
)


def _as_count(value) -> int:
    """Coerce a verdict count to a non-negative int (garbage -> 0)."""
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def parse_verdict(report: str) -> tuple[str, Optional[dict]]:
    """Split a synthesis report into (human markdown, machine verdict).

    Returns ``(report_without_block, verdict)`` when a well-formed
    ``qa-verdict`` block is present, else ``(report_unchanged, None)``. A
    ``None`` verdict means "no curated verdict available" (fallback report,
    model ignored the instruction, malformed JSON) — distinct from a verdict
    with an empty ``critical`` list, which means "curated report has zero
    criticals". The report is only stripped when parsing succeeds, so a
    malformed block stays visible to a human reader rather than vanishing.

    The normalized verdict always has the shape::

        {"critical": [{"title": str, "flow": str | None}, ...],
         "critical_count": int, "major": int, "known_observed_again": int,
         "new": int | None, "recurring": int | None}

    ``new`` / ``recurring`` are the run-over-run labels (Report Quality
    Standard 11) and are ``None`` when the block carries no such keys — the
    run had no previous report to compare against.
    """
    if not report:
        return report, None

    matches = list(_VERDICT_BLOCK.finditer(report))
    if not matches:
        return report, None
    match = matches[-1]

    try:
        raw = json.loads(match.group("body"))
    except (ValueError, TypeError):
        logger.warning("Synthesis qa-verdict block is not valid JSON; ignoring it")
        return report, None
    if not isinstance(raw, dict):
        logger.warning("Synthesis qa-verdict block is not a JSON object; ignoring it")
        return report, None

    critical_raw = raw.get("critical")
    if critical_raw is None:
        critical_raw = []
    if not isinstance(critical_raw, list):
        logger.warning("Synthesis qa-verdict 'critical' is not a list; ignoring verdict")
        return report, None

    critical = []
    for entry in critical_raw:
        if isinstance(entry, dict):
            title = str(entry.get("title") or "").strip()
            flow = entry.get("flow")
            flow = str(flow).strip() if flow not in (None, "") else None
        else:
            title, flow = str(entry).strip(), None
        if not title:
            # A titleless entry still means "the model listed a critical".
            # Dropping it would silently decrement the count the deploy gate
            # fails on while the report body still describes the finding —
            # the gate must never read lower than what the model claimed
            # (2026-09-03 review). Keep it, named so a reader can tell why.
            logger.warning(
                "Synthesis qa-verdict critical entry has no title; "
                "counting it anyway so the gate is not silently lowered"
            )
            title = "(untitled critical finding — see the report body)"
        critical.append({"title": title, "flow": flow})

    verdict = {
        "critical": critical,
        "critical_count": len(critical),
        "major": _as_count(raw.get("major", 0)),
        "known_observed_again": _as_count(raw.get("known_observed_again", 0)),
        "new": _as_count(raw["new"]) if raw.get("new") is not None else None,
        "recurring": (
            _as_count(raw["recurring"]) if raw.get("recurring") is not None else None
        ),
    }

    stripped = (report[:match.start()] + report[match.end():]).rstrip() + "\n"
    return stripped, verdict


def drop_unfounded_run_over_run_labels(
    verdict: Optional[dict], previous_report: str
) -> Optional[dict]:
    """Null the verdict's ``new``/``recurring`` when no previous report was given.

    The prompt says to omit both keys without a "Previous Run Findings"
    section, and the model writes ``"new": 0, "recurring": 0`` anyway — the
    onlinestoryservices PR #1080 run had no previous report and its PR
    footer still read "Since previous run: 0 new, 0 recurring" under a
    critical that was, by any reading, new. Labels the model had nothing to
    compare against are not labels; ``None`` is what every consumer
    (``cli.py`` counts, action outputs, the footer) already treats as "no
    previous report".
    """
    if verdict is None or (previous_report or "").strip():
        return verdict
    if verdict.get("new") is None and verdict.get("recurring") is None:
        return verdict
    logger.info(
        "Synthesis qa-verdict carried new/recurring counts (%s/%s) but no "
        "previous report was supplied; dropping them",
        verdict.get("new"), verdict.get("recurring"),
    )
    return {**verdict, "new": None, "recurring": None}


# Prepended to the fallback report persisted before the AI synthesis call.
# Only ever read if the run died before the AI report overwrote it.
PROVISIONAL_REPORT_BANNER = (
    "> **Note:** Provisional report, written before AI synthesis started. "
    "If you are reading this, the run ended (server restart or crash) before "
    "the AI-written report could replace it. The findings below are complete "
    "but unpolished.\n\n"
)


def _plural(n: int, unit: str) -> str:
    return f"{n} {unit}" if n == 1 else f"{n} {unit}s"


def format_duration(seconds: float) -> str:
    """Human-readable run duration ("1 minute", "45 seconds", "1h 05m")."""
    seconds = int(seconds)
    if seconds < 60:
        return _plural(seconds, "second")
    if seconds < 3600:
        return _plural(seconds // 60, "minute")
    return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m"


# Substrings (lower-cased) in a flow's completion reason that mean the run's
# cost/time budget cut it off, as opposed to a worker error.
_LIMIT_REASON_MARKERS = ("limit", "cost", "budget", "time", "pause", "duration")

# Smoke mode (SharedFlowState.smoke): every report — AI, fallback, or the
# deterministic NOT TESTED one — carries this tag in its Goal Assessment /
# status line so nobody mistakes a smoke pass for regression coverage. The
# synthesis prompt asks the model for it; apply_smoke_tag() is the
# deterministic backstop that inserts it when the model didn't.
SMOKE_REPORT_TAG = "SMOKE TEST ONLY — not a regression test"
SMOKE_STATUS_LINE = (
    f"**{SMOKE_REPORT_TAG}.** A single worker loaded the target and "
    "opened each top-level navigation link once (no sign-up, login, form "
    "submission or flow exploration). A pass here means only that the "
    "deployed UI serves without server errors, console exceptions or broken "
    "layout — it is NOT evidence that the deploy's own changes work."
)

_GOAL_ASSESSMENT_HEADING = re.compile(r"^##+\s*Goal Assessment\s*$", re.MULTILINE)
_FIRST_H1 = re.compile(r"^#\s+.+$", re.MULTILINE)
_ANY_HEADING = re.compile(r"^#{1,6}\s", re.MULTILINE)


def strip_smoke_tag(report: str) -> str:
    """Remove a model-written SMOKE tag from a non-smoke report.

    The tag is only ever true for smoke-mode runs, where apply_smoke_tag adds
    it deterministically. A model that sees a small goal ("check the homepage
    loads") can still decide the run was a smoke test and head the report
    with the tag — which tells the reader the run was "not a regression test"
    when it was a normal exploration.

    Scoped to the report's STATUS REGION: everything through the title and the
    Goal Assessment, plus any heading that is itself the tag. That is exactly
    where apply_smoke_tag puts the status line, and the only place the tag
    claims *this* run was a smoke test. Past the first ordinary section
    heading it is context, not a claim: run-over-run memory feeds the previous
    report into synthesis, so a previous smoke run gets quoted under "Not
    Observed This Run" — blanking that phrase loses the reason those findings
    are not comparable. apply_smoke_tag anchors its idempotence check the same
    way.
    """
    if SMOKE_REPORT_TAG not in report:
        return report
    first_h1 = _FIRST_H1.search(report)
    first_h1_line = report[: first_h1.start()].count("\n") if first_h1 else -1

    kept = []
    in_status_region = True
    for index, line in enumerate(report.splitlines()):
        if (
            in_status_region
            and index > first_h1_line
            and _ANY_HEADING.match(line)
            and SMOKE_REPORT_TAG not in line
            and not _GOAL_ASSESSMENT_HEADING.match(line)
        ):
            # An ordinary section heading — everything below it is report
            # body, where the tag can only ever be quoted context.
            in_status_region = False
        if not in_status_region or SMOKE_REPORT_TAG not in line:
            kept.append(line)
            continue
        remainder = line.replace(SMOKE_REPORT_TAG, "")
        if re.search(r"[A-Za-z0-9]", remainder):
            # The tag is embedded in a sentence: drop the phrase, keep the
            # rest — the report is the gate's source of truth.
            kept.append(
                remainder.replace("**.**", "").replace("** **", "").strip()
                if remainder.strip("*#. ")
                else remainder
            )
        # else: a heading / bold line that is nothing but the tag — drop it
    cleaned = "\n".join(kept)
    while "\n\n\n" in cleaned:
        cleaned = cleaned.replace("\n\n\n", "\n\n")
    return cleaned + ("\n" if report.endswith("\n") else "")


def apply_smoke_tag(report: str) -> str:
    """Make sure a smoke-mode report says so, deterministically.

    Idempotent: a report that already carries SMOKE_REPORT_TAG (the model
    followed the prompt) is returned unchanged. Otherwise the status line is
    inserted right after the "## Goal Assessment" heading when there is one,
    else after the report's H1 title, else prepended — so the tag is the
    first thing a reader sees whatever shape the report took.
    """
    block = f"\n\n{SMOKE_STATUS_LINE}"
    match = _GOAL_ASSESSMENT_HEADING.search(report) or _FIRST_H1.search(report)
    if match:
        end = match.end()
        # Idempotence is judged where the tag belongs, not anywhere in the
        # report: a previous smoke report quoted under "Not Observed This
        # Run" must not suppress the status line.
        following = report[end:].lstrip()
        if following.startswith(SMOKE_STATUS_LINE) or following.startswith(f"**{SMOKE_REPORT_TAG}"):
            return report
        return report[:end] + block + report[end:]
    if report.lstrip().startswith(SMOKE_STATUS_LINE):
        return report
    return f"{SMOKE_STATUS_LINE}\n\n{report}"


# How a resume clone's FlowExplorationData.resume_kind reads in the report.
# The clone carries the original flow's name (the original is RESUMED and
# excluded from every section), so this suffix is the only trace of the
# interruption: "Checkout: Checkout verified (resumed after pause)".
_RESUME_KIND_LABELS = {
    "pause": "pause",
    "credentials": "credentials were provided",
    "data": "data was provided",
    "approval": "the approval prompt was answered",
}


def _with_resume_note(text: str, resume_kind: Optional[str]) -> str:
    """Append "(resumed after ...)" once to a flow's status text."""
    if not resume_kind:
        return text
    label = _RESUME_KIND_LABELS.get(resume_kind, resume_kind)
    return f"{text} (resumed after {label})"


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

    def _summarize_actions(self, actions: list) -> list[dict]:
        """Summarize a flow's actions in order, carrying planted-canary evidence.

        `_describe_action` cuts a `type` text to 27 chars, and a long flow is
        windowed head+tail — so a canary the worker planted in a sentence
        ("…a well named Vorlith-ab12cd-4471…") vanished from the history
        synthesis reads while the later `find_text` match survived, and
        Report Quality Standard 15 then downgraded a PROVEN leak (PR #636
        review). A non-sensitive `type` whose text contains the run nonce,
        or a substring a later `find_text` in the same flow searched for, is
        annotated with that token (and the search step) and flagged
        `pinned` so the windowing keeps it.
        """
        searches = []
        for step, action in enumerate(actions, 1):
            if action.get("action_type") == "find_text":
                needle = str(action.get("text") or "").strip()
                if needle:
                    searches.append((step, needle))
        nonce = str(getattr(self.ai, "run_nonce", "") or "")
        return [
            self._summarize_action(
                action,
                later_searches=[(s, n) for s, n in searches if s > step],
                run_nonce=nonce,
            )
            for step, action in enumerate(actions, 1)
        ]

    @staticmethod
    def _carried_tokens(text: str, later_searches: list, run_nonce: str) -> list[str]:
        """Phrases of a typed text that later evidence keys on, as annotations."""
        lowered = text.lower()
        found: dict[str, dict] = {}
        if run_nonce and run_nonce.lower() in lowered:
            for word in text.split():
                if run_nonce.lower() in word.lower():
                    token = word.strip("\"'.,;:!?()[]{}<>")
                    found.setdefault(token.lower(), {"token": token, "run_id": True, "step": None})
        for step, needle in later_searches:
            if needle.lower() in lowered:
                entry = found.setdefault(needle.lower(), {"token": needle, "run_id": False, "step": None})
                if entry["step"] is None:
                    entry["step"] = step
        out = []
        for entry in found.values():
            label = f"run-ID token '{entry['token'][:40]}'" if entry["run_id"] else f"'{entry['token'][:40]}'"
            if entry["step"] is not None:
                label += f", searched at step {entry['step']}"
            out.append(label)
        return out

    def _summarize_action(
        self, action: dict, later_searches: list | None = None, run_nonce: str = ""
    ) -> dict:
        """
        Create a human-readable summary of an action.

        Handles the current JSON action format with action_type, ref, text, etc.
        `later_searches` / `run_nonce` come from `_summarize_actions` (see there).
        """
        success = action.get("success", True)
        url = action.get("page_url", action.get("url", ""))

        description = self._describe_action(action)
        pinned = False
        if action.get("action_type") == "type" and not is_sensitive_field_label(
            action.get("element") or action.get("ref") or ""
        ):
            tokens = self._carried_tokens(
                str(action.get("text") or ""), later_searches or [], run_nonce
            )
            if tokens:
                description = f"{description} (text contains {'; '.join(tokens)})"
                pinned = True
        # The worker's crafted note ("Field holds 68 chars (DOM-verified), but
        # its first text line is NOT visible…", "Triggered file download:
        # report.pdf") is evidence the synthesis prompt's quality standards
        # tell the model to look for in the flow's action history. Without
        # this it was visible to the worker only, and a standard that cites
        # it in synthesis was a standard about evidence synthesis never saw
        # (PR #607 review).
        # Capped like the worker's history line (claude_provider._format_history):
        # the download note is joined over site-controlled filenames, and one
        # pathological name must not bloat or steer the flow summary.
        note = action.get("note")
        if note:
            description = f"{description} [{str(note)[:240]}]"

        summary = {
            "description": description,
            "success": success,
            "url": url,
        }
        if pinned:
            summary["pinned"] = True
        return summary

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
            if is_sensitive_field_label(element or ref or ""):
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

        # Whole-page text search — its note carries the result
        if action_type == "find_text":
            needle = action.get("text", "")
            return f"Find text '{needle[:40]}' on the page"

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

        # Set HTTP auth
        if action_type == "set_http_auth":
            return "Applied HTTP Basic Auth credentials"

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
        """Generate the final QA synthesis report (markdown only).

        Convenience wrapper over :meth:`generate_report_with_verdict` for
        callers that don't need the machine-readable verdict.
        """
        report, _verdict = await self.generate_report_with_verdict(shared_state)
        return report

    async def generate_report_with_verdict(
        self, shared_state: SharedFlowState
    ) -> tuple[str, Optional[dict]]:
        """Generate the report plus verdict; smoke-mode reports get the SMOKE tag.

        See :meth:`_generate_report_with_verdict` for the body. This wrapper
        exists so the deterministic smoke tag is applied on every path (AI
        report, fallback, NOT TESTED) without touching each return.
        """
        report, verdict = await self._generate_report_with_verdict(shared_state)
        if getattr(shared_state, "smoke", False):
            report = apply_smoke_tag(report)
        else:
            report = strip_smoke_tag(report)
        return report, verdict

    async def _generate_report_with_verdict(
        self, shared_state: SharedFlowState
    ) -> tuple[str, Optional[dict]]:
        """
        Generate final QA synthesis report plus its curated verdict.

        Args:
            shared_state: The shared state containing all exploration data

        Returns:
            ``(report, verdict)`` — the markdown report (with the model's
            ``qa-verdict`` block stripped) and the parsed verdict dict from
            :func:`parse_verdict`. ``verdict`` is ``None`` whenever no AI
            curation happened or its verdict could not be parsed: the
            deterministic NOT TESTED report, the non-AI fallback report, or
            an AI report without a well-formed block.
        """
        # Calculate duration
        duration_seconds = (datetime.now() - shared_state.start_time).total_seconds()
        duration = format_duration(duration_seconds)

        all_flows = await shared_state.get_all_flows()

        # Get all issues, with the trace the report needs to cite them: the
        # flow, the step in that flow's action summary, and the screenshot.
        all_issues = await shared_state.get_all_issues()
        issues = []
        for issue in all_issues:
            issue_dict = {
                "description": issue.description,
                "severity": issue.severity,
                "url": issue.url,
                "context": issue.action_context,
                "source": issue.source,
            }
            if issue.flow_name:
                issue_dict["flow_name"] = issue.flow_name
            if issue.turn is not None:
                # Issue.turn is the worker's 0-based turn (chat transcript
                # "TURN n"); the flow action summary below is numbered from 1
                # and the triggering action is already recorded when the
                # issue is filed, so step = turn + 1 points at that action.
                issue_dict["step"] = issue.turn + 1
            if issue.screenshot_path:
                issue_dict["screenshot"] = issue.screenshot_path
            issues.append(issue_dict)

        # Get completed flows with rich action summaries. The Root/first-worker
        # flow is excluded: its only job is flow enumeration (worker.py
        # FIRST_WORKER_MAX_TURNS), so listing it under "Flows Tested" let a
        # run where nothing but flow-mapping finished read as 1 flow tested.
        # Any issues it reported still arrive via get_all_issues below.
        completed_flows = []
        for flow in all_flows:
            if flow.status == FlowStatus.COMPLETED and not flow.is_first_worker:
                # Create action summary for reproduction steps
                action_summary = self._summarize_actions(flow.actions)

                flow_summary = {
                    "flow_name": flow.flow_name,
                    "completion_reason": _with_resume_note(
                        flow.completion_reason or "Completed", flow.resume_kind
                    ),
                    "action_count": len(flow.actions),
                    "issue_count": len(flow.issues),
                    "urls_visited": flow.urls_visited,
                    "action_summary": action_summary
                }
                # A flow that failed and only succeeded on the automatic
                # retry is flakiness signal the report should see.
                if flow.failure_history:
                    flow_summary["earlier_failed_attempts"] = flow.failure_history
                completed_flows.append(flow_summary)

        # Get blocked flows (credentials/approval) so synthesis can report on them
        blocked_flows = []
        for flow in all_flows:
            if flow.status in (FlowStatus.BLOCKED_FOR_CREDENTIALS, FlowStatus.BLOCKED_FOR_APPROVAL):
                blocked_flows.append({
                    "flow_name": flow.flow_name,
                    "status": flow.status.value,
                    "reason": flow.completion_reason or "Blocked",
                })

        # Get flows that never finished: failed (worker/AI errors), interrupted
        # by the cost/time limit (BLOCKED_FOR_PAUSE in non-interactive runs), or
        # still exploring when the run ended. Without these the report silently
        # overstates coverage — a flow that crashed at step 5 simply vanishes
        # and reads as if it was never attempted.
        incomplete_flows = []
        for flow in all_flows:
            if flow.status == FlowStatus.FAILED:
                reason = flow.completion_reason or "Failed"
            elif flow.status == FlowStatus.BLOCKED_FOR_PAUSE:
                # Skip the placeholder resume clones created by
                # add_pause_checkpoint: they mirror an original paused flow
                # (which has its own registry entry) and never ran an action.
                if flow.started_at is None:
                    continue
                reason = flow.completion_reason or "Interrupted by pause/limit"
            elif flow.status == FlowStatus.EXPLORING:
                reason = flow.completion_reason or "Interrupted while still exploring"
            elif flow.status == FlowStatus.PENDING:
                if flow.failure_history:
                    # Failed, was re-queued for its automatic retry, but the
                    # run ended before a worker claimed the retry. "Never
                    # started" would contradict the recorded actions and drop
                    # the real crash reason.
                    reason = (
                        f"Failed once ({flow.failure_history[-1]}); automatic "
                        f"retry was queued but the run ended before it started"
                    )
                else:
                    # Discovered but never picked up: the run ended (cost/time
                    # limit, user stop) while this flow was still queued.
                    # Without this the report silently understates
                    # discovered-but-untested coverage.
                    reason = "Never started — run ended before this flow was explored"
            else:
                continue

            incomplete_flows.append({
                "flow_name": flow.flow_name,
                "status": flow.status.value,
                "reason": _with_resume_note(reason, flow.resume_kind),
                "action_count": len(flow.actions),
                "action_summary": self._summarize_actions(flow.actions),
            })

        # Nothing ran and nothing was found: skip the AI call entirely.
        # Burning synthesis tokens to wrap "0 flows tested" in prose helped
        # nobody on mono PR #356 — a deterministic report that names the
        # blockers and how to clear them is faster, free, and more actionable.
        if not completed_flows and not issues:
            report = self.generate_not_tested_report(
                target_url=shared_state.target_url,
                duration=duration,
                blocked_flows=blocked_flows,
                incomplete_flows=incomplete_flows,
                goal=shared_state.goal or "",
                interactive=shared_state.interactive,
            )
            if shared_state.chat_logger:
                shared_state.chat_logger.log_synthesis_call(
                    context=(
                        "0 flows completed, 0 issues — deterministic "
                        "NOT TESTED / INCOMPLETE report, no AI synthesis call made"
                    ),
                    report=report,
                    input_tokens=0,
                    output_tokens=0,
                )
            return report, None

        # Persist a deterministic report before the AI call. Synthesis is the
        # longest single await of the run, and a server restart or SIGKILL
        # here used to lose the whole run (mono run 694ba81f: 2 flows, $0.27,
        # no report.md on disk). Overwritten by the AI report on success, so
        # /api/flow/{id}/report always has something to serve.
        if shared_state.chat_logger:
            provisional = self._generate_fallback_report(
                    shared_state.target_url,
                    duration,
                    issues,
                    completed_flows,
                    blocked_flows,
                    incomplete_flows=incomplete_flows,
                    goal=shared_state.goal or "",
                    known_issues=shared_state.known_issues or "",
                    previous_report=shared_state.previous_report or "",
            )
            if getattr(shared_state, "smoke", False):
                provisional = apply_smoke_tag(provisional)
            shared_state.chat_logger.write_report(PROVISIONAL_REPORT_BANNER + provisional)

        # Call AI to generate report
        try:
            result = await self.ai.generate_synthesis_report(
                target_url=shared_state.target_url,
                duration=duration,
                flows_tested=len(completed_flows),
                issues=issues,
                completed_flows=completed_flows,
                blocked_flows=blocked_flows,
                incomplete_flows=incomplete_flows,
                goal=shared_state.goal or "",
                known_issues=shared_state.known_issues or "",
                previous_report=shared_state.previous_report or "",
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

            report, verdict = parse_verdict(result.get("report", ""))
            verdict = drop_unfounded_run_over_run_labels(
                verdict, shared_state.previous_report or ""
            )
            if verdict is None:
                logger.warning(
                    "Synthesis report carried no parseable qa-verdict block; "
                    "curated critical count unavailable for this run"
                )

            # Log the synthesis call (the human report, verdict stripped)
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

            return report, verdict
        except Exception as e:
            # Fallback to simple report if AI fails
            logger.warning(f"Synthesis AI call failed, using fallback report: {e}")
            return self._generate_fallback_report(
                shared_state.target_url,
                duration,
                issues,
                completed_flows,
                blocked_flows,
                incomplete_flows=incomplete_flows,
                goal=shared_state.goal or "",
                error=str(e),
                known_issues=shared_state.known_issues or "",
                previous_report=shared_state.previous_report or "",
            ), None

    def generate_not_tested_report(
        self,
        target_url: str,
        duration: str,
        blocked_flows: list[dict] | None = None,
        incomplete_flows: list[dict] | None = None,
        goal: str = "",
        reason: str | None = None,
        interactive: bool = False,
    ) -> str:
        """Deterministic report for runs where no flow finished and nothing was found.

        No AI involved: there are no findings to synthesize, only blockers to
        surface. Keeps the PR comment short and actionable instead of a long
        AI-written report that reads like a test happened.

        The status line and the "How to Fix" advice are derived from what
        actually happened (flows cut off by the cost/time limit, worker
        errors, missing credentials, a failed pre-flight probe) rather than a
        fixed template — a run that took 19 actions before the time limit hit
        must not claim "no flows were executed" and point at credentials.
        ``interactive`` selects web-UI wording (Advanced Options) over CI
        wording (workflow inputs/secrets).
        """
        blocked_flows = blocked_flows or []
        incomplete_flows = incomplete_flows or []

        started = [f for f in incomplete_flows if f.get("action_count", 0) > 0]
        total_actions = sum(f.get("action_count", 0) for f in incomplete_flows)

        # Classify why the flows never finished.
        limit_hit = failed = interrupted = 0
        for flow in incomplete_flows:
            text = (flow.get("reason") or "").lower()
            # "Failed once (...); automatic retry was queued ..." is a
            # PENDING flow whose crash reason may mention a timeout.
            if flow.get("status") == FlowStatus.FAILED.value or text.startswith("failed once"):
                failed += 1
            elif any(marker in text for marker in _LIMIT_REASON_MARKERS):
                limit_hit += 1
            elif flow.get("action_count", 0) > 0:
                interrupted += 1  # e.g. user stop while still exploring

        if started:
            causes = []
            if limit_hit:
                causes.append("cost/time limit reached")
            if failed:
                causes.append("worker errors")
            if interrupted:
                causes.append("run interrupted")
            cause_text = f" ({', '.join(causes)})" if causes else ""
            status_line = (
                f"**Status: INCOMPLETE** — {_plural(len(started), 'flow')} "
                f"started but none finished{cause_text}. "
                f"{_plural(total_actions, 'action')} taken, no issues recorded, "
                "and no flow ran to completion — this deploy has NOT been verified."
            )
        else:
            status_line = (
                "**Status: NOT TESTED** — no flows were executed and no issues "
                "were recorded. This deploy has NOT been verified."
            )

        lines = [
            "# QA Test Report",
            "",
            "## Goal Assessment",
            "",
            status_line,
            "",
        ]
        if reason:
            lines.extend([f"**Reason**: {reason}", ""])
        lines.append(f"**Target URL:** {target_url}")
        if goal:
            lines.append(f"**Testing Goal:** {goal}")
        lines.extend([f"**Duration:** {duration}", ""])

        if blocked_flows:
            lines.append("## Blockers")
            for flow in blocked_flows:
                status_label = (
                    "missing credentials"
                    if "credentials" in flow["status"]
                    else "pending approval"
                )
                lines.append(
                    f"- **{flow['flow_name']}**: {flow['reason']} ({status_label})"
                )
            lines.append("")

        if incomplete_flows:
            lines.append("## Flows That Never Finished")
            for flow in incomplete_flows:
                count = flow.get("action_count", 0)
                suffix = f" — {_plural(count, 'action')} taken" if count else ""
                lines.append(
                    f"- **{flow['flow_name']}**: {flow.get('reason', 'Incomplete')}{suffix}"
                )
            lines.append("")

        # How to Fix: only the advice that matches an observed cause.
        fixes = []
        if blocked_flows:
            if interactive:
                fixes.append(
                    "- **Missing credentials**: enter the named keys under "
                    "**Advanced Options → Test Credentials** (one `KEY=value` "
                    "per line; HTTP Basic Auth uses `HTTP_USERNAME` / "
                    "`HTTP_PASSWORD`), or answer the credential request while "
                    "the run is waiting."
                )
            else:
                fixes.append(
                    "- **Missing credentials**: add the named keys to the QA "
                    "workflow's `credentials` secret (one `KEY=value` per line; "
                    "HTTP Basic Auth uses `HTTP_USERNAME` / `HTTP_PASSWORD`)."
                )
        if limit_hit:
            if interactive:
                fixes.append(
                    "- **Cost/time limit reached**: raise **Max Cost** / "
                    "**Max Duration** under Advanced Options, or narrow the "
                    "testing goal so fewer flows are needed."
                )
            else:
                fixes.append(
                    "- **Cost/time limit reached**: raise the workflow's "
                    "`max-cost` / `max-duration` inputs, or narrow the testing "
                    "goal so fewer flows are needed."
                )
        if failed:
            where = "the run's activity log" if interactive else "the Actions log"
            fixes.append(
                f"- **Worker errors**: {_plural(failed, 'flow')} failed before "
                f"finishing — see {where} for the error and re-run."
            )
        if interrupted:
            fixes.append(
                "- **Run interrupted**: the run was stopped while flows were "
                "still exploring — re-run and let it finish."
            )
        if reason or not fixes:
            url_where = (
                "the URL you entered" if interactive
                else "the workflow's `url` input"
            )
            fixes.append(
                "- **Unreachable target or wrong URL**: verify the deployment "
                f"is up and {url_where} points at it."
            )
        fixes.append(
            "- Then start a new run." if interactive
            else "- Then re-run the QA workflow."
        )

        lines.extend(["## How to Fix", "", *fixes])
        return "\n".join(lines)

    def _generate_fallback_report(
        self,
        target_url: str,
        duration: str,
        issues: list[dict],
        completed_flows: list[dict],
        blocked_flows: list[dict] | None = None,
        incomplete_flows: list[dict] | None = None,
        goal: str = "",
        error: str | None = None,
        known_issues: str = "",
        previous_report: str = "",
    ) -> str:
        """Generate a simple report without AI."""
        lines = [
            "# QA Test Report",
            "",
        ]

        if error:
            lines.extend([
                "> **Warning:** QA Bot encountered errors during this run. "
                "The results below may be incomplete. Check the Actions log for details.",
                ">",
                f"> `{error[:300]}`",
                "",
            ])

        lines.extend([
            f"**Target URL:** {target_url}",
        ])
        if goal:
            lines.append(f"**Testing Goal:** {goal}")
        lines.extend([
            f"**Duration:** {duration}",
            f"**Flows Tested:** {len(completed_flows)}",
            f"**Issues Found:** {len(issues)}",
            "",
        ])

        # Issues by severity
        critical = [i for i in issues if i.get("severity") == "critical"]
        major = [i for i in issues if i.get("severity") == "major"]
        minor = [i for i in issues if i.get("severity") == "minor"]
        cosmetic = [i for i in issues if i.get("severity") == "cosmetic"]

        def issue_bullet(issue: dict) -> str:
            trace = format_issue_trace(issue)
            return f"- {issue['description']}" + (f" ({trace})" if trace else "")

        if critical:
            lines.append("## Critical Issues")
            for issue in critical:
                lines.append(issue_bullet(issue))
                if issue.get("url"):
                    lines.append(f"  URL: {issue['url']}")
            lines.append("")

        if major:
            lines.append("## Major Issues")
            for issue in major:
                lines.append(issue_bullet(issue))
                if issue.get("url"):
                    lines.append(f"  URL: {issue['url']}")
            lines.append("")

        if minor:
            lines.append("## Minor Issues")
            for issue in minor:
                lines.append(issue_bullet(issue))
            lines.append("")

        if cosmetic:
            lines.append("## Cosmetic Issues")
            for issue in cosmetic:
                lines.append(issue_bullet(issue))
            lines.append("")

        if blocked_flows:
            lines.append("## Blocked Flows")
            for flow in blocked_flows:
                status_label = "missing credentials" if "credentials" in flow["status"] else "pending approval"
                lines.append(f"- {flow['flow_name']}: {flow['reason']} ({status_label})")
            lines.append("")

        if incomplete_flows:
            lines.append("## Incomplete / Untested Flows")
            lines.append("These flows were NOT fully tested:")
            for flow in incomplete_flows:
                status_label = "failed" if flow.get("status") == "failed" else "interrupted"
                lines.append(
                    f"- {flow['flow_name']}: {flow.get('reason', 'Incomplete')} "
                    f"({status_label} after {flow.get('action_count', 0)} actions)"
                )
            lines.append("")

        lines.append("## Flows Tested")
        for flow in completed_flows:
            lines.append(f"- {flow['flow_name']}: {flow['completion_reason']}")
        lines.append("")

        if not issues:
            lines.append("**No issues found during testing.**")

        if known_issues and known_issues.strip():
            # The AI-side known-issues curation didn't run (fallback path), so
            # surface the operator's list: issues above may match it.
            lines.extend([
                "",
                "## Known Issues (operator-provided)",
                "The following were already known before this run; issues "
                "listed above may be re-observations of them (AI curation "
                "was unavailable for this report):",
                "",
                known_issues.strip(),
            ])

        if previous_report and previous_report.strip():
            # NEW / recurring labelling is done by the AI; say plainly that it
            # didn't happen rather than let the reader assume everything is new.
            lines.extend([
                "",
                "**Previous run:** a previous report was supplied but findings "
                "were not compared against it (AI curation was unavailable for "
                "this report), so nothing above is labelled new or recurring.",
            ])

        return "\n".join(lines)
