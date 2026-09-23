"""Independent re-check of CRITICAL findings before they can fail the gate.

A curated critical fails the deploy gate, and onlinestoryservices' unattended
nightly auto-deploy now holds production (and pages the owner) on one. Across
both CI repos the gate went red on findings 14 times with 0 real bugs, and
every false critical was ONE worker misreading ONE observation: placeholder
text read as data loss (onlinestoryservices #1067), a headless repaint
artifact read as "the textarea doesn't render typed text" (#1080), a missing
ref read as "button non-functional" (mono #356).

So in non-interactive runs (CLI/CI; ``SharedFlowState.recheck_criticals``)
every new AI-reported critical gets a second, independent look during the
run: ``SharedFlowState.add_issue`` queues a short verification flow at
``PRIORITY_RECHECK`` — fresh browser context on the issue's page, a goal
carrying the finding and the mechanical steps that led to it (not the
reporter's reasoning), RECHECK_MAX_TURNS turns, no forking — whose single
terminal action, ``recheck_result``, records a machine-readable outcome.
``apply_outcome`` turns that outcome into deterministic changes on the
ORIGINAL issue, so the curated gate, the raw-count fallback, the report and
summary.json all tell the same story:

- ``not_reproduced`` -> severity ``major`` + UNCONFIRMED_TAG
- ``reproduced``     -> stays ``critical`` + REPRODUCED_TAG
- ``inconclusive``   -> stays ``critical`` (the gate stays conservative),
  annotated with why (turn cap, run ended, verifier error, no verdict)
- ``skipped``        -> stays ``critical``, "[NOT RE-CHECKED — <reason>]"
  (re-check cap reached, too little time or cost budget left)

At most MAX_RECHECKS_PER_RUN re-checks run per run. A new critical that is
a re-report of one already re-checked (``is_rereport``: same page, near-
identical wording, and filed by the same attempt of the same flow) joins
that re-check and shares its outcome instead of spending another. An
outcome never transfers to an independent sighting (another flow, or a
retry of the same flow), open re-check or settled.
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse

from qa_bot.agent.state import Issue

logger = logging.getLogger(__name__)

# Budget guards. A re-check is ~RECHECK_MAX_TURNS vision turns (roughly
# 2-3 minutes and $0.15-0.30 at Sonnet 5 prices); scheduling one that the
# run's own limits will cut off buys nothing but cost.
MAX_RECHECKS_PER_RUN = 3
RECHECK_MIN_REMAINING_SECONDS = 180
RECHECK_MIN_REMAINING_COST_USD = 0.25
# Checked again when a worker picks the re-check up: it may have queued
# behind a busy worker slot while the clock and budget ran down.
RECHECK_MIN_START_SECONDS = 60
RECHECK_MIN_START_COST_USD = 0.10

# Worker slots reserved for re-checks on top of max_agents, so a re-check
# starts as soon as it is queued instead of waiting for a user flow to end
# (with smoke mode's single worker it would otherwise start only after the
# smoke pass, too late in a 5-minute budget to reach a verdict). A queued
# re-check can also take the next free regular slot.
RECHECK_SLOTS = 1

# Turn cap for the verifier, shaped like the first-worker / smoke caps: a
# nudge note from RECHECK_NUDGE_TURNS, force-complete (as inconclusive) at
# RECHECK_MAX_TURNS.
RECHECK_NUDGE_TURNS = 11
RECHECK_MAX_TURNS = 15

# How many of the reporter's actions (ending at the report) the verifier sees.
RECHECK_STEPS_SHOWN = 10

REPRODUCED = "reproduced"
NOT_REPRODUCED = "not_reproduced"
INCONCLUSIVE = "inconclusive"
OUTCOMES = (REPRODUCED, NOT_REPRODUCED, INCONCLUSIVE)
PENDING = "pending"
RUNNING = "running"
SKIPPED = "skipped"
OPEN_STATUSES = (PENDING, RUNNING)

UNCONFIRMED_TAG = "[UNCONFIRMED — independent re-check did not reproduce]"
REPRODUCED_TAG = "[REPRODUCED by independent re-check]"

# A verdict entry carrying the UNCONFIRMED tag is the demoted finding copied
# into the critical list against Report Quality Standard 16. Anchored on the
# bracketed tag / the tag's own wording, never the bare word "unconfirmed"
# ("order placed but unconfirmed" is a real critical headline).
_UNCONFIRMED_TITLE = re.compile(
    r"\[\s*unconfirmed\b|independent re-check did not reproduce", re.IGNORECASE
)


def clip(text: str, limit: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def issue_title(description: str, limit: int = 80) -> str:
    """Short headline for logs/events: the first sentence, clipped."""
    text = " ".join(str(description or "").split())
    match = re.search(r"[.!?](?:\s|$)", text)
    if match and match.start() >= 20:
        text = text[: match.start()]
    return clip(text, limit) or "(untitled critical finding)"


_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset({
    "the", "and", "for", "with", "that", "this", "from", "after", "when",
    "was", "are", "but", "not", "into", "has", "have", "its", "which",
    "then", "than", "while", "page", "user", "users", "does", "did",
})


def _significant_words(text: str) -> set[str]:
    return {
        w for w in _WORD.findall(str(text or "").lower())
        if len(w) >= 3 and w not in _STOPWORDS
    }


def similar_findings(a: str, b: str, threshold: float = 0.5) -> bool:
    """Whether two critical descriptions are near-identical wordings.

    Deliberately strict (word-set Jaccard >= 0.5): a match makes the new
    issue SHARE the other's re-check outcome — a not_reproduced demotes it
    too — so only rewordings of the same report may match. A paraphrase that
    falls below the bar gets its own re-check (or, past the cap, stays
    critical), which is the safe side.
    """
    wa, wb = _significant_words(a), _significant_words(b)
    if not wa or not wb:
        return " ".join(str(a).split()).lower() == " ".join(str(b).split()).lower()
    return len(wa & wb) / len(wa | wb) >= threshold


def page_key(url: str) -> str:
    """A URL reduced to the page it names: scheme://host[:port]/path.

    Query, fragment, credentials, a default port and a trailing slash are
    ignored (``/story/7/edit?v=2#top`` is the same page as
    ``/story/7/edit``); the path keeps its case. A non-web value (empty,
    ``about:blank``) is compared as-is.
    """
    text = str(url or "").strip()
    parsed = urlparse(text)
    if parsed.scheme.lower() not in ("http", "https"):
        return text
    scheme = parsed.scheme.lower()
    try:
        port = parsed.port
    except ValueError:  # malformed port: keep the raw netloc
        return f"{scheme}://{parsed.netloc.lower()}{parsed.path.rstrip('/') or '/'}"
    default_port = 443 if scheme == "https" else 80
    netloc = (parsed.hostname or "") + (f":{port}" if port and port != default_port else "")
    return f"{scheme}://{netloc}{parsed.path.rstrip('/') or '/'}"


def is_rereport(record: "RecheckRecord", issue: Issue, attempt: Optional[tuple]) -> bool:
    """Whether a new critical re-reports ``record``'s finding, and so joins
    its re-check and shares the outcome instead of getting its own.

    All three must hold: the same page (``page_key``), near-identical
    wording (``similar_findings``), and the same ATTEMPT of the same flow
    filing it (``attempt``, from ``SharedFlowState._attempt_key``) — a
    rewording of what that attempt already reported. Anything else is an
    independent sighting, which is corroboration, not a rewording: another
    flow, or a retry of the same flow re-observing it from a fresh start
    (``fail_flow`` re-queues the same flow_id). It gets its own re-check —
    or, past the cap or budget, stays critical — whether the record is
    still open or settled, so a ``not_reproduced`` verdict only ever
    demotes the attempt it checked. Joining on text alone, then on "still
    open or same flow id", demoted such sightings unchecked (2026-09-23
    reviews).
    """
    if not attempt or attempt != record.origin_attempt:
        return False
    return (
        page_key(record.url) == page_key(issue.url)
        and similar_findings(record.original_description, issue.description)
    )


def describe_step(action: dict) -> str:
    """One mechanical line for a recorded worker action (no reasoning).

    The verifier gets what the reporter DID, not what it concluded — the
    conclusion is the thing being checked.
    """
    kind = action.get("action_type", "?")
    target = action.get("element") or action.get("ref") or ""
    if not target and action.get("coordinate"):
        target = f"point {tuple(action['coordinate'])}"
    if kind == "navigate":
        line = f"navigate to {clip(action.get('target_url', ''), 120)}"
    elif kind == "type":
        line = f"type \"{clip(action.get('text', ''), 40)}\" into {target or 'a field'}"
    elif kind == "find_text":
        line = f"find_text \"{clip(action.get('text', ''), 40)}\""
    elif target:
        line = f"{kind} {clip(target, 60)}"
    else:
        line = kind
    if action.get("success") is False:
        error = clip(action.get("error", ""), 100)
        line += f" (failed{': ' + error if error else ''})"
    note = action.get("note")
    if note:
        line += f" [{clip(note, 140)}]"
    return line


EARLIER_STEPS_NOT_RECORDED = (
    "(earlier steps not recorded: this flow continued from browser state "
    "another flow set up — if the finding depends on state you cannot "
    "recreate, the verdict is inconclusive)"
)


def original_steps(actions: list[dict], complete: bool = True) -> list[str]:
    """The reporter's last RECHECK_STEPS_SHOWN steps up to the report.

    ``actions`` is the trail that led to the report, oldest first, ending
    with the reporting action (``SharedFlowState._action_trail``, taken
    when the issue is filed): the reporting flow's current attempt,
    prefixed — for a flow restored from a checkpoint — with its parent's
    steps up to the fork. ``complete`` is False when that prefix was not
    recorded; the first line then says so, so the verifier treats state it
    cannot recreate as inconclusive rather than as the bug not happening.
    Earlier steps that exist but fall outside the window get a line too,
    and the shown steps keep their position in the trail. Issue reports
    and flow creation are not steps. A page line is added whenever the
    page changes, so the verifier knows where each step ran.
    """
    steps = [a for a in actions if a.get("action_type") not in ("report_issue", "add_flow")]
    window = steps[-RECHECK_STEPS_SHOWN:]
    hidden = len(steps) - len(window)
    lines = [] if complete else [EARLIER_STEPS_NOT_RECORDED]
    if hidden:
        lines.append(f"({hidden} earlier step{'s' if hidden != 1 else ''} not shown)")
    last_page = None
    for index, action in enumerate(window, hidden + 1):
        page = action.get("page_url") or ""
        where = f" — on {clip(page, 120)}" if page and page != last_page else ""
        last_page = page or last_page
        lines.append(f"{index}. {describe_step(action)}{where}")
    return lines


def recheck_start_url(issue_url: str, target_url: str) -> str:
    """Where the verifier starts: the issue's page when it is a web URL."""
    if urlparse(issue_url or "").scheme in ("http", "https"):
        # Lazy import: browser_pool pulls in Playwright
        from qa_bot.orchestrator.browser_pool import strip_url_credentials
        return strip_url_credentials(issue_url)
    return target_url


@dataclass
class RecheckRecord:
    """One scheduled re-check and the issues that share its outcome."""
    recheck_id: str
    flow_id: str
    title: str
    original_description: str
    url: str
    origin_flow_name: str
    steps: list[str]
    issues: list[Issue] = field(default_factory=list)  # primary first, then linked
    status: str = PENDING
    reason: str = ""
    # The attempt that filed the primary issue, (flow_id, index into the
    # flow's actions where that attempt began): only its own re-reports
    # share the outcome (see is_rereport)
    origin_attempt: Optional[tuple] = None

    @classmethod
    def new(
        cls, issue: Issue, steps: list[str], origin_flow_name: str,
        origin_attempt: Optional[tuple] = None,
    ) -> "RecheckRecord":
        return cls(
            recheck_id=uuid.uuid4().hex[:12],
            flow_id=str(uuid.uuid4()),
            title=issue_title(issue.description),
            original_description=issue.description,
            url=issue.url,
            origin_flow_name=origin_flow_name,
            steps=steps,
            issues=[issue],
            origin_attempt=origin_attempt,
        )


def apply_outcome(issue: Issue, status: str, reason: str = "", **extra) -> None:
    """Deterministically annotate an issue with its re-check status.

    Always rebuilt from the pristine description kept in ``issue.recheck``,
    so re-annotation never stacks tags. Only ``not_reproduced`` changes
    severity (critical -> major); every other status keeps the issue
    critical so the gate stays conservative.
    """
    record = dict(issue.recheck or {})
    original = record.get("original_description") or issue.description
    record.setdefault("original_severity", issue.severity)
    record.update(extra)
    record.update(status=status, reason=reason, original_description=original)
    if status == NOT_REPRODUCED:
        if issue.severity == "critical":
            issue.severity = "major"
        tag = UNCONFIRMED_TAG
    elif status == REPRODUCED:
        tag = REPRODUCED_TAG
    elif status == INCONCLUSIVE:
        tag = f"[RE-CHECK INCONCLUSIVE — kept critical: {clip(reason, 90)}]"
    elif status == SKIPPED:
        tag = f"[NOT RE-CHECKED — {clip(reason, 90)}]"
    else:  # pending / running: nothing settled yet
        tag = ""
    issue.description = f"{tag} {original}" if tag else original
    issue.recheck = record


def issue_payload(issue: Issue) -> dict:
    """What an event consumer (CLI, web UI) needs to update its copy of an
    issue: match on (original_description, url), then take the rest."""
    recheck = issue.recheck or {}
    return {
        "original_description": recheck.get("original_description", issue.description),
        "description": issue.description,
        "severity": issue.severity,
        "url": issue.url,
        "flow_id": issue.flow_id,
        "flow_name": issue.flow_name,
        "recheck": dict(recheck),
    }


def result_event_data(record: RecheckRecord, issues: list[Issue]) -> dict:
    return {
        "recheck_id": record.recheck_id,
        "flow_id": record.flow_id,
        "title": record.title,
        "outcome": record.status,
        "reason": record.reason,
        "issues": [issue_payload(i) for i in issues],
    }


def new_issue_event(issue: Issue, worker_id: str, flow_id: str) -> Optional[dict]:
    """The re-check event to emit right after a new critical was filed.

    Derived from the decision ``SharedFlowState.add_issue`` recorded on
    ``issue.recheck``: scheduled (or joined an open re-check), skipped with
    a reason, or joined a re-check that has already settled (its outcome is
    applied at once, so this is a ``recheck_result``).
    """
    recheck = issue.recheck
    if not recheck:
        return None
    status = recheck.get("status")
    title = issue_title(recheck.get("original_description") or issue.description)
    base = {"worker_id": worker_id, "flow_id": flow_id}
    if status in OPEN_STATUSES:
        return {**base, "type": "recheck_scheduled", "data": {
            "recheck_id": recheck.get("recheck_id"),
            "recheck_flow_id": recheck.get("recheck_flow_id"),
            "title": title,
            "linked": bool(recheck.get("linked")),
            "issue": issue_payload(issue),
        }}
    if status == SKIPPED:
        return {**base, "type": "recheck_skipped", "data": {
            "title": title,
            "reason": recheck.get("reason", ""),
            "issue": issue_payload(issue),
        }}
    if status in OUTCOMES:
        return {**base, "type": "recheck_result", "data": {
            "recheck_id": recheck.get("recheck_id"),
            "flow_id": recheck.get("recheck_flow_id"),
            "title": title,
            "outcome": status,
            "reason": recheck.get("reason", ""),
            "issues": [issue_payload(issue)],
        }}
    return None


def summarize(issues: list[Issue]) -> Optional[dict]:
    """Per-issue re-check counts for summary.json / the CLI result / the PR
    footer; None when no issue was ever a re-check candidate."""
    counts = {REPRODUCED: 0, NOT_REPRODUCED: 0, INCONCLUSIVE: 0, "not_rechecked": 0}
    seen = False
    for issue in issues:
        status = (issue.recheck or {}).get("status")
        if not status:
            continue
        seen = True
        if status == SKIPPED:
            counts["not_rechecked"] += 1
        elif status in counts:
            counts[status] += 1
        else:  # still open: finalize_rechecks has not run yet
            counts[INCONCLUSIVE] += 1
    return counts if seen else None


# Title-vs-description matching for the untagged backstop below. A verdict
# title is a short headline and a description a paragraph, so the measure
# is containment: shared significant words over the SMALLER word set.
# Dropping needs a strong match to the not-reproduced finding; one weak
# match to any still-critical issue is enough to keep the entry.
DEMOTED_TITLE_MIN_SHARED_WORDS = 3
DEMOTED_TITLE_MATCH = 0.8
STILL_CRITICAL_TITLE_MIN_SHARED_WORDS = 2
STILL_CRITICAL_TITLE_MATCH = 0.5
_RECHECK_TAG_TEXT = re.compile(
    r"\[\s*(?:unconfirmed|reproduced|re-check|not re-checked)\b[^\]]*\]", re.IGNORECASE
)


def _title_match(title: str, description: str, min_shared: int, threshold: float) -> bool:
    words_title = _significant_words(_RECHECK_TAG_TEXT.sub(" ", title))
    words_description = _significant_words(description)
    if not words_title or not words_description:
        return False
    shared = len(words_title & words_description)
    return (
        shared >= min_shared
        and shared / min(len(words_title), len(words_description)) >= threshold
    )


def _pristine(issue: Issue) -> str:
    return (issue.recheck or {}).get("original_description") or issue.description


def _is_retitled_demoted(entry: dict, issues: list[Issue]) -> Optional[Issue]:
    """The not-reproduced issue a verdict critical re-titles, or None.

    None whenever the entry could stand for any issue that is still
    critical, or when its ``flow`` names a different flow than the
    not-reproduced one — a false drop would let a real critical through.
    """
    title = entry.get("title", "")
    if any(
        issue.severity == "critical"
        and _title_match(title, _pristine(issue),
                         STILL_CRITICAL_TITLE_MIN_SHARED_WORDS, STILL_CRITICAL_TITLE_MATCH)
        for issue in issues
    ):
        return None
    flow = (entry.get("flow") or "").lower()
    for issue in issues:
        if (issue.recheck or {}).get("status") != NOT_REPRODUCED:
            continue
        if flow and issue.flow_name and issue.flow_name.lower() not in flow:
            continue
        if _title_match(title, _pristine(issue), DEMOTED_TITLE_MIN_SHARED_WORDS, DEMOTED_TITLE_MATCH):
            return issue
    return None


def drop_unconfirmed_criticals(
    verdict: Optional[dict], issues: Optional[list[Issue]] = None
) -> Optional[dict]:
    """Deterministic backstop for Report Quality Standard 16: a finding the
    independent re-check did not reproduce never reaches the gate's list.

    Drops a verdict critical when its title carries the UNCONFIRMED tag
    (the demoted finding copied back), or — given the run's ``issues`` —
    when it is that finding re-titled or merged with the tag left off: its
    title strongly matches a not_reproduced issue's original description
    (DEMOTED_TITLE_MATCH of the smaller word set, at least
    DEMOTED_TITLE_MIN_SHARED_WORDS words, same flow when the entry names
    one) and matches NO issue still critical after re-check, even weakly.
    The tag check alone let "Chapter text lost after reload" through as a
    curated critical and fail the gate (2026-09-23 review).
    """
    if not verdict or not verdict.get("critical"):
        return verdict
    kept = []
    for entry in verdict["critical"]:
        title = entry.get("title", "")
        if _UNCONFIRMED_TITLE.search(title):
            continue
        demoted = _is_retitled_demoted(entry, issues or [])
        if demoted is not None:
            # clip() flattens newlines: the title is model output from
            # site-influenced text and must not start a workflow-command line
            logger.warning(
                "Verdict backstop dropped critical %r: it re-titles a finding the "
                "independent re-check did not reproduce (%r) and matches no "
                "finding that is still critical",
                clip(title, 160), clip(_pristine(demoted), 160),
            )
            continue
        kept.append(entry)
    if len(kept) == len(verdict["critical"]):
        return verdict
    return {**verdict, "critical": kept, "critical_count": len(kept)}
