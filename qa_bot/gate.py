"""The deploy gate: what a finished run's result means for a deploy.

``evaluate_gate`` is the one place that turns a run's result dict (the CLI's
``--output json``) into a verdict. ``qa_bot.cli.main`` takes its exit code and
its final stderr line from it and writes it into the result JSON as ``gate``.

The exit-code contract:

    0  the site was tested and no critical finding gates
    1  critical findings (and nothing else)
    2  no verdict: the run aborted, or nothing was tested (no user flow
       completed, or no report was produced). The CLI uses 2 for a
       configuration or run error as well: no result, so no verdict.

Precedence: aborted (2), then criticals (1), then nothing tested (2). A run
that died part-way says so even when it salvaged a critical, and a run that
found a critical before it could finish a single flow still reports the
critical.

action/entrypoint.sh computes the GitHub check's result in bash from the same
result file (it has to re-emit workflow commands and honour the
``fail-on-critical`` input). tests/test_gate_parity.py runs that shipped bash
and this function over one table of result files, so the two cannot drift
apart unnoticed.
"""

import re
from dataclasses import asdict, dataclass
from typing import Optional

EXIT_PASS = 0
EXIT_CRITICAL = 1
EXIT_NO_VERDICT = 2

# What run_exploration puts in `report` when synthesis never delivered one.
NO_REPORT = "No report generated"

REASON_ABORTED = "aborted"
REASON_CRITICAL = "critical"
REASON_ZERO_FLOWS = "zero_flows"
REASON_NO_REPORT = "no_report"

STATUS_BY_EXIT = {EXIT_PASS: "pass", EXIT_CRITICAL: "fail", EXIT_NO_VERDICT: "no_verdict"}
_EXIT_BY_REASON = {
    REASON_ABORTED: EXIT_NO_VERDICT,
    REASON_CRITICAL: EXIT_CRITICAL,
    REASON_ZERO_FLOWS: EXIT_NO_VERDICT,
    REASON_NO_REPORT: EXIT_NO_VERDICT,
}
_TITLES_MAX_CHARS = 600  # the cap entrypoint.sh puts on its ::error:: line
_LINE_BREAKS = re.compile(r"[\r\n]+")


@dataclass(frozen=True)
class GateVerdict:
    """The gate's answer for one run.

    ``reasons`` lists every failing rule that applies, in precedence order;
    the first decides ``exit_code``. ``source`` says which critical count
    gated: ``curated`` (the synthesis verdict) or ``raw`` (worker findings,
    used when synthesis produced no verdict). ``count`` is that count,
    ``display`` the same sentence the Action's PR footer shows after
    "Gate:", and ``message`` the one-line explanation of the verdict.
    """

    status: str
    exit_code: int
    reasons: tuple[str, ...]
    source: str
    count: int
    raw_critical_count: int
    display: str
    message: str

    @property
    def curated_critical_count(self) -> Optional[int]:
        return self.count if self.source == "curated" else None

    def as_dict(self) -> dict:
        data = asdict(self)
        data["reasons"] = list(self.reasons)
        return data


def _count(value) -> Optional[int]:
    """A non-negative whole number, or None: a malformed value means
    "absent" (``True`` is not a count). entrypoint.sh reads the same fields
    through ``jq -r`` and ``^[0-9]+$``, which also takes a digits-only
    string (``"0"``); this does not. The CLI never writes one, and reading
    it as absent is the stricter answer (tests/test_gate_parity.py pins the
    difference)."""
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _one_line(text) -> str:
    """Model- or site-influenced text flattened the way entrypoint.sh
    flattens its copy (jq ``gsub("[\\r\\n]+"; " ")``), so it stays on the
    one line it is embedded in."""
    return _LINE_BREAKS.sub(" ", str(text))


def evaluate_gate(result: dict, fail_on_zero_flows: bool = True) -> GateVerdict:
    """The verdict for a run's result dict. Pure: no I/O, no globals.

    ``fail_on_zero_flows=False`` accepts a run in which no user flow was
    tested (the CLI's ``--allow-zero-flows``, the Action's
    ``fail-on-zero-flows: 'false'``); nothing else is negotiable.
    """
    issues = result.get("issues") or []
    raw = sum(
        1 for issue in issues
        if isinstance(issue, dict) and str(issue.get("severity", "")).lower() == "critical"
    )
    # Curated when synthesis produced a verdict (the report is the source of
    # truth), else the raw worker count (owner decision, 2026-09-02).
    curated = _count(result.get("curated_critical_count"))
    if curated is not None:
        source, count = "curated", curated
        display = f"{curated} critical after curation ({raw} raw worker finding(s))"
    else:
        source, count = "raw", raw
        display = f"{raw} critical (raw worker findings; synthesis produced no curated verdict)"

    flows = _count(result.get("flows_explored")) or 0
    error = result.get("error")
    never_completed = result.get("completed") is False
    report = result.get("report")
    has_report = isinstance(report, str) and report.strip() not in ("", NO_REPORT)

    # Insertion order below is the precedence order.
    texts = {}
    if never_completed:
        # Startup failure or a crash with nothing salvaged: no result file is
        # written, and that one fact is the whole verdict.
        texts[REASON_ABORTED] = (
            f"QA exploration failed: {_one_line(error or 'exploration ended before completing')}"
        )
    else:
        if error:
            texts[REASON_ABORTED] = (
                f"QA exploration aborted: {_one_line(error)} — the report covers only "
                "what ran before the abort"
            )
        if count > 0 and source == "curated":
            titles = " | ".join(
                _one_line(t) for t in (result.get("curated_critical_titles") or [])
                if isinstance(t, str)
            )[:_TITLES_MAX_CHARS]
            texts[REASON_CRITICAL] = (
                f"Found {count} critical issue(s) after curation"
                f"{': ' + titles if titles else ''} ({raw} raw worker finding(s))"
            )
        elif count > 0:
            texts[REASON_CRITICAL] = (
                f"Found {raw} critical issue(s) (raw worker findings; synthesis "
                "produced no curated verdict)"
            )
        if flows == 0 and fail_on_zero_flows:
            texts[REASON_ZERO_FLOWS] = (
                "No flows were tested — the target was NOT verified"
                + (" (the report names the blockers)" if has_report else "")
            )
        if not has_report:
            texts[REASON_NO_REPORT] = (
                "No report was produced, so there is nothing to base a verdict on"
            )

    reasons = tuple(texts)
    if reasons:
        exit_code = _EXIT_BY_REASON[reasons[0]]
        message = texts[reasons[0]]
        if len(reasons) > 1:
            also = "; ".join(texts[r][0].lower() + texts[r][1:] for r in reasons[1:])
            message = f"{message}; also: {also}"
    else:
        exit_code = EXIT_PASS
        if flows == 0:
            message = (
                "No flows were tested — the target was NOT verified; accepted "
                "because zero-flow runs are allowed"
            )
        else:
            message = f"{flows} flow(s) tested, {display}"

    return GateVerdict(
        status=STATUS_BY_EXIT[exit_code],
        exit_code=exit_code,
        reasons=reasons,
        source=source,
        count=count,
        raw_critical_count=raw,
        display=display,
        message=message,
    )
