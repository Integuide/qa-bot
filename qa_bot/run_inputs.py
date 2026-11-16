"""The inputs a person gives a run, checked by one rule on every surface.

The web API (``server/schemas.py``) and the CLI (``qa_bot/cli.py``, and
through it the GitHub Action) both take a target URL and run limits from
whoever starts the run. Each check here raises ``ValueError`` with a message
fit to show that person when the value can never work, and returns the value
the run will use otherwise: an over-range limit is capped, not refused,
because a workflow that passes ``max-agents: 25`` runs today.

The coordinator keeps its own clamps as a backstop; a caller that checks
here first knows the capped value and can show it.
"""

import re

from qa_bot.config import MAX_COST_CAP_USD

MAX_AGENTS_CAP = 20

_SCHEME = re.compile(r"^([A-Za-z][A-Za-z0-9+.\-]*)://")


def normalize_target_url(value: str) -> str:
    """The URL a run will open: stripped, with ``https://`` added when no
    scheme was given (``example.com`` is how people type a site).

    Raises ValueError for an empty value or a scheme other than http(s):
    the browser could never test either, and the run would end as an
    "unreachable target" report minutes later.
    """
    url = (value or "").strip()
    if not url:
        raise ValueError("the target URL is empty")
    if url.lower().startswith(("http://", "https://")):
        return url
    scheme = _SCHEME.match(url)
    if scheme:
        raise ValueError(
            f"the target URL must start with http:// or https://, not {scheme.group(1)}://"
        )
    return "https://" + url


def _require_positive(value, name: str):
    # `not value > 0` rather than `value <= 0`: NaN compares false both ways,
    # and a NaN cost limit is a limit that never trips.
    if not value > 0:
        raise ValueError(f"{name} must be positive")
    return value


def check_max_cost(value: float, name: str = "max cost") -> float:
    """A positive cost limit in USD, capped at MAX_COST_CAP_USD."""
    return min(_require_positive(value, name), MAX_COST_CAP_USD)


def check_max_duration(value: int, name: str = "max duration") -> int:
    """A positive duration limit in minutes."""
    return _require_positive(value, name)


def check_max_agents(value: int, name: str = "max agents") -> int:
    """A positive number of parallel agents, capped at MAX_AGENTS_CAP."""
    return min(_require_positive(value, name), MAX_AGENTS_CAP)
