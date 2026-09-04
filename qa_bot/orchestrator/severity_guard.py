"""Deterministic severity guard for QA issues.

The GitHub Action fails a deploy whenever the bot reports >=1 issue with
``severity == "critical"`` (see ``action/entrypoint.sh``). That count is taken
from the **raw, worker-reported** issues that stream out during exploration --
it does NOT pass through the synthesis agent's dedup/curation step. So a single
Haiku worker over-escalating one observation to "critical" is enough to turn a
clean deploy red.

The worker/synthesis *prompts* ask the model to calibrate severity correctly,
but prompt guidance alone is not a reliable gate for a deploy-blocking check.
This module is the defense-in-depth backstop: a small, deterministic classifier
that recognises a few well-understood NON-regression signatures and caps their
severity so they can never, on their own, fail a deploy:

1. **Self-inflicted automation/tooling failures** -- the bot's own click /
   navigation tooling timed out (e.g. Playwright "Timeout 30000ms exceeded",
   or the model's narrative that an element "is visible/rendered but the click
   won't register"). That is a flake in the *tester*, not a defect in the app
   under test. Capped to ``minor`` (it's an inconclusive observation, not a
   verified app bug).

2. **Environment limitations** -- errors that are inherent to the staging
   environment rather than the code: third-party OAuth providers (Google,
   Apple, etc.) refusing a bare-IP / non-HTTPS / disallowed ``redirect_uri``
   (Google's ``Error 400: invalid_request`` / "doesn't comply with ... OAuth
   2.0 policy"). These cannot pass on a plain-HTTP raw-IP host by design and
   work on the real HTTPS production domain. Capped to ``minor``.

3. **Guessed-URL 404s** -- a 404 reached by the bot *typing a guessed path*
   (e.g. ``/contact/``) rather than by following a real link in the rendered
   UI. A guessed path that 404s is not evidence of a missing feature. Capped
   to ``minor``.

4. **Worker-tagged known issues** -- a description starting with ``[KNOWN]``
   (tolerating markdown decoration a half-obeying model plausibly adds:
   ``**[KNOWN]**``, ``[Known]``, ``- [KNOWN]``), the marker the worker prompt
   tells the model to emit when an observation
   matches the operator-provided known-issues list. The semantic *matching* of
   findings against that list deliberately stays with the model (no reliable
   textual signature), but the prefix itself is deterministic: an issue the
   worker has explicitly tagged as already-acknowledged must never fail a
   deploy on its own, even if the model then contradicts itself on severity.
   Safe because the prompt forbids tagging materially-worse/different behavior
   ``[KNOWN]`` -- a real regression in a known-issue area arrives untagged and
   keeps its severity. Capped to ``minor``.

5. **Signup test-data collisions** -- a signup/registration attempt rejected
   because the email/username "already exists / is taken / is registered".
   The runner reuses test identities across runs, so this is almost always a
   collision with an account a previous run created, not a broken signup
   (the synthesis prompt's "Likely False Positives" #1). Requires BOTH a
   signup context word AND the already-exists phrase; a description that
   says the site *accepted* / *allowed* a duplicate is a real bug and is
   excluded. Capped to ``minor``.

6. **Post-signup anti-abuse artifacts** -- a freshly created account being
   "restricted", "flagged", "suspended", "associated with another account"
   or "pattern-matched" right after signup. The runner's single IP address
   and browser fingerprint create many accounts, so the site's anti-abuse
   system is reacting -- correctly -- to the tester (synthesis "Likely False
   Positives" #4). Requires BOTH a restriction signal AND a new/just-created
   account signal, so an *existing* user being suspended keeps its severity.
   Capped to ``minor``.

This guard only ever **lowers** severity, never raises it, and it is
intentionally conservative: it matches specific, high-confidence phrases so it
won't mask genuine app regressions (a real broken click that surfaces a JS
error, a real 5xx, a real broken link the bot actually followed, etc.). Genuine
issues keep whatever severity the worker assigned.

There is deliberately **NO** signature for the mono PR #592 class -- "field X
has no client-side validation", "this link opens in the same tab instead of a
new one". Those read *identically* whether they are true or false; what
separates them is EVIDENCE, which this module cannot see. Two attempts at a
signature masked, between them, every one of ten plausible real-bug
descriptions an LLM worker would write ("the checkout endpoint has no input
validation -- it silently accepts a negative quantity and issues a refund",
"the Terms link opens in the same tab during checkout, wiping the order
form"), because the true and the false finding share their whole vocabulary.
The right fix is upstream and is what this class of false positive actually
got: the ref list now carries the attributes (browser/controller.py) so the
model can check instead of guess, and the worker/synthesis prompts tell it to.
See docs/troubleshooting.md, "Bot Reports a Missing HTML Attribute That Is
Actually There".

For the same reason there is **NO** signature for data-loss claims
(onlinestoryservices PR #1067): "reloading destroyed previously-saved chapter
content" is word-for-word what a REAL data-loss regression reads like. What
separates the true finding from the false one is whether the worker ever saved
the content -- evidence that lives in the flow's action history, which this
module cannot see. That class is handled upstream too: the worker prompt
section "Data-Loss Claims Require a Verified Save First" and synthesis Report
Quality Standard 12 / "Likely False Positives" #7 make the save an explicit,
checkable precondition. See docs/troubleshooting.md, "Bot Reports Data Loss It
Never Established Was Saved".
"""

from __future__ import annotations

import re

# Severity ordering, lowest -> highest. Used to enforce "cap, never raise".
_SEVERITY_RANK = {
    "cosmetic": 0,
    "minor": 1,
    "major": 2,
    "critical": 3,
}

# The ceiling we apply to a recognised non-regression. ``minor`` (not
# ``cosmetic``) so the observation still shows up in the report for a human to
# notice, but can never fail the deploy on its own.
_NON_REGRESSION_CEILING = "minor"


# 1. Self-inflicted automation / tooling failures ----------------------------
# Playwright surfaces click/navigation timeouts as "Timeout 30000ms exceeded".
# The model frequently narrates the same root cause as "element is visible but
# the click won't register / click detection fails / not clickable" -- which is
# a tester flake, not an app defect.
# A tooling failure is distinguished from a real app bug by the presence of a
# TIMEOUT / "won't register" / "click detection" signal -- NOT by the generic
# word "fails" (a real broken link can also "fail"). We deliberately anchor on
# those tester-flake signatures so genuine regressions keep their severity.
_TOOLING_FAILURE_PATTERNS = [
    re.compile(r"timeout\s*\d+\s*ms\s*exceeded", re.IGNORECASE),
    # "N-second timeout" counts only next to the bot's OWN interaction
    # vocabulary (click/tap/element/locator/selector). A bare "30-second
    # timeout" is just as often a narrated backend hang ("the request hits the
    # 30-second timeout and the order is never placed") -- a real outage that
    # must keep its severity.
    re.compile(
        r"\b(?:click\w*|tap\w*|element|locator|selector)\b[^.]{0,60}?\b\d+\s*-?\s*second\s+timeout",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b\d+\s*-?\s*second\s+timeout\b[^.]{0,60}?\b(?:click\w*|tap\w*|element|locator|selector)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bclick\s+detection\b", re.IGNORECASE),
    # "click(ing) ... times out / timed out" -- a click TIMEOUT (allow a short
    # gap such as a quoted element name between the verb and the timeout phrase).
    re.compile(r"click(?:ing|s)?\b[^.]{0,40}?(?:times?\s+out|timed\s+out|timing\s+out)", re.IGNORECASE),
    # A click/tap/element "won't/doesn't/did not register|respond" -- the
    # silent-click flake. Anchored on the bot's OWN interaction (click/tap/
    # element) because the bare phrase over-matches real outages: "the API
    # server does not respond" / "search box does not respond to input" are
    # genuine app regressions that must keep their severity.
    re.compile(
        r"\b(?:click|tap|element)\w*[^.]{0,40}?"
        r"(?:won'?t|does(?:\s*n'?t| not)|did(?:\s*n'?t| not)|will not)\s+(?:register|respond)",
        re.IGNORECASE,
    ),
    # NOTE: a bare "not clickable" is deliberately NOT a tooling signal --
    # Playwright's "element is not clickable at point (x,y): another element
    # intercepts" is frequently a genuine overlay/z-index regression (a
    # cookie banner covering the Buy button). The true flake phrasings carry a
    # timeout / "click detection" / "won't register" signal, matched above.
    re.compile(r"element\s+(?:is\s+)?(?:visible|rendered)[^.]*(?:but|yet|however)[^.]*click", re.IGNORECASE),
    # Automation-vocabulary timeouts only. A bare "waiting for ... timeout" is
    # NOT enough: "hangs waiting for /api/pay ... 504 gateway timeout" is a real
    # backend failure. "waiting for locator/selector/element" is Playwright's
    # own wait idiom, so that phrasing (with or without the timeout word) is a
    # tester flake.
    re.compile(r"\b(?:playwright|locator)\b.*\btimeout", re.IGNORECASE),
    re.compile(r"\bwaiting\s+for\s+(?:locator|selector|element)\b", re.IGNORECASE),
]

# 2. Environment limitations (third-party OAuth refusing non-HTTPS/IP host) ---
# This class is the easiest to over-match, so it requires BOTH an auth-flow
# context AND a redirect-URI / host-policy refusal signal (see
# classify_non_regression). The brand name alone (e.g. "google") is deliberately
# NOT a sufficient provider signal -- many ordinary app pages mention Google,
# and a real in-app bug like "avatar upload on the Google-linked profile returns
# Error 400 from our API" must keep its severity. Only an actual OAuth/SSO
# sign-in context counts.
_OAUTH_PROVIDER_PATTERNS = [
    re.compile(r"\boauth\b", re.IGNORECASE),
    re.compile(r"\bopenid\b", re.IGNORECASE),
    re.compile(r"\bsso\b", re.IGNORECASE),
    re.compile(r"\bredirect[_\s-]?uri\b", re.IGNORECASE),
    re.compile(r"\bsign[\s-]?in\s+with\b", re.IGNORECASE),
    re.compile(r"\b(?:google|apple|facebook|microsoft|github)\s+(?:sign[\s-]?in|login|log[\s-]?in|oauth|auth|sso)\b", re.IGNORECASE),
]
# The redirect-URI / host-policy refusal signal. ``error 400`` on its own is NOT
# here -- a bare 400 is too generic; it only contributes when it co-occurs with
# one of these OAuth-redirect-specific phrases (which the AND-gate enforces).
# NOTE: a bare "redirect_uri" is NOT a refusal signal (it lives in the provider
# list above, as auth-flow *context*). Keeping it in both lists collapsed the
# AND-gate to a single token: "our callback drops the redirect_uri query param"
# -- a real in-app OAuth regression -- satisfied both halves and was downgraded.
# The policy half must name the PROVIDER'S refusal, not merely the parameter.
_OAUTH_POLICY_PATTERNS = [
    re.compile(r"invalid_request", re.IGNORECASE),
    re.compile(r"doesn'?t\s+comply\s+with", re.IGNORECASE),
    re.compile(r"oauth\s*2\.0\s*policy", re.IGNORECASE),
    re.compile(r"\b(?:https?\s+required|must\s+use\s+https|not\s+a\s+valid\s+(?:origin|redirect(?:[_\s-]?uri)?))", re.IGNORECASE),
]

# 3. Guessed-URL 404s --------------------------------------------------------
_NOT_FOUND_PATTERNS = [
    re.compile(r"\b404\b"),
    re.compile(r"not\s+found", re.IGNORECASE),
    re.compile(r"\bmissing\b", re.IGNORECASE),
    re.compile(r"does(?:\s*n'?t| not)\s+exist", re.IGNORECASE),
]
# Signals that the bot reached the 404 by GUESSING/typing a path rather than by
# following a real link in the rendered UI.
# NOTE: a BARE "navigated to X" is NOT a guess signal -- it's the most common way
# a worker describes a link it actually FOLLOWED (e.g. "clicked the pricing link,
# which navigated to /pricing and returned 404" is a real broken link that must
# keep its severity). We require an explicit guess marker -- "directly to",
# "typed/entered/guessed/assumed", "common path" -- so a followed link is never
# silently downgraded. "tried/attempted" are NOT guess markers ("the app tried
# to load /reports/123 and returned 404" narrates a link the bot followed), and
# neither is "expected path" ("redirect goes to the wrong URL; expected path
# /order/confirm" describes a real broken redirect).
_GUESSED_PATH_PATTERNS = [
    re.compile(r"\bnavigat(?:e|ed|ing)\s+directly\s+to\b", re.IGNORECASE),
    re.compile(r"\b(?:typed|entered|guessed|assumed)\b.*\b(?:url|path|/\w)", re.IGNORECASE),
    re.compile(r"\bcommon\s+(?:url|path|route)\b", re.IGNORECASE),
    re.compile(r"\bendpoint\s+(?:is\s+)?missing\b", re.IGNORECASE),
]


# 5. Signup test-data collisions ---------------------------------------------
# Both halves are required: the signup CONTEXT (so "login returns 500 for
# existing users" -- no collision phrase -- keeps its severity) and the
# already-exists PHRASE (so "signup returns 500" keeps its severity).
_SIGNUP_CONTEXT_PATTERNS = [
    re.compile(r"\bsign[\s-]?up\b", re.IGNORECASE),
    re.compile(r"\bregist(?:er|ration|ering)\b", re.IGNORECASE),
    re.compile(r"\b(?:creat(?:e|ing|ed)|new)\s+(?:an?\s+|new\s+)?(?:account|user)\b", re.IGNORECASE),
    re.compile(r"\baccount\s+creation\b", re.IGNORECASE),
]
_ALREADY_EXISTS_PATTERNS = [
    re.compile(
        r"\balready\s+(?:exists?|taken|registered|in\s+use|been\s+(?:registered|taken|used)|has\s+an\s+account)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:email|username|e-mail|address)\s+(?:is\s+)?(?:taken|in\s+use|unavailable)\b", re.IGNORECASE),
]
# A description saying the site ACCEPTED a duplicate ("signup accepts an
# email that already exists -- duplicate accounts created") is a genuine bug
# that merely quotes the collision phrase. Never cap those.
_DUPLICATE_ACCEPTED_PATTERNS = [
    re.compile(r"\bduplicate\b", re.IGNORECASE),
    re.compile(r"\b(?:accept(?:s|ed)?|allow(?:s|ed)?|permit(?:s|ted)?)\b", re.IGNORECASE),
    re.compile(r"\b(?:second|two|multiple)\s+accounts?\b", re.IGNORECASE),
]

# Rule 5, second escape: the collision phrasing describes a REAL failure.
# "Signup returns 500 — a brand-new address is rejected with 'email already
# registered'" carries both halves of the collision signature and none of the
# duplicate-accepted wording, so without this it would be capped to minor and
# could never fail a deploy — the exact masking the guard promises not to do
# (2026-09-03 review). Deliberately narrow: a server-error signal or an
# explicitly-fresh identity, never a bare "error" (a genuine collision is
# almost always narrated as one).
_SIGNUP_REAL_FAILURE_PATTERNS = [
    re.compile(r"\b5\d{2}\b"),
    re.compile(r"\binternal server error\b", re.IGNORECASE),
    re.compile(r"\bcrash(?:es|ed|ing)?\b", re.IGNORECASE),
    re.compile(r"\b(?:brand[-\s]?new|fresh|unused|never[-\s]+(?:been[-\s]+)?"
               r"(?:registered|used|seen))\b", re.IGNORECASE),
    re.compile(r"\b(?:random(?:ly)?|newly|uniquely)[-\s]*generated\b", re.IGNORECASE),
]

# 6. Post-signup anti-abuse artifacts ----------------------------------------
# Both halves are required: the restriction SIGNAL and the FRESH-ACCOUNT
# signal. "Existing user suspended after login" has no fresh-account signal
# and keeps its severity; "new account creation returns 500" has no
# restriction signal and keeps its severity.
# A bare "restricted/suspended/locked" is NOT enough: "newly created account
# is locked out — login after registration returns 'account suspended'" is a
# broken signup, not an anti-abuse artifact. The restriction must come with
# an anti-abuse/review context.
_ANTI_ABUSE_PATTERNS = [
    re.compile(r"\b(?:restricted|flagged|suspended|locked)\b.{0,80}\b(?:pending|under|manual|security)\s+review\b", re.IGNORECASE | re.DOTALL),
    re.compile(r"\bassociated\s+with\s+(?:another|an\s+existing|a\s+different)\s+(?:account|user)\b", re.IGNORECASE),
    re.compile(r"\bpattern[\s-]?match(?:ed|ing)?\b", re.IGNORECASE),
    re.compile(r"\b(?:fraud|anti-?abuse|abuse\s+(?:detection|prevention)|unusual\s+activity|suspicious\s+activity)\b", re.IGNORECASE),
]
_FRESH_ACCOUNT_PATTERNS = [
    re.compile(r"\b(?:after|following|upon|post)[\s-]+(?:a\s+|the\s+|successful\s+|completing\s+)*(?:sign[\s-]?up|registration|registering|account\s+creation|creating\s+(?:an?\s+|the\s+)?account)\b", re.IGNORECASE),
    re.compile(r"\b(?:new|newly[\s-]created|just[\s-]created|fresh|freshly[\s-]created|brand[\s-]new)\s+(?:user\s+)?account\b", re.IGNORECASE),
    re.compile(r"\baccount\s+(?:that\s+)?(?:was\s+)?just\s+(?:created|registered|signed\s+up)\b", re.IGNORECASE),
    re.compile(r"\bimmediately\s+after\s+(?:creating|registering|signing\s+up)\b", re.IGNORECASE),
]

# 4. Worker-tagged known issues ----------------------------------------------
# The tag must sit at the START of the description -- that anchor is what makes
# the cap safe (a mid-sentence mention of a known issue is not a self-tag). But
# this class exists as a backstop for a HALF-obeying model, and a half-obeying
# model plausibly decorates the tag: "**[KNOWN]**", "[Known]", "- [KNOWN]". So
# tolerate leading whitespace, a list marker ("-"/"*" plus space), and
# bold/emphasis asterisks, and match the tag itself case-insensitively.
_KNOWN_TAG_PATTERN = re.compile(r"^\s*(?:[-*]\s+)?\*{0,2}\[known\]", re.IGNORECASE)


def _matches_any(patterns, text: str) -> bool:
    return any(p.search(text) for p in patterns)


def classify_non_regression(text: str) -> str | None:
    """Return a short reason code if ``text`` describes a known non-regression.

    Returns one of ``"tooling_failure"``, ``"environment_limitation"``,
    ``"guessed_url_404"``, ``"signup_collision"``,
    ``"post_signup_antiabuse"`` or ``None`` if no signature matches.

    ``text`` should be the combined human-readable signal for the issue
    (description + action context). Matching is conservative and high-confidence
    so genuine app bugs are not misclassified.
    """
    if not text:
        return None

    # 1. Tooling/automation failure (highest priority -- a click timeout that
    #    the model narrated as "broken" should be downgraded even if it also
    #    mentions a feature name).
    if _matches_any(_TOOLING_FAILURE_PATTERNS, text):
        return "tooling_failure"

    # 2. Third-party OAuth refusing the staging host. Require BOTH an OAuth/
    #    provider signal AND an OAuth-policy/redirect-uri signal so we don't
    #    swallow a genuine in-app auth bug that merely mentions "login".
    if _matches_any(_OAUTH_PROVIDER_PATTERNS, text) and _matches_any(_OAUTH_POLICY_PATTERNS, text):
        return "environment_limitation"

    # 3. A 404 the bot reached by guessing a URL. Require BOTH a not-found
    #    signal AND a guessed-path signal -- a 404 on a link the bot actually
    #    followed is a real broken link and keeps its severity.
    if _matches_any(_NOT_FOUND_PATTERNS, text) and _matches_any(_GUESSED_PATH_PATTERNS, text):
        return "guessed_url_404"

    # 5. Signup rejected because the test identity already exists. Require
    #    BOTH the signup context AND the collision phrase, and never cap a
    #    description that says a duplicate was ACCEPTED, or that reports the
    #    collision as a real failure (a 5xx, a crash, a brand-new address).
    if (
        _matches_any(_SIGNUP_CONTEXT_PATTERNS, text)
        and _matches_any(_ALREADY_EXISTS_PATTERNS, text)
        and not _matches_any(_DUPLICATE_ACCEPTED_PATTERNS, text)
        and not _matches_any(_SIGNUP_REAL_FAILURE_PATTERNS, text)
    ):
        return "signup_collision"

    # 6. Anti-abuse system reacting to a freshly created account. Require
    #    BOTH the restriction signal AND the fresh-account signal.
    if _matches_any(_ANTI_ABUSE_PATTERNS, text) and _matches_any(_FRESH_ACCOUNT_PATTERNS, text):
        return "post_signup_antiabuse"

    return None


def cap_severity(severity: str, description: str, action_context: str = "") -> tuple[str, str | None]:
    """Cap an issue's severity if it matches a known non-regression signature.

    Args:
        severity: The worker-assigned severity (critical/major/minor/cosmetic).
        description: The issue description text.
        action_context: Optional extra context (e.g. the action that triggered
            the issue) to widen the matching surface.

    Returns:
        ``(new_severity, reason)`` where ``new_severity`` is the (possibly
        lowered) severity and ``reason`` is the matched non-regression reason
        code (or ``None`` if nothing matched). Severity is only ever lowered. If
        the issue already sits at/below the ceiling, severity is returned
        unchanged; ``reason`` is still returned (the classification is real even
        when no cap was needed) so a future caller *could* annotate, though the
        current caller (``add_issue``) only acts on it when it actually lowers
        severity.
    """
    normalized = (severity or "minor").strip().lower()
    text = f"{description or ''}\n{action_context or ''}"

    # 4. Worker-tagged known issue. Checked on the DESCRIPTION only (a
    #    "[KNOWN]" appearing mid-sentence, or only in the action context,
    #    is not the worker's deliberate self-tag).
    if _KNOWN_TAG_PATTERN.match(description or ""):
        reason = "known_issue"
    else:
        reason = classify_non_regression(text)
    if reason is None:
        return normalized, None

    current_rank = _SEVERITY_RANK.get(normalized, _SEVERITY_RANK["minor"])
    ceiling_rank = _SEVERITY_RANK[_NON_REGRESSION_CEILING]

    if current_rank > ceiling_rank:
        return _NON_REGRESSION_CEILING, reason

    # Already at/below the ceiling -- leave severity, but report the reason so
    # callers can annotate the issue ("environment limitation", etc.).
    return normalized, reason
