"""Shared rules for recognising and masking secrets.

Two different questions get asked in the codebase and they need different
heuristics:

- "Is this credential / user-data *key* a secret?" — matched against keys
  like ``PASSWORD`` or ``ADMIN_PW`` that the operator or user supplied.
  Deliberately tight: a false positive here hides a value from the AI
  (``format_user_data_for_prompt``), so ``pin`` (matches ``shipping``) and
  ``key`` (matches ``keyword``) are excluded.
- "Does this UI element *label* look like a secret field?" — matched
  against labels like ``Password input`` or ``API key field`` before a typed
  value is printed to a console/log. Broader, because a false positive only
  hides a typed value in a log line.

Masking by *value* (``SharedFlowState.get_secret_values`` →
``ChatLogger.register_secrets`` / worker action events) is the primary
defence; these keyword lists are the fallback for values we never learned.
"""

import re
from typing import Iterable

# Substrings that mark a credential / user-data key as secret.
SENSITIVE_KEY_PATTERNS: tuple[str, ...] = (
    "password", "passwd", "passphrase", "secret", "token",
    "api_key", "apikey", "credential", "auth", "bearer",
    "private_key", "privatekey", "cvv", "cvc", "ssn",
)
# Short tokens that are only secrets as whole words: ``user_pass``, ``PASS``,
# ``ADMIN_PW`` and ``pwd`` are passwords; ``passenger_name`` and
# ``passport_number`` are not (a false positive here hides a value from the
# AI and, via by-value redaction, mangles the report).
_SENSITIVE_KEY_TOKENS = re.compile(r"(?<![a-z])(?:pass|pwd|pw)(?![a-z])")

# Substrings that mark a UI element label as a secret field.
SENSITIVE_FIELD_KEYWORDS: tuple[str, ...] = (
    "password", "passwd", "pwd", "pw", "secret", "token", "key", "auth",
    "credential", "pin", "api", "private", "ssn", "cvv", "cvc",
)

# Values shorter than this are never redacted by value: a 1-3 char secret
# would match ordinary text everywhere and shred the logs.
MIN_SECRET_LENGTH = 4


def is_sensitive_key(key: str) -> bool:
    """True if a credential / user-data key name looks like a secret."""
    key_lower = key.lower()
    if any(pattern in key_lower for pattern in SENSITIVE_KEY_PATTERNS):
        return True
    return bool(_SENSITIVE_KEY_TOKENS.search(key_lower))


def is_sensitive_field_label(label: str) -> bool:
    """True if a UI element label / ref description looks like a secret field."""
    label_lower = (label or "").lower()
    return any(keyword in label_lower for keyword in SENSITIVE_FIELD_KEYWORDS)


def looks_like_secret_value(value: str) -> bool:
    """Is ``value`` safe to redact by VALUE everywhere it appears?

    A short dictionary word used as a password ("admin", "test") would turn
    every ordinary "admin" in a report into "[MASKED - 5 chars]" — mangling
    the report and telling the reader exactly which word the password was.
    Such values are still masked by KEY NAME on the prompt/log paths; only
    by-value redaction is skipped for them.
    """
    if not value or len(value) < MIN_SECRET_LENGTH:
        return False
    if len(value) >= 8:
        return True
    return any(not ch.isalpha() for ch in value)


def mask_value(value: str) -> str:
    """Length-hinting placeholder shown wherever a secret value is masked."""
    return f"[MASKED - {len(value)} chars]"


def redact_values(text: str, secrets: Iterable[str]) -> str:
    """Replace every occurrence of each secret value in ``text``.

    Longest secrets are replaced first so a secret that contains another
    (``hunter2-admin`` / ``hunter2``) is masked whole.
    """
    if not text:
        return text
    for secret in sorted(set(secrets), key=len, reverse=True):
        if looks_like_secret_value(secret) and secret in text:
            text = text.replace(secret, mask_value(secret))
    return text


def redact_obj(obj, secrets: Iterable[str]):
    """``redact_values`` over every string inside a nested dict/list (keys
    are left alone). Redact before JSON-encoding, so a secret containing
    quotes or non-ASCII isn't hidden from the match by escaping."""
    if isinstance(obj, str):
        return redact_values(obj, secrets)
    if isinstance(obj, dict):
        return {k: redact_obj(v, secrets) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact_obj(v, secrets) for v in obj]
    return obj
