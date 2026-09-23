"""OpenRouter provider — runs QA Bot on a non-Claude model (GPT-6 Luna) by id.

Selected by model id only: a "vendor/slug" id (``openai/gpt-6-luna``) routes
here through ``qa_bot.ai.create_provider``; Claude ids keep ClaudeProvider,
which stays the default. Built for a like-for-like A/B: the prompts, the
action/supervisor parsers and the history windowing are ClaudeProvider's own
(borrowed below, not copied), so this module only translates the request to
OpenAI chat format and the reply back into the AIProvider event protocol.

Wire facts (OpenRouter docs, and ai_news/llm/openrouter.py which runs Luna in
production since 2026-09-23): one non-streaming POST to /chat/completions;
``usage.prompt_tokens`` INCLUDES ``prompt_tokens_details.cached_tokens`` and
``cache_write_tokens``; ``usage.cost`` is the real charge, summed here as
``charged_cost_usd``; ``error.metadata.error_type`` is the stable error
field: a policy block is HTTP 403 with error_type ``refusal`` or
``content_policy_violation`` (moderation adds ``reasons``/``flagged_input``),
while a permission 403 has neither; a 402 with ``Retry-After`` is the
in-flight spending budget (transient), one without is no credit. A provider
error that interrupts generation arrives as a 200 whose ``choices[0]`` has
``finish_reason == "error"`` and an ``error`` object beside the partial
content. A model's own refusal is ``message.refusal`` or
``finish_reason == "content_filter"``.
OpenAI caches prompt prefixes automatically (there is no cache_control), so
the worker system prompt goes out as ONE message with the run-wide base
first, and the history keeps ClaudeProvider's stepped window so the prefix
stays stable.

Data path: screenshots, page text and any test credentials in prompts go to
OpenRouter and the model vendor instead of Anthropic.
"""

import asyncio
import base64
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import AsyncGenerator, Optional

import httpx

from qa_bot.config import (
    OPENROUTER_SUPERVISOR_EFFORT,
    OPENROUTER_SYNTHESIS_EFFORT,
    OPENROUTER_TIMEOUT_SECONDS,
    OPENROUTER_WORKER_EFFORT,
)
from .base import AIProvider, FatalProviderError
from .claude_provider import (
    BACKOFF_MULTIPLIER,
    INITIAL_BACKOFF,
    MAX_BACKOFF,
    MAX_PARSE_RETRIES,
    MAX_RETRIES,
    PARSE_CORRECTION_PROMPT,
    SYNTHESIS_MAX_TOKENS_ALWAYS_ON,
    ClaudeProvider,
    _calculate_wait_time,
)
from .prompts import (
    SUPERVISOR_SYSTEM_PROMPT,
    SYNTHESIS_SYSTEM_PROMPT,
    format_supervisor_context,
    format_synthesis_context,
    get_worker_action_prompt,
    get_worker_system_prompt_parts,
)

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# Transient: timeouts, conflicts, rate limits, gateway/upstream hiccups (52x
# are Cloudflare's, which OpenRouter sits behind). See _failure_kind for the
# rest: content-triggered 403s fail the turn, the in-flight budget's 402 is
# retried, other 401/402/403s are account errors (fatal), and any other 4xx
# is a bad request no retry will fix.
RETRY_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529})
# OpenRouter's two policy error_types, both HTTP 403 (errors doc, "Content
# policy"): a filter around the model flagged the input or output, or the
# provider reported the model's refusal as an error.
POLICY_ERROR_TYPES = frozenset({"refusal", "content_policy_violation"})
# What a content-triggered 403 carries even without one of those types:
# moderation's reasons / flagged_input, a content guardrail's patterns.
CONTENT_BLOCK_KEYS = ("reasons", "flagged_input", "patterns")
# limit_source of the in-flight spending budget's 402: transient, sent with
# Retry-After while recent requests settle (limits doc).
IN_FLIGHT_BUDGET = "openrouter_in_flight_budget"
# Same ceilings as ClaudeProvider; reasoning tokens count against them.
WORKER_MAX_TOKENS = 16000
SUPERVISOR_MAX_TOKENS = 12000

_ACCOUNT_HINTS = {
    401: "OpenRouter rejected this API key — check OPENROUTER_API_KEY at openrouter.ai/settings/keys",
    402: "OpenRouter account has no credit — add credits at openrouter.ai/settings/credits",
    403: "OpenRouter refused this API key (permission denied or key limit reached)",
}


class OpenRouterError(Exception):
    """No usable reply. ``usage`` holds the four AIProvider usage counts the
    failed call's responses reported (a refused turn still processed its
    input; a reply cut off mid-generation was still billed)."""

    def __init__(self, message: str, *, usage: Optional[dict] = None):
        super().__init__(message)
        self.usage = usage or usage_counts(None)


class OpenRouterRefusal(OpenRouterError):
    """The provider's policy layer or the model itself declined the request."""


class OpenRouterAccountError(OpenRouterError, FatalProviderError):
    """401 bad key / 402 no credit / a 403 not triggered by content: every
    later call fails the same way, so the worker aborts the run instead of
    retrying."""


@dataclass
class _Reply:
    text: str
    reasoning: str
    finish_reason: Optional[str]
    usage: dict  # the four AIProvider usage keys


def usage_counts(usage) -> dict:
    """OpenAI-shaped usage → the four AIProvider usage keys.

    ``prompt_tokens`` includes cached and cache-write tokens, so the uncached
    input is what remains; ``completion_tokens`` includes reasoning tokens
    (billed at the output rate).
    """
    usage = usage if isinstance(usage, dict) else {}
    details = usage.get("prompt_tokens_details")
    details = details if isinstance(details, dict) else {}

    def _int(d: dict, key: str) -> int:
        try:
            return max(0, int(d.get(key) or 0))
        except (TypeError, ValueError):
            return 0

    cached = _int(details, "cached_tokens")
    written = _int(details, "cache_write_tokens")
    return {
        "input_tokens": max(0, _int(usage, "prompt_tokens") - cached - written),
        "output_tokens": _int(usage, "completion_tokens"),
        "cache_read_tokens": cached,
        "cache_creation_tokens": written,
    }


def _add_usage(total: dict, usage: dict) -> None:
    for key, value in usage.items():
        total[key] = total.get(key, 0) + value


def _text_of(content) -> str:
    """A message's text: a plain string, or the text parts of a part list."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            p.get("text") or p.get("summary") or ""
            for p in content if isinstance(p, dict)
        )
    return ""


def to_openai_messages(messages: list[dict]) -> list[dict]:
    """Anthropic-format messages → OpenAI chat messages (no system message).

    Text blocks keep their text (``cache_control`` dropped), image blocks
    become ``image_url`` data URLs at high detail (the worker reads small UI
    text off the screenshot; "auto" may downscale it), thinking blocks are
    dropped (another model's reasoning can't be replayed). Consecutive
    same-role messages are merged, so the result strictly alternates when
    the worker's one-off user notes (approval result on resume, data already
    available) follow an assistant turn and meet the next prompt. The worker
    stores each turn prompt-then-action; a history in the pre-2026-09
    action-then-prompt order would alternate too.
    """
    out: list[dict] = []
    for msg in messages:
        role = msg.get("role")
        if role not in ("user", "assistant"):
            continue
        content = msg.get("content")
        blocks = [{"type": "text", "text": content}] if isinstance(content, str) else (content or [])
        parts = []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and block.get("text"):
                parts.append({"type": "text", "text": block["text"]})
            elif block.get("type") == "image" and role == "user":
                source = block.get("source") or {}
                if source.get("type") == "base64" and source.get("data"):
                    media_type = source.get("media_type") or "image/png"
                    parts.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{source['data']}",
                            "detail": "high",
                        },
                    })
        if not parts:
            continue
        if out and out[-1]["role"] == role:
            out[-1]["content"].extend(parts)
        else:
            out.append({"role": role, "content": parts})
    for msg in out:
        if msg["role"] == "assistant":
            # Plain strings: the most portable assistant content form.
            msg["content"] = "\n\n".join(p["text"] for p in msg["content"])
    return out


def _error_of(payload) -> Optional[dict]:
    """The error a response body reports, or None when it holds a reply.

    Either the top-level ``error`` (a failed request, or a 200 whose body
    holds only an error) or, when a provider error interrupted generation,
    the ``error`` on ``choices[0]`` beside ``finish_reason: "error"`` (errors
    doc, Chat Completions). That choice's partial content is never the
    answer.
    """
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    if not error:
        choices = payload.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
        error = choice.get("error")
        if not error and choice.get("finish_reason") == "error":
            error = {"message": "generation failed (finish_reason=error)"}
    if not error:
        return None
    return error if isinstance(error, dict) else {"message": str(error)}


def _failure_kind(status: int, metadata: dict, retry_after: Optional[str]) -> str:
    """How a failed response ends. The stable ``metadata.error_type`` decides
    first, then the status:

    - "refusal": a block triggered by this turn's content (a policy
      error_type, or a 403 carrying moderation / guardrail matches). It
      fails the turn only, because the next turn's content may pass.
    - "retry": transient. RETRY_STATUSES, the in-flight budget's 402, or an
      error in a 200 body that names no status.
    - "account": 401, any other 402 (no wait hint means no credit), any
      other 403. Every later call fails the same way, so the run aborts.
    - "error": anything else; no retry will fix it.
    """
    if metadata.get("error_type") in POLICY_ERROR_TYPES or (
        status == 403 and any(key in metadata for key in CONTENT_BLOCK_KEYS)
    ):
        return "refusal"
    if status == 402 and (retry_after or metadata.get("limit_source") == IN_FLIGHT_BUDGET):
        return "retry"
    if status in _ACCOUNT_HINTS:
        return "account"
    if status == 200 or status in RETRY_STATUSES:
        return "retry"
    return "error"


class OpenRouterProvider(AIProvider):
    """OpenRouter chat-completions implementation of the AIProvider surface."""

    # ClaudeProvider's provider-neutral helpers (action-history recap, action
    # and supervisor parsing, screenshot stripping + stepped history window),
    # borrowed so both providers format and parse identically.
    _format_history = ClaudeProvider._format_history
    _extract_action_from_response = ClaudeProvider._extract_action_from_response
    _parse_supervisor_response = ClaudeProvider._parse_supervisor_response
    _coerce_supervisor_action = staticmethod(ClaudeProvider._coerce_supervisor_action)
    _build_messages_from_history = staticmethod(ClaudeProvider._build_messages_from_history)

    def __init__(
        self,
        api_key: str,
        model: str,
        max_concurrent_calls: int = 2,
        *,
        worker_effort: str = OPENROUTER_WORKER_EFFORT,
        supervisor_effort: str = OPENROUTER_SUPERVISOR_EFFORT,
        synthesis_effort: str = OPENROUTER_SYNTHESIS_EFFORT,
        timeout: float = OPENROUTER_TIMEOUT_SECONDS,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ):
        self.api_key = api_key
        self.model = model
        self.worker_effort = worker_effort
        self.supervisor_effort = supervisor_effort
        self.synthesis_effort = synthesis_effort
        self.timeout = timeout
        self._transport = transport  # test seam (httpx.MockTransport)
        self._api_semaphore = asyncio.Semaphore(max_concurrent_calls)
        self.charged_cost_usd = 0.0
        # Per-run test-data nonce + date, stamped once (see ClaudeProvider).
        self.run_nonce = uuid.uuid4().hex[:6]
        self.run_date = datetime.now().strftime("%Y-%m-%d (%A)")

    def _note_charge(self, usage) -> None:
        """Add a response's ``usage.cost`` (the real charge) to the run total."""
        cost = usage.get("cost") if isinstance(usage, dict) else None
        if isinstance(cost, (int, float)) and not isinstance(cost, bool):
            self.charged_cost_usd += float(cost)

    async def _post(self, operation: str, body: dict) -> tuple[dict, dict]:
        """POST one chat completion with ClaudeProvider's retry semantics →
        (the reply payload, usage counts summed over every attempt).

        Retries transient failures (see _failure_kind), connection errors and
        timeouts with jittered exponential backoff (an exact ``retry-after``
        wins), holding the concurrency semaphore only for the request itself.
        Raises OpenRouterRefusal on a content-triggered block,
        OpenRouterAccountError on a bad key / no credit / other 403, and
        OpenRouterError otherwise. A failed attempt's tokens (a reply cut off
        mid-generation was still billed) stay in the usage returned or raised.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": "Integuide QA Bot",  # attribution in OpenRouter's activity log
        }
        backoff = INITIAL_BACKOFF
        last = "no attempt"
        spent = usage_counts(None)
        for attempt in range(MAX_RETRIES):
            async with self._api_semaphore:
                try:
                    async with httpx.AsyncClient(timeout=self.timeout, transport=self._transport) as client:
                        resp = await client.post(OPENROUTER_URL, headers=headers, json=body)
                except httpx.HTTPError as e:  # connection errors and timeouts
                    last = f"{type(e).__name__}: {e}".rstrip(": ")
                    wait_time = _calculate_wait_time(backoff)
                else:
                    try:
                        payload = resp.json()
                    except ValueError:
                        payload = None
                    raw_usage = payload.get("usage") if isinstance(payload, dict) else None
                    self._note_charge(raw_usage)
                    _add_usage(spent, usage_counts(raw_usage))
                    error = _error_of(payload)
                    if resp.status_code == 200 and isinstance(payload, dict) and error is None:
                        return payload, spent
                    error = error or {}
                    status = resp.status_code
                    if status == 200 and isinstance(error.get("code"), int):
                        status = error["code"]  # an error reported inside a 200 body
                    message = str(error.get("message") or "")[:300]
                    metadata = error.get("metadata") if isinstance(error.get("metadata"), dict) else {}
                    retry_after = resp.headers.get("retry-after")
                    kind = _failure_kind(status, metadata, retry_after)
                    last = f"HTTP {status}" + (f": {message}" if message else "")
                    if kind == "refusal":
                        error_type = metadata.get("error_type")
                        label = error_type if error_type in POLICY_ERROR_TYPES else "flagged content"
                        raise OpenRouterRefusal(
                            f"refused by the provider ({label}, {last})", usage=spent
                        )
                    if kind == "account":
                        reason = _ACCOUNT_HINTS[status] + (f": {message}" if message else "")
                        raise OpenRouterAccountError(f"{reason} (HTTP {status})", usage=spent)
                    if kind == "error":
                        raise OpenRouterError(f"OpenRouter {last}", usage=spent)
                    wait_time = _calculate_wait_time(backoff)
                    if retry_after:
                        try:
                            wait_time = min(float(retry_after), MAX_BACKOFF)
                        except ValueError:
                            pass
            if attempt < MAX_RETRIES - 1:
                # Sleep outside the semaphore so other workers can proceed
                logger.warning(
                    f"{operation}: OpenRouter {last} (attempt {attempt + 1}/{MAX_RETRIES}), "
                    f"retrying in {wait_time:.1f}s..."
                )
                await asyncio.sleep(wait_time)
                backoff *= BACKOFF_MULTIPLIER
        logger.error(f"{operation}: All {MAX_RETRIES} retries exhausted")
        raise OpenRouterError(f"API error after {MAX_RETRIES} retries: OpenRouter {last}", usage=spent)

    async def _complete(
        self, operation: str, system: str, messages: list[dict], *, effort: str, max_tokens: int
    ) -> _Reply:
        """One chat completion → its text, reasoning and usage counts.

        Raises OpenRouterRefusal when the model itself declined."""
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "max_tokens": max_tokens,
            "usage": {"include": True},  # usage.cost, the real charge
        }
        if effort:
            body["reasoning"] = {"effort": effort}
        payload, usage = await self._post(operation, body)
        choices = payload.get("choices") or [{}]
        choice = choices[0] if isinstance(choices[0], dict) else {}
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        finish_reason = choice.get("finish_reason")
        if message.get("refusal") or finish_reason == "content_filter":
            raise OpenRouterRefusal(f"the model declined ({finish_reason or 'refusal'})", usage=usage)
        return _Reply(
            text=_text_of(message.get("content")),
            reasoning=_text_of(message.get("reasoning")) or _text_of(message.get("reasoning_details")),
            finish_reason=finish_reason,
            usage=usage,
        )

    async def analyze_for_worker_stream(
        self,
        screenshot_bytes: bytes,
        ref_list: str,
        current_url: str,
        flow_name: str,
        flow_goal: str,
        action_history: list[dict],
        conversation_history: list[dict] = None,
        prior_context: str = "",
        additional_context: str = "",
        is_first_worker: bool = False,
        worker_number: int = 0,
        flow_description: str = "",
        parent_flow_name: str = "",
        target_domain: str = "",
        viewport_width: int = 1280,
        viewport_height: int = 720,
        credentials: dict[str, str] | None = None,
        user_data: dict[str, dict[str, str]] | None = None,
        known_issues: str = "",
        recheck_context: str = "",
    ) -> AsyncGenerator[dict, None]:
        """The worker turn — same prompts, events and corrective retries as
        ClaudeProvider.analyze_for_worker_stream, over one non-streaming call
        (reasoning, when the reply carries it, is yielded as one thinking
        block after the call returns)."""
        system_base, system_worker_context = get_worker_system_prompt_parts(
            is_first_worker=is_first_worker,
            worker_number=worker_number,
            flow_name=flow_name,
            flow_description=flow_description,
            parent_flow_name=parent_flow_name,
            target_domain=target_domain,
            run_nonce=self.run_nonce,
            current_date=self.run_date,
            known_issues=known_issues,
            recheck_context=recheck_context,
        )
        # One system message, run-wide base first: OpenAI's automatic prefix
        # cache then shares the base across every worker in the run.
        system_prompt = system_base + system_worker_context
        history_text = self._format_history(action_history)
        user_prompt = get_worker_action_prompt(
            url=current_url,
            flow_name=flow_name,
            goal=flow_goal,
            history=history_text if history_text else "No actions taken yet.",
            ref_list=ref_list,
            viewport_width=viewport_width,
            viewport_height=viewport_height,
            prior_context=prior_context,
            additional_context=additional_context,
            credentials=credentials,
            user_data=user_data,
        )
        # Stored by the worker in Anthropic block form, like ClaudeProvider's
        # turns, so its screenshot stripping and checkpoints work unchanged;
        # translated to OpenAI format per request.
        current_content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.standard_b64encode(screenshot_bytes).decode("utf-8"),
                },
            },
            {"type": "text", "text": user_prompt},
        ]
        messages = to_openai_messages(
            self._build_messages_from_history(conversation_history or [])
            + [{"role": "user", "content": current_content}]
        )

        cumulative_usage = usage_counts(None)
        action = None
        reply = None
        for parse_attempt in range(MAX_PARSE_RETRIES + 1):
            try:
                reply = await self._complete(
                    "Worker", system_prompt, messages,
                    effort=self.worker_effort, max_tokens=WORKER_MAX_TOKENS,
                )
            except OpenRouterAccountError:
                raise  # fatal: the worker aborts the run (fatal_api_error_reason)
            except OpenRouterRefusal as e:
                # Like ClaudeProvider's refusal branch: re-sending the same
                # request would be refused the same way, so fail the turn
                # with an honest reason (the flow-level retry still applies).
                _add_usage(cumulative_usage, e.usage)
                logger.warning(f"Worker: {self.model} declined the turn ({e}); not retrying the identical request")
                yield {
                    "type": "error",
                    "error": (
                        f"{self.model} declined to generate this turn ({e}) — the "
                        "test content most likely tripped a safety filter. Retry "
                        "with mundane, inoffensive free text."
                    ),
                    "thinking": "",
                    "raw_response": "",
                    **cumulative_usage,
                }
                return
            except OpenRouterError as e:
                _add_usage(cumulative_usage, e.usage)
                yield {
                    "type": "error",
                    "error": str(e),
                    "thinking": "",
                    "raw_response": "",
                    **cumulative_usage,
                }
                return
            _add_usage(cumulative_usage, reply.usage)

            if reply.reasoning:
                yield {"type": "thinking_start"}
                yield {"type": "thinking_delta", "text": reply.reasoning}
                yield {"type": "thinking_complete", "text": reply.reasoning}

            if not reply.text.strip():
                # Reasoning but no answer (e.g. the budget ran out mid-thought):
                # re-sample the identical request within the parse budget.
                if parse_attempt < MAX_PARSE_RETRIES:
                    logger.warning(
                        f"Worker: response contained no text (finish_reason="
                        f"{reply.finish_reason}, attempt {parse_attempt + 1}/"
                        f"{MAX_PARSE_RETRIES + 1}), retrying"
                    )
                    continue
                yield {
                    "type": "error",
                    "error": "No text response from AI",
                    "thinking": reply.reasoning,
                    "raw_response": "",
                    **cumulative_usage,
                }
                return

            try:
                action = self._extract_action_from_response(reply.text)
                break
            except ValueError as e:
                if parse_attempt < MAX_PARSE_RETRIES:
                    logger.warning(
                        f"Worker: response was not a valid JSON action "
                        f"(attempt {parse_attempt + 1}/{MAX_PARSE_RETRIES + 1}), "
                        f"asking model to reformat"
                    )
                    # Local to this turn: the worker stores only the final
                    # successful exchange in its history.
                    messages = messages + [
                        {"role": "assistant", "content": reply.text},
                        {"role": "user", "content": [{
                            "type": "text",
                            "text": f"{PARSE_CORRECTION_PROMPT}\n\nSpecifically: {str(e)[:500]}",
                        }]},
                    ]
                    continue
                yield {
                    "type": "error",
                    "error": str(e),
                    "thinking": reply.reasoning,
                    "raw_response": reply.text,
                    **cumulative_usage,
                }
                return

        yield {
            "type": "complete",
            "action": action,
            "thinking": reply.reasoning,
            "assistant_content": [{"type": "text", "text": reply.text}],
            "user_content": current_content,
            **cumulative_usage,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        }

    async def analyze_for_supervisor(
        self,
        active_workers: list[dict],
        blocked_workers: list[dict],
        pending_flows: list[dict],
        completed_flows: list[dict],
        issues: list[dict],
        goal: str = "",
    ) -> dict:
        """Supervisor decision — same prompt and parsing as ClaudeProvider."""
        context = format_supervisor_context(
            active_workers=active_workers,
            blocked_workers=blocked_workers,
            pending_flows=pending_flows,
            completed_flows=completed_flows,
            issues=issues,
            goal=goal,
        )
        reply = await self._complete(
            "Supervisor", SUPERVISOR_SYSTEM_PROMPT, [{"role": "user", "content": context}],
            effort=self.supervisor_effort, max_tokens=SUPERVISOR_MAX_TOKENS,
        )
        if not reply.text.strip():
            return {"action": "observe", "reasoning": "No response from AI", **reply.usage}
        result = self._parse_supervisor_response(reply.text)
        result.update(reply.usage)
        return result

    async def generate_synthesis_report(
        self,
        target_url: str,
        duration: str,
        flows_tested: int,
        issues: list[dict],
        completed_flows: list[dict],
        blocked_flows: list[dict] | None = None,
        incomplete_flows: list[dict] | None = None,
        goal: str = "",
        known_issues: str = "",
        previous_report: str = "",
    ) -> dict:
        """Final QA report — same prompt as ClaudeProvider.

        A refusal, an empty reply or one cut off by a provider error
        mid-generation raises, so SynthesisAgent falls back to its
        deterministic report (and the gate to the raw critical count) with
        the cause named, instead of shipping an empty or partial report.
        """
        context = format_synthesis_context(
            target_url=target_url,
            duration=duration,
            flows_tested=flows_tested,
            issues=issues,
            completed_flows=completed_flows,
            blocked_flows=blocked_flows,
            incomplete_flows=incomplete_flows,
            goal=goal,
            known_issues=known_issues,
            previous_report=previous_report,
        )
        # Reasoning is spent out of max_tokens before the report, and the
        # qa-verdict fence is the report's LAST section: take the headroom.
        reply = await self._complete(
            "Synthesis", SYNTHESIS_SYSTEM_PROMPT, [{"role": "user", "content": context}],
            effort=self.synthesis_effort, max_tokens=SYNTHESIS_MAX_TOKENS_ALWAYS_ON,
        )
        report = reply.text
        if not report.strip():
            raise OpenRouterError(
                f"synthesis reply had no text (finish_reason={reply.finish_reason})",
                usage=reply.usage,
            )
        if reply.finish_reason == "length":
            logger.warning("Synthesis report hit the max_tokens output limit and was truncated")
            if report.count("```") % 2 == 1:
                report += "\n```"  # close a fence the truncation left open
            report += (
                "\n\n> **Warning:** This report was truncated at the model's "
                "output token limit — later sections (e.g. Test Coverage, "
                "Recommendations) may be missing."
            )
        return {"report": report, **reply.usage}
