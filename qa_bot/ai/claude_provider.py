import asyncio
import base64
import json
import logging
import random
import re
import uuid
from datetime import datetime
from typing import AsyncGenerator, Optional
from urllib.parse import urlparse
from anthropic import AsyncAnthropic, RateLimitError, APIError, APIConnectionError, APITimeoutError
from pydantic import ValidationError

from qa_bot.browser.controller import DUPLICATE_REF_PREFIX
from qa_bot.config import DEFAULT_MODEL, MODEL_SONNET, MODEL_OPUS
from .base import AIProvider, AgentAction
from .prompts import (
    get_worker_system_prompt_parts,
    get_worker_action_prompt,
    SUPERVISOR_SYSTEM_PROMPT,
    format_supervisor_context,
    SYNTHESIS_SYSTEM_PROMPT,
    format_synthesis_context
)

logger = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES = 5
INITIAL_BACKOFF = 2.0  # seconds
MAX_BACKOFF = 60.0  # seconds
BACKOFF_MULTIPLIER = 5.0
JITTER_FACTOR = 0.5  # Add up to 50% randomness to backoff

# Corrective retries when a worker response contains no parseable JSON
# action — or no text at all (a thinking-only response, an intermittent
# model failure mode). One bad reply must not kill the whole flow, so we
# ask the model to reformat (or simply re-sample, when there is no text
# to correct) before giving up.
MAX_PARSE_RETRIES = 2
PARSE_CORRECTION_PROMPT = (
    "Your last reply did not contain a valid JSON action object. "
    "Respond again with ONLY a single JSON object describing your chosen "
    'action (including the "action_type" field) - no prose, no markdown, '
    "no explanation outside the JSON."
)

# Prompt caching configuration
# Ephemeral cache has a 5-minute TTL - sufficient for multi-turn conversations
CACHE_CONTROL_EPHEMERAL = {"type": "ephemeral"}

# History windowing for long worker flows.
#
# Every turn re-sends the full text conversation (each user turn carries the
# complete action prompt: ref list, recap, boilerplate — several K tokens), so
# an uncapped flow grows quadratically in cost and, deep into a flow, the
# request gets large enough to feed the intermittent "thinking-only / no text"
# response failure mode (2026-07-17 prod QA run: the image-fallback flow died
# to it twice, ~46 turns deep, with the run's token counter moving ~100K per
# turn). Windowing bounds both. Continuity survives trimming because every
# turn's prompt independently carries the flow goal (system prompt) and a
# recent-actions recap (action_history), and an omission note tells the model
# the tail was cut.
#
# The boundary is stepped, not a pure sliding tail: dropping `len - KEEP`
# messages each call would shift the kept prefix every turn and defeat prompt
# caching. Instead the drop count rounds up to a multiple of
# HISTORY_TRIM_STEP, so the boundary only jumps every STEP/2 turns (a turn
# appends 2 messages) — one cache re-write per jump instead of one per turn.
# The kept window oscillates between (WINDOW - STEP) and WINDOW messages.
HISTORY_WINDOW_MESSAGES = 60  # ~30 exchanges; only histories beyond this are trimmed
HISTORY_TRIM_STEP = 20  # drop count rounds up to a multiple of this
HISTORY_OMITTED_NOTE = (
    "[Context note: the earliest {count} messages of this session were "
    "omitted to keep the conversation compact. Rely on the Recent Actions "
    "list in the current prompt and your flow goal for continuity.]"
)


# Model-id prefixes that REQUIRE adaptive thinking (they reject the legacy
# enabled+budget_tokens form with HTTP 400). Keyed on the config constants so it
# stays correct as they're bumped: today these are Sonnet 5 / Opus 4.8. Matched
# as prefixes so dated snapshot ids (e.g. "claude-sonnet-5-20260514") route the
# same way. Any other model — Haiku 4.5, or a legacy Sonnet/Opus passed via
# --model / AI_MODEL — takes the fixed-budget form, which those models accept.
# NOTE: add any future adaptive-only model here when its constant is introduced.
_ADAPTIVE_THINKING_MODEL_PREFIXES = (MODEL_SONNET, MODEL_OPUS)


def _requires_adaptive_thinking(model: str) -> bool:
    """True for models that reject the legacy enabled+budget_tokens thinking form
    and require {"type": "adaptive"}. Single source of truth for the model-family
    split so the worker/supervisor call sites and the synthesis path can't drift.
    """
    return model.startswith(_ADAPTIVE_THINKING_MODEL_PREFIXES)


def _thinking_config(model: str, budget_tokens: int, *, stream_to_ui: bool = False) -> dict:
    """Return the thinking parameter appropriate for the model.

    The two forms are mutually exclusive and each 400s on the wrong model:
    - Adaptive-only models (Sonnet 5 / Opus 4.8) require {"type": "adaptive"}.
    - Everything else (Haiku 4.5, legacy Sonnet/Opus) takes the fixed-budget form
      {"type": "enabled", "budget_tokens": N}.

    ``stream_to_ui`` adds display="summarized" so adaptive thinking actually streams
    readable text — without it, Sonnet 5 / Opus 4.8 stream thinking blocks with empty
    text (display defaults to "omitted"), leaving the live "AI Thinking" pane blank.
    """
    if _requires_adaptive_thinking(model):
        config = {"type": "adaptive"}
        if stream_to_ui:
            config["display"] = "summarized"
        return config
    return {"type": "enabled", "budget_tokens": budget_tokens}


def _calculate_wait_time(backoff: float) -> float:
    """Calculate wait time with jitter to prevent thundering herd."""
    # Add randomness: wait_time = backoff * (1 + random(0, JITTER_FACTOR))
    jitter = random.random() * JITTER_FACTOR
    return min(backoff * (1 + jitter), MAX_BACKOFF)


def _extract_token_usage(usage) -> dict:
    """Extract all token types from an Anthropic usage object.

    Returns dict with: input_tokens, output_tokens, cache_read_tokens, cache_creation_tokens
    """
    if not usage:
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_creation_tokens": 0,
        }
    return {
        "input_tokens": getattr(usage, "input_tokens", 0) or 0,
        "output_tokens": getattr(usage, "output_tokens", 0) or 0,
        "cache_read_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
        "cache_creation_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
    }


def simplify_url(url: str) -> str:
    """Simplify URL to just domain + path (strip query params and fragments)."""
    try:
        parsed = urlparse(url)
        simplified = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        if simplified.endswith('/') and parsed.path == '/':
            simplified = simplified[:-1]
        return simplified
    except Exception:
        return url


class ClaudeProvider(AIProvider):
    """Anthropic Claude implementation with vision capabilities."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        max_concurrent_calls: int = 2
    ):
        # Disable SDK-level retries — we handle retries ourselves with
        # _retry_with_backoff / the streaming retry loop. The SDK's default
        # max_retries=2 would add a hidden retry layer that compounds backoff
        # and generates extra 429 pressure during rate limit windows.
        self.client = AsyncAnthropic(api_key=api_key, max_retries=0)
        self.model = model
        # Semaphore to limit concurrent API calls and avoid rate limiting
        self._api_semaphore = asyncio.Semaphore(max_concurrent_calls)
        self._max_concurrent_calls = max_concurrent_calls
        # Per-run nonce + date workers use to build test data (signup emails
        # etc.) that can't collide with data registered by previous QA runs.
        # Both are stamped once and reused for the whole run: prompt caching
        # depends on a stable system prompt, and recomputing the date per
        # worker call would shift "today" mid-run (and break the cache) when
        # a long run crosses midnight.
        self.run_nonce = uuid.uuid4().hex[:6]
        self.run_date = datetime.now().strftime("%Y-%m-%d (%A)")

    @staticmethod
    def _build_messages_from_history(
        conversation_history: list[dict],
    ) -> list[dict]:
        """Build messages list from conversation history, stripping screenshots
        and windowing long histories.

        Replaces **all** image blocks with a lightweight text placeholder.
        Only the *current turn* (appended by the caller) carries a real
        screenshot.  This keeps the text prefix stable across turns so that
        Anthropic prompt-caching can reuse it, and avoids resending large
        base64 payloads that the model has already seen.

        Histories longer than HISTORY_WINDOW_MESSAGES are trimmed to a recent
        window (stepped boundary — see the constant's comment for the caching
        rationale) with a leading user note stating how many messages were
        omitted.  Windowing happens here, at message-build time, so the
        stored history stays complete for checkpoints and flow forking.

        The original ``conversation_history`` is NOT mutated.
        """
        if not conversation_history:
            return []

        messages: list[dict] = []

        window = conversation_history
        if len(conversation_history) > HISTORY_WINDOW_MESSAGES:
            over = len(conversation_history) - HISTORY_WINDOW_MESSAGES
            # Round the drop count up to a step multiple so the boundary is
            # stable across several turns (prompt-cache friendly).
            drop = ((over + HISTORY_TRIM_STEP - 1) // HISTORY_TRIM_STEP) * HISTORY_TRIM_STEP
            window = conversation_history[drop:]
            messages.append({
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": HISTORY_OMITTED_NOTE.format(count=drop),
                }],
            })

        for msg in window:
            content = msg.get("content")
            if isinstance(content, list):
                new_blocks = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "image":
                        new_blocks.append({"type": "text", "text": "[Screenshot of page]"})
                    else:
                        new_blocks.append(block.copy())
                messages.append({"role": msg["role"], "content": new_blocks})
            else:
                messages.append({"role": msg["role"], "content": content})

        return messages

    @staticmethod
    def _apply_cache_breakpoint(messages: list[dict]) -> None:
        """Mark the last assistant message as the prompt-cache breakpoint.

        History is stored as [assistant_N, user_N, ...] so the last message
        is always a user message.  Walk backwards to find the last assistant
        turn and add cache_control to its final content block.

        Mutates ``messages`` in place.
        """
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "assistant":
                content = messages[i]["content"]
                if isinstance(content, list) and content:
                    content[-1]["cache_control"] = CACHE_CONTROL_EPHEMERAL
                break

    async def _retry_with_backoff(self, operation_name: str, operation):
        """
        Execute an async operation with exponential backoff retry for transient errors.

        Uses a semaphore to limit concurrent API calls across all workers.
        The semaphore is released during retry sleep to avoid blocking other workers.

        Retries on:
        - RateLimitError (429)
        - APIConnectionError (network issues)
        - APITimeoutError
        - Overloaded errors (529)
        """
        last_exception = None
        backoff = INITIAL_BACKOFF

        for attempt in range(MAX_RETRIES):
            # Acquire semaphore only for the API call, not during sleep
            async with self._api_semaphore:
                try:
                    return await operation()
                except RateLimitError as e:
                    last_exception = e
                    wait_time = _calculate_wait_time(backoff)
                    # Try to extract retry-after from headers if available (use exact value, no jitter)
                    if hasattr(e, 'response') and e.response:
                        retry_after = e.response.headers.get('retry-after')
                        if retry_after:
                            try:
                                wait_time = min(float(retry_after), MAX_BACKOFF)
                            except ValueError:
                                pass
                    logger.warning(
                        f"{operation_name}: Rate limited (attempt {attempt + 1}/{MAX_RETRIES}), "
                        f"retrying in {wait_time:.1f}s..."
                    )
                except (APIConnectionError, APITimeoutError) as e:
                    last_exception = e
                    wait_time = _calculate_wait_time(backoff)
                    logger.warning(
                        f"{operation_name}: Connection/timeout error (attempt {attempt + 1}/{MAX_RETRIES}), "
                        f"retrying in {wait_time:.1f}s: {e}"
                    )
                except APIError as e:
                    # Check for overloaded (529) or server errors (5xx)
                    if hasattr(e, 'status_code') and e.status_code in (529, 500, 502, 503, 504):
                        last_exception = e
                        wait_time = _calculate_wait_time(backoff)
                        logger.warning(
                            f"{operation_name}: Server error {e.status_code} (attempt {attempt + 1}/{MAX_RETRIES}), "
                            f"retrying in {wait_time:.1f}s..."
                        )
                    else:
                        # Non-retryable API error
                        raise

            # Sleep outside the semaphore context to allow other workers to proceed
            await asyncio.sleep(wait_time)
            backoff *= BACKOFF_MULTIPLIER

        # All retries exhausted
        logger.error(f"{operation_name}: All {MAX_RETRIES} retries exhausted")
        if last_exception is None:
            raise RuntimeError(f"{operation_name}: All retries exhausted but no exception captured")
        raise last_exception

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
    ) -> AsyncGenerator[dict, None]:
        """
        Stream AI analysis for worker-based exploration.

        The AI receives a screenshot (for visual analysis) and a text list
        of interactive elements with ref numbers, and outputs a structured
        JSON action to execute.

        Yields events:
            {"type": "thinking_start"}
            {"type": "thinking_delta", "text": "..."}
            {"type": "thinking_complete", "text": "full thinking"}
            {"type": "complete", "action": AgentAction, "thinking": str,
             "assistant_content": list[dict]}
        """
        # Encode screenshot as base64
        image_data = base64.standard_b64encode(screenshot_bytes).decode("utf-8")

        # Format action history
        history_text = self._format_history(action_history)

        # Build the worker system prompt (no credentials - those go in user
        # message). Two parts: the base is identical for every worker in a
        # run, the context is per-worker. They go in separate system blocks
        # with their own cache breakpoints so all workers share one cache
        # entry for the ~4-5k-token base instead of each writing their own.
        system_base, system_worker_context = get_worker_system_prompt_parts(
            is_first_worker=is_first_worker,
            worker_number=worker_number,
            flow_name=flow_name,
            flow_description=flow_description,
            parent_flow_name=parent_flow_name,
            target_domain=target_domain,
            run_nonce=self.run_nonce,
            current_date=self.run_date,
        )
        system_prompt = system_base + system_worker_context  # for logging

        # Build user prompt with ref list and credentials
        # Credentials/user_data are in user message (not system prompt) to prevent
        # prompt injection where malicious input could be treated as instructions
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

        # Build current user message with screenshot
        current_message = {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_data
                    }
                },
                {
                    "type": "text",
                    "text": user_prompt
                }
            ]
        }

        # Build complete messages: history (screenshots stripped) + current turn
        messages = self._build_messages_from_history(conversation_history or [])

        self._apply_cache_breakpoint(messages)

        messages.append(current_message)

        # Track content as we stream
        full_thinking = ""
        full_response = ""
        thinking_signature = ""
        in_thinking_block = False

        # Outer corrective-retry loop: when a completed response contains no
        # parseable JSON action, re-ask the model to reformat — and when it
        # contains no text at all (thinking-only), re-sample the same request —
        # up to MAX_PARSE_RETRIES times, instead of failing the whole flow.
        action = None
        cumulative_usage = _extract_token_usage(None)

        for parse_attempt in range(MAX_PARSE_RETRIES + 1):
            # Retry logic for stream creation and processing
            # Use semaphore to limit concurrent API calls across all workers
            # Semaphore is released during retry sleep to avoid blocking other workers
            last_exception = None
            backoff = INITIAL_BACKOFF
            stream_completed = False
            token_usage = _extract_token_usage(None)

            for attempt in range(MAX_RETRIES):
                # Reset state for each attempt
                full_thinking = ""
                full_response = ""
                thinking_signature = ""
                in_thinking_block = False
                wait_time = None

                # Acquire semaphore only for the API call, not during sleep
                async with self._api_semaphore:
                    try:
                        async with self.client.messages.stream(
                            model=self.model,
                            max_tokens=16000,
                            system=[
                                {
                                    # Shared base — one cache entry serves
                                    # every worker in the run
                                    "type": "text",
                                    "text": system_base,
                                    "cache_control": CACHE_CONTROL_EPHEMERAL
                                },
                                {
                                    # Per-worker assignment — cached per
                                    # worker across its own turns
                                    "type": "text",
                                    "text": system_worker_context,
                                    "cache_control": CACHE_CONTROL_EPHEMERAL
                                },
                            ],
                            messages=messages,
                            thinking=_thinking_config(
                                self.model, 10000, stream_to_ui=True
                            )
                        ) as stream:
                            async for event in stream:
                                if event.type == "content_block_start":
                                    if hasattr(event.content_block, 'type'):
                                        if event.content_block.type == "thinking":
                                            in_thinking_block = True
                                            yield {"type": "thinking_start"}
                                        elif event.content_block.type == "text":
                                            in_thinking_block = False

                                elif event.type == "content_block_delta":
                                    if hasattr(event.delta, 'type'):
                                        if event.delta.type == "thinking_delta":
                                            thinking_text = event.delta.thinking
                                            full_thinking += thinking_text
                                            yield {"type": "thinking_delta", "text": thinking_text}
                                        elif event.delta.type == "text_delta":
                                            full_response += event.delta.text
                                        elif event.delta.type == "signature_delta":
                                            thinking_signature = event.delta.signature

                                elif event.type == "content_block_stop":
                                    if in_thinking_block:
                                        yield {"type": "thinking_complete", "text": full_thinking}
                                        in_thinking_block = False

                            # Get final message for token usage
                            final_message = await stream.get_final_message()
                            token_usage = _extract_token_usage(final_message.usage)

                        # Stream completed successfully
                        stream_completed = True
                        break

                    except RateLimitError as e:
                        last_exception = e
                        wait_time = _calculate_wait_time(backoff)
                        # Use retry-after header if provided (exact value, no jitter)
                        if hasattr(e, 'response') and e.response:
                            retry_after = e.response.headers.get('retry-after')
                            if retry_after:
                                try:
                                    wait_time = min(float(retry_after), MAX_BACKOFF)
                                except ValueError:
                                    pass
                        logger.warning(
                            f"Worker stream: Rate limited (attempt {attempt + 1}/{MAX_RETRIES}), "
                            f"retrying in {wait_time:.1f}s..."
                        )

                    except (APIConnectionError, APITimeoutError) as e:
                        last_exception = e
                        wait_time = _calculate_wait_time(backoff)
                        logger.warning(
                            f"Worker stream: Connection error (attempt {attempt + 1}/{MAX_RETRIES}), "
                            f"retrying in {wait_time:.1f}s: {e}"
                        )

                    except APIError as e:
                        if hasattr(e, 'status_code') and e.status_code in (529, 500, 502, 503, 504):
                            last_exception = e
                            wait_time = _calculate_wait_time(backoff)
                            logger.warning(
                                f"Worker stream: Server error {e.status_code} (attempt {attempt + 1}/{MAX_RETRIES}), "
                                f"retrying in {wait_time:.1f}s..."
                            )
                        else:
                            # Non-retryable API error (e.g., 400, 401, 403)
                            raise

                # Sleep outside the semaphore context to allow other workers to proceed
                if wait_time is not None:
                    await asyncio.sleep(wait_time)
                    backoff *= BACKOFF_MULTIPLIER

            # Accumulate usage across parse attempts. The totals ride on the
            # terminal event either way — "complete" or "error" — and the
            # worker bills both, so cost tracking sees every API call made
            # for this turn even when parse retries are exhausted.
            for usage_key, usage_value in token_usage.items():
                cumulative_usage[usage_key] = cumulative_usage.get(usage_key, 0) + usage_value

            if not stream_completed:
                # All retries exhausted
                logger.error(f"Worker stream: All {MAX_RETRIES} retries exhausted")
                yield {
                    "type": "error",
                    "error": f"API error after {MAX_RETRIES} retries: {last_exception}",
                    "thinking": "",
                    "raw_response": "",
                    **cumulative_usage,
                }
                return

            # Parse the final response - extract action JSON
            if not full_response:
                # A completed stream can carry thinking but no text block —
                # an intermittent model failure mode (seen live 2026-07-12:
                # "No text response from AI" killed the same flow twice, once
                # 23 actions deep). One text-less response must not be
                # flow-fatal, so retry within the same corrective budget as a
                # parse failure. There is no assistant text to echo back (the
                # API rejects empty text blocks), so the retry re-sends the
                # identical request — a fresh sample almost always carries
                # text.
                if parse_attempt < MAX_PARSE_RETRIES:
                    logger.warning(
                        f"Worker stream: response contained no text "
                        f"(attempt {parse_attempt + 1}/{MAX_PARSE_RETRIES + 1}), "
                        f"retrying"
                    )
                    continue
                yield {
                    "type": "error",
                    "error": "No text response from AI",
                    "thinking": full_thinking,
                    "raw_response": "",
                    **cumulative_usage,
                }
                return

            try:
                action = self._extract_action_from_response(full_response)
                break
            except ValueError as e:
                if parse_attempt < MAX_PARSE_RETRIES:
                    logger.warning(
                        f"Worker stream: response was not a valid JSON action "
                        f"(attempt {parse_attempt + 1}/{MAX_PARSE_RETRIES + 1}), "
                        f"asking model to reformat"
                    )
                    # Echo the malformed reply and ask for JSON only. These
                    # corrective messages stay local to this turn — the worker
                    # only stores the final successful exchange in history.
                    messages.append({
                        "role": "assistant",
                        "content": [{"type": "text", "text": full_response}]
                    })
                    # Include the specific failure (e.g. a pydantic validation
                    # message like "left_click needs a ref ... or a
                    # coordinate") — the generic prompt alone only mentions
                    # JSON formatting, which isn't always the actual problem.
                    correction = (
                        f"{PARSE_CORRECTION_PROMPT}\n\n"
                        f"Specifically: {str(e)[:500]}"
                    )
                    messages.append({
                        "role": "user",
                        "content": [{"type": "text", "text": correction}]
                    })
                    continue
                # Yield error event with raw response for debugging
                yield {
                    "type": "error",
                    "error": str(e),
                    "thinking": full_thinking,
                    "raw_response": full_response,
                    **cumulative_usage,
                }
                return

        # Build the full assistant content for storage
        assistant_content = []
        # Preserve the thinking block whenever the turn produced one, even if its
        # text is empty (Sonnet 5 / Opus 4.8 return summarized thinking; a
        # signature with empty text still represents a real block that must be
        # echoed back unchanged on the next turn).
        if full_thinking or thinking_signature:
            thinking_block = {"type": "thinking", "thinking": full_thinking}
            if thinking_signature:
                thinking_block["signature"] = thinking_signature
            assistant_content.append(thinking_block)
        assistant_content.append({"type": "text", "text": full_response})

        yield {
            "type": "complete",
            "action": action,
            "thinking": full_thinking,
            "assistant_content": assistant_content,
            "user_content": current_message["content"],
            **cumulative_usage,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        }

    def _format_history(self, action_history: list[dict]) -> str:
        """Format action history for prompts."""
        if not action_history:
            return ""

        lines = []
        for i, action in enumerate(action_history[-10:], 1):  # Last 10 actions
            action_type = action.get("action_type", "unknown")
            reasoning = action.get("reasoning", "")
            url = action.get("url", "")
            error = action.get("error", "")
            success = action.get("success", True)

            # Build action summary
            status = "✓" if success else "✗"
            line = f"{i}. {status} [{action_type}]"

            # Add relevant details based on action type
            if action_type in ("left_click", "right_click", "double_click", "triple_click", "hover"):
                ref = action.get("ref")
                coordinate = action.get("coordinate")
                if ref:
                    line += f" ref={ref}"
                elif coordinate:
                    line += f" coordinate={coordinate}"
                else:
                    line += " ref=?"
            elif action_type == "type":
                ref = action.get("ref", "?")
                text = action.get("text", "")[:30]
                line += f" ref={ref} text=\"{text}\""
            elif action_type == "navigate":
                nav_url = action.get("url", "")
                line += f" → {simplify_url(nav_url)}"
            elif action_type in ("done", "block"):
                reason = action.get("reason", "")[:50]
                line += f" \"{reason}\""

            if reasoning:
                line += f"\n   {reasoning[:80]}"
            if url:
                line += f"\n   URL: {simplify_url(url)}"
            if error:
                # Crafted corrective messages (duplicate-ref disambiguation,
                # ref-timeout guidance, missing-target instructions) must
                # arrive intact; only raw exception text is truncated hard.
                is_corrective = (
                    action.get("error_corrective")
                    or error.startswith(DUPLICATE_REF_PREFIX)
                )
                max_err = 300 if is_corrective else 100
                line += f"\n   Error: {error[:max_err]}"
            note = action.get("note", "")
            if note:
                # e.g. "Triggered file download: report.pdf" — without this
                # the AI never learns why the screenshot didn't change
                line += f"\n   Note: {note[:150]}"

            lines.append(line)

        return "\n\n".join(lines)

    def _extract_action_from_response(self, text: str) -> AgentAction:
        """
        Extract JSON action from AI response.

        Args:
            text: Raw AI response text

        Returns:
            Parsed AgentAction

        Raises:
            ValueError: If no valid JSON action found
        """
        text = text.strip()

        # Remember the last ValidationError raised by a candidate that was
        # clearly meant to be the action (a dict carrying "action_type"): if
        # no later candidate parses cleanly, its message — worded as an
        # instruction by AgentAction's validators — is what the parse
        # correction retry should feed back to the model, not a generic
        # "no valid JSON" complaint.
        validation_error: ValidationError | None = None

        def _try_build(data) -> Optional[AgentAction]:
            nonlocal validation_error
            try:
                return AgentAction(**data)
            except TypeError:
                # data isn't a mapping — this candidate isn't the action
                return None
            except ValidationError as e:
                if isinstance(data, dict) and "action_type" in data:
                    validation_error = e
                return None

        # Try to parse as direct JSON. Schema mismatch or non-mapping data
        # means "this candidate isn't the (valid) action", so fall through
        # to the next extraction layer rather than aborting.
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = None
        if data is not None:
            action = _try_build(data)
            if action is not None:
                return action

        # Try to extract JSON from markdown code block
        match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                data = None
            if data is not None:
                action = _try_build(data)
                if action is not None:
                    return action

        # Fall back to scanning for a JSON object containing "action_type".
        # raw_decode is string-aware, so braces inside string values (e.g.
        # code/CSS snippets quoted in issue descriptions like "missing {
        # before color:red") cannot derail it the way regex matching or
        # naive brace counting could.
        decoder = json.JSONDecoder()
        for key_match in re.finditer(r'"action_type"', text):
            # Try each opening brace before this occurrence, nearest first
            search_end = key_match.start()
            while search_end > 0:
                start = text.rfind('{', 0, search_end)
                if start < 0:
                    break
                try:
                    data, _ = decoder.raw_decode(text, start)
                except json.JSONDecodeError:
                    data = None
                if isinstance(data, dict) and "action_type" in data:
                    # Schema-invalid candidates (e.g. a quoted example or
                    # truncated object) are recorded and scanning continues;
                    # a valid action may still appear later in the response.
                    action = _try_build(data)
                    if action is not None:
                        return action
                search_end = start

        if validation_error is not None:
            # The response DID contain an action object — it just failed
            # schema validation (e.g. a click with neither ref nor
            # coordinate). Surface the validator's own message so the
            # correction retry tells the model exactly what to fix.
            details = "; ".join(
                err.get("msg", "invalid value")
                for err in validation_error.errors()[:3]
            )
            raise ValueError(f"Action JSON failed validation: {details}")

        # Show context in error
        preview = text[:500] if len(text) > 500 else text
        raise ValueError(f"No valid JSON action found in AI response. Response preview:\n{preview}")

    async def analyze_for_supervisor(
        self,
        active_workers: list[dict],
        blocked_workers: list[dict],
        pending_flows: list[dict],
        completed_flows: list[dict],
        issues: list[dict],
    ) -> dict:
        """
        Call AI for supervisor decisions.

        Returns parsed action dict with:
        - action: "message" | "stop" | "ask_user" | "unblock" | "skip_flow" | "observe"
        - Plus action-specific fields (worker_id, message, reason, flow_id, etc.)
        """
        # Build context
        context = format_supervisor_context(
            active_workers=active_workers,
            blocked_workers=blocked_workers,
            pending_flows=pending_flows,
            completed_flows=completed_flows,
            issues=issues
        )

        async def _make_supervisor_request():
            return await self.client.messages.create(
                model=self.model,
                # Headroom for adaptive thinking: on Sonnet 5 / Opus 4.8 thinking
                # tokens count against max_tokens with no fixed budget, and the
                # newer tokenizer runs ~30% heavier, so give the decision room.
                max_tokens=12000,
                system=[{
                    "type": "text",
                    "text": SUPERVISOR_SYSTEM_PROMPT,
                    "cache_control": CACHE_CONTROL_EPHEMERAL
                }],
                messages=[
                    {
                        "role": "user",
                        "content": context
                    }
                ],
                # Supervisor thinking is never displayed, so no summarized display.
                thinking=_thinking_config(self.model, 5000)
            )

        response = await self._retry_with_backoff("Supervisor", _make_supervisor_request)

        # Extract token usage
        token_usage = _extract_token_usage(response.usage)

        # Extract text response
        response_text = None
        for block in response.content:
            if block.type == "text":
                response_text = block.text
                break

        if not response_text:
            return {
                "action": "observe",
                "reasoning": "No response from AI",
                **token_usage,
            }

        result = self._parse_supervisor_response(response_text)
        result.update(token_usage)
        return result

    @staticmethod
    def _coerce_supervisor_action(parsed) -> dict | None:
        """Coerce a parsed JSON value into a single action dict.

        The prompt asks for one JSON object, but a more agentic model may emit a
        JSON array of actions (e.g. several skip_flow decisions). The caller does
        ``result.update(...)`` on this, so a bare list would raise AttributeError
        and kill the supervisor cycle. Take the first dict from a list; reject
        anything that isn't a usable action dict.
        """
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            if len(parsed) > 1:
                # The prompt asks for one action per response, so this is rare —
                # but make the drop visible rather than silently losing the tail.
                logger.warning(
                    f"Supervisor returned {len(parsed)} actions in one response; "
                    f"taking the first and dropping {len(parsed) - 1}"
                )
            return parsed[0]
        return None

    def _parse_supervisor_response(self, text: str) -> dict:
        """Parse supervisor response text into action dict."""
        if not text:
            return {"action": "observe", "reasoning": "Empty response"}

        # Try to parse as JSON
        try:
            action = self._coerce_supervisor_action(json.loads(text))
            if action is not None:
                return action
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code block
        match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
        if match:
            try:
                action = self._coerce_supervisor_action(json.loads(match.group(1).strip()))
                if action is not None:
                    return action
            except json.JSONDecodeError:
                pass

        # Try to find JSON object with brace matching (try each { position)
        search_start = 0
        while search_start < len(text):
            start = text.find('{', search_start)
            if start < 0:
                break
            depth = 0
            for i in range(start, len(text)):
                if text[i] == '{':
                    depth += 1
                elif text[i] == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start:i + 1])
                        except json.JSONDecodeError:
                            break
            search_start = start + 1

        return {"action": "observe", "reasoning": f"Could not parse response: {text[:200]}"}

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
    ) -> dict:
        """
        Generate final QA synthesis report.

        Returns dict with:
            - report: markdown-formatted report string
            - input_tokens: tokens used for input
            - output_tokens: tokens used for output
        """
        # Build context
        context = format_synthesis_context(
            target_url=target_url,
            duration=duration,
            flows_tested=flows_tested,
            issues=issues,
            completed_flows=completed_flows,
            blocked_flows=blocked_flows,
            incomplete_flows=incomplete_flows,
            goal=goal,
        )

        async def _make_synthesis_request():
            # 16000 matches the worker-stream budget: the synthesis report is
            # the product's primary deliverable and large multi-agent runs
            # routinely need more than the old 4000-token cap. max_tokens is a
            # ceiling, not pre-spent — typical reports cost the same as before.
            # Synthesis doesn't need thinking, and it protects the 10% synthesis
            # cost reserve. On adaptive-only models (Sonnet 5 / Opus 4.8), omitting
            # the param would silently enable adaptive thinking (extra cost + a
            # thinking block ahead of the report), so disable it explicitly. Haiku
            # and legacy models default to no thinking when the param is omitted.
            # Same family check as _thinking_config so the two can't drift.
            synthesis_kwargs = {}
            if _requires_adaptive_thinking(self.model):
                synthesis_kwargs["thinking"] = {"type": "disabled"}
            return await self.client.messages.create(
                model=self.model,
                max_tokens=16000,
                system=[{
                    "type": "text",
                    "text": SYNTHESIS_SYSTEM_PROMPT,
                    "cache_control": CACHE_CONTROL_EPHEMERAL
                }],
                messages=[
                    {
                        "role": "user",
                        "content": context
                    }
                ],
                **synthesis_kwargs,
            )

        response = await self._retry_with_backoff("Synthesis", _make_synthesis_request)

        # Extract token usage
        token_usage = _extract_token_usage(response.usage)

        # Scan for the text block rather than assuming content[0]: with adaptive
        # thinking the response can lead with a thinking block, and content[0]
        # would then be a ThinkingBlock (no .text) — losing the whole report.
        report = next((b.text for b in response.content if b.type == "text"), "")
        if response.stop_reason == "max_tokens":
            # Surface truncation instead of silently delivering a report that
            # ends mid-sentence with its tail sections missing.
            logger.warning(
                "Synthesis report hit the max_tokens output limit and was truncated"
            )
            if report.count("```") % 2 == 1:
                # Truncation landed inside a code fence (the prompt asks for
                # fenced repro steps) — close it so the warning renders as a
                # blockquote instead of disappearing into the code block.
                report += "\n```"
            report += (
                "\n\n> **Warning:** This report was truncated at the model's "
                "output token limit — later sections (e.g. Test Coverage, "
                "Recommendations) may be missing."
            )

        return {
            "report": report,
            **token_usage,
        }
