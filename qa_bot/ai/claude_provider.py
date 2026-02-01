import asyncio
import base64
import json
import logging
import random
import re
from typing import AsyncGenerator
from urllib.parse import urlparse
from anthropic import AsyncAnthropic, RateLimitError, APIError, APIConnectionError, APITimeoutError

from .base import AIProvider, AgentAction
from .prompts import (
    get_worker_system_prompt,
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

# Prompt caching configuration
# Ephemeral cache has a 5-minute TTL - sufficient for multi-turn conversations
CACHE_CONTROL_EPHEMERAL = {"type": "ephemeral"}


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
        model: str = "claude-haiku-4-5-20251001",
        max_concurrent_calls: int = 2
    ):
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model
        # Semaphore to limit concurrent API calls and avoid rate limiting
        self._api_semaphore = asyncio.Semaphore(max_concurrent_calls)
        self._max_concurrent_calls = max_concurrent_calls

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

        # Build the worker system prompt (no credentials - those go in user message)
        system_prompt = get_worker_system_prompt(
            is_first_worker=is_first_worker,
            worker_number=worker_number,
            flow_name=flow_name,
            flow_description=flow_description,
            parent_flow_name=parent_flow_name,
            target_domain=target_domain,
        )

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

        # Build complete messages: history + current turn
        # Use deep copy to avoid mutating the original conversation history
        messages = []
        if conversation_history:
            for msg in conversation_history:
                messages.append({
                    "role": msg["role"],
                    "content": [block.copy() for block in msg["content"]] if isinstance(msg["content"], list) else msg["content"]
                })

        # Add cache_control to the last assistant message for prompt caching
        # This caches the entire conversation prefix (system + all previous turns)
        # reducing costs by ~90% for cached tokens on subsequent turns
        if messages and messages[-1]["role"] == "assistant":
            content = messages[-1]["content"]
            if isinstance(content, list) and content:
                # Add cache_control to the last content block
                content[-1]["cache_control"] = CACHE_CONTROL_EPHEMERAL

        messages.append(current_message)

        # Track content as we stream
        full_thinking = ""
        full_response = ""
        thinking_signature = ""
        in_thinking_block = False
        input_tokens = 0
        output_tokens = 0
        cache_creation_tokens = 0
        cache_read_tokens = 0

        # Retry logic for stream creation and processing
        # Use semaphore to limit concurrent API calls across all workers
        # Semaphore is released during retry sleep to avoid blocking other workers
        last_exception = None
        backoff = INITIAL_BACKOFF
        stream_completed = False

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
                        system=[{
                            "type": "text",
                            "text": system_prompt,
                            "cache_control": CACHE_CONTROL_EPHEMERAL
                        }],
                        messages=messages,
                        thinking={
                            "type": "enabled",
                            "budget_tokens": 10000
                        }
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

        if not stream_completed:
            # All retries exhausted
            logger.error(f"Worker stream: All {MAX_RETRIES} retries exhausted")
            yield {
                "type": "error",
                "error": f"API error after {MAX_RETRIES} retries: {last_exception}",
                "thinking": "",
                "raw_response": "",
                **_extract_token_usage(None),
            }
            return

        # Parse the final response - extract action JSON
        if not full_response:
            yield {
                "type": "error",
                "error": "No text response from AI",
                "thinking": full_thinking,
                "raw_response": "",
                **token_usage,
            }
            return

        try:
            action = self._extract_action_from_response(full_response)
        except ValueError as e:
            # Yield error event with raw response for debugging
            yield {
                "type": "error",
                "error": str(e),
                "thinking": full_thinking,
                "raw_response": full_response,
                **token_usage,
            }
            return

        # Build the full assistant content for storage
        assistant_content = []
        if full_thinking:
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
            **token_usage,
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
                ref = action.get("ref", "?")
                line += f" ref={ref}"
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
                line += f"\n   Error: {error[:100]}"

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

        # Try to parse as direct JSON
        try:
            data = json.loads(text)
            return AgentAction(**data)
        except (json.JSONDecodeError, TypeError):
            pass

        # Try to extract JSON from markdown code block
        match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1).strip())
                return AgentAction(**data)
            except (json.JSONDecodeError, TypeError):
                pass

        # Try to find JSON object in text
        match = re.search(r'\{[^{}]*"action_type"[^{}]*\}', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                return AgentAction(**data)
            except (json.JSONDecodeError, TypeError):
                pass

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
                max_tokens=8000,  # Must be > budget_tokens
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
                thinking={
                    "type": "enabled",
                    "budget_tokens": 5000
                }
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

    def _parse_supervisor_response(self, text: str) -> dict:
        """Parse supervisor response text into action dict."""
        if not text:
            return {"action": "observe", "reasoning": "Empty response"}

        # Try to parse as JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown
        match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return {"action": "observe", "reasoning": f"Could not parse response: {text[:200]}"}

    async def generate_synthesis_report(
        self,
        target_url: str,
        duration: str,
        flows_tested: int,
        issues: list[dict],
        completed_flows: list[dict],
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
        )

        async def _make_synthesis_request():
            return await self.client.messages.create(
                model=self.model,
                max_tokens=4000,
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
                ]
            )

        response = await self._retry_with_backoff("Synthesis", _make_synthesis_request)

        # Extract token usage
        token_usage = _extract_token_usage(response.usage)

        return {
            "report": response.content[0].text,
            **token_usage,
        }
