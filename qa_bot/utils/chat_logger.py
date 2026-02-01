"""Chat history and operations logger for debugging AI interactions."""

import json
import logging
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse


# Query parameters that may contain sensitive data and should be masked
SENSITIVE_PARAMS = {
    'token', 'access_token', 'refresh_token', 'api_key', 'apikey', 'key',
    'secret', 'password', 'pwd', 'auth', 'authorization', 'bearer',
    'session', 'sessionid', 'sid', 'credential', 'credentials'
}

# Credential keys that should have values masked in logs
CREDENTIAL_KEYS = {
    'password', 'pwd', 'secret', 'token', 'api_key', 'apikey',
    'card_number', 'cardnumber', 'cvv', 'cvc', 'card_cvv',
    'expiry', 'card_expiry', 'expiration', 'pin',
    'ssn', 'social_security', 'account_number'
}


def _sanitize_url(url: str) -> str:
    """Mask sensitive query parameters in URLs to prevent logging secrets."""
    try:
        parsed = urlparse(url)
        if not parsed.query:
            return url

        params = parse_qs(parsed.query, keep_blank_values=True)
        sanitized = {}
        for key, values in params.items():
            if key.lower() in SENSITIVE_PARAMS:
                sanitized[key] = ['***REDACTED***'] * len(values)
            else:
                sanitized[key] = values

        # Rebuild URL with sanitized params
        sanitized_query = urlencode(sanitized, doseq=True)
        return urlunparse(parsed._replace(query=sanitized_query))
    except Exception:
        # If parsing fails, return original URL
        return url


def _is_sensitive_key(key: str) -> bool:
    """Check if a key name suggests sensitive credential data."""
    key_lower = key.lower()
    return any(sensitive in key_lower for sensitive in CREDENTIAL_KEYS)


def _mask_credentials_in_dict(data: dict) -> dict:
    """
    Recursively mask credential values in a dictionary.

    Keys containing 'password', 'secret', 'token', etc. will have
    their values replaced with '***MASKED***'.
    """
    if not isinstance(data, dict):
        return data

    masked = {}
    for key, value in data.items():
        if isinstance(value, dict):
            masked[key] = _mask_credentials_in_dict(value)
        elif isinstance(value, list):
            masked[key] = [
                _mask_credentials_in_dict(v) if isinstance(v, dict) else v
                for v in value
            ]
        elif _is_sensitive_key(key) and isinstance(value, str):
            masked[key] = '***MASKED***'
        else:
            masked[key] = value
    return masked


def _mask_credentials_in_text(text: str) -> str:
    """
    Mask credential values that appear in text as KEY: value or KEY=value patterns.

    This catches credentials that might appear in AI prompts or responses.
    """
    if not text:
        return text

    # Pattern for KEY: value (with optional spaces, common in formatted output)
    # Matches things like "PASSWORD: secret123" or "  CARD_NUMBER: 4111..."
    for key_pattern in CREDENTIAL_KEYS:
        # Case-insensitive pattern for KEY: value
        pattern = rf'({key_pattern})\s*:\s*(\S+)'
        text = re.sub(pattern, r'\1: ***MASKED***', text, flags=re.IGNORECASE)

        # Also handle KEY=value format
        pattern = rf'({key_pattern})\s*=\s*(\S+)'
        text = re.sub(pattern, r'\1=***MASKED***', text, flags=re.IGNORECASE)

    return text


class ChatLogger:
    """
    Logs AI chat interactions and operational events for debugging.

    Creates structured logs for:
    - Worker conversations (one file per worker/flow)
    - Supervisor calls
    - Synthesis agent calls
    - Operational events (SSE events persisted to disk)
    - Screenshots (optional, development mode only)

    File structure:
        logs/
        └── {exploration_id}/
            ├── operations.log       # All SSE events, human-readable
            └── chats/
                ├── worker-1_flow-abc.txt
                ├── worker-2_flow-def.txt
                ├── supervisor.txt
                └── synthesis.txt
            └── screenshots/         # Only if save_screenshots=True
                ├── worker-1_abc_turn000.png
                └── ...
    """

    def __init__(
        self,
        log_dir: str,
        exploration_id: str,
        save_screenshots: bool = False,
        max_logs: int = 10
    ):
        self.exploration_id = exploration_id
        self._log_dir = Path(log_dir)
        self.base_dir = self._log_dir / exploration_id
        self.chats_dir = self.base_dir / "chats"
        self._save_screenshots = save_screenshots
        self._max_logs = max_logs

        # Clean up old logs before creating new ones
        self._cleanup_old_logs()

        # Create directories
        self.chats_dir.mkdir(parents=True, exist_ok=True)
        if self._save_screenshots:
            self.screenshots_dir = self.base_dir / "screenshots"
            self.screenshots_dir.mkdir(parents=True, exist_ok=True)
            logging.warning(
                f"Screenshot saving enabled - screenshots may contain sensitive information. "
                f"Saving to: {self.screenshots_dir}"
            )

        # Track which files we've initialized (to write headers)
        self._initialized_files: set[str] = set()

        # Set up Python logger for operations.log
        self._ops_logger = self._setup_ops_logger()

    def _cleanup_old_logs(self):
        """Remove old exploration logs, keeping only the most recent ones."""
        if not self._log_dir.exists():
            return

        # Get all exploration directories (UUIDs)
        exploration_dirs = [
            d for d in self._log_dir.iterdir()
            if d.is_dir() and d.name != self.exploration_id
        ]

        if len(exploration_dirs) < self._max_logs:
            return

        # Sort by modification time (oldest first)
        exploration_dirs.sort(key=lambda d: d.stat().st_mtime)

        # Delete oldest directories to keep only max_logs - 1 (leaving room for current)
        dirs_to_delete = exploration_dirs[:len(exploration_dirs) - self._max_logs + 1]

        for old_dir in dirs_to_delete:
            try:
                shutil.rmtree(old_dir)
            except Exception as e:
                # Log at debug level - cleanup is best-effort and may race with other processes
                logging.debug(f"Failed to cleanup old log directory {old_dir}: {e}")

    def _setup_ops_logger(self) -> logging.Logger:
        """Set up Python logger for operations.log."""
        logger = logging.getLogger(f"qa-bot.{self.exploration_id}")
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()  # Avoid duplicate handlers

        # File handler for operations.log
        ops_log_path = self.base_dir / "operations.log"
        file_handler = logging.FileHandler(ops_log_path, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s | %(message)s",
            datefmt="%H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Prevent propagation to root logger
        logger.propagate = False

        return logger

    def _get_chat_file(self, worker_id: str, flow_id: str) -> Path:
        """Get the path for a worker/flow chat file."""
        # Truncate flow_id to keep filenames reasonable
        short_flow_id = flow_id[:8] if len(flow_id) > 8 else flow_id
        return self.chats_dir / f"{worker_id}_{short_flow_id}.txt"

    def _get_supervisor_file(self) -> Path:
        """Get the path for supervisor chat file."""
        return self.chats_dir / "supervisor.txt"

    def _get_synthesis_file(self) -> Path:
        """Get the path for synthesis chat file."""
        return self.chats_dir / "synthesis.txt"

    def _write_header(self, file_path: Path, title: str, metadata: dict):
        """Write a header to a new chat file."""
        with open(file_path, "w", encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"{title}\n")
            f.write("=" * 80 + "\n")
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
            f.write("=" * 80 + "\n\n")

    def _append(self, file_path: Path, content: str):
        """Append content to a file."""
        with open(file_path, "a", encoding='utf-8') as f:
            f.write(content)

    # =========================================================================
    # Operations Log (SSE Events)
    # =========================================================================

    def log_event(self, event: dict):
        """
        Log an SSE event to operations.log.

        This is the central method for persisting all events that are
        streamed to the UI Activity Log.
        """
        event_type = event.get("type", "unknown")
        data = event.get("data", {})

        # Format message based on event type
        message = self._format_event_message(event_type, data, event)

        if message:
            self._ops_logger.info(f"{event_type:20} | {message}")

    def _format_event_message(
        self,
        event_type: str,
        data: dict,
        event: dict
    ) -> Optional[str]:
        """Format an event into a human-readable log message."""

        # Lifecycle events
        if event_type == "exploration_started":
            target_url = _sanitize_url(data.get('target_url', 'unknown'))
            return f"Exploration started: {target_url}"

        elif event_type == "exploration_complete":
            issues = data.get("issues_found", 0)
            flows = data.get("flows_explored", 0)
            return f"Exploration complete: {issues} issues, {flows} flows"

        elif event_type == "stopping":
            return data.get("message", "Stopping exploration...")

        # Worker events
        elif event_type == "worker_started":
            worker = data.get("worker_id", "unknown")
            flow = data.get("flow_name", "unknown")
            return f"{worker} started flow: {flow}"

        elif event_type == "worker_stopped":
            worker = data.get("worker_id", "unknown")
            return f"{worker} stopped"

        elif event_type == "worker_blocked":
            worker = data.get("worker_id", "unknown")
            reason = data.get("reason", "unknown")
            return f"{worker} blocked: {reason}"

        elif event_type == "worker_error":
            worker = data.get("worker_id", "unknown")
            error = data.get("error", "unknown")
            return f"{worker} error: {error}"

        # Flow events
        elif event_type == "flow_started":
            flow = data.get("flow_name", "unknown")
            worker = data.get("worker_id", "")
            return f"Flow started: {flow}" + (f" ({worker})" if worker else "")

        elif event_type == "flow_completed":
            flow = data.get("flow_name", "unknown")
            actions = data.get("action_count", 0)
            reason = data.get("completion_reason", "")
            msg = f"Flow completed: {flow} ({actions} actions)"
            if reason:
                msg += f" - {reason}"
            return msg

        elif event_type == "flow_failed":
            flow = data.get("flow_name", "unknown")
            error = data.get("error", "unknown")
            return f"Flow failed: {flow} - {error}"

        elif event_type == "flow_created":
            flow = data.get("flow_name", "unknown")
            parent = data.get("parent_flow", "")
            return f"Flow created: {flow}" + (f" (from {parent})" if parent else "")

        elif event_type == "flow_skipped_by_supervisor":
            flow = data.get("flow_name", data.get("flow_id", "unknown"))
            reason = data.get("reason", "duplicate")
            return f"Flow skipped: {flow} - {reason}"

        # Checkpoint events
        elif event_type == "checkpoint_created":
            branch = data.get("branch_name", "unknown")
            parent = data.get("parent_flow_name", "")
            return f"Checkpoint created: {branch}" + (f" (from {parent})" if parent else "")

        elif event_type == "checkpoint_claimed":
            worker = data.get("worker_id", "unknown")
            branch = data.get("branch_name", data.get("checkpoint_id", "unknown")[:8])
            return f"{worker} claimed checkpoint: {branch}"

        # Action events
        elif event_type == "action":
            worker = event.get("worker_id", data.get("worker_id", "unknown"))
            action = data.get("action_type", "unknown")
            ref = data.get("ref", "")
            text = data.get("text", "")
            element = data.get("element", "")

            if action in ("left_click", "right_click", "double_click", "triple_click", "hover"):
                target = element or ref or "element"
                return f"{worker}: {action.replace('_', ' ')} {target}"
            elif action == "type" and text:
                return f"{worker}: type '{text[:30]}...'" if len(text) > 30 else f"{worker}: type '{text}'"
            elif action == "form_input":
                value = data.get("value", "")
                return f"{worker}: form_input {value}"
            elif action == "scroll":
                direction = data.get("scroll_direction", "down")
                return f"{worker}: scroll {direction}"
            elif action == "scroll_to":
                target = element or ref or "element"
                return f"{worker}: scroll_to {target}"
            elif action == "key":
                key = data.get("key", "")
                modifiers = data.get("modifiers", [])
                key_str = "+".join(modifiers + [key]) if modifiers else key
                return f"{worker}: key {key_str}"
            elif action == "navigate":
                url = _sanitize_url(data.get("url", ""))
                return f"{worker}: navigate to {url[:50]}"
            elif action == "wait":
                duration = data.get("duration", 0)
                return f"{worker}: wait {duration}s"
            elif action == "resize":
                width = data.get("width", 0)
                height = data.get("height", 0)
                return f"{worker}: resize {width}x{height}"
            else:
                return f"{worker}: {action}"

        elif event_type == "issue":
            severity = data.get("severity", "unknown")
            desc = data.get("description", "unknown")
            if len(desc) > 60:
                desc = desc[:60] + "..."
            return f"[{severity}] {desc}"

        # AI thinking events (skip deltas, only log complete)
        elif event_type == "ai_thinking_start":
            return None  # Don't log, too noisy

        elif event_type == "ai_thinking_delta":
            return None  # Don't log, too noisy

        elif event_type == "ai_thinking_complete":
            return None  # Don't log, we log the action instead

        # Supervisor events
        elif event_type == "supervisor_started":
            return "Supervisor started"

        elif event_type == "supervisor_stopped":
            decisions = data.get("decisions_made", 0)
            return f"Supervisor stopped ({decisions} decisions)"

        elif event_type == "supervisor_reviewing_block":
            worker = data.get("worker_id", "unknown")
            return f"Supervisor reviewing blocked worker: {worker}"

        elif event_type == "supervisor_reviewing_flows":
            count = data.get("count", 0)
            return f"Supervisor reviewing {count} pending flows"

        elif event_type == "supervisor_action":
            action = data.get("action", "unknown")
            target = data.get("target_worker", "")
            return f"Supervisor action: {action}" + (f" for {target}" if target else "")

        elif event_type == "supervisor_error":
            error = data.get("error", "unknown")
            return f"Supervisor error: {error}"

        # Synthesis events
        elif event_type == "synthesis_started":
            return "Generating synthesis report..."

        elif event_type == "synthesis_complete":
            return "Synthesis report complete"

        elif event_type == "synthesis_error":
            error = data.get("error", "unknown")
            return f"Synthesis error: {error}"

        # AI error events
        elif event_type == "ai_error":
            worker = event.get("worker_id", data.get("worker_id", "unknown"))
            error = data.get("error", "unknown")
            # Truncate long error messages for operations log (full details in chat log)
            if len(error) > 100:
                error = error[:100] + "..."
            return f"{worker} AI error: {error}"

        # Progress events
        elif event_type == "progress":
            return None  # Don't log periodic progress, too noisy

        # Screenshot events
        elif event_type == "screenshot":
            return None  # Don't log, too noisy

        # Navigation events
        elif event_type == "navigation_error":
            url = _sanitize_url(data.get("url", "unknown"))
            error = data.get("error", "unknown")
            return f"Navigation error: {url} - {error}"

        # Error events
        elif event_type == "error":
            message = data.get("message", "unknown")
            fatal = data.get("fatal", False)
            prefix = "FATAL ERROR" if fatal else "ERROR"
            return f"{prefix}: {message}"

        # Default: log unknown events for debugging
        else:
            return f"(event data: {json.dumps(data, default=str)[:100]})"

    # =========================================================================
    # Worker Chat Logging
    # =========================================================================

    def log_worker_system_prompt(
        self,
        worker_id: str,
        flow_id: str,
        flow_name: str,
        system_prompt: str
    ):
        """Log the system prompt for a worker (called once at start of flow)."""
        file_path = self._get_chat_file(worker_id, flow_id)
        file_key = str(file_path)

        if file_key not in self._initialized_files:
            self._write_header(file_path, f"FLOW: {flow_name}", {
                "Worker": worker_id,
                "Flow ID": flow_id,
                "Started": datetime.now().isoformat()
            })
            self._initialized_files.add(file_key)

        self._append(file_path, "\n--- SYSTEM PROMPT ---\n")
        self._append(file_path, system_prompt)
        self._append(file_path, "\n\n")

    def log_worker_turn(
        self,
        worker_id: str,
        flow_id: str,
        turn_number: int,
        current_url: str,
        user_prompt: str,
        thinking: Optional[str],
        response: dict,
        input_tokens: int = 0,
        output_tokens: int = 0,
        screenshot_size_bytes: int = 0,
        cache_creation_tokens: int = 0,
        cache_read_tokens: int = 0
    ):
        """Log a single turn of worker conversation."""
        file_path = self._get_chat_file(worker_id, flow_id)

        # Turn header
        self._append(file_path, "=" * 80 + "\n")
        self._append(file_path, f"TURN {turn_number} | {datetime.now().strftime('%H:%M:%S')} | ")
        token_info = f"Tokens: {input_tokens} in, {output_tokens} out"
        if cache_creation_tokens or cache_read_tokens:
            token_info += f" | Cache: {cache_creation_tokens} created, {cache_read_tokens} read"
        self._append(file_path, token_info + "\n")
        self._append(file_path, f"URL: {_sanitize_url(current_url)}\n")
        if screenshot_size_bytes:
            self._append(file_path, f"Screenshot: {screenshot_size_bytes // 1024}KB")
            if self._save_screenshots:
                short_flow_id = flow_id[:8] if len(flow_id) > 8 else flow_id
                self._append(file_path, f" (saved: {worker_id}_{short_flow_id}_turn{turn_number:03d}.png)")
            self._append(file_path, "\n")
        self._append(file_path, "=" * 80 + "\n\n")

        # User message (mask credentials)
        self._append(file_path, "--- USER ---\n")
        self._append(file_path, _mask_credentials_in_text(user_prompt))
        self._append(file_path, "\n\n")

        # Assistant thinking (mask credentials)
        if thinking:
            self._append(file_path, "--- ASSISTANT (thinking) ---\n")
            self._append(file_path, _mask_credentials_in_text(thinking))
            self._append(file_path, "\n\n")

        # Assistant response (the action - mask credentials in dict)
        self._append(file_path, "--- ASSISTANT (action) ---\n")
        self._append(file_path, json.dumps(_mask_credentials_in_dict(response), indent=2))
        self._append(file_path, "\n\n")

    def log_worker_completion(
        self,
        worker_id: str,
        flow_id: str,
        reason: str,
        total_turns: int
    ):
        """Log flow completion."""
        file_path = self._get_chat_file(worker_id, flow_id)

        self._append(file_path, "=" * 80 + "\n")
        self._append(file_path, "FLOW COMPLETED\n")
        self._append(file_path, "=" * 80 + "\n")
        self._append(file_path, f"Reason: {reason}\n")
        self._append(file_path, f"Total turns: {total_turns}\n")
        self._append(file_path, f"Ended: {datetime.now().isoformat()}\n")

    def log_worker_ai_error(
        self,
        worker_id: str,
        flow_id: str,
        turn_number: int,
        current_url: str,
        error: str,
        raw_response: Optional[str] = None,
        thinking: Optional[str] = None
    ):
        """Log an AI error during a worker turn (for debugging parsing failures)."""
        file_path = self._get_chat_file(worker_id, flow_id)

        # Error header
        self._append(file_path, "=" * 80 + "\n")
        self._append(file_path, f"ERROR TURN {turn_number} | {datetime.now().strftime('%H:%M:%S')}\n")
        self._append(file_path, f"URL: {current_url}\n")
        self._append(file_path, "=" * 80 + "\n\n")

        # Error message
        self._append(file_path, "--- ERROR ---\n")
        self._append(file_path, error)
        self._append(file_path, "\n\n")

        # Assistant thinking (if available)
        if thinking:
            self._append(file_path, "--- ASSISTANT (thinking) ---\n")
            self._append(file_path, thinking)
            self._append(file_path, "\n\n")

        # Raw response (for debugging)
        if raw_response:
            self._append(file_path, "--- RAW RESPONSE ---\n")
            self._append(file_path, raw_response)
            self._append(file_path, "\n\n")

    # =========================================================================
    # Screenshot Saving
    # =========================================================================

    def save_screenshot(
        self,
        worker_id: str,
        flow_id: str,
        turn_number: int,
        screenshot_bytes: bytes
    ) -> Optional[Path]:
        """
        Save a screenshot PNG to the screenshots directory.

        Returns the path where screenshot was saved, or None if saving disabled.
        """
        if not self._save_screenshots:
            return None

        short_flow_id = flow_id[:8] if len(flow_id) > 8 else flow_id
        filename = f"{worker_id}_{short_flow_id}_turn{turn_number:03d}.png"
        filepath = self.screenshots_dir / filename

        with open(filepath, "wb") as f:
            f.write(screenshot_bytes)

        return filepath

    # =========================================================================
    # Supervisor Chat Logging
    # =========================================================================

    def log_supervisor_call(
        self,
        context: str,
        response: dict,
        input_tokens: int = 0,
        output_tokens: int = 0
    ):
        """Log a supervisor AI call."""
        file_path = self._get_supervisor_file()
        file_key = str(file_path)

        if file_key not in self._initialized_files:
            self._write_header(file_path, "SUPERVISOR LOG", {
                "Exploration": self.exploration_id,
                "Started": datetime.now().isoformat()
            })
            self._initialized_files.add(file_key)

        self._append(file_path, "=" * 80 + "\n")
        self._append(file_path, f"SUPERVISOR CALL | {datetime.now().strftime('%H:%M:%S')} | ")
        self._append(file_path, f"Tokens: {input_tokens} in, {output_tokens} out\n")
        self._append(file_path, "=" * 80 + "\n\n")

        # Mask credentials in context and response
        self._append(file_path, "--- CONTEXT ---\n")
        self._append(file_path, _mask_credentials_in_text(context))
        self._append(file_path, "\n\n")

        self._append(file_path, "--- RESPONSE ---\n")
        self._append(file_path, json.dumps(_mask_credentials_in_dict(response), indent=2))
        self._append(file_path, "\n\n")

    # =========================================================================
    # Synthesis Chat Logging
    # =========================================================================

    def log_synthesis_call(
        self,
        context: str,
        report: str,
        input_tokens: int = 0,
        output_tokens: int = 0
    ):
        """Log a synthesis agent AI call."""
        file_path = self._get_synthesis_file()
        file_key = str(file_path)

        if file_key not in self._initialized_files:
            self._write_header(file_path, "SYNTHESIS LOG", {
                "Exploration": self.exploration_id,
                "Started": datetime.now().isoformat()
            })
            self._initialized_files.add(file_key)

        self._append(file_path, "=" * 80 + "\n")
        self._append(file_path, f"SYNTHESIS CALL | {datetime.now().strftime('%H:%M:%S')} | ")
        self._append(file_path, f"Tokens: {input_tokens} in, {output_tokens} out\n")
        self._append(file_path, "=" * 80 + "\n\n")

        self._append(file_path, "--- CONTEXT ---\n")
        self._append(file_path, context)
        self._append(file_path, "\n\n")

        self._append(file_path, "--- REPORT ---\n")
        self._append(file_path, report)
        self._append(file_path, "\n\n")

    def close(self):
        """Clean up logger resources to prevent memory leaks."""
        if self._ops_logger:
            # Close and remove all handlers
            for handler in self._ops_logger.handlers[:]:
                handler.close()
                self._ops_logger.removeHandler(handler)
            # Remove logger from manager to prevent accumulation
            logger_name = f"qa-bot.{self.exploration_id}"
            if logger_name in logging.Logger.manager.loggerDict:
                del logging.Logger.manager.loggerDict[logger_name]
            self._ops_logger = None
