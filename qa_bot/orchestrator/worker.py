"""Flow-based exploration worker - one worker per flow."""

import asyncio
import logging
import uuid
from typing import AsyncGenerator
from datetime import datetime
from dataclasses import dataclass
from urllib.parse import urlparse

from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from qa_bot.ai.base import AIProvider, AgentAction
from qa_bot.agent.state import Issue
from qa_bot.browser.controller import BrowserController, BENIGN_FAILURE_REASONS, DuplicateRefError
from qa_bot.orchestrator.browser_pool import BrowserPool, strip_url_credentials
from qa_bot.orchestrator.shared_state import SharedFlowState, UserDataField, UserDataRequest, format_user_data_for_prompt
from qa_bot.orchestrator.flow import FlowTask, FlowCheckpoint, FlowStatus

logger = logging.getLogger(__name__)


# ── Network failure classification ──────────────────────────────────────


def classify_network_failure(
    req_url: str,
    req_method: str,
    req_status: int,
    req_resource_type: str,
    req_failure_reason: str,
    page_url: str,
) -> tuple[str, str] | None:
    """Classify a failed network request into (severity, description) or None to skip.

    Returns None if the failure is benign (browser abort, ad-blocker, etc.)
    and should not be reported as an issue.
    """
    failure = req_failure_reason or ""

    # Skip benign browser-initiated aborts entirely
    if failure in BENIGN_FAILURE_REASONS:
        return None

    # Skip 4xx on the page the AI itself navigated to: the AI is looking at
    # the error page in its screenshot and can report it with real context.
    # Auto-filing these turned 9 deliberate /admin-variant probes into 9
    # separate "404 Not Found" issues on mono PR #356. Sub-resource and
    # background-request failures (invisible to the AI) are still auto-filed,
    # as are 5xx everywhere.
    if (
        req_resource_type == "document"
        and 400 <= req_status < 500
        and _normalize_url_for_comparison(req_url)
        == _normalize_url_for_comparison(page_url)
    ):
        return None

    # Determine if the failed request is same-origin.
    # Only strips "www." — subdomains like api.example.com are treated as
    # third-party. This is intentional: most cross-subdomain failures are
    # external services, and same-site APIs typically share the main domain.
    page_origin = urlparse(page_url).netloc.removeprefix("www.")
    req_origin = urlparse(req_url).netloc.removeprefix("www.")
    is_same_origin = req_origin == page_origin

    if req_status >= 500:
        severity = "major" if is_same_origin else "minor"
        desc = f"Server error {req_status}: {req_method} {req_url[:100]}"
    elif req_status == 0:
        # Real connection failures (not aborted — those were filtered above)
        if is_same_origin:
            severity = "major"
        elif req_resource_type in ("document", "xhr", "fetch"):
            severity = "minor"
        else:
            # Third-party image/font/style failures — not actionable
            return None
        desc = f"Network failure: {req_method} {req_url[:100]} - {failure}"
    elif req_status == 404:
        severity = "minor" if is_same_origin else "cosmetic"
        desc = f"404 Not Found: {req_method} {req_url[:100]}"
    else:
        severity = "minor"
        desc = f"HTTP {req_status}: {req_method} {req_url[:100]}"

    return severity, desc


def _strip_old_screenshots(history: list[dict]) -> None:
    """Replace all base64 image blocks with a lightweight text placeholder.

    Mutates the list in-place to free memory.  The current turn's screenshot
    lives in ``current_message`` (not in ``conversation_history``), so every
    image in history is safe to strip.  ``_build_messages_from_history`` in
    claude_provider performs the same replacement when building the API
    request, but this function frees the actual bytes from the Python heap
    between turns so they don't accumulate over long flows.
    """
    for msg in history:
        content = msg.get("content")
        if isinstance(content, list):
            for j, block in enumerate(content):
                if isinstance(block, dict) and block.get("type") == "image":
                    content[j] = {"type": "text", "text": "[Screenshot of page]"}


# Explicit budget for the per-turn screenshot. The browser context's default
# action timeout is deliberately short (fail fast on stale refs), but the
# main-loop screenshot is not a ref-based action: a heavy animation or busy
# main thread can transiently exceed the context default, and that must cost
# a retry, not the whole flow.
TURN_SCREENSHOT_TIMEOUT_MS = 15000

# Max individually auto-filed network issues per (status, origin) per flow.
# Past this, a single rollup issue is filed and further repeats are dropped —
# one broken asset pattern should be one finding, not one issue per URL.
MAX_NETWORK_ISSUES_PER_PATTERN = 3


def _normalize_url_for_comparison(url: str) -> str:
    """Normalize a URL for reload/commit equality checks.

    Strips the fragment and trailing slash: Playwright's page.url is
    normalized and servers append/remove slashes or fragments on redirect,
    so 'https://x.com/page', 'https://x.com/page/' and
    'https://x.com/page#top' are the same page for navigation-commit
    purposes. Exact string equality would misclassify a reload of such a
    page as a failed navigation, or a cosmetic redirect as a real one.
    """
    base, _, _ = url.partition("#")
    return base.rstrip("/")


# How long the document gets to reach DOMContentLoaded after the server has
# started responding. A hanging third-party script can hold the parser open
# far longer — that's a page-weight problem, not a failed navigation.
NAVIGATION_SETTLE_TIMEOUT_MS = 10_000


async def _safe_goto(page, url: str, timeout: int = 30000) -> None:
    """Navigate to a URL, failing only when the server never responds.

    Two-phase wait:

    1. ``wait_until="commit"`` with the full timeout — fails only if the
       server never starts responding (hanging server, or a beforeunload
       guard cancelled the navigation). This is the only case worth telling
       the AI "navigation failed".
    2. A bounded ``domcontentloaded`` wait afterwards. If the document
       doesn't settle in time (e.g. a hung third-party script holding the
       parser open), log and continue — the server answered, so reporting
       "timed out" would be a false positive. onlinestoryservices PR #742
       had exactly this: nginx answered story URLs instantly with 200/301,
       yet the bot reported 30s timeouts and a critical "story links broken".

    If the commit wait itself times out but the page URL still changed (the
    navigation raced the timeout), continue too. A goto targeting the URL the
    page is already on is a reload — its URL cannot change, so the
    URL-unchanged signal carries no information there and re-raising would
    fail every reload of a slow page; treat it as committed.

    All URL comparisons are fragment- and trailing-slash-insensitive (see
    _normalize_url_for_comparison) so cosmetic differences between the AI's
    target URL and Playwright's normalized page.url don't misclassify.
    """
    url_before = page.url
    try:
        await page.goto(url, wait_until="commit", timeout=timeout)
    except (TimeoutError, PlaywrightTimeoutError):
        norm_before = _normalize_url_for_comparison(url_before)
        is_reload = _normalize_url_for_comparison(url) == norm_before
        committed = _normalize_url_for_comparison(page.url) != norm_before
        if not (page.url and page.url != "about:blank" and (is_reload or committed)):
            raise
        logger.warning(
            f"Navigation to {url} timed out waiting for commit, "
            f"but page is on {page.url}. Continuing."
        )
    try:
        await page.wait_for_load_state(
            "domcontentloaded", timeout=NAVIGATION_SETTLE_TIMEOUT_MS
        )
    except (TimeoutError, PlaywrightTimeoutError):
        logger.warning(
            f"Page {page.url} committed but domcontentloaded didn't fire "
            f"within {NAVIGATION_SETTLE_TIMEOUT_MS}ms. Continuing with "
            f"whatever has rendered."
        )
        await asyncio.sleep(2)


async def _settle_after_interaction(browser, sleep_s: float = 0.5) -> None:
    """Pause after an interaction; if it triggered a navigation, wait
    (bounded) for the new document so the next screenshot isn't a blank
    mid-load frame the AI misreads as a hang or broken page.

    wait_for_load_state returns immediately when the current document is
    already past domcontentloaded, so this only costs time mid-navigation.
    """
    await asyncio.sleep(sleep_s)
    page = browser.active_page
    if not page:
        return
    try:
        await page.wait_for_load_state(
            "domcontentloaded", timeout=NAVIGATION_SETTLE_TIMEOUT_MS
        )
    except (TimeoutError, PlaywrightTimeoutError):
        logger.warning(
            f"Document on {page.url} not settled after interaction; continuing."
        )
    except Exception:
        # Page may have closed (popup flows) — never fail the turn here.
        pass


def _target_url_for_auth(page_url: str, fallback_url: str) -> str:
    """Return the best URL for HTTP auth scoping.

    Avoids using about:blank or empty strings, which would cause
    route interception patterns to not match the actual domain.
    """
    if page_url and page_url != "about:blank":
        return page_url
    return fallback_url


@dataclass
class ActionResult:
    """Result of executing an action."""
    success: bool
    message: str
    error: str = ""
    flow_done: bool = False
    done_reason: str = ""
    flow_blocked: bool = False
    block_reason: str = ""
    pending_flows: list = None
    reported_issues: list = None
    screenshot: bytes = None  # For explicit screenshot action
    data_request: "UserDataRequest" = None  # For request_data action
    needs_context_rebuild: bool = False  # For set_http_auth: recreate browser context

    def __post_init__(self):
        if self.pending_flows is None:
            self.pending_flows = []
        if self.reported_issues is None:
            self.reported_issues = []


@dataclass
class PendingFlow:
    """A flow to be created."""
    name: str
    description: str
    keep_state: bool = False


@dataclass
class ReportedIssue:
    """An issue reported during action execution."""
    description: str
    severity: str


class FlowExplorationWorker:
    """
    Worker that executes a single flow exploration task.

    Architecture: One worker per flow
    - Worker is created with a specific FlowTask
    - AI analyzes screenshots and returns JSON actions
    - Actions are executed via simple handlers matching Chrome MCP format
    - Runs until AI returns done/block action, or time/token limits reached
    - Then exits (coordinator handles spawning new workers for new flows)

    Available actions (Chrome MCP aligned):
    - left_click: Click an element by ref
    - right_click: Right-click for context menu
    - double_click: Double-click an element
    - triple_click: Triple-click (select paragraph)
    - hover: Mouse hover over element
    - type: Type text into an element
    - form_input: Set form value (checkbox, select)
    - scroll: Scroll the page
    - scroll_to: Scroll element into view
    - key: Press keyboard key(s)
    - left_click_drag: Drag from start to end coordinate
    - screenshot: Take explicit screenshot
    - zoom: Zoom into region for inspection
    - wait: Wait/pause
    - navigate: Go to URL or back/forward
    - resize: Resize viewport

    QA-specific actions (extensions):
    - report_issue: Report a QA issue found
    - add_flow: Create a new flow to explore
    - done: Mark current flow as complete
    - block: Request help from supervisor
    """

    def __init__(
        self,
        worker_id: str,
        worker_number: int,
        ai_provider: AIProvider,
        browser_pool: BrowserPool,
        shared_state: SharedFlowState,
        flow_task: FlowTask,
        is_first_worker: bool = False
    ):
        self.worker_id = worker_id
        self.worker_number = worker_number
        self.ai = ai_provider
        self.browser_pool = browser_pool
        self.state = shared_state
        self.flow_task = flow_task
        self.is_first_worker = is_first_worker
        self._blocked = False
        # (status, origin) -> count of auto-filed network issues this flow;
        # used to cap repeat spam (one broken asset pattern = one finding,
        # not one issue per URL variant)
        self._network_issue_counts: dict[tuple[int, str], int] = {}

    async def run(self) -> AsyncGenerator[dict, None]:
        """Execute the assigned flow task."""
        yield {
            "type": "worker_started",
            "data": {
                "worker_id": self.worker_id,
                "worker_number": self.worker_number,
                "flow_id": self.flow_task.flow_id,
                "flow_name": self.flow_task.flow_name,
                "is_first_worker": self.is_first_worker,
                "timestamp": datetime.now().isoformat()
            }
        }

        try:
            async for event in self._execute_flow():
                yield event
        except asyncio.CancelledError:
            raise  # Let cancellation propagate after _execute_flow handled checkpoint
        except Exception as e:
            retried = await self.state.fail_flow(
                self.flow_task.flow_id, str(e), task=self.flow_task
            )
            yield {
                "type": "flow_retrying" if retried else "flow_failed",
                "data": {
                    "worker_id": self.worker_id,
                    "flow_id": self.flow_task.flow_id,
                    "flow_name": self.flow_task.flow_name,
                    "error": str(e),
                    **({"attempt": self.flow_task.attempt} if retried else {}),
                }
            }

    async def _execute_flow(self) -> AsyncGenerator[dict, None]:
        """Execute the flow exploration."""
        task = self.flow_task

        flow_data = await self.state.start_flow_exploration(task, self.worker_id)
        if flow_data is None:
            # The supervisor marked this flow skip after the coordinator's
            # re-checks but before we claimed it. Honor the skip: the run
            # already emitted flow_skipped for it, so exploring it anyway
            # would spend budget on a flow the report says was deduplicated.
            logger.info(
                f"{self.worker_id}: flow {task.flow_name} ({task.flow_id}) "
                f"was marked skip before exploration started; discarding"
            )
            return

        yield {
            "type": "flow_started",
            "data": {
                "flow_id": task.flow_id,
                "flow_name": task.flow_name,
                "worker_id": self.worker_id,
                "is_root": task.is_root,
                "parent_flow_id": task.parent_flow_id,
                "flow_path": task.flow_path.to_list()
            }
        }

        context = None
        page = None
        browser = None
        conversation_history = []
        parent_flow_name = ""

        try:
            # Check for HTTP Basic Auth credentials
            http_auth = await self.state.get_http_auth()
            pw_credentials = (
                {"username": http_auth["username"], "password": http_auth["password"]}
                if http_auth else None
            )

            if task.is_root or task.checkpoint_id is None:
                # Root flow or fresh flow: create fresh context
                # Flows created via add_flow_task have checkpoint_id=None and start fresh
                context = await self.browser_pool.create_isolated_context(
                    http_credentials=pw_credentials,
                )

                # Also apply via route interception to proactively send the header
                # (avoids 401 round-trip on subrequests after initial navigation)
                if http_auth:
                    await self.browser_pool.apply_http_auth(
                        context, http_auth["username"], http_auth["password"], http_auth["target_url"]
                    )

                page = await context.new_page()

                try:
                    await _safe_goto(page, task.start_url)
                except Exception as e:
                    yield {
                        "type": "navigation_error",
                        "worker_id": self.worker_id,
                        "flow_id": task.flow_id,
                        "data": {
                            "flow_id": task.flow_id,
                            "url": task.start_url,
                            "error": str(e)
                        }
                    }
                    raise

            else:
                # Continuation flow: restore from checkpoint
                checkpoint = await self.state.get_checkpoint(task.checkpoint_id)
                if checkpoint is None:
                    raise ValueError(f"Checkpoint {task.checkpoint_id} not found for flow {task.flow_id}")

                await self.state.claim_checkpoint(task.checkpoint_id, self.worker_id)
                parent_flow_name = checkpoint.branch_name or ""

                context = await self.browser_pool.create_isolated_context(
                    storage_state=checkpoint.browser_storage_state,
                    http_credentials=pw_credentials,
                )

                # Also apply via route interception for checkpoint-restored flows
                if http_auth:
                    await self.browser_pool.apply_http_auth(
                        context, http_auth["username"], http_auth["password"], http_auth["target_url"]
                    )

                page = await context.new_page()

                try:
                    await _safe_goto(page, strip_url_credentials(checkpoint.current_url))
                except Exception as e:
                    yield {
                        "type": "checkpoint_restore_error",
                        "worker_id": self.worker_id,
                        "flow_id": task.flow_id,
                        "data": {
                            "flow_id": task.flow_id,
                            "error": str(e)
                        }
                    }
                    raise

                conversation_history = checkpoint.conversation_history.copy()

                # Check if this is a resume from approval request
                approval_result = await self.state.get_approval_result(task.flow_id)
                if approval_result:
                    if approval_result.get('approved'):
                        # Add message indicating user approved the action
                        conversation_history.append({
                            "role": "user",
                            "content": [{"type": "text", "text": "User APPROVED the action. You may proceed with the action you requested approval for."}]
                        })
                    else:
                        # Add message indicating user denied the action
                        conversation_history.append({
                            "role": "user",
                            "content": [{"type": "text", "text": "User DENIED the action. Skip the action you requested approval for and continue testing other aspects of the flow."}]
                        })

            browser = BrowserController.from_page(page)
            prior_context = self._build_flow_context(task, conversation_history)

            action_count = 0
            action_history = []
            flow_completed = False
            last_error = ""
            # Why the exploration loop broke, for accurate flow accounting after
            # the loop. None means the loop ended by its own condition
            # (flow done, or should_stop() cost/time cutoff). Abnormal breaks set
            # this so the flow is recorded as failed/interrupted, not completed.
            exit_reason = None
            supervisor_stop_reason = ""

            # Main exploration loop - runs until AI returns done/block action, or global limits hit
            while not flow_completed and not self.state.should_stop():
                # Check if we're blocked
                if self._blocked:
                    response = await self.state.wait_for_unblock(self.worker_id, timeout=300)
                    if response is None:
                        exit_reason = "block_timeout"
                        yield {
                            "type": "block_timeout",
                            "data": {"worker_id": self.worker_id, "flow_id": task.flow_id}
                        }
                        break
                    elif response.get("action") == "stop":
                        exit_reason = "supervisor_stop"
                        supervisor_stop_reason = response.get("reason", "")
                        break
                    elif response.get("action") == "checkpoint_and_exit":
                        # Save checkpoint and exit - flow will resume when credentials arrive
                        reason = response.get("reason", "Waiting for credentials")
                        storage_state = await context.storage_state()
                        _strip_old_screenshots(conversation_history)
                        checkpoint = FlowCheckpoint(
                            parent_flow_id=task.flow_id,
                            browser_storage_state=storage_state,
                            # Resume on the page the AI was actually looking at
                            # (active popup when one is open) — page.url would
                            # silently point at the opener while the inherited
                            # history references the popup.
                            current_url=browser.current_url,
                            conversation_history=conversation_history.copy(),
                            flow_path=task.flow_path,
                            branch_name=f"credential_resume:{task.flow_name}",
                            created_by_worker=self.worker_id
                        )
                        # Save checkpoint without queueing - task will be created
                        # by provide_credentials() when credentials arrive
                        saved_checkpoint = await self.state.add_checkpoint(
                            checkpoint,
                            f"Resume after credentials: {reason}",
                            queue_task=False
                        )

                        if saved_checkpoint:
                            # Link checkpoint to credential request using the checkpoint's ID
                            await self.state.set_credential_request_checkpoint(
                                task.flow_id,
                                checkpoint.checkpoint_id
                            )

                        yield {
                            "type": "flow_blocked_for_credentials",
                            "data": {
                                "worker_id": self.worker_id,
                                "flow_id": task.flow_id,
                                "flow_name": task.flow_name,
                                "checkpoint_id": checkpoint.checkpoint_id,
                                "reason": reason
                            }
                        }

                        # Mark flow as blocked (not completed/failed)
                        await self.state.block_flow_for_credentials(task.flow_id, reason)
                        break  # Exit loop - worker can pick up new flow
                    elif response.get("action") == "checkpoint_and_exit_for_approval":
                        # Save checkpoint and exit - flow will resume when user approves/denies
                        reason = response.get("reason", "Waiting for approval")
                        storage_state = await context.storage_state()
                        _strip_old_screenshots(conversation_history)
                        checkpoint = FlowCheckpoint(
                            parent_flow_id=task.flow_id,
                            browser_storage_state=storage_state,
                            # Active page (popup-aware), matching the history
                            current_url=browser.current_url,
                            conversation_history=conversation_history.copy(),
                            flow_path=task.flow_path,
                            branch_name=f"approval_resume:{task.flow_name}",
                            created_by_worker=self.worker_id
                        )
                        # Save checkpoint without queueing - task will be created
                        # by provide_approval() when user responds
                        saved_checkpoint = await self.state.add_checkpoint(
                            checkpoint,
                            f"Resume after approval: {reason}",
                            queue_task=False
                        )

                        if saved_checkpoint:
                            # Link checkpoint to approval request
                            await self.state.set_approval_request_checkpoint(
                                task.flow_id,
                                checkpoint.checkpoint_id
                            )

                        yield {
                            "type": "flow_blocked_for_approval",
                            "data": {
                                "worker_id": self.worker_id,
                                "flow_id": task.flow_id,
                                "flow_name": task.flow_name,
                                "checkpoint_id": checkpoint.checkpoint_id,
                                "reason": reason
                            }
                        }

                        # Mark flow as blocked for approval
                        await self.state.block_flow_for_approval(task.flow_id, reason)
                        break  # Exit loop - worker can pick up new flow
                    elif response.get("action") == "checkpoint_and_exit_for_data":
                        # Save checkpoint and exit - flow will resume when user provides data
                        reason = response.get("reason", "Waiting for user data")
                        request_name = response.get("request_name", "")
                        storage_state = await context.storage_state()
                        _strip_old_screenshots(conversation_history)
                        checkpoint = FlowCheckpoint(
                            parent_flow_id=task.flow_id,
                            browser_storage_state=storage_state,
                            # Active page (popup-aware), matching the history
                            current_url=browser.current_url,
                            conversation_history=conversation_history.copy(),
                            flow_path=task.flow_path,
                            branch_name=f"data_resume:{task.flow_name}",
                            created_by_worker=self.worker_id
                        )
                        # Save checkpoint without queueing - task will be created
                        # by provide_user_data() when data arrives
                        saved_checkpoint = await self.state.add_checkpoint(
                            checkpoint,
                            f"Resume after data: {reason}",
                            queue_task=False
                        )

                        if saved_checkpoint and request_name:
                            # Link checkpoint to data request
                            await self.state.set_data_request_checkpoint(
                                request_name,
                                checkpoint.checkpoint_id
                            )

                        yield {
                            "type": "flow_blocked_for_data",
                            "data": {
                                "worker_id": self.worker_id,
                                "flow_id": task.flow_id,
                                "flow_name": task.flow_name,
                                "checkpoint_id": checkpoint.checkpoint_id,
                                "request_name": request_name,
                                "reason": reason
                            }
                        }

                        # Mark flow as blocked for credentials (reusing existing status)
                        await self.state.block_flow_for_credentials(task.flow_id, reason)
                        break  # Exit loop - worker can pick up new flow
                    else:
                        supervisor_message = response.get("message", "")
                        prior_context = f"Supervisor response: {supervisor_message}\n\n{prior_context}"
                        self._blocked = False

                # Pin the active page for the whole turn: a popup opening
                # mid-turn would otherwise flip active_page between the
                # screenshot and the ref list (AI sees one page, refs point
                # at another) or between observation and the action. The
                # popup becomes active on the next turn's pin.
                browser.pin_active_page()

                # Take screenshot (isolated: one slow render must not kill the flow)
                screenshot = await self._take_turn_screenshot(browser)

                # Get ref list and viewport size
                ref_list = await browser.get_ref_list()
                viewport_size = await browser.get_viewport_size()

                # Save screenshot to disk if enabled
                if self.state.chat_logger:
                    self.state.chat_logger.save_screenshot(
                        worker_id=self.worker_id,
                        flow_id=task.flow_id,
                        turn_number=action_count,
                        screenshot_bytes=screenshot
                    )

                yield {
                    "type": "screenshot",
                    "worker_id": self.worker_id,
                    "flow_id": task.flow_id,
                    "data": {
                        "flow_id": task.flow_id,
                        "index": action_count,
                        # The screenshot shows the active page (popup when one
                        # is open), so report that page's URL, not the opener's
                        "url": browser.current_url
                    }
                }

                # Call AI for analysis
                try:
                    current_thinking = ""
                    current_screenshot_size = len(screenshot)

                    # Get credentials and user data for the AI to know what's available
                    credentials = await self.state.get_credentials()
                    user_data = await self.state.get_all_user_data()

                    async for ai_event in self.ai.analyze_for_worker_stream(
                        screenshot_bytes=screenshot,
                        ref_list=ref_list,
                        # Must match the screenshot: the URL of the active
                        # page (popup when one is open), not the main page
                        current_url=browser.current_url,
                        flow_name=task.flow_name,
                        flow_goal=task.goal,
                        action_history=action_history,
                        conversation_history=conversation_history,
                        prior_context=prior_context,
                        additional_context="",
                        is_first_worker=self.is_first_worker,
                        worker_number=self.worker_number,
                        flow_description=task.goal,
                        parent_flow_name=parent_flow_name,
                        target_domain=self.state.target_domain,
                        viewport_width=viewport_size["width"],
                        viewport_height=viewport_size["height"],
                        credentials=credentials,
                        user_data=user_data,
                    ):
                        # Handle AI events
                        result = await self._handle_ai_event(
                            ai_event=ai_event,
                            page=page,
                            context=context,
                            browser=browser,
                            task=task,
                            action_count=action_count,
                            action_history=action_history,
                            conversation_history=conversation_history,
                            current_screenshot_size=current_screenshot_size,
                        )

                        if result is None:
                            continue  # Not a complete event, keep streaming

                        # Yield any events from the handler
                        for event in result.get("events", []):
                            yield event

                        # Update browser references if context was rebuilt (e.g., after set_http_auth)
                        if "new_page" in result:
                            page = result["new_page"]
                            context = result["new_context"]
                            browser = result["new_browser"]

                        # Check for flow completion or error
                        if result.get("flow_completed"):
                            flow_completed = True
                            break
                        if result.get("should_break"):
                            break

                        # Update action count if action was executed
                        if result.get("action_executed"):
                            action_count += 1

                        # Update current thinking
                        if result.get("thinking"):
                            current_thinking = result["thinking"]

                    # Clear prior context after first turn
                    prior_context = ""

                except Exception as e:
                    last_error = str(e)
                    exit_reason = "ai_exception"
                    yield {
                        "type": "ai_error",
                        "worker_id": self.worker_id,
                        "flow_id": task.flow_id,
                        "data": {"error": last_error}
                    }
                    break

            # The loop exited without the AI marking the flow done. Record the
            # flow honestly by exit reason instead of always calling it
            # COMPLETED — a crashed or cut-off flow reported as completed
            # silently inflates the coverage the QA report claims.
            if not flow_completed and flow_data.status == FlowStatus.EXPLORING:
                if exit_reason in ("ai_exception", "block_timeout"):
                    # The worker could not finish testing — a mid-flow failure.
                    # Route to FAILED so synthesis lists it as a coverage gap,
                    # matching the retries-exhausted path (fail_flow, above).
                    reason = (
                        f"AI error: {last_error[:200]}" if last_error
                        else "Worker blocked with no response from supervisor"
                    )
                    retried = await self.state.fail_flow(
                        task.flow_id, reason, task=task
                    )
                    yield {
                        "type": "flow_retrying" if retried else "flow_failed",
                        "worker_id": self.worker_id,
                        "data": {
                            "flow_id": task.flow_id,
                            "flow_name": task.flow_name,
                            "error": reason,
                            "action_count": action_count,
                            **({"attempt": task.attempt} if retried else {}),
                        }
                    }
                elif exit_reason == "supervisor_stop":
                    # Deliberate stop — never retry what the supervisor killed
                    reason = supervisor_stop_reason or "Stopped by supervisor"
                    await self.state.fail_flow(
                        task.flow_id, reason, task=task, allow_retry=False
                    )
                    yield {
                        "type": "flow_failed",
                        "worker_id": self.worker_id,
                        "data": {
                            "flow_id": task.flow_id,
                            "flow_name": task.flow_name,
                            "error": reason,
                            "action_count": action_count,
                        }
                    }
                elif self.state.should_stop():
                    # Cost/time cutoff mid-exploration. How to record it depends
                    # on the run mode, because BLOCKED_FOR_PAUSE is a
                    # resume-pending status that is_complete() blocks on:
                    reason = self.state.get_stop_reason()
                    if self.state.interactive:
                        # Interactive (web UI): the run pauses for possible
                        # resume, and the coordinator's cancellation path owns
                        # checkpoint creation. If the worker self-exits on
                        # should_stop() before that cancellation lands, marking
                        # the flow BLOCKED_FOR_PAUSE here (with no checkpoint)
                        # would wedge is_complete() forever — nothing can resume a
                        # checkpoint-less flow, so the run would never synthesize.
                        # Complete it instead (the pre-existing behavior); the
                        # coordinator still drives the pause/resume UX.
                        await self.state.complete_flow(task.flow_id, reason)
                        yield {
                            "type": "flow_completed",
                            "worker_id": self.worker_id,
                            "data": {
                                "flow_id": task.flow_id,
                                "flow_name": task.flow_name,
                                "completion_reason": reason,
                                "action_count": action_count,
                            }
                        }
                    else:
                        # Non-interactive (CI): a terminal cutoff, no resume. Mark
                        # it BLOCKED_FOR_PAUSE so synthesis reports it as untested
                        # rather than verified coverage (started_at is set, so it
                        # routes into incomplete flows). The coordinator breaks
                        # straight to synthesis on should_stop() here, bypassing
                        # is_complete(), so this can't hang.
                        await self.state.block_flow_for_pause(task.flow_id, reason)
                        yield {
                            "type": "flow_interrupted",
                            "worker_id": self.worker_id,
                            "data": {
                                "flow_id": task.flow_id,
                                "flow_name": task.flow_name,
                                "completion_reason": reason,
                                "action_count": action_count,
                            }
                        }
                else:
                    # Genuinely ambiguous exit with no error and no stop signal —
                    # treat as completed but keep whatever reason we can surface.
                    completion_reason = self.state.get_stop_reason()
                    if completion_reason == "Unknown" and last_error:
                        completion_reason = f"AI error: {last_error[:200]}"
                    await self.state.complete_flow(task.flow_id, completion_reason)
                    yield {
                        "type": "flow_completed",
                        "worker_id": self.worker_id,
                        "data": {
                            "flow_id": task.flow_id,
                            "flow_name": task.flow_name,
                            "completion_reason": completion_reason,
                            "action_count": action_count
                        }
                    }

        except asyncio.CancelledError:
            # Check if pause (should checkpoint) vs user stop (no checkpoint)
            # Note: We can't yield here - the generator is being torn down.
            # The checkpoint is saved and coordinator will emit events for saved checkpoints.
            if self.state.is_pause_cancellation():
                # Always mark the flow as blocked for pause, even if we can't checkpoint
                # This prevents the flow from showing as "exploring" after pause
                await self.state.block_flow_for_pause(task.flow_id, self.state.get_stop_reason())

                # Only try to create checkpoint if browser context is available
                if context and page:
                    try:
                        storage_state = await asyncio.wait_for(context.storage_state(), timeout=5.0)
                        _strip_old_screenshots(conversation_history)
                        checkpoint = FlowCheckpoint(
                            parent_flow_id=task.flow_id,
                            browser_storage_state=storage_state,
                            # Active page (popup-aware). browser may be unbound
                            # here: cancellation can land before the controller
                            # is created (during the initial _safe_goto), so
                            # fall back to the main page's URL.
                            current_url=browser.current_url if browser else page.url,
                            conversation_history=conversation_history.copy(),
                            flow_path=task.flow_path,
                            branch_name=f"pause_resume:{task.flow_name}",
                            created_by_worker=self.worker_id
                        )
                        await self.state.add_pause_checkpoint(checkpoint, f"Resume after pause: {task.goal}")
                        # Checkpoint saved - coordinator will emit flow_paused_checkpoint events
                    except Exception as e:
                        logger.warning(f"Failed to save pause checkpoint for {task.flow_name}: {e}")
            raise  # Re-raise CancelledError

        finally:
            if browser:
                browser.unpin_active_page()
            if context:
                try:
                    await asyncio.wait_for(context.close(), timeout=5.0)
                except Exception:
                    pass  # Ignore errors/timeouts when closing context

    def _cap_network_issue(
        self, status: int, origin: str, severity: str, desc: str
    ) -> tuple[str, str] | None:
        """Apply the per-flow cap on auto-filed network issues.

        The first MAX_NETWORK_ISSUES_PER_PATTERN failures per (status, origin)
        file individually, the next one becomes a single rollup issue, the
        rest are dropped. One broken asset pattern = one finding, not one
        issue per URL variant.

        Returns (severity, desc) to file, or None to skip.
        """
        key = (status, origin)
        seen = self._network_issue_counts.get(key, 0)
        self._network_issue_counts[key] = seen + 1
        if seen < MAX_NETWORK_ISSUES_PER_PATTERN:
            return severity, desc
        if seen == MAX_NETWORK_ISSUES_PER_PATTERN:
            return "minor", (
                f"Multiple additional HTTP {status} failures on {origin} — "
                f"further occurrences not individually reported"
            )
        return None

    async def _take_turn_screenshot(self, browser: BrowserController) -> bytes:
        """Take the per-turn screenshot with an explicit timeout and one retry.

        page.screenshot inherits the context's short default action timeout,
        so a single >timeout stall (heavy animation, busy main thread) would
        otherwise fail the entire flow from the main loop, outside any
        per-action error handling. A persistent failure still propagates —
        the AI cannot act without seeing the page.
        """
        try:
            return await browser.screenshot(timeout_ms=TURN_SCREENSHOT_TIMEOUT_MS)
        except Exception as e:
            logger.warning(
                f"{self.worker_id}: per-turn screenshot failed ({e}); retrying once"
            )
            await asyncio.sleep(1.0)
            return await browser.screenshot(timeout_ms=TURN_SCREENSHOT_TIMEOUT_MS)

    def _build_flow_context(self, task: FlowTask, conversation_history: list[dict]) -> str:
        """Build context string from flow path and history."""
        context_parts = []

        if len(task.flow_path) > 1:
            context_parts.append(f"## Flow Context")
            context_parts.append(f"Flow path: {task.flow_path.as_string()}")
            context_parts.append("")

        if conversation_history:
            context_parts.append("## Previous Context")
            context_parts.append(f"{len(conversation_history)} previous exchanges preserved.")
            context_parts.append("")

        return "\n".join(context_parts) if context_parts else ""

    async def _handle_ai_event(
        self,
        ai_event: dict,
        page,
        context,
        browser: BrowserController,
        task: FlowTask,
        action_count: int,
        action_history: list[dict],
        conversation_history: list[dict],
        current_screenshot_size: int,
    ) -> dict | None:
        """
        Handle a single AI event.

        Returns dict with:
        - events: list of events to yield
        - flow_completed: True if flow should end
        - should_break: True if loop should break
        - action_executed: True if an action was executed
        - thinking: current thinking text (for streaming updates)

        All event types return a result dict. For streaming events (thinking_start,
        thinking_delta, thinking_complete), only the events list is populated.
        Returns None only for unrecognized event types.
        """
        events = []
        result = {"events": events}

        if ai_event["type"] == "thinking_start":
            events.append({
                "type": "ai_thinking_start",
                "worker_id": self.worker_id,
                "flow_id": task.flow_id,
                "data": {}
            })
            return result

        elif ai_event["type"] == "thinking_delta":
            events.append({
                "type": "ai_thinking_delta",
                "worker_id": self.worker_id,
                "flow_id": task.flow_id,
                "data": {"text": ai_event["text"]}
            })
            return result

        elif ai_event["type"] == "thinking_complete":
            result["thinking"] = ai_event.get("text", "")
            events.append({
                "type": "ai_thinking_complete",
                "worker_id": self.worker_id,
                "flow_id": task.flow_id,
                "data": {"text": result["thinking"]}
            })
            return result

        elif ai_event["type"] == "error":
            error_msg = ai_event.get("error", "Unknown AI error")

            # Terminal error events still carry the cumulative usage of every
            # API call made this turn (up to 3 full vision calls when parse
            # retries are exhausted). Bill them so the cost tracker, cost cap
            # and estimated_cost_usd account for failed turns too.
            input_tokens = ai_event.get("input_tokens", 0)
            output_tokens = ai_event.get("output_tokens", 0)
            cache_creation_tokens = ai_event.get("cache_creation_tokens", 0)
            cache_read_tokens = ai_event.get("cache_read_tokens", 0)
            if input_tokens or output_tokens or cache_creation_tokens or cache_read_tokens:
                await self.state.add_token_usage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cache_read_tokens=cache_read_tokens,
                    cache_creation_tokens=cache_creation_tokens,
                )

            if self.state.chat_logger:
                self.state.chat_logger.log_worker_ai_error(
                    worker_id=self.worker_id,
                    flow_id=task.flow_id,
                    turn_number=action_count,
                    current_url=browser.current_url,
                    error=error_msg,
                    raw_response=ai_event.get("raw_response"),
                    thinking=ai_event.get("thinking")
                )

            retried = await self.state.fail_flow(
                task.flow_id, f"AI error: {error_msg}", task=task
            )

            events.append({
                "type": "ai_error",
                "worker_id": self.worker_id,
                "flow_id": task.flow_id,
                "data": {"error": error_msg}
            })
            if retried:
                events.append({
                    "type": "flow_retrying",
                    "worker_id": self.worker_id,
                    "data": {
                        "flow_id": task.flow_id,
                        "flow_name": task.flow_name,
                        "error": f"AI error: {error_msg}",
                        "attempt": task.attempt,
                    }
                })

            result["flow_completed"] = True
            result["should_break"] = True
            return result

        elif ai_event["type"] == "complete":
            # Got action from AI - execute it
            action: AgentAction = ai_event["action"]
            thinking = ai_event.get("thinking", "")

            # Track token usage by type for accurate cost calculation
            input_tokens = ai_event.get("input_tokens", 0)
            output_tokens = ai_event.get("output_tokens", 0)
            cache_creation_tokens = ai_event.get("cache_creation_tokens", 0)
            cache_read_tokens = ai_event.get("cache_read_tokens", 0)
            if input_tokens or output_tokens or cache_creation_tokens or cache_read_tokens:
                await self.state.add_token_usage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cache_read_tokens=cache_read_tokens,
                    cache_creation_tokens=cache_creation_tokens,
                )

            # Log to chat history file
            if self.state.chat_logger:
                if action_count == 0:
                    self.state.chat_logger.log_worker_system_prompt(
                        worker_id=self.worker_id,
                        flow_id=task.flow_id,
                        flow_name=task.flow_name,
                        system_prompt=ai_event.get("system_prompt", "")
                    )
                self.state.chat_logger.log_worker_turn(
                    worker_id=self.worker_id,
                    flow_id=task.flow_id,
                    turn_number=action_count,
                    current_url=browser.current_url,
                    user_prompt=ai_event.get("user_prompt", ""),
                    thinking=thinking,
                    response=action.model_dump(),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    screenshot_size_bytes=current_screenshot_size,
                    cache_creation_tokens=cache_creation_tokens,
                    cache_read_tokens=cache_read_tokens
                )

            # Store conversation history
            if ai_event.get("assistant_content"):
                conversation_history.append({
                    "role": "assistant",
                    "content": ai_event["assistant_content"]
                })
            if ai_event.get("user_content"):
                conversation_history.append({
                    "role": "user",
                    "content": ai_event["user_content"]
                })

            # Free memory from old screenshots no longer needed for API calls
            _strip_old_screenshots(conversation_history)

            # Execute the action
            exec_result = await self._execute_action(action, page, browser, context, conversation_history, task)

            # Handle context rebuild (e.g., after set_http_auth applies credentials)
            if exec_result.needs_context_rebuild:
                http_auth = await self.state.get_http_auth()
                if http_auth:
                    reload_url = _target_url_for_auth(page.url, task.start_url)
                    try:
                        await page.close()
                        await context.close()
                        pw_credentials = {"username": http_auth["username"], "password": http_auth["password"]}
                        context = await self.browser_pool.create_isolated_context(
                            http_credentials=pw_credentials,
                        )
                        await self.browser_pool.apply_http_auth(
                            context, http_auth["username"], http_auth["password"], http_auth["target_url"]
                        )
                        page = await context.new_page()
                        await _safe_goto(page, reload_url)
                        browser = BrowserController.from_page(page)
                    except Exception as e:
                        logger.error(f"Context rebuild failed after set_http_auth: {e}")
                        # Close the just-created context before bailing so a
                        # failed apply_http_auth / new_page / goto doesn't leak
                        # it — on success the new context is handed back to the
                        # caller via result["new_context"], but on this error
                        # path nothing else owns it. (If create_isolated_context
                        # itself failed, `context` is the already-closed old one;
                        # the double close is caught and harmless.)
                        try:
                            await context.close()
                        except Exception:
                            pass
                        # Re-raise so the worker exits cleanly via the outer try/except
                        # rather than continuing with stale page/context references
                        raise
                    # Return new page/context/browser so caller can update its references
                    result["new_page"] = page
                    result["new_context"] = context
                    result["new_browser"] = browser

            # Detect file downloads triggered by the action. Without this,
            # clicking an export/PDF link looks like a dead button: the
            # screenshot doesn't change and nothing records the download.
            downloads = browser.get_downloads(clear=True)
            for download in downloads:
                events.append({
                    "type": "download",
                    "worker_id": self.worker_id,
                    "flow_id": task.flow_id,
                    "data": {
                        "url": download.url,
                        "suggested_filename": download.suggested_filename,
                        "page_url": download.page_url,
                        "timestamp": download.timestamp,
                    }
                })

            # Build action record for history
            action_dict = {
                "action_type": action.action_type,
                "reasoning": action.reasoning,
                # URL where the action was executed — the active page (popup
                # when one is open), matching where ref actions are routed
                "page_url": browser.current_url,
                "timestamp": datetime.now().isoformat(),
                "success": exec_result.success,
            }
            # Add action-specific fields
            if action.ref is not None:
                action_dict["ref"] = action.ref
            if action.text:
                action_dict["text"] = action.text
            if action.url:
                action_dict["target_url"] = action.url  # Target URL for navigate action
            if action.reason:
                action_dict["reason"] = action.reason
            if downloads:
                # Surface the download in action history so the next AI turn
                # knows the action worked (the screenshot won't show it).
                filenames = ", ".join(
                    d.suggested_filename or d.url for d in downloads
                )
                action_dict["note"] = f"Triggered file download: {filenames}"

            if exec_result.error:
                action_dict["error"] = exec_result.error
                events.append({
                    "type": "action_error",
                    "worker_id": self.worker_id,
                    "flow_id": task.flow_id,
                    "data": {
                        "error": exec_result.error,
                        "action_type": action.action_type
                    }
                })

            action_history.append(action_dict)
            await self.state.record_action_for_flow(task.flow_id, action_dict)
            result["action_executed"] = True

            # Handle flow control from action result
            if exec_result.flow_done:
                completion_reason = exec_result.done_reason
                await self.state.complete_flow(task.flow_id, completion_reason)

                if self.state.chat_logger:
                    self.state.chat_logger.log_worker_completion(
                        worker_id=self.worker_id,
                        flow_id=task.flow_id,
                        reason=completion_reason,
                        total_turns=action_count
                    )

                events.append({
                    "type": "flow_completed",
                    "worker_id": self.worker_id,
                    "data": {
                        "flow_id": task.flow_id,
                        "flow_name": task.flow_name,
                        "completion_reason": completion_reason,
                        "action_count": action_count
                    }
                })
                result["flow_completed"] = True
                result["should_break"] = True

            elif exec_result.flow_blocked:
                reason = exec_result.block_reason
                await self.state.block_worker(self.worker_id, task.flow_id, reason)
                self._blocked = True

                events.append({
                    "type": "worker_blocked",
                    "data": {
                        "worker_id": self.worker_id,
                        "flow_id": task.flow_id,
                        "reason": reason
                    }
                })

            # Handle data request
            if exec_result.data_request:
                data_request = exec_result.data_request

                # Check if we already have data for this request name
                existing_data = await self.state.get_user_data(data_request.request_name)
                if existing_data:
                    # Data already available - add to conversation history for AI to use
                    # Note: Password fields are masked in logs but AI still receives raw values
                    data_lines = [f"Data for '{data_request.request_name}' is available:"]
                    data_lines.extend(format_user_data_for_prompt(existing_data, data_request.fields))
                    conversation_history.append({
                        "role": "user",
                        "content": [{"type": "text", "text": "\n".join(data_lines) + "\n\nUse this data to continue."}]
                    })
                    events.append({
                        "type": "data_already_available",
                        "worker_id": self.worker_id,
                        "flow_id": task.flow_id,
                        "data": {
                            "request_name": data_request.request_name,
                            "message": "Data already available, continuing"
                        }
                    })
                else:
                    # Need to request data - add to pending and block
                    # Note: We don't emit data_request here - the supervisor handles that
                    # after receiving the block notification. This prevents duplicate events.
                    await self.state.add_data_request(data_request)
                    await self.state.block_worker(self.worker_id, task.flow_id, f"Requesting data: {data_request.request_name}")
                    self._blocked = True

            # Process pending flows
            for pending_flow in exec_result.pending_flows:
                if not await self.state.is_flow_skipped_by_name(pending_flow.name):
                    flow_created = False
                    # Fork from the page the AI is actually looking at
                    # (active popup when one is open, main page otherwise) so
                    # the child worker starts where the branch point really
                    # was — page.url would silently point at the opener.
                    branch_url = strip_url_credentials(browser.current_url)
                    if pending_flow.keep_state:
                        storage_state = await context.storage_state()
                        _strip_old_screenshots(conversation_history)
                        checkpoint = FlowCheckpoint(
                            parent_flow_id=task.flow_id,
                            browser_storage_state=storage_state,
                            current_url=branch_url,
                            conversation_history=conversation_history.copy(),
                            flow_path=task.flow_path.extend(pending_flow.name),
                            branch_name=pending_flow.name,
                            created_by_worker=self.worker_id
                        )
                        new_flow_task = await self.state.add_checkpoint(checkpoint, pending_flow.description)
                        flow_created = new_flow_task is not None
                    else:
                        new_flow_task = await self.state.add_flow_task(
                            flow_name=pending_flow.name,
                            start_url=branch_url,
                            goal=pending_flow.description,
                            parent_flow_id=task.flow_id
                        )
                        flow_created = new_flow_task is not None

                    if flow_created:
                        events.append({
                            "type": "flow_created",
                            "worker_id": self.worker_id,
                            "data": {
                                "flow_id": new_flow_task.flow_id,
                                "flow_name": pending_flow.name,
                                "description": pending_flow.description,
                                "keep_state": pending_flow.keep_state,
                                "created_by": self.worker_id,
                                "parent_flow": task.flow_name,
                                "parent_flow_id": task.flow_id,
                                "parent_action_index": action_count
                            }
                        })
                    else:
                        events.append({
                            "type": "flow_skipped",
                            "worker_id": self.worker_id,
                            "data": {
                                "flow_name": pending_flow.name,
                                "reason": "Flow with this name already exists"
                            }
                        })

            # Process reported issues
            issue_screenshot_path = None
            for reported_issue in exec_result.reported_issues:
                if issue_screenshot_path is None:
                    issue_screenshot = await browser.screenshot()
                    idx = await self.state.get_next_issue_screenshot_index()
                    if self.state.chat_logger:
                        issue_screenshot_path = self.state.chat_logger.save_issue_screenshot(issue_screenshot, idx) or ""
                    else:
                        issue_screenshot_path = ""
                        logger.debug("chat_logger unavailable, issue screenshot discarded")

                issue = Issue(
                    description=reported_issue.description,
                    severity=reported_issue.severity,
                    # Attribute the issue to the page the AI was looking at
                    # (popup when one is open), not the opener page
                    url=browser.current_url,
                    action_context=f"{action.action_type}: {action.reasoning[:100]}",
                    screenshot_path=issue_screenshot_path
                )
                is_new = await self.state.add_issue(issue)

                events.append({
                    "type": "issue",
                    "worker_id": self.worker_id,
                    "flow_id": task.flow_id,
                    "data": {
                        "description": issue.description,
                        "severity": issue.severity,
                        "url": issue.url,
                        "is_new": is_new,
                        "screenshot_path": issue_screenshot_path
                    }
                })

            # Auto-report console errors
            console_errors = browser.get_console_errors()
            console_screenshot_path = None
            for error in console_errors:
                if any(keyword in error.text.lower() for keyword in ["uncaught", "exception", "typeerror", "referenceerror"]):
                    if console_screenshot_path is None:
                        console_screenshot = await browser.screenshot()
                        idx = await self.state.get_next_issue_screenshot_index()
                        if self.state.chat_logger:
                            console_screenshot_path = self.state.chat_logger.save_issue_screenshot(console_screenshot, idx) or ""
                        else:
                            console_screenshot_path = ""
                            logger.debug("chat_logger unavailable, console error screenshot discarded")

                    issue = Issue(
                        description=f"JavaScript error: {error.text[:200]}",
                        severity="major",
                        url=error.url or browser.current_url,
                        action_context=f"Console error at {error.location}" if error.location else "Console error",
                        screenshot_path=console_screenshot_path
                    )
                    is_new = await self.state.add_issue(issue)
                    if is_new:
                        events.append({
                            "type": "issue",
                            "worker_id": self.worker_id,
                            "flow_id": task.flow_id,
                            "data": {
                                "description": issue.description,
                                "severity": issue.severity,
                                "url": issue.url,
                                "source": "console",
                                "is_new": True,
                                "screenshot_path": console_screenshot_path
                            }
                        })
                        await self.state.record_console_error()
            browser.clear_console_messages()

            # Auto-report failed network requests
            failed_requests = browser.get_failed_requests(clear=True)
            network_screenshot_path = None

            for req in failed_requests:
                # NOTE: must not be named `result` — that would shadow the
                # function's return dict and crash at `result["thinking"]`.
                classification = classify_network_failure(
                    req_url=req.url,
                    req_method=req.method,
                    req_status=req.status,
                    req_resource_type=req.resource_type,
                    req_failure_reason=req.failure_reason,
                    page_url=browser.current_url,
                )
                if classification is None:
                    continue
                severity, desc = classification

                capped = self._cap_network_issue(
                    req.status, urlparse(req.url).netloc, severity, desc
                )
                if capped is None:
                    continue
                severity, desc = capped

                if network_screenshot_path is None:
                    network_screenshot = await browser.screenshot()
                    idx = await self.state.get_next_issue_screenshot_index()
                    if self.state.chat_logger:
                        network_screenshot_path = self.state.chat_logger.save_issue_screenshot(network_screenshot, idx) or ""
                    else:
                        network_screenshot_path = ""
                        logger.debug("chat_logger unavailable, network error screenshot discarded")

                issue = Issue(
                    description=desc,
                    severity=severity,
                    url=browser.current_url,
                    action_context=f"Network request to {req.url[:50]}",
                    screenshot_path=network_screenshot_path
                )
                is_new = await self.state.add_issue(issue)
                if is_new:
                    events.append({
                        "type": "issue",
                        "worker_id": self.worker_id,
                        "flow_id": task.flow_id,
                        "data": {
                            "description": issue.description,
                            "severity": issue.severity,
                            "url": issue.url,
                            "source": "network",
                            "is_new": True,
                            "screenshot_path": network_screenshot_path
                        }
                    })
                    await self.state.record_network_failure()

            # Clear all accumulated network requests to free memory
            browser.clear_network_requests()

            # Detect HTTP Basic Auth challenges (401 responses)
            http_auth_challenges = browser.get_http_auth_challenges(clear=True)
            for challenge in http_auth_challenges:
                events.append({
                    "type": "http_auth_required",
                    "worker_id": self.worker_id,
                    "flow_id": task.flow_id,
                    "data": {
                        "url": challenge.url,
                        "auth_type": challenge.auth_type,
                        "realm": challenge.realm,
                        "timestamp": challenge.timestamp
                    }
                })

            # Report popup windows
            popups = browser.get_popups(clear=True)
            for popup in popups:
                events.append({
                    "type": "popup_opened",
                    "worker_id": self.worker_id,
                    "flow_id": task.flow_id,
                    "data": {
                        "url": popup.url,
                        "opener_url": popup.opener_url,
                        "timestamp": popup.timestamp,
                        "is_active": browser.has_active_popup
                    }
                })

            # Report dialogs
            dialogs = browser.get_dialogs(clear=True)
            dialog_screenshot_path = None
            for dialog in dialogs:
                events.append({
                    "type": "dialog",
                    "worker_id": self.worker_id,
                    "flow_id": task.flow_id,
                    "data": {
                        "dialog_type": dialog.dialog_type,
                        "message": dialog.message,
                        "was_accepted": dialog.was_accepted,
                        "url": dialog.url
                    }
                })
                await self.state.record_dialog()

                if dialog.dialog_type == "alert" and any(kw in dialog.message.lower() for kw in ["error", "failed", "invalid"]):
                    if dialog_screenshot_path is None:
                        dialog_screenshot = await browser.screenshot()
                        idx = await self.state.get_next_issue_screenshot_index()
                        if self.state.chat_logger:
                            dialog_screenshot_path = self.state.chat_logger.save_issue_screenshot(dialog_screenshot, idx) or ""
                        else:
                            dialog_screenshot_path = ""
                            logger.debug("chat_logger unavailable, dialog screenshot discarded")

                    issue = Issue(
                        description=f"Alert dialog with error: {dialog.message[:100]}",
                        severity="minor",
                        url=dialog.url or browser.current_url,
                        action_context="Dialog appeared during testing",
                        screenshot_path=dialog_screenshot_path
                    )
                    is_new = await self.state.add_issue(issue)
                    if is_new:
                        events.append({
                            "type": "issue",
                            "worker_id": self.worker_id,
                            "flow_id": task.flow_id,
                            "data": {
                                "description": issue.description,
                                "severity": issue.severity,
                                "url": issue.url,
                                "source": "dialog",
                                "is_new": True,
                                "screenshot_path": dialog_screenshot_path
                            }
                        })

            # Increment action counter and emit event
            await self.state.increment_actions()
            events.append({
                "type": "action",
                "worker_id": self.worker_id,
                "flow_id": task.flow_id,
                "data": action_dict
            })

            result["thinking"] = thinking
            return result

        return None

    async def _execute_action(
        self,
        action: AgentAction,
        page,
        browser: BrowserController,
        context,
        conversation_history: list[dict],
        task: FlowTask,
    ) -> ActionResult:
        """
        Execute a single action on the page.

        Action types match Claude for Chrome MCP tools exactly.
        Returns ActionResult with success status and any control flow signals.
        """
        try:
            # Click actions
            if action.action_type == "left_click":
                if action.ref is None:
                    return ActionResult(
                        success=False,
                        message="left_click requires ref",
                        error="No ref provided"
                    )
                await browser.click_ref(action.ref)
                await _settle_after_interaction(browser)
                return ActionResult(success=True, message=f"Clicked {action.ref}")

            elif action.action_type == "right_click":
                if action.ref is None:
                    return ActionResult(
                        success=False,
                        message="right_click requires ref",
                        error="No ref provided"
                    )
                await browser.right_click_ref(action.ref)
                await asyncio.sleep(0.3)
                return ActionResult(success=True, message=f"Right-clicked {action.ref}")

            elif action.action_type == "double_click":
                if action.ref is None:
                    return ActionResult(
                        success=False,
                        message="double_click requires ref",
                        error="No ref provided"
                    )
                await browser.double_click_ref(action.ref)
                await asyncio.sleep(0.5)
                return ActionResult(success=True, message=f"Double-clicked {action.ref}")

            elif action.action_type == "triple_click":
                if action.ref is None:
                    return ActionResult(
                        success=False,
                        message="triple_click requires ref",
                        error="No ref provided"
                    )
                await browser.triple_click_ref(action.ref)
                await asyncio.sleep(0.3)
                return ActionResult(success=True, message=f"Triple-clicked {action.ref}")

            elif action.action_type == "hover":
                if action.ref is None:
                    return ActionResult(
                        success=False,
                        message="hover requires ref",
                        error="No ref provided"
                    )
                await browser.hover_ref(action.ref)
                await asyncio.sleep(0.3)
                return ActionResult(success=True, message=f"Hovered over {action.ref}")

            elif action.action_type == "type":
                if action.ref is None or action.text is None:
                    return ActionResult(
                        success=False,
                        message="type requires ref and text",
                        error="Missing ref or text"
                    )
                await browser.fill_ref(action.ref, action.text)
                return ActionResult(success=True, message=f"Typed into {action.ref}")

            elif action.action_type == "form_input":
                if action.ref is None or action.value is None:
                    return ActionResult(
                        success=False,
                        message="form_input requires ref and value",
                        error="Missing ref or value"
                    )
                await browser.set_form_value(action.ref, action.value)
                return ActionResult(success=True, message=f"Set form value on {action.ref}")

            elif action.action_type == "scroll":
                direction = action.scroll_direction or "down"
                # Clamp scroll_amount to 1-10 range, default 3, convert ticks to pixels
                scroll_ticks = max(1, min(action.scroll_amount or 3, 10))
                amount = scroll_ticks * 100
                delta_y = amount if direction == "down" else -amount if direction == "up" else 0
                delta_x = amount if direction == "right" else -amount if direction == "left" else 0
                # Route through active_page: the AI is looking at the popup
                # screenshot when one is open, so scrolling must target it too
                await browser.active_page.mouse.wheel(delta_x, delta_y)
                return ActionResult(success=True, message=f"Scrolled {direction}")

            elif action.action_type == "scroll_to":
                if action.ref is None:
                    return ActionResult(
                        success=False,
                        message="scroll_to requires ref",
                        error="No ref provided"
                    )
                await browser.scroll_to_ref(action.ref)
                return ActionResult(success=True, message=f"Scrolled to {action.ref}")

            elif action.action_type == "key":
                key = action.key or "Enter"
                if action.modifiers:
                    modifier_map = {
                        "ctrl": "Control",
                        "control": "Control",
                        "shift": "Shift",
                        "alt": "Alt",
                        "cmd": "Meta",
                        "meta": "Meta",
                    }
                    modifiers = [modifier_map.get(m.lower(), m) for m in action.modifiers]
                    key_combo = "+".join(modifiers + [key])
                    # Key presses must go to the page the AI is looking at
                    # (the popup when one is open), e.g. Enter to submit a form
                    await browser.active_page.keyboard.press(key_combo)
                    if key == "Enter":
                        # Ctrl/Cmd+Enter submits forms in plenty of apps →
                        # may trigger a full navigation, same as bare Enter
                        await _settle_after_interaction(browser, sleep_s=0.2)
                    return ActionResult(success=True, message=f"Pressed {key_combo}")
                else:
                    await browser.active_page.keyboard.press(key)
                    if key == "Enter":
                        # Enter commonly submits a form → full navigation
                        await _settle_after_interaction(browser, sleep_s=0.2)
                    return ActionResult(success=True, message=f"Pressed {key}")

            elif action.action_type == "left_click_drag":
                if action.start_coordinate is None or action.coordinate is None:
                    return ActionResult(
                        success=False,
                        message="left_click_drag requires start_coordinate and coordinate",
                        error="Missing coordinates"
                    )
                await browser.drag(action.start_coordinate, action.coordinate)
                return ActionResult(
                    success=True,
                    message=f"Dragged from {action.start_coordinate} to {action.coordinate}"
                )

            elif action.action_type == "screenshot":
                # Take a screenshot - could be stored/returned for explicit capture
                full_page = action.full_page or False
                screenshot_bytes = await browser.screenshot(full_page=full_page)
                return ActionResult(
                    success=True,
                    message=f"Took screenshot (full_page={full_page})",
                    screenshot=screenshot_bytes
                )

            elif action.action_type == "zoom":
                # Zoom is primarily for inspection - we just acknowledge it
                # The actual cropping would be done if we stored the screenshot
                if action.region is None:
                    return ActionResult(
                        success=False,
                        message="zoom requires region [x0, y0, x1, y1]",
                        error="Missing region"
                    )
                return ActionResult(
                    success=True,
                    message=f"Zoom region captured: {action.region}"
                )

            elif action.action_type == "navigate":
                url = action.url
                if not url:
                    return ActionResult(
                        success=False,
                        message="Navigate action requires URL",
                        error="No URL provided for navigate action"
                    )
                # Navigate the page the AI is currently looking at (the popup
                # when one is open, otherwise the main page)
                target_page = browser.active_page
                # Handle special navigation commands (Chrome MCP style)
                if url.lower() == "back":
                    await target_page.go_back()
                    return ActionResult(success=True, message="Navigated back")
                elif url.lower() == "forward":
                    await target_page.go_forward()
                    return ActionResult(success=True, message="Navigated forward")
                # Validate URL scheme - block dangerous schemes
                parsed = urlparse(url)
                if parsed.scheme.lower() not in ("http", "https", ""):
                    return ActionResult(
                        success=False,
                        message=f"Navigation blocked: unsafe URL scheme '{parsed.scheme}'",
                        error=f"Only http/https URLs are allowed, got: {parsed.scheme}"
                    )
                # Add https if no scheme provided
                if not parsed.scheme:
                    url = f"https://{url}"

                try:
                    await _safe_goto(target_page, url)
                except (TimeoutError, PlaywrightTimeoutError):
                    # Navigation never committed (hanging server, or a
                    # beforeunload guard cancelled it). Tell the AI where it
                    # actually is so its world model stays correct.
                    return ActionResult(
                        success=False,
                        message=f"No server response for {url} within 30s",
                        error=(
                            f"No server response for {url}; still on "
                            f"{target_page.url}. Retry once before reporting."
                        ),
                    )
                except Exception as e:
                    # A URL that serves a file aborts the navigation: Chromium
                    # raises "Download is starting" (net::ERR_ABORTED on other
                    # engines/versions). Report it as a download, not as a raw
                    # low-level navigation error.
                    msg = str(e)
                    if "Download is starting" in msg or "net::ERR_ABORTED" in msg:
                        # Give the download event a moment to be captured.
                        # Peek without clearing — _handle_ai_event drains the
                        # downloads afterwards and emits the download event.
                        await asyncio.sleep(0.5)
                        downloads = browser.get_downloads()
                        if downloads:
                            names = ", ".join(
                                d.suggested_filename or d.url for d in downloads
                            )
                            return ActionResult(
                                success=True,
                                message=f"URL triggered a file download ({names}) "
                                        f"instead of a page navigation",
                            )
                        if "Download is starting" in msg:
                            return ActionResult(
                                success=True,
                                message="URL triggered a file download instead "
                                        "of a page navigation",
                            )
                    raise
                return ActionResult(success=True, message=f"Navigated to {url}")

            elif action.action_type == "wait":
                # Use duration if provided (Chrome MCP style), default to 2 seconds, max 30
                duration = min(action.duration if action.duration is not None else 2, 30)
                await asyncio.sleep(duration)
                return ActionResult(success=True, message=f"Waited {duration} seconds")

            elif action.action_type == "resize":
                width = action.width or 1280
                height = action.height or 720
                # Resize the page the AI is looking at (screenshots and
                # viewport size already come from active_page)
                await browser.active_page.set_viewport_size({"width": width, "height": height})
                return ActionResult(success=True, message=f"Resized to {width}x{height}")

            elif action.action_type == "report_issue":
                if not action.issue_description:
                    return ActionResult(
                        success=False,
                        message="Report issue requires description",
                        error="No description provided"
                    )
                severity = action.severity or "minor"
                return ActionResult(
                    success=True,
                    message=f"Reported issue: {action.issue_description}",
                    reported_issues=[ReportedIssue(
                        description=action.issue_description,
                        severity=severity
                    )]
                )

            elif action.action_type == "add_flow":
                if not action.flow_name:
                    return ActionResult(
                        success=False,
                        message="Add flow requires name",
                        error="No flow name provided"
                    )
                return ActionResult(
                    success=True,
                    message=f"Added flow: {action.flow_name}",
                    pending_flows=[PendingFlow(
                        name=action.flow_name,
                        description=action.flow_description or action.flow_name,
                        keep_state=action.keep_state
                    )]
                )

            elif action.action_type == "done":
                reason = action.reason or "Flow completed"
                return ActionResult(
                    success=True,
                    message=f"Flow done: {reason}",
                    flow_done=True,
                    done_reason=reason
                )

            elif action.action_type == "block":
                reason = action.reason or "Need assistance"
                return ActionResult(
                    success=True,
                    message=f"Flow blocked: {reason}",
                    flow_blocked=True,
                    block_reason=reason
                )

            elif action.action_type == "request_data":
                if not action.request_name or not action.request_fields:
                    return ActionResult(
                        success=False,
                        message="request_data requires request_name and request_fields",
                        error="Missing request_name or request_fields"
                    )
                # Create UserDataRequest from action fields
                fields = []
                for field_dict in action.request_fields:
                    fields.append(UserDataField(
                        key=field_dict.get("key", ""),
                        label=field_dict.get("label", field_dict.get("key", "")),
                        field_type=field_dict.get("type", "text"),
                        placeholder=field_dict.get("placeholder", ""),
                        required=field_dict.get("required", True),
                        description=field_dict.get("description", ""),
                    ))
                data_request = UserDataRequest(
                    request_id=str(uuid.uuid4()),
                    request_name=action.request_name,
                    description=action.request_description or "Data needed to continue",
                    fields=fields,
                    flow_id=task.flow_id,
                    worker_id=self.worker_id,
                    requested_at=datetime.now().isoformat(),
                )
                return ActionResult(
                    success=True,
                    message=f"Requesting data: {action.request_name}",
                    data_request=data_request
                )

            elif action.action_type == "close_popup":
                closed = await browser.close_active_popup()
                if closed:
                    return ActionResult(success=True, message="Closed active popup window")
                else:
                    return ActionResult(
                        success=False,
                        message="No active popup to close",
                        error="No popup window is currently active"
                    )

            elif action.action_type == "set_http_auth":
                username_key = action.username_key
                password_key = action.password_key
                if not username_key or not password_key:
                    return ActionResult(
                        success=False,
                        message="set_http_auth requires both username_key and password_key",
                        error="Missing username_key or password_key"
                    )
                credentials = await self.state.get_credentials()
                username = credentials.get(username_key)
                password = credentials.get(password_key)
                if username is None or password is None:
                    available = list(credentials.keys()) if credentials else []
                    return ActionResult(
                        success=False,
                        message=f"Credential keys not found. Available keys: {available}",
                        error=f"Key {username_key!r} or {password_key!r} not in available credentials"
                    )
                # Store HTTP auth for this and future flows
                target_url = _target_url_for_auth(page.url, task.start_url)
                await self.state.set_http_auth(username, password, target_url)
                logger.info(f"HTTP auth set via AI action: username_key={username_key!r}, password_key={password_key!r}")
                return ActionResult(
                    success=True,
                    message="HTTP Basic Auth credentials applied. The page will reload with authentication.",
                    needs_context_rebuild=True,
                )

            else:
                return ActionResult(
                    success=False,
                    message=f"Unknown action type: {action.action_type}",
                    error=f"Unsupported action type: {action.action_type}"
                )

        except DuplicateRefError as e:
            return ActionResult(
                success=False,
                message=f"Multiple elements match {e.ref}",
                error=str(e)
            )

        except Exception as e:
            return ActionResult(
                success=False,
                message=f"Action failed: {action.action_type}",
                error=str(e)
            )
