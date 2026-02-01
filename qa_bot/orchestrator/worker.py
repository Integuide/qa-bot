"""Flow-based exploration worker - one worker per flow."""

import asyncio
import base64
import logging
import uuid
from typing import AsyncGenerator
from datetime import datetime
from dataclasses import dataclass
from urllib.parse import urlparse

from qa_bot.ai.base import AIProvider, AgentAction
from qa_bot.agent.state import Issue
from qa_bot.browser.controller import BrowserController
from qa_bot.orchestrator.browser_pool import BrowserPool
from qa_bot.orchestrator.shared_state import SharedFlowState, UserDataField, UserDataRequest, format_user_data_for_prompt
from qa_bot.orchestrator.flow import FlowTask, FlowCheckpoint, FlowStatus

logger = logging.getLogger(__name__)


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


def screenshot_to_base64(screenshot_bytes: bytes) -> str:
    """Convert screenshot bytes to base64 string for SSE transport."""
    return base64.b64encode(screenshot_bytes).decode('utf-8')


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
            await self.state.fail_flow(self.flow_task.flow_id, str(e))
            yield {
                "type": "flow_failed",
                "data": {
                    "worker_id": self.worker_id,
                    "flow_id": self.flow_task.flow_id,
                    "flow_name": self.flow_task.flow_name,
                    "error": str(e)
                }
            }

    async def _execute_flow(self) -> AsyncGenerator[dict, None]:
        """Execute the flow exploration."""
        task = self.flow_task

        flow_data = await self.state.start_flow_exploration(task, self.worker_id)

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
        conversation_history = []
        parent_flow_name = ""

        try:
            if task.is_root or task.checkpoint_id is None:
                # Root flow or fresh flow: create fresh context
                # Flows created via add_flow_task have checkpoint_id=None and start fresh
                context = await self.browser_pool.create_isolated_context()
                page = await context.new_page()

                try:
                    await page.goto(
                        task.start_url,
                        wait_until="domcontentloaded",
                        timeout=30000
                    )
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
                    storage_state=checkpoint.browser_storage_state
                )
                page = await context.new_page()

                try:
                    await page.goto(
                        checkpoint.current_url,
                        wait_until="domcontentloaded",
                        timeout=30000
                    )
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

            # Main exploration loop - runs until AI returns done/block action, or global limits hit
            while not flow_completed and not self.state.should_stop():
                # Check if we're blocked
                if self._blocked:
                    response = await self.state.wait_for_unblock(self.worker_id, timeout=300)
                    if response is None:
                        yield {
                            "type": "block_timeout",
                            "data": {"worker_id": self.worker_id, "flow_id": task.flow_id}
                        }
                        break
                    elif response.get("action") == "stop":
                        break
                    elif response.get("action") == "checkpoint_and_exit":
                        # Save checkpoint and exit - flow will resume when credentials arrive
                        reason = response.get("reason", "Waiting for credentials")
                        storage_state = await context.storage_state()
                        checkpoint = FlowCheckpoint(
                            parent_flow_id=task.flow_id,
                            browser_storage_state=storage_state,
                            current_url=page.url,
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
                        checkpoint = FlowCheckpoint(
                            parent_flow_id=task.flow_id,
                            browser_storage_state=storage_state,
                            current_url=page.url,
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
                        checkpoint = FlowCheckpoint(
                            parent_flow_id=task.flow_id,
                            browser_storage_state=storage_state,
                            current_url=page.url,
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

                # Take screenshot
                screenshot = await browser.screenshot()

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
                        "url": page.url
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
                        current_url=page.url,
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
                    yield {
                        "type": "ai_error",
                        "worker_id": self.worker_id,
                        "flow_id": task.flow_id,
                        "data": {"error": str(e)}
                    }
                    break

            # Mark done if not already (loop exited due to global stop condition)
            if not flow_completed and flow_data.status == FlowStatus.EXPLORING:
                completion_reason = self.state.get_stop_reason()
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
                        storage_state = await context.storage_state()
                        checkpoint = FlowCheckpoint(
                            parent_flow_id=task.flow_id,
                            browser_storage_state=storage_state,
                            current_url=page.url,
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
            if context:
                try:
                    await context.close()
                except Exception:
                    pass  # Ignore errors when closing context

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

            if self.state.chat_logger:
                self.state.chat_logger.log_worker_ai_error(
                    worker_id=self.worker_id,
                    flow_id=task.flow_id,
                    turn_number=action_count,
                    current_url=page.url,
                    error=error_msg,
                    raw_response=ai_event.get("raw_response"),
                    thinking=ai_event.get("thinking")
                )

            await self.state.fail_flow(task.flow_id, f"AI error: {error_msg}")

            events.append({
                "type": "ai_error",
                "worker_id": self.worker_id,
                "flow_id": task.flow_id,
                "data": {"error": error_msg}
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
                    current_url=page.url,
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

            # Execute the action
            exec_result = await self._execute_action(action, page, browser, context, conversation_history, task)

            # Build action record for history
            action_dict = {
                "action_type": action.action_type,
                "reasoning": action.reasoning,
                "page_url": page.url,  # URL where action was executed
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
                    if pending_flow.keep_state:
                        storage_state = await context.storage_state()
                        checkpoint = FlowCheckpoint(
                            parent_flow_id=task.flow_id,
                            browser_storage_state=storage_state,
                            current_url=page.url,
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
                            start_url=page.url,
                            goal=pending_flow.description,
                            parent_flow_id=task.flow_id
                        )
                        flow_created = new_flow_task is not None

                    if flow_created:
                        events.append({
                            "type": "flow_created",
                            "worker_id": self.worker_id,
                            "data": {
                                "flow_name": pending_flow.name,
                                "description": pending_flow.description,
                                "keep_state": pending_flow.keep_state,
                                "created_by": self.worker_id,
                                "parent_flow": task.flow_name
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
            issue_screenshot_b64 = None
            for reported_issue in exec_result.reported_issues:
                if issue_screenshot_b64 is None:
                    issue_screenshot = await browser.screenshot()
                    issue_screenshot_b64 = screenshot_to_base64(issue_screenshot)

                issue = Issue(
                    description=reported_issue.description,
                    severity=reported_issue.severity,
                    url=page.url,
                    action_context=f"{action.action_type}: {action.reasoning[:100]}",
                    screenshot_base64=issue_screenshot_b64
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
                        "screenshot": issue_screenshot_b64
                    }
                })

            # Auto-report console errors
            console_errors = browser.get_console_errors()
            console_screenshot_b64 = None
            for error in console_errors:
                if any(keyword in error.text.lower() for keyword in ["uncaught", "exception", "typeerror", "referenceerror"]):
                    if console_screenshot_b64 is None:
                        console_screenshot = await browser.screenshot()
                        console_screenshot_b64 = screenshot_to_base64(console_screenshot)

                    issue = Issue(
                        description=f"JavaScript error: {error.text[:200]}",
                        severity="major",
                        url=error.url or page.url,
                        action_context=f"Console error at {error.location}" if error.location else "Console error",
                        screenshot_base64=console_screenshot_b64
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
                                "screenshot": console_screenshot_b64
                            }
                        })
                        await self.state.record_console_error()
            browser.clear_console_messages()

            # Auto-report failed network requests
            failed_requests = browser.get_failed_requests(clear=True)
            network_screenshot_b64 = None
            for req in failed_requests:
                if req.status >= 500 or req.status == 0:
                    severity = "major"
                    desc = f"Server error {req.status}: {req.method} {req.url[:100]}" if req.status else f"Network failure: {req.method} {req.url[:100]} - {req.failure_reason}"
                elif req.status == 404:
                    severity = "minor"
                    desc = f"404 Not Found: {req.method} {req.url[:100]}"
                else:
                    severity = "minor"
                    desc = f"HTTP {req.status}: {req.method} {req.url[:100]}"

                if network_screenshot_b64 is None:
                    network_screenshot = await browser.screenshot()
                    network_screenshot_b64 = screenshot_to_base64(network_screenshot)

                issue = Issue(
                    description=desc,
                    severity=severity,
                    url=page.url,
                    action_context=f"Network request to {req.url[:50]}",
                    screenshot_base64=network_screenshot_b64
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
                            "screenshot": network_screenshot_b64
                        }
                    })
                    await self.state.record_network_failure()

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
            dialog_screenshot_b64 = None
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
                    if dialog_screenshot_b64 is None:
                        dialog_screenshot = await browser.screenshot()
                        dialog_screenshot_b64 = screenshot_to_base64(dialog_screenshot)

                    issue = Issue(
                        description=f"Alert dialog with error: {dialog.message[:100]}",
                        severity="minor",
                        url=dialog.url or page.url,
                        action_context="Dialog appeared during testing",
                        screenshot_base64=dialog_screenshot_b64
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
                                "screenshot": dialog_screenshot_b64
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
                await asyncio.sleep(0.5)
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
                await page.mouse.wheel(delta_x, delta_y)
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
                    await page.keyboard.press(key_combo)
                    return ActionResult(success=True, message=f"Pressed {key_combo}")
                else:
                    await page.keyboard.press(key)
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
                # Handle special navigation commands (Chrome MCP style)
                if url.lower() == "back":
                    await page.go_back()
                    return ActionResult(success=True, message="Navigated back")
                elif url.lower() == "forward":
                    await page.go_forward()
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
                await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                return ActionResult(success=True, message=f"Navigated to {url}")

            elif action.action_type == "wait":
                # Use duration if provided (Chrome MCP style), default to 2 seconds, max 30
                duration = min(action.duration if action.duration is not None else 2, 30)
                await asyncio.sleep(duration)
                return ActionResult(success=True, message=f"Waited {duration} seconds")

            elif action.action_type == "resize":
                width = action.width or 1280
                height = action.height or 720
                await page.set_viewport_size({"width": width, "height": height})
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

            else:
                return ActionResult(
                    success=False,
                    message=f"Unknown action type: {action.action_type}",
                    error=f"Unsupported action type: {action.action_type}"
                )

        except Exception as e:
            return ActionResult(
                success=False,
                message=f"Action failed: {action.action_type}",
                error=str(e)
            )
