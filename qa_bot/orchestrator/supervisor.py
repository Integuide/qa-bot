"""Supervisor agent that monitors and coordinates workers."""

import asyncio
import uuid
from typing import AsyncGenerator
from datetime import datetime

from qa_bot.ai.base import AIProvider
from qa_bot.orchestrator.shared_state import SharedFlowState, format_user_data_for_prompt, UserDataRequest, UserDataField
from qa_bot.orchestrator.flow import FlowStatus


class SupervisorAgent:
    """
    Supervisor that monitors exploration and manages workers.

    Runs event-driven (triggered on flow status changes):
    - Flow created
    - Flow completed
    - Worker blocked
    - Worker finished

    Actions available:
    - MESSAGE(worker_id, message) - Send instructions
    - STOP(worker_id, reason) - Stop a worker
    - ASK_USER(question) - Ask user for clarification via data request
    - UNBLOCK(worker_id, message) - Respond to blocked worker
    - SKIP_FLOW(flow_id, reason) - Skip a pending flow
    - OBSERVE - No action needed
    """

    def __init__(
        self,
        ai_provider: AIProvider,
        shared_state: SharedFlowState,
    ):
        self.ai = ai_provider
        self.state = shared_state
        self._running = False
        # Track which data request names have had events emitted to UI
        # to prevent duplicate data_request events when multiple workers
        # request the same data before it's provided
        self._emitted_data_requests: set[str] = set()

    async def run(self) -> AsyncGenerator[dict, None]:
        """
        Main supervisor loop.

        Waits for triggers and takes action when needed.
        Handles pause/resume by waiting instead of exiting when limits are reached.
        """
        self._running = True

        yield {
            "type": "supervisor_started",
            "data": {"timestamp": datetime.now().isoformat()}
        }

        while self._running:
            # Check if we should stop (user requested stop)
            if self.state.should_stop() and not self.state.is_paused():
                # If paused, we'll wait for resume below
                # If not paused but should_stop, check if it's a user-requested stop
                # or just a limit that might be resumed
                if self.state.stop_requested:
                    break

            # If paused (cost/time limit reached), wait for resume
            if self.state.is_paused():
                resumed = await self.state.wait_for_resume(timeout=5.0)
                if resumed:
                    # Clear the emitted data requests set since we're starting fresh
                    self._emitted_data_requests.clear()
                    yield {
                        "type": "supervisor_started",
                        "data": {"timestamp": datetime.now().isoformat(), "resumed": True}
                    }
                continue

            # Wait for trigger or timeout
            triggered = await self.state.wait_for_supervisor_trigger(timeout=5.0)

            # Check if there's work to do
            blocked_workers = await self.state.get_blocked_workers()
            pending_flows = await self.state.get_pending_flows()

            # Prioritize blocked workers - they're waiting
            if blocked_workers:
                async for event in self._handle_blocked_workers(blocked_workers):
                    yield event

            # Review pending flows periodically for duplicates
            elif triggered and len(pending_flows) > 3:
                async for event in self._review_pending_flows(pending_flows):
                    yield event

            # Check for completion
            if self.state.is_complete():
                break

        self._running = False

        yield {
            "type": "supervisor_stopped",
            "data": {"timestamp": datetime.now().isoformat()}
        }

    def _is_credential_block(self, reason: str) -> bool:
        """Check if the block reason is related to needing credentials (legacy).

        NOTE: This is kept for backwards compatibility with the block action.
        The preferred approach is to use request_data action which creates
        explicit data requests that don't require keyword guessing.
        """
        reason_lower = reason.lower()
        credential_keywords = [
            'credential', 'login', 'password', 'username', 'auth',
            'sign in', 'signin', 'log in', 'authentication',
            'payment', 'card', 'checkout',
            # HTTP Basic/Digest Auth keywords
            'http auth', 'basic auth', 'digest auth', '401',
            'http_username', 'http_password', 'www-authenticate'
        ]
        return any(keyword in reason_lower for keyword in credential_keywords)

    def _is_data_request_block(self, reason: str) -> bool:
        """Check if the block reason is from a request_data action."""
        return reason.lower().startswith("requesting data:")

    def _is_approval_block(self, reason: str) -> bool:
        """Check if the block reason is requesting user approval for an irreversible action."""
        import re
        reason_lower = reason.lower()
        # Explicit approval request prefix
        if 'approval_needed' in reason_lower:
            return True
        # Keywords suggesting irreversible actions - use word boundaries to avoid
        # false positives (e.g., "unsubscribed" should not match "subscribe")
        approval_keywords = [
            'confirm', 'delete', 'remove', 'publish', 'send email',
            'purchase', 'pay', 'subscribe', 'cancel', 'unsubscribe',
            'upgrade', 'downgrade'
        ]
        return any(re.search(rf'\b{keyword}\b', reason_lower) for keyword in approval_keywords)

    def _format_credentials_for_worker(self, credentials: dict[str, str]) -> str:
        """Format credentials for inclusion in worker message."""
        lines = []
        for key, value in credentials.items():
            lines.append(f"  {key}: {value}")
        return "Available credentials:\n" + "\n".join(lines)

    async def _handle_blocked_workers(
        self,
        blocked_workers: list[dict]
    ) -> AsyncGenerator[dict, None]:
        """Handle blocked workers by calling AI for guidance."""

        for blocked in blocked_workers:
            worker_id = blocked["worker_id"]
            reason = blocked.get("reason", "Unknown")
            flow_id = blocked.get("flow_id", "")

            yield {
                "type": "supervisor_reviewing_block",
                "data": {
                    "worker_id": worker_id,
                    "reason": reason
                }
            }

            # Check if this is a data request block (from request_data action)
            if self._is_data_request_block(reason):
                # Get the pending data request for this worker
                data_request = await self.state.get_data_request_by_worker(worker_id)

                if data_request:
                    # Check if data is already available
                    existing_data = await self.state.get_user_data(data_request.request_name)
                    if existing_data:
                        # Data already available - unblock with data
                        # Note: Password fields are masked in logged messages
                        data_lines = [f"Data for '{data_request.request_name}' is available:"]
                        data_lines.extend(format_user_data_for_prompt(existing_data, data_request.fields))
                        await self.state.unblock_worker(worker_id, {
                            "action": "unblock",
                            "message": "\n".join(data_lines) + "\n\nUse this data to continue."
                        })
                        yield {
                            "type": "supervisor_action",
                            "data": {
                                "action": "unblock_with_data",
                                "target_worker": worker_id,
                                "request_name": data_request.request_name
                            }
                        }
                        continue
                    else:
                        # Need to wait for data - checkpoint and emit event if not already emitted
                        request_name = data_request.request_name

                        # Only emit data_request event if we haven't already for this request_name
                        # This prevents duplicate modals when multiple workers request the same data
                        if request_name not in self._emitted_data_requests:
                            self._emitted_data_requests.add(request_name)
                            yield {
                                "type": "data_request",
                                "data": data_request.to_dict()
                            }

                        await self.state.unblock_worker(worker_id, {
                            "action": "checkpoint_and_exit_for_data",
                            "request_name": request_name,
                            "reason": f"Waiting for data: {request_name}"
                        })

                        yield {
                            "type": "supervisor_action",
                            "data": {
                                "action": "checkpoint_and_exit_for_data",
                                "target_worker": worker_id,
                                "flow_id": flow_id,
                                "request_name": request_name,
                                "reason": "Flow blocked for data - checkpoint saved, worker released"
                            }
                        }
                        continue

            # Check if this is a credential-related block (legacy fallback)
            if self._is_credential_block(reason):
                credentials = await self.state.get_credentials()

                if credentials:
                    # We have credentials - provide them directly without AI call
                    creds_message = self._format_credentials_for_worker(credentials)
                    await self.state.unblock_worker(worker_id, {
                        "action": "unblock",
                        "message": f"Use these credentials to continue:\n{creds_message}"
                    })

                    yield {
                        "type": "supervisor_action",
                        "data": {
                            "action": "unblock_with_credentials",
                            "target_worker": worker_id,
                            "credential_keys": list(credentials.keys())
                        }
                    }
                    continue

                else:
                    # No credentials available - request from user
                    # Worker will checkpoint state and exit, allowing it to pick up other flows
                    # When credentials arrive, a new flow will be created to resume from checkpoint

                    # Determine what credentials are likely needed
                    needed = ["USERNAME", "PASSWORD"]  # Default
                    reason_lower = reason.lower()
                    if "payment" in reason_lower or "card" in reason_lower or "checkout" in reason_lower:
                        needed = ["CARD_NUMBER", "CARD_EXPIRY", "CARD_CVV"]

                    # Record the credential request (checkpoint_id will be set by worker)
                    await self.state.request_credentials(
                        worker_id=worker_id,
                        flow_id=flow_id,
                        needed=needed,
                        reason=reason
                    )

                    # Emit credential request event for UI
                    yield {
                        "type": "credential_request",
                        "data": {
                            "worker_id": worker_id,
                            "flow_id": flow_id,
                            "needed": needed,
                            "reason": reason
                        }
                    }

                    # Tell worker to save checkpoint and exit
                    # Worker will be released to pick up other pending flows
                    # This flow will resume when credentials are provided
                    await self.state.unblock_worker(worker_id, {
                        "action": "checkpoint_and_exit",
                        "reason": reason
                    })

                    yield {
                        "type": "supervisor_action",
                        "data": {
                            "action": "checkpoint_and_exit",
                            "target_worker": worker_id,
                            "flow_id": flow_id,
                            "needed": needed,
                            "reason": "Flow blocked for credentials - checkpoint saved, worker released"
                        }
                    }
                    continue

            # Check if this is an approval request for irreversible action
            if self._is_approval_block(reason):
                # Extract action description from the reason
                action_description = reason
                if 'approval_needed:' in reason.lower():
                    # Extract part after "APPROVAL_NEEDED:"
                    idx = reason.lower().find('approval_needed:')
                    action_description = reason[idx + len('approval_needed:'):].strip()

                # Request approval from user (may auto-approve if skip_permissions or always-approve pattern)
                result = await self.state.request_approval(
                    worker_id=worker_id,
                    flow_id=flow_id,
                    reason=reason,
                    action_description=action_description
                )

                if result.get("auto_approved"):
                    # Auto-approved - unblock worker immediately
                    await self.state.unblock_worker(worker_id, {
                        "action": "unblock",
                        "message": "Action auto-approved. Proceed with the action."
                    })
                    yield {
                        "type": "supervisor_action",
                        "data": {
                            "action": "auto_approved",
                            "target_worker": worker_id,
                            "flow_id": flow_id,
                            "reason": "Action auto-approved (skip_permissions or always-approve pattern)"
                        }
                    }
                    continue
                else:
                    # Need user approval - emit event for UI
                    yield {
                        "type": "approval_request",
                        "data": {
                            "worker_id": worker_id,
                            "flow_id": flow_id,
                            "action_description": action_description,
                            "reason": reason
                        }
                    }

                    # Tell worker to save checkpoint and exit
                    await self.state.unblock_worker(worker_id, {
                        "action": "checkpoint_and_exit_for_approval",
                        "reason": reason
                    })

                    yield {
                        "type": "supervisor_action",
                        "data": {
                            "action": "checkpoint_and_exit_for_approval",
                            "target_worker": worker_id,
                            "flow_id": flow_id,
                            "action_description": action_description,
                            "reason": "Flow blocked for approval - checkpoint saved, worker released"
                        }
                    }
                    continue

            # Get current state for AI (for non-credential/non-approval blocks)
            active_workers = await self._format_active_workers()
            all_pending = await self.state.get_pending_flows()
            all_flows = await self.state.get_all_flows()
            all_issues = await self.state.get_all_issues()

            pending_flows = [
                {"flow_id": f.flow_id[:8], "flow_name": f.flow_name}
                for f in all_pending[:10]
            ]
            completed_flows = [
                {"flow_id": f.flow_id[:8], "flow_name": f.flow_name}
                for f in all_flows
                if f.status == FlowStatus.COMPLETED
            ][:10]
            issues = [
                {"severity": i.severity, "description": i.description[:100]}
                for i in all_issues[:5]
            ]

            # Call AI
            try:
                response = await self.ai.analyze_for_supervisor(
                    active_workers=active_workers,
                    blocked_workers=[blocked],
                    pending_flows=pending_flows,
                    completed_flows=completed_flows,
                    issues=issues,
                )

                # Track token usage by type
                input_tokens = response.get("input_tokens", 0)
                output_tokens = response.get("output_tokens", 0)
                cache_read_tokens = response.get("cache_read_tokens", 0)
                cache_creation_tokens = response.get("cache_creation_tokens", 0)
                if input_tokens or output_tokens or cache_read_tokens or cache_creation_tokens:
                    await self.state.add_token_usage(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cache_read_tokens=cache_read_tokens,
                        cache_creation_tokens=cache_creation_tokens,
                    )

                # Log supervisor call
                if self.state.chat_logger:
                    context_summary = f"Blocked worker: {worker_id}\nReason: {reason}\nActive workers: {len(active_workers)}\nPending flows: {len(pending_flows)}"
                    self.state.chat_logger.log_supervisor_call(
                        context=context_summary,
                        response=response,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens
                    )

                action = response.get("action", "observe")

                yield {
                    "type": "supervisor_action",
                    "data": {
                        "action": action,
                        "target_worker": worker_id,
                        "response": response
                    }
                }

                # Execute the action
                if action == "unblock":
                    message = response.get("message", "Please continue")
                    await self.state.unblock_worker(worker_id, {
                        "action": "unblock",
                        "message": message
                    })

                elif action == "stop":
                    reason = response.get("reason", "Supervisor requested stop")
                    await self.state.stop_worker(worker_id, reason)

                elif action == "ask_user":
                    question = response.get("question", response.get("reason", ""))
                    request_id = str(uuid.uuid4())

                    # Create a data request for the question (use UUID to avoid collisions)
                    data_request = UserDataRequest(
                        request_id=request_id,
                        request_name=f"supervisor_question_{request_id[:8]}",
                        description=question,
                        fields=[UserDataField(
                            key="response",
                            label="Your response",
                            field_type="textarea",
                            required=True
                        )],
                        flow_id=flow_id,
                        worker_id=worker_id,
                        requested_at=datetime.now().isoformat()
                    )
                    await self.state.add_data_request(data_request)

                    # Emit supervisor_action so frontend can track the question count
                    yield {
                        "type": "supervisor_action",
                        "data": {
                            "action": "ask_user",
                            "question": question,
                            "target_worker": worker_id,
                            "flow_id": flow_id
                        }
                    }

                    yield {"type": "data_request", "data": data_request.to_dict()}

                    await self.state.unblock_worker(worker_id, {
                        "action": "checkpoint_and_exit_for_data",
                        "request_name": data_request.request_name,
                        "reason": f"Waiting for user guidance: {question}"
                    })

                elif action == "message":
                    # Just send a message without unblocking
                    message = response.get("message", "")
                    if message:
                        await self.state.send_message_to_worker(worker_id, message)

                else:  # observe
                    # Default: just unblock with generic message
                    await self.state.unblock_worker(worker_id, {
                        "action": "unblock",
                        "message": "Please continue as best you can"
                    })

            except Exception as e:
                yield {
                    "type": "supervisor_error",
                    "data": {
                        "error": str(e),
                        "worker_id": worker_id
                    }
                }

                # Unblock with error message
                await self.state.unblock_worker(worker_id, {
                    "action": "unblock",
                    "message": f"Supervisor error: {str(e)}. Please try an alternative approach."
                })

    async def _review_pending_flows(
        self,
        pending_flows: list
    ) -> AsyncGenerator[dict, None]:
        """Review pending flows for duplicates."""

        # Only review if we have multiple pending flows
        if len(pending_flows) < 3:
            return

        yield {
            "type": "supervisor_reviewing_flows",
            "data": {"count": len(pending_flows)}
        }

        # Get context
        active_workers = await self._format_active_workers()
        all_flows = await self.state.get_all_flows()
        all_issues = await self.state.get_all_issues()

        completed_flows = [
            {"flow_id": f.flow_id[:8], "flow_name": f.flow_name}
            for f in all_flows
            if f.status == FlowStatus.COMPLETED
        ][:10]
        issues = [
            {"severity": i.severity, "description": i.description[:100]}
            for i in all_issues[:5]
        ]

        pending_flow_dicts = [
            {"flow_id": f.flow_id, "flow_name": f.flow_name}
            for f in pending_flows[:15]
        ]

        try:
            response = await self.ai.analyze_for_supervisor(
                active_workers=active_workers,
                blocked_workers=[],
                pending_flows=pending_flow_dicts,
                completed_flows=completed_flows,
                issues=issues,
            )

            # Track token usage by type
            input_tokens = response.get("input_tokens", 0)
            output_tokens = response.get("output_tokens", 0)
            cache_read_tokens = response.get("cache_read_tokens", 0)
            cache_creation_tokens = response.get("cache_creation_tokens", 0)
            if input_tokens or output_tokens or cache_read_tokens or cache_creation_tokens:
                await self.state.add_token_usage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cache_read_tokens=cache_read_tokens,
                    cache_creation_tokens=cache_creation_tokens,
                )

            # Log supervisor call
            if self.state.chat_logger:
                pending_names = [f["flow_name"] for f in pending_flow_dicts[:5]]
                context_summary = f"Review pending flows\nPending: {pending_names}\nActive workers: {len(active_workers)}\nCompleted: {len(completed_flows)}"
                self.state.chat_logger.log_supervisor_call(
                    context=context_summary,
                    response=response,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens
                )

            action = response.get("action", "observe")

            if action == "skip_flow":
                flow_id = response.get("flow_id", "")
                reason = response.get("reason", "Duplicate")

                if flow_id:
                    # Get flow name before skipping
                    flow_data = await self.state.get_flow_data(flow_id)
                    flow_name = flow_data.flow_name if flow_data else flow_id

                    success = await self.state.mark_flow_skip(flow_id, reason)
                    if success:
                        yield {
                            "type": "flow_skipped",
                            "data": {
                                "flow_id": flow_id,
                                "flow_name": flow_name,
                                "reason": reason
                            }
                        }

            # Note: ask_user is not supported in flow review context (no blocked worker to resume)
            # The supervisor should use skip_flow or observe when reviewing pending flows

        except Exception as e:
            yield {
                "type": "supervisor_error",
                "data": {"error": str(e)}
            }

    async def _format_active_workers(self) -> list[dict]:
        """Format active workers for AI context."""
        workers = []
        all_flows = await self.state.get_all_flows()
        for flow_data in all_flows:
            if flow_data.status == FlowStatus.EXPLORING and flow_data.worker_id:
                workers.append({
                    "worker_id": flow_data.worker_id,
                    "flow_name": flow_data.flow_name,
                    "status": "exploring",
                    "action_count": len(flow_data.actions)
                })
        return workers

    def stop(self):
        """Request supervisor to stop."""
        self._running = False
