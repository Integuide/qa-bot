"""Flow-based exploration orchestrator that spawns workers per flow."""

import asyncio
import logging
from typing import AsyncGenerator, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

from qa_bot.ai.base import AIProvider
from qa_bot.orchestrator.browser_pool import BrowserPool
from qa_bot.orchestrator.shared_state import SharedFlowState
from qa_bot.orchestrator.worker import FlowExplorationWorker
from qa_bot.orchestrator.supervisor import SupervisorAgent
from qa_bot.orchestrator.synthesis import SynthesisAgent
from qa_bot.orchestrator.flow import FlowTask, FlowStatus
from qa_bot.config import VIEWPORT_WIDTH, VIEWPORT_HEIGHT, LOG_CHAT_HISTORY, LOG_DIR, LOG_SCREENSHOTS, LOG_MAX_RUNS, ENVIRONMENT, TESTMAIL_API_KEY, TESTMAIL_NAMESPACE, MAX_COST_CAP_USD
from qa_bot.utils.chat_logger import ChatLogger
from qa_bot.services.email_service import EmailService


def _should_save_screenshots() -> bool:
    """Determine if screenshots should be saved based on config."""
    if LOG_SCREENSHOTS == "true":
        return True
    elif LOG_SCREENSHOTS == "false":
        return False
    else:  # "auto"
        return ENVIRONMENT == "development"


class FlowExplorationOrchestrator:
    """
    Coordinates flow-based parallel exploration of a website.

    Architecture: Spawn-per-flow model
    - Each flow gets a dedicated worker that runs until the AI says DONE
    - Workers are spawned when flows are created, not upfront
    - Maximum concurrent workers is enforced via semaphore
    - When a worker completes, resources are freed for new flows

    Key features:
    - Starts with single root flow
    - Workers create new flows when they find branches
    - New flows trigger new worker spawns (up to max_agents)
    - AI determines flow completion
    """

    def __init__(
        self,
        ai_provider: AIProvider,
        max_agents: int = 5,
        max_branches_per_flow: int = 10,
        headless: bool = True,
        credentials: Optional[dict[str, str]] = None,
        skip_permissions: bool = False
    ):
        self.ai = ai_provider
        self.max_agents = max_agents
        self.max_branches_per_flow = max_branches_per_flow
        self.headless = headless
        self._initial_credentials = credentials
        self._skip_permissions = skip_permissions

        self._browser_pool: Optional[BrowserPool] = None
        self._shared_state: Optional[SharedFlowState] = None
        self._worker_semaphore: Optional[asyncio.Semaphore] = None
        self._active_worker_tasks: set[asyncio.Task] = set()
        # Worker number pool: recycles numbers 1 through max_agents
        self._available_worker_numbers: set[int] = set()
        self._supervisor: Optional[SupervisorAgent] = None
        self._supervisor_task: Optional[asyncio.Task] = None
        self._event_queue: Optional[asyncio.Queue] = None

    @property
    def shared_state(self) -> Optional[SharedFlowState]:
        """Access to shared state for external credential injection."""
        return self._shared_state

    async def run_exploration(
        self,
        target_url: str,
        goal: str = "Explore user flows through this website and find any bugs, broken elements, or usability problems.",
        max_agents: Optional[int] = None,
        max_branches_per_flow: Optional[int] = None,
        max_duration_minutes: int = 30,
        max_cost_usd: float = 5.0,
        testmail_api_key: Optional[str] = None,
        testmail_namespace: Optional[str] = None
    ) -> AsyncGenerator[dict, None]:
        """
        Run flow-based parallel exploration of target URL.

        Yields unified event stream from all workers.
        """
        num_agents = max_agents or self.max_agents
        max_branches = max_branches_per_flow or self.max_branches_per_flow

        # Cap cost at maximum allowed (defense-in-depth)
        if max_cost_usd is not None and max_cost_usd > MAX_COST_CAP_USD:
            max_cost_usd = MAX_COST_CAP_USD

        # Semaphore limits concurrent workers
        self._worker_semaphore = asyncio.Semaphore(num_agents)
        self._event_queue = asyncio.Queue()

        # Initialize worker number pool (1 through num_agents)
        self._available_worker_numbers = set(range(1, num_agents + 1))

        # Get model from AI provider for cost tracking
        model = getattr(self.ai, 'model', 'claude-haiku-4-5')

        # Initialize shared flow state
        self._shared_state = SharedFlowState.create(
            target_url=target_url,
            goal=goal,
            max_branches_per_flow=max_branches,
            max_duration_minutes=max_duration_minutes,
            max_cost_usd=max_cost_usd,
            model=model,
            skip_permissions=self._skip_permissions,
        )

        # Set initial credentials if provided
        if self._initial_credentials:
            await self._shared_state.set_credentials(self._initial_credentials)

        # Initialize chat logger if enabled
        if LOG_CHAT_HISTORY:
            self._shared_state.chat_logger = ChatLogger(
                log_dir=LOG_DIR,
                exploration_id=self._shared_state.exploration_id,
                save_screenshots=_should_save_screenshots(),
                max_logs=LOG_MAX_RUNS
            )

        # Initialize email service if Testmail.app credentials are provided or configured
        email_key = testmail_api_key or TESTMAIL_API_KEY
        email_namespace = testmail_namespace or TESTMAIL_NAMESPACE
        if email_key and email_namespace:
            self._shared_state.email_service = EmailService(email_key, email_namespace)

        # Helper to yield and log events
        def _log_event(event: dict):
            if self._shared_state.chat_logger:
                self._shared_state.chat_logger.log_event(event)

        event = {
            "type": "exploration_started",
            "data": {
                "exploration_id": self._shared_state.exploration_id,
                "target_url": target_url,
                "goal": goal,
                "max_agents": num_agents,
                "max_branches_per_flow": max_branches,
                "max_duration_minutes": max_duration_minutes,
                "max_cost_usd": max_cost_usd,
                "has_credentials": bool(self._initial_credentials),
                "credential_keys": list(self._initial_credentials.keys()) if self._initial_credentials else [],
                "timestamp": datetime.now().isoformat()
            }
        }
        _log_event(event)
        yield event

        # Initialize browser pool
        self._browser_pool = BrowserPool(
            max_pages=num_agents * 2,
            headless=self.headless,
            viewport_width=VIEWPORT_WIDTH,
            viewport_height=VIEWPORT_HEIGHT
        )
        await self._browser_pool.start()

        try:
            # Spawn supervisor
            self._supervisor = SupervisorAgent(
                ai_provider=self.ai,
                shared_state=self._shared_state,
            )
            self._supervisor_task = asyncio.create_task(
                self._run_supervisor_with_queue(self._supervisor, self._event_queue)
            )

            # Get the root flow and spawn first worker
            root_flow = await self._shared_state.claim_pending_flow()
            if root_flow:
                await self._spawn_worker_for_flow(root_flow, is_first_worker=True)

            # Main event loop
            last_progress_time = datetime.now()
            progress_interval = 5.0
            final_report_already_generated = False  # Track if we used interim as final
            report = None  # Will hold synthesis report

            while True:
                # Check if we should stop
                if self._shared_state.should_stop():
                    stop_reason = self._shared_state.get_stop_reason()

                    if stop_reason == "Stop requested":
                        # User requested stop - break and run final synthesis
                        event = {
                            "type": "stopping",
                            "data": {"message": "Stop requested, waiting for workers to finish..."}
                        }
                        _log_event(event)
                        yield event
                        # Cancel active workers
                        for task in self._active_worker_tasks:
                            task.cancel()
                        break
                    else:
                        # Limit reached - generate interim report then pause
                        event = {
                            "type": "stopping",
                            "data": {"message": f"Limit reached: {stop_reason}. Waiting for workers..."}
                        }
                        _log_event(event)
                        yield event

                        # Set paused state BEFORE cancelling workers so they can checkpoint
                        await self._shared_state.pause(stop_reason)

                        # Cancel active workers gracefully - they will checkpoint because paused=True
                        for task in self._active_worker_tasks:
                            task.cancel()

                        # Wait for workers to finish
                        if self._active_worker_tasks:
                            try:
                                await asyncio.wait_for(
                                    asyncio.gather(*self._active_worker_tasks, return_exceptions=True),
                                    timeout=15.0
                                )
                            except asyncio.TimeoutError:
                                pass

                        # Drain events from queue before generating report
                        while not self._event_queue.empty():
                            try:
                                queued_event = self._event_queue.get_nowait()
                                _log_event(queued_event)
                                yield queued_event
                            except asyncio.QueueEmpty:
                                break

                        # Emit events for any pause checkpoints saved by workers
                        pause_checkpoints = self._shared_state.get_pause_checkpoint_info()
                        for cp_info in pause_checkpoints:
                            event = {
                                "type": "flow_paused_checkpoint",
                                "worker_id": cp_info.get("created_by_worker", "unknown"),
                                "data": {
                                    "flow_id": cp_info["flow_id"],
                                    "flow_name": cp_info["flow_name"],
                                    "checkpoint_id": cp_info["checkpoint_id"],
                                    "url": cp_info.get("url", ""),
                                    "parent_flow_id": cp_info.get("parent_flow_id", "")  # Original flow being paused
                                }
                            }
                            _log_event(event)
                            yield event

                        # Generate interim synthesis report
                        event = {
                            "type": "synthesis_started",
                            "data": {"timestamp": datetime.now().isoformat(), "interim": True}
                        }
                        _log_event(event)
                        yield event

                        synthesis = SynthesisAgent(self.ai)
                        try:
                            interim_report = await synthesis.generate_report(self._shared_state)
                            event = {
                                "type": "synthesis_complete",
                                "data": {
                                    "report": interim_report,
                                    "timestamp": datetime.now().isoformat(),
                                    "interim": True
                                }
                            }
                            _log_event(event)
                            yield event
                        except Exception as e:
                            interim_report = f"Error generating interim report: {e}"
                            event = {
                                "type": "synthesis_error",
                                "data": {"error": str(e), "interim": True}
                            }
                            _log_event(event)
                            yield event

                        # Emit paused event (pause state already set before worker cancellation)
                        progress = await self._shared_state.get_progress()
                        event = {
                            "type": "exploration_paused",
                            "data": {
                                "reason": stop_reason,
                                "progress": progress,
                                "interim_report": interim_report,
                                "timestamp": datetime.now().isoformat()
                            }
                        }
                        _log_event(event)
                        yield event

                        # Wait for resume signal (30 minute timeout)
                        PAUSE_TIMEOUT_SECONDS = 30 * 60  # 30 minutes
                        resumed = await self._shared_state.wait_for_resume(timeout=PAUSE_TIMEOUT_SECONDS)

                        if not resumed or self._shared_state.stop_requested:
                            # Timed out or user stopped during pause - use interim report as final
                            if not resumed:
                                event = {
                                    "type": "exploration_pause_timeout",
                                    "data": {"message": "Pause timed out after 30 minutes"}
                                }
                                _log_event(event)
                                yield event
                            # Skip to final summary (interim report already generated)
                            final_report_already_generated = True
                            report = interim_report
                            break

                        # Resume exploration
                        event = {
                            "type": "exploration_resumed",
                            "data": {
                                "new_max_cost_usd": self._shared_state.max_cost_usd,
                                "new_max_duration_minutes": self._shared_state.max_duration_minutes,
                                "timestamp": datetime.now().isoformat()
                            }
                        }
                        _log_event(event)
                        yield event

                        # Reinitialize worker tracking for resumption
                        # At this point, all cancelled workers should have completed their cleanup
                        # (releasing semaphore slots, returning worker numbers, decrementing active count)
                        # We reset our tracking to a clean state and let the semaphore/shared state
                        # remain as workers left them (should be fully released)
                        active_workers = self._shared_state._active_workers
                        if active_workers > 0:
                            logger.warning(f"Resuming with {active_workers} active workers still tracked in shared state")
                        self._active_worker_tasks.clear()
                        self._available_worker_numbers = set(range(1, num_agents + 1))
                        # Reset semaphore to full capacity for fresh start
                        self._worker_semaphore = asyncio.Semaphore(num_agents)

                        # Continue the main loop - will spawn workers for any pending flows
                        continue

                # Check for new pending flows and spawn workers
                await self._spawn_workers_for_pending_flows()

                # Process events from queue
                try:
                    event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)

                    # Track worker completion
                    if event["type"] == "worker_stopped":
                        # Worker finished - check for more pending flows
                        await self._spawn_workers_for_pending_flows()

                    _log_event(event)
                    yield event

                except asyncio.TimeoutError:
                    pass

                # Emit progress periodically
                now = datetime.now()
                if (now - last_progress_time).total_seconds() >= progress_interval:
                    event = {
                        "type": "progress",
                        "data": await self._shared_state.get_progress()
                    }
                    _log_event(event)
                    yield event
                    last_progress_time = now

                # Check if exploration is complete (no active workers, no pending flows)
                if self._is_exploration_complete():
                    break

            # Wait for remaining workers to finish
            if self._active_worker_tasks:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*self._active_worker_tasks, return_exceptions=True),
                        timeout=15.0
                    )
                except asyncio.TimeoutError:
                    for task in self._active_worker_tasks:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*self._active_worker_tasks, return_exceptions=True)

            # Drain remaining events from queue
            while not self._event_queue.empty():
                try:
                    event = self._event_queue.get_nowait()
                    _log_event(event)
                    yield event
                except asyncio.QueueEmpty:
                    break

            # Generate synthesis report (skip if we already used interim report as final)
            if not final_report_already_generated:
                event = {
                    "type": "synthesis_started",
                    "data": {"timestamp": datetime.now().isoformat()}
                }
                _log_event(event)
                yield event

                synthesis = SynthesisAgent(self.ai)
                try:
                    report = await synthesis.generate_report(self._shared_state)
                    event = {
                        "type": "synthesis_complete",
                        "data": {
                            "report": report,
                            "timestamp": datetime.now().isoformat()
                        }
                    }
                    _log_event(event)
                    yield event
                except Exception as e:
                    event = {
                        "type": "synthesis_error",
                        "data": {"error": str(e)}
                    }
                    _log_event(event)
                    yield event
                    report = None
            else:
                # report was already set from interim report during pause handling
                pass

            # Final summary
            all_issues = await self._shared_state.get_all_issues()
            all_flows = await self._shared_state.get_all_flows()
            flow_tree = await self._shared_state.get_flow_tree()
            final_progress = await self._shared_state.get_progress()

            event = {
                "type": "exploration_complete",
                "data": {
                    "exploration_id": self._shared_state.exploration_id,
                    "target_url": target_url,
                    "issues_found": len(all_issues),
                    "issues": [issue.to_dict() for issue in all_issues],
                    "flows_explored": len([
                        f for f in all_flows
                        if f.status == FlowStatus.COMPLETED
                    ]),
                    "total_flows": len(all_flows),
                    "flow_tree": flow_tree,
                    "total_actions": self._shared_state.total_actions,
                    "duration_seconds": (datetime.now() - self._shared_state.start_time).total_seconds(),
                    "final_progress": final_progress,
                    "synthesis_report": report
                }
            }
            _log_event(event)
            yield event

        except Exception as e:
            event = {
                "type": "error",
                "data": {
                    "message": str(e),
                    "fatal": True
                }
            }
            _log_event(event)
            yield event
        finally:
            await self._cleanup()

    async def _spawn_worker_for_flow(self, flow_task: FlowTask, is_first_worker: bool = False):
        """Spawn a dedicated worker for a specific flow.

        The is_first_worker flag is read from the flow_task if set there,
        otherwise falls back to the parameter (for backward compatibility).
        """
        # Acquire semaphore slot (blocks if at max workers)
        await self._worker_semaphore.acquire()

        # Track active worker count in shared state (for UI progress)
        await self._shared_state.increment_active_workers()

        # Get a worker number from the pool (recycles 1-N)
        # Safety: The semaphore acquired above guarantees a number is available,
        # since pool size equals semaphore count. Assert defensively.
        assert self._available_worker_numbers, "Bug: No available worker numbers despite semaphore acquisition"
        worker_number = min(self._available_worker_numbers)
        self._available_worker_numbers.remove(worker_number)
        worker_id = f"worker-{worker_number}"

        # Use is_first_worker from flow_task if set, otherwise use parameter
        # This ensures resumed flows retain their first_worker status
        effective_is_first_worker = flow_task.is_first_worker or is_first_worker

        worker = FlowExplorationWorker(
            worker_id=worker_id,
            worker_number=worker_number,
            ai_provider=self.ai,
            browser_pool=self._browser_pool,
            shared_state=self._shared_state,
            flow_task=flow_task,
            is_first_worker=effective_is_first_worker
        )

        # Create task that runs worker and releases semaphore when done
        task = asyncio.create_task(
            self._run_worker_and_release(worker, worker_number)
        )
        self._active_worker_tasks.add(task)
        task.add_done_callback(self._active_worker_tasks.discard)

    async def _run_worker_and_release(self, worker: FlowExplorationWorker, worker_number: int):
        """Run a worker and release semaphore when done."""
        try:
            async for event in worker.run():
                await self._event_queue.put(event)
        except asyncio.CancelledError:
            pass  # Worker handled checkpoint in _execute_flow, just cleanup
        except Exception as e:
            await self._event_queue.put({
                "type": "worker_error",
                "data": {
                    "worker_id": worker.worker_id,
                    "error": str(e)
                }
            })
        finally:
            await self._event_queue.put({
                "type": "worker_stopped",
                "data": {
                    "worker_id": worker.worker_id,
                    "flow_id": worker.flow_task.flow_id,
                    "timestamp": datetime.now().isoformat()
                }
            })
            await self._shared_state.decrement_active_workers()
            # Return worker number to the pool for reuse
            self._available_worker_numbers.add(worker_number)
            self._worker_semaphore.release()

    async def _spawn_workers_for_pending_flows(self):
        """Check for pending flows and spawn workers for them."""
        while True:
            # Check if semaphore has available slots (non-blocking)
            if self._worker_semaphore.locked():
                # All worker slots in use
                break

            # Try to claim a pending flow first
            flow_task = await self._shared_state.claim_pending_flow()
            if flow_task is None:
                # No pending flows
                break

            # Spawn worker - this will acquire the semaphore
            # If another coroutine grabbed it between our check and now,
            # this will block until a slot is free, which is acceptable
            await self._spawn_worker_for_flow(flow_task)

    def _is_exploration_complete(self) -> bool:
        """Check if exploration is complete.

        Uses shared_state.is_complete() which checks:
        - No pending flows in task queue
        - No active workers
        - No pause checkpoints waiting
        - No flows blocked for credentials, approval, or pause
        """
        # Also check our local active worker tasks as a safety net
        # (shared_state._active_workers counter should match, but belt-and-suspenders)
        if len(self._active_worker_tasks) > 0:
            return False
        return self._shared_state.is_complete()

    async def _run_supervisor_with_queue(
        self,
        supervisor: SupervisorAgent,
        queue: asyncio.Queue
    ):
        """Run the supervisor and put all its events into the queue."""
        try:
            async for event in supervisor.run():
                await queue.put(event)
        except Exception as e:
            await queue.put({
                "type": "supervisor_error",
                "data": {"error": str(e)}
            })

    async def _cleanup(self):
        """Clean up all resources."""
        # Stop supervisor
        if self._supervisor:
            self._supervisor.stop()
        if self._supervisor_task and not self._supervisor_task.done():
            self._supervisor_task.cancel()
            try:
                await self._supervisor_task
            except asyncio.CancelledError:
                pass
        self._supervisor = None
        self._supervisor_task = None

        # Cancel any remaining worker tasks
        for task in list(self._active_worker_tasks):
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self._active_worker_tasks.clear()

        # Stop browser pool
        if self._browser_pool:
            await self._browser_pool.stop()
            self._browser_pool = None

        # Close chat logger to prevent resource leaks
        if self._shared_state and self._shared_state.chat_logger:
            self._shared_state.chat_logger.close()

    def request_stop(self):
        """Request exploration to stop gracefully."""
        if self._shared_state:
            self._shared_state.stop_requested = True
            # If paused, wake up wait_for_resume() so coordinator can check stop_requested
            if self._shared_state.is_paused():
                self._shared_state._resume_event.set()
