"""Flow-based exploration orchestrator that spawns workers per flow."""

import asyncio
import gc
import logging
from typing import AsyncGenerator, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

from qa_bot.ai.base import AIProvider
from qa_bot.orchestrator.browser_pool import BrowserPool, BROWSER_USER_AGENT
from qa_bot.orchestrator.shared_state import SharedFlowState, SMOKE_GOAL
from qa_bot.orchestrator.worker import FlowExplorationWorker
from qa_bot.orchestrator.supervisor import SupervisorAgent
from qa_bot.orchestrator.synthesis import SynthesisAgent, apply_smoke_tag, format_duration
from qa_bot.orchestrator.flow import FlowTask, FlowStatus
from qa_bot.orchestrator.recheck import RECHECK_SLOTS
from qa_bot.config import VIEWPORT_WIDTH, VIEWPORT_HEIGHT, LOG_CHAT_HISTORY, LOG_DIR, LOG_SCREENSHOTS, LOG_MAX_RUNS, ENVIRONMENT, TESTMAIL_API_KEY, TESTMAIL_NAMESPACE, MAX_COST_CAP_USD, DEFAULT_MODEL, recheck_criticals_default
from qa_bot.utils.chat_logger import ChatLogger
from qa_bot.utils.secrets import redact_obj, redact_values
from qa_bot.services.email_service import EmailService

# summary.json statuses that mean the run has finished. Anything else left on
# disk ("running", "synthesizing") means the process died mid-run without
# reaching the code that writes a terminal summary.
_TERMINAL_SUMMARY_STATUSES = frozenset({"completed", "error", "preflight_failed", "interrupted"})


def _count_flows_explored(all_flows) -> int:
    """Number of user flows actually tested to completion.

    The Root/first-worker flow (and its resume/retry clones, which inherit
    ``is_first_worker``) is excluded: its only job is flow enumeration, so
    counting it let a run where nothing but flow-mapping finished report
    ``flows_explored=1`` and slip past the action's fail-on-zero-flows gate.
    Independent re-check flows (``recheck_of``) verify one finding and are
    excluded for the same reason.
    """
    return len([
        f for f in all_flows
        if f.status == FlowStatus.COMPLETED
        and not f.is_first_worker
        and not f.recheck_of
    ])


def _should_save_screenshots() -> bool:
    """Determine if screenshots should be saved based on config."""
    if LOG_SCREENSHOTS == "true":
        return True
    elif LOG_SCREENSHOTS == "false":
        return False
    else:  # "auto"
        return ENVIRONMENT == "development"


# Keep this at least as lenient as the browser's navigation budget
# (_safe_goto's 30s default in worker.py) — a probe stricter than the real
# navigation would abort a slow-cold-start target as "unreachable" (a NOT TESTED
# report that fails the CI deploy check) even though the browser would load it.
PREFLIGHT_TIMEOUT_SECONDS = 30.0


def _detect_http_auth_credentials(
    credentials: dict[str, str],
) -> tuple[str, str] | None:
    """Find HTTP Basic Auth credentials among the provided keys.

    Mirrors the key heuristics the worker prompt teaches the AI for
    set_http_auth: exact HTTP_USERNAME/HTTP_PASSWORD first, then any
    key pair containing http+user / http+pass.
    """
    if "HTTP_USERNAME" in credentials and "HTTP_PASSWORD" in credentials:
        return credentials["HTTP_USERNAME"], credentials["HTTP_PASSWORD"]

    username = password = None
    for key, value in credentials.items():
        lowered = key.lower()
        if "http" in lowered and "user" in lowered and username is None:
            username = value
        elif "http" in lowered and "pass" in lowered and password is None:
            password = value
    if username is not None and password is not None:
        return username, password
    return None


async def preflight_check(
    target_url: str,
    credentials: dict[str, str] | None,
    interactive: bool = False,
) -> str | None:
    """Probe the target URL before spending anything on browsers or AI calls.

    Returns a human-actionable fatal reason, or None when exploration should
    proceed. Only conditions that make EVERY flow useless abort the run:

    - the target doesn't respond at all (DNS failure, refused, timeout)
    - the target demands HTTP Basic Auth and no usable credentials were
      provided (or the provided ones are rejected) — non-interactive runs
      only, since a web-UI user can supply credentials mid-run

    Anything else (404/5xx on the root, redirects elsewhere) is left to the
    exploration itself — a reachable server is worth testing.

    Rationale: runs on mono PR #356 burned full explorations producing
    "NOT TESTED" reports for exactly these two conditions.
    """
    import httpx

    from qa_bot.config import IGNORE_HTTPS_ERRORS

    auth = _detect_http_auth_credentials(credentials or {})

    def _unreachable(err: Exception) -> str:
        return (
            f"Target {target_url} is unreachable: "
            f"{type(err).__name__}: {err}. Verify the deployment is up and the "
            f"URL is correct."
        )

    # A cold-starting staging box often times out or resets the very first
    # request but serves the second. Retry once on those transient errors;
    # connection-refused / DNS failures (ConnectError) won't recover in a
    # second, so fail fast on them.
    transient = (httpx.TimeoutException, httpx.ReadError, httpx.RemoteProtocolError)
    for attempt in range(2):
        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=PREFLIGHT_TIMEOUT_SECONDS,
                # Match the browser's TLS posture so the probe never fails a
                # target the browser itself would happily load.
                verify=not IGNORE_HTTPS_ERRORS,
                # Send the same User-Agent the real browser uses so WAF/bot
                # protection that drops or resets non-browser clients doesn't
                # make the probe read a target as "unreachable" when the real
                # Chromium would load it fine. Matching exactly means the probe
                # fails iff the browser would.
                headers={"User-Agent": BROWSER_USER_AGENT},
            ) as client:
                response = await client.get(target_url)
                if response.status_code == 401 and not interactive:
                    if auth is None:
                        return (
                            f"Target {target_url} requires HTTP Basic Auth but no "
                            f"HTTP credentials were provided. Add HTTP_USERNAME "
                            f"and HTTP_PASSWORD to the run's credentials."
                        )
                    response = await client.get(target_url, auth=auth)
                    if response.status_code == 401:
                        return (
                            f"Target {target_url} rejected the provided HTTP "
                            f"Basic Auth credentials (still 401). Check "
                            f"HTTP_USERNAME / HTTP_PASSWORD."
                        )
            return None
        except httpx.ConnectError as e:
            # Refused / DNS resolution failure — retrying won't help.
            return _unreachable(e)
        except transient as e:
            if attempt == 0:
                await asyncio.sleep(1.0)
                continue
            return _unreachable(e)
        except httpx.HTTPError as e:
            return _unreachable(e)
    return None


class FatalExplorationError(RuntimeError):
    """The run was aborted for a reason no retry can fix (see SharedFlowState.abort)."""


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
        skip_permissions: bool = False,
        interactive: bool = True
    ):
        self.ai = ai_provider
        self.max_agents = max_agents
        self.max_branches_per_flow = max_branches_per_flow
        self.headless = headless
        self._initial_credentials = credentials
        self._skip_permissions = skip_permissions
        self._interactive = interactive

        self._browser_pool: Optional[BrowserPool] = None
        self._shared_state: Optional[SharedFlowState] = None
        self._worker_semaphore: Optional[asyncio.Semaphore] = None
        self._active_worker_tasks: set[asyncio.Task] = set()
        # Worker number pool: recycles numbers 1 through max_agents
        self._available_worker_numbers: set[int] = set()
        # Slot(s) reserved for independent re-checks of critical findings, on
        # top of max_agents, with their own worker numbers (see
        # _reset_recheck_slots)
        self._recheck_semaphore: Optional[asyncio.Semaphore] = None
        self._recheck_worker_numbers: set[int] = set()
        self._supervisor: Optional[SupervisorAgent] = None
        self._supervisor_task: Optional[asyncio.Task] = None
        self._event_queue: Optional[asyncio.Queue] = None
        # Status of the last summary.json written for the current run (None
        # until one is written) — lets the finally-block tell an interrupted
        # run from one that already recorded its outcome.
        self._last_summary_status: Optional[str] = None

    @property
    def shared_state(self) -> Optional[SharedFlowState]:
        """Access to shared state for external credential injection."""
        return self._shared_state

    @property
    def is_synthesizing(self) -> bool:
        """True from synthesis_started until the run's outcome is recorded.

        Lets the web server tell a client attaching mid-synthesis that the
        run is in its (long, quiet) report-generation phase rather than
        still exploring — the shared state alone can't distinguish them.
        """
        return self._last_summary_status == "synthesizing"

    async def run_exploration(
        self,
        target_url: str,
        goal: str = "Explore user flows through this website and find any bugs, broken elements, or usability problems.",
        max_agents: Optional[int] = None,
        max_branches_per_flow: Optional[int] = None,
        max_duration_minutes: int = 30,
        max_cost_usd: float = 5.0,
        testmail_api_key: Optional[str] = None,
        testmail_namespace: Optional[str] = None,
        exploration_id: Optional[str] = None,
        known_issues: str = "",
        smoke: bool = False,
        previous_report: str = "",
        recheck_criticals: Optional[bool] = None,
    ) -> AsyncGenerator[dict, None]:
        """
        Run flow-based parallel exploration of target URL.

        Yields unified event stream from all workers.

        ``smoke`` (CLI ``--smoke``, action ``mode: smoke``) turns the run
        into a bounded smoke pass: the preflight probe, then ONE worker
        with the fixed SMOKE_GOAL and a hard turn cap, no flow forking, and
        a report tagged "SMOKE TEST ONLY — not a regression test". The
        caller's goal is ignored so a smoke run can't be steered into a
        full exploration.

        ``previous_report`` (CLI ``--previous-report``, action
        ``previous-report``) is the previous run's report for the same
        target; synthesis labels findings NEW / recurring against it. It is
        deliberately NOT merged into ``known_issues`` — a regression found in
        the previous run must not be demoted on this one.

        ``recheck_criticals`` turns on the independent re-check of CRITICAL
        findings (orchestrator/recheck.py); None means the default —
        ``RECHECK_CRITICALS`` env, else on for non-interactive runs only.
        """
        if smoke:
            goal = SMOKE_GOAL
        if recheck_criticals is None:
            recheck_criticals = recheck_criticals_default(self._interactive)
        num_agents = max_agents or self.max_agents
        # Clamp to the same 1-20 range config.py enforces on the MAX_AGENTS env
        # var (defense-in-depth): the CLI --max-agents flag and the GitHub
        # Action's max-agents input reach here unclamped, and an out-of-range
        # value would size the semaphore and worker-number pool nonsensically
        # (0 or negative deadlocks; a huge value floods the target with workers).
        num_agents = max(1, min(20, num_agents))
        max_branches = max_branches_per_flow or self.max_branches_per_flow

        # Cap cost at maximum allowed (defense-in-depth)
        if max_cost_usd is not None and max_cost_usd > MAX_COST_CAP_USD:
            max_cost_usd = MAX_COST_CAP_USD

        # Semaphore limits concurrent workers
        self._worker_semaphore = asyncio.Semaphore(num_agents)
        self._event_queue = asyncio.Queue()

        # Initialize worker number pool (1 through num_agents)
        self._available_worker_numbers = set(range(1, num_agents + 1))
        self._reset_recheck_slots(num_agents)

        # Get model from AI provider for cost tracking
        model = getattr(self.ai, 'model', DEFAULT_MODEL)

        self._last_summary_status = None

        # Initialize shared flow state
        self._shared_state = SharedFlowState.create(
            target_url=target_url,
            goal=goal,
            max_branches_per_flow=max_branches,
            max_duration_minutes=max_duration_minutes,
            max_cost_usd=max_cost_usd,
            model=model,
            skip_permissions=self._skip_permissions,
            interactive=self._interactive,
            exploration_id=exploration_id,
            known_issues=known_issues,
            smoke=smoke,
            previous_report=previous_report,
            recheck_criticals=recheck_criticals,
        )

        # Initialize chat logger if enabled (before credentials are set, so
        # set_credentials can register their values for by-value redaction)
        if LOG_CHAT_HISTORY:
            self._shared_state.chat_logger = ChatLogger(
                log_dir=LOG_DIR,
                exploration_id=self._shared_state.exploration_id,
                save_screenshots=_should_save_screenshots(),
                max_logs=LOG_MAX_RUNS
            )
            # Every run directory carries a machine-readable state from the
            # start: a summary.json still at "running" is how an operator tells
            # a run the process died under from one that is still going.
            await self._write_provisional_summary("running")

        # Set initial credentials if provided
        if self._initial_credentials:
            await self._shared_state.set_credentials(self._initial_credentials)

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
                "has_known_issues": bool(known_issues),
                "has_previous_report": bool(previous_report),
                "mode": "smoke" if smoke else "full",
                "recheck_criticals": recheck_criticals,
                "credential_keys": list(self._initial_credentials.keys()) if self._initial_credentials else [],
                "timestamp": datetime.now().isoformat()
            }
        }
        _log_event(event)
        yield event

        # Pre-flight: if the target can't be tested at all (unreachable, or
        # behind HTTP auth with no usable credentials), fail in seconds for
        # ~$0 instead of spinning up browsers and AI workers that can only
        # produce a "NOT TESTED" report.
        preflight_error = await preflight_check(
            target_url, self._initial_credentials, interactive=self._interactive
        )
        if preflight_error:
            async for event in self._abort_for_preflight(
                target_url, preflight_error, _log_event
            ):
                yield event
            return

        # Initialize browser pool
        self._browser_pool = BrowserPool(
            max_pages=num_agents * 2,
            headless=self.headless,
            viewport_width=VIEWPORT_WIDTH,
            viewport_height=VIEWPORT_HEIGHT
        )

        try:
            # Inside the try so a startup failure (e.g. Playwright not
            # installed) still runs _cleanup() and yields a fatal error event.
            await self._browser_pool.start()

            # Spawn supervisor
            self._supervisor = SupervisorAgent(
                ai_provider=self.ai,
                shared_state=self._shared_state,
            )
            self._supervisor_task = asyncio.create_task(
                self._run_supervisor_with_queue(self._supervisor, self._event_queue)
            )

            # Get the root flow and spawn first worker
            # In smoke mode the root flow IS the smoke pass (no enumeration
            # step, no children), so it runs as a normal, counted worker.
            root_flow = await self._shared_state.claim_pending_flow()
            if root_flow:
                await self._spawn_worker_for_flow(
                    root_flow, is_first_worker=not smoke
                )

            # Main event loop
            last_progress_time = datetime.now()
            progress_interval = 5.0
            report = None  # Will hold synthesis report

            while True:
                # A worker hit an error no retry can fix (rejected API key,
                # no credit). Surface what the workers already reported, then
                # route through the fatal-error path below: salvage whatever
                # was found and end with a fatal `error` event — the web UI
                # shows the reason instead of a green "Complete", the CLI
                # exits non-zero, and no further flow is attempted.
                if self._shared_state.fatal_error:
                    while not self._event_queue.empty():
                        try:
                            queued_event = self._event_queue.get_nowait()
                            _log_event(queued_event)
                            yield queued_event
                        except asyncio.QueueEmpty:
                            break
                    raise FatalExplorationError(self._shared_state.fatal_error)

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
                        # Cancel active workers (copy to avoid mutation during iteration)
                        for task in list(self._active_worker_tasks):
                            task.cancel()
                        break
                    else:
                        # Limit reached - pause and let user decide
                        event = {
                            "type": "stopping",
                            "data": {"message": f"Limit reached: {stop_reason}. Waiting for workers..."}
                        }
                        _log_event(event)
                        yield event

                        # Set paused state BEFORE cancelling workers so they can checkpoint
                        await self._shared_state.pause(stop_reason)

                        # Cancel active workers gracefully - they will checkpoint because paused=True
                        for task in list(self._active_worker_tasks):
                            task.cancel()

                        # Wait for workers to finish (they may be saving checkpoints)
                        if self._active_worker_tasks:
                            try:
                                await asyncio.wait_for(
                                    asyncio.gather(*self._active_worker_tasks, return_exceptions=True),
                                    timeout=15.0
                                )
                            except asyncio.TimeoutError:
                                logger.warning("Pause: workers did not finish within 15s — proceeding")

                        # Drain events from queue before pausing
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

                        # Emit paused event (pause state already set before worker cancellation)
                        # Synthesis is deferred until the user decides to stop or exploration completes
                        progress = await self._shared_state.get_progress()
                        event = {
                            "type": "exploration_paused",
                            "data": {
                                "reason": stop_reason,
                                "progress": progress,
                                "timestamp": datetime.now().isoformat()
                            }
                        }
                        _log_event(event)
                        yield event

                        if not self._interactive:
                            # Non-interactive (CI/GitHub Actions): no user to
                            # resume, so proceed straight to synthesis.
                            break

                        # Wait for resume signal (30 minute timeout)
                        PAUSE_TIMEOUT_SECONDS = 30 * 60  # 30 minutes
                        resumed = await self._shared_state.wait_for_resume(timeout=PAUSE_TIMEOUT_SECONDS)

                        if not resumed or self._shared_state.stop_requested:
                            # Timed out or user stopped during pause - run synthesis after break
                            if not resumed:
                                event = {
                                    "type": "exploration_pause_timeout",
                                    "data": {"message": "Pause timed out after 30 minutes"}
                                }
                                _log_event(event)
                                yield event
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
                            logger.warning(f"Resuming with {active_workers} active workers still tracked - resetting to 0")
                            self._shared_state._active_workers = 0
                        self._active_worker_tasks.clear()
                        self._available_worker_numbers = set(range(1, num_agents + 1))
                        # Reset semaphore to full capacity for fresh start
                        self._worker_semaphore = asyncio.Semaphore(num_agents)
                        self._reset_recheck_slots(num_agents)

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
                    for task in list(self._active_worker_tasks):
                        if not task.done():
                            task.cancel()
                    try:
                        await asyncio.wait_for(
                            asyncio.gather(*self._active_worker_tasks, return_exceptions=True),
                            timeout=10.0
                        )
                    except asyncio.TimeoutError:
                        logger.warning("Workers did not exit after cancellation — proceeding without them")
            # Drain remaining events from queue
            while not self._event_queue.empty():
                try:
                    event = self._event_queue.get_nowait()
                    _log_event(event)
                    yield event
                except asyncio.QueueEmpty:
                    break

            # Final progress before synthesis: every worker has exited, so
            # this is the first progress payload with active_workers == 0
            # (the periodic one above may be up to progress_interval stale).
            # The web UI's stat cards read progress payloads, not their own
            # increments, so without it "Active Workers: 3" would persist
            # through synthesis.
            event = {
                "type": "progress",
                "data": await self._shared_state.get_progress()
            }
            _log_event(event)
            yield event

            # Generate synthesis report
            # The main loop is the only consumer of _event_queue; once it has
            # exited, anything still producing (the supervisor) would fill the
            # queue unbounded for the whole synthesis await. Stop it first.
            await self._stop_supervisor()

            # Every worker has exited: a re-check still queued or cut off
            # settles inconclusive (its critical stays critical) so no
            # finding reaches the report or the gate still "pending"
            for event in await self._finalize_rechecks():
                _log_event(event)
                yield event

            event = {
                "type": "synthesis_started",
                "data": {"timestamp": datetime.now().isoformat()}
            }
            _log_event(event)
            yield event

            # Synthesis is the longest single await of the run and used to be
            # the point where a server restart lost everything — nothing was
            # on disk until the AI call returned. Persist what is known now
            # (SynthesisAgent likewise writes a provisional report.md before
            # its AI call); both are overwritten on success.
            await self._write_provisional_summary("synthesizing")

            synthesis = SynthesisAgent(self.ai)
            verdict = None
            try:
                report, verdict = await synthesis.generate_report_with_verdict(
                    self._shared_state
                )
                report, verdict = self._mask_secrets(report, verdict)
                self._persist_report(report)
                event = {
                    "type": "synthesis_complete",
                    "data": {
                        "report": report,
                        # Curated verdict parsed from the AI report (None when
                        # synthesis fell back to a deterministic report or the
                        # block was unparseable). The deploy gate still counts
                        # RAW worker criticals — this is surfaced alongside so
                        # both numbers are visible.
                        "verdict": verdict,
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
                    "flows_explored": _count_flows_explored(all_flows),
                    "total_flows": len(all_flows),
                    "flow_tree": flow_tree,
                    "total_actions": self._shared_state.total_actions,
                    "duration_seconds": (datetime.now() - self._shared_state.start_time).total_seconds(),
                    "final_progress": final_progress,
                    "synthesis_report": report,
                    "recheck_summary": self._shared_state.recheck_summary(),
                }
            }
            _log_event(event)

            # Write summary.json before yielding final event — code after
            # yield in an async generator may not execute if consumer stops iterating
            self._write_summary(
                final_progress,
                status="completed",
                issues=[issue.to_dict() for issue in all_issues],
                verdict=verdict,
            )

            yield event

        except Exception as e:
            if isinstance(e, FatalExplorationError):
                logger.error(f"Exploration aborted: {e}")  # expected, no traceback
            else:
                logger.exception("Fatal error in exploration main loop")

            # Salvage whatever the workers already found: a late crash must
            # not discard an entire run's findings. Synthesize a report from
            # collected state and emit exploration_complete so the CLI and
            # the GitHub Action PR comment still receive the results.
            # The main loop (sole consumer of _event_queue) is gone: stop the
            # supervisor first so it cannot fill the queue during synthesis.
            try:
                await self._stop_supervisor()
            except Exception:
                logger.exception("Failed to stop supervisor before salvage")
            try:
                salvage_events = await self._build_salvage_events(target_url, str(e))
            except Exception:
                logger.exception("Failed to salvage results after fatal error")
                salvage_events = []

            # Write summary.json even on error. Done after salvage so the
            # recorded cost includes the salvage synthesis call's usage.
            try:
                error_progress = await self._shared_state.get_progress()
                error_issues = await self._shared_state.get_all_issues()
                self._write_summary(
                    error_progress,
                    status="error",
                    error=str(e),
                    issues=[issue.to_dict() for issue in error_issues],
                )
            except Exception:
                pass  # Best-effort, don't mask the original error
            for salvage_event in salvage_events:
                _log_event(salvage_event)
                yield salvage_event

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
            if self._last_summary_status not in _TERMINAL_SUMMARY_STATUSES:
                # Cancelled (Ctrl-C, SIGTERM from a server restart, runner
                # timeout, consumer stopped iterating): CancelledError and
                # GeneratorExit are not Exceptions, so the except-path summary
                # never ran. Record that — and where — the run died.
                await self._write_provisional_summary(
                    "interrupted",
                    error=(
                        "Run interrupted during synthesis"
                        if self._last_summary_status == "synthesizing"
                        else "Run interrupted before exploration finished"
                    ),
                )
            await self._cleanup()

    def _reset_recheck_slots(self, num_agents: int):
        """(Re)create the re-check slot(s): numbered after the regular
        workers, so worker ids stay unique across both pools."""
        self._recheck_semaphore = asyncio.Semaphore(RECHECK_SLOTS)
        self._recheck_worker_numbers = set(
            range(num_agents + 1, num_agents + 1 + RECHECK_SLOTS)
        )

    async def _spawn_worker_for_flow(
        self,
        flow_task: FlowTask,
        is_first_worker: bool = False,
        recheck_slot: bool = False,
    ):
        """Spawn a dedicated worker for a specific flow.

        The is_first_worker flag is read from the flow_task if set there,
        otherwise falls back to the parameter (for backward compatibility).
        ``recheck_slot`` runs the worker in a reserved re-check slot instead
        of a regular one.
        """
        # Acquire semaphore slot (blocks if at max workers).
        # Capture the semaphore and number pool objects so the worker's
        # release goes to the SAME objects it acquired from. The resume path
        # replaces both after a 15s grace period; without this, a straggler
        # worker finishing after resume would over-credit the new semaphore
        # beyond max_agents and exhaust the worker number pool.
        if recheck_slot:
            semaphore = self._recheck_semaphore
            number_pool = self._recheck_worker_numbers
        else:
            semaphore = self._worker_semaphore
            number_pool = self._available_worker_numbers
        await semaphore.acquire()

        # Re-check the supervisor skip set after the (possibly long) semaphore
        # wait: claim_pending_flow leaves the flow's registry status PENDING,
        # so the supervisor can mark a claimed-but-unspawned flow as a
        # duplicate while all worker slots are busy. Without this re-check the
        # flow would emit flow_skipped yet still be explored once a slot
        # freed, spending budget on the duplicate.
        if await self._shared_state.is_flow_skipped(flow_task.flow_id):
            logger.info(
                f"Discarding flow skipped while waiting for a worker slot: "
                f"{flow_task.flow_name} ({flow_task.flow_id})"
            )
            semaphore.release()
            return

        # Track active worker count in shared state (for UI progress)
        await self._shared_state.increment_active_workers()

        # Get a worker number from the pool (recycles 1-N)
        # Safety: The semaphore acquired above guarantees a number is available,
        # since pool size equals semaphore count. Assert defensively.
        assert number_pool, "Bug: No available worker numbers despite semaphore acquisition"
        worker_number = min(number_pool)
        number_pool.remove(worker_number)
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
            self._run_worker_and_release(worker, worker_number, semaphore, number_pool)
        )
        self._active_worker_tasks.add(task)
        task.add_done_callback(self._active_worker_tasks.discard)

    async def _run_worker_and_release(
        self,
        worker: FlowExplorationWorker,
        worker_number: int,
        semaphore: asyncio.Semaphore,
        number_pool: set[int],
    ):
        """Run a worker and release the semaphore/number pool it acquired from.

        The semaphore and pool are passed explicitly (not read from self) so
        a worker that outlives a pause/resume cycle credits the objects it
        was spawned under, not the fresh ones created by the resume path.
        """
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
            number_pool.add(worker_number)
            semaphore.release()

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

            # Discard flows the supervisor marked as duplicates instead of
            # exploring them. The task stays out of the queue and keeps its
            # SKIPPED status; a flow_skipped event was already emitted by
            # the supervisor when it made the decision.
            if await self._shared_state.is_flow_skipped(flow_task.flow_id):
                logger.info(
                    f"Discarding supervisor-skipped flow: {flow_task.flow_name} "
                    f"({flow_task.flow_id})"
                )
                continue

            # Spawn worker - this will acquire the semaphore
            # If another coroutine grabbed it between our check and now,
            # this will block until a slot is free, which is acceptable
            await self._spawn_worker_for_flow(flow_task)

        # A queued re-check of a critical finding never waits for a user flow
        # to finish: it decides the gate and must end inside the run's
        # budget, so it also has slot(s) of its own.
        while self._recheck_semaphore is not None and not self._recheck_semaphore.locked():
            recheck_task = await self._shared_state.claim_pending_recheck()
            if recheck_task is None:
                break
            await self._spawn_worker_for_flow(recheck_task, recheck_slot=True)

    def _is_exploration_complete(self) -> bool:
        """Check if exploration is complete.

        Uses shared_state.is_complete() which checks:
        - No pending flows in task queue
        - No active workers
        - No pause checkpoints waiting
        - No flows blocked for pause

        Note: Credential/approval blocks are terminal — they don't prevent
        completion. Those flows need external input that may never arrive.
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

    async def _abort_for_preflight(
        self, target_url: str, reason: str, _log_event
    ) -> AsyncGenerator[dict, None]:
        """End the run before any browser/AI cost when preflight fails.

        Emits the same terminal events as a normal run (synthesis_complete,
        exploration_complete) so every consumer — CLI result file, GitHub
        Action PR comment, web UI — receives a definitive, actionable
        NOT TESTED report instead of a generic crash.
        """
        logger.error(f"Preflight check failed: {reason}")
        event = {
            "type": "preflight_failed",
            "data": {
                "reason": reason,
                "timestamp": datetime.now().isoformat(),
            },
        }
        _log_event(event)
        yield event

        duration_seconds = (
            datetime.now() - self._shared_state.start_time
        ).total_seconds()
        report = SynthesisAgent(self.ai).generate_not_tested_report(
            target_url=target_url,
            duration=format_duration(duration_seconds),
            goal=self._shared_state.goal or "",
            reason=reason,
            interactive=self._interactive,
        )
        if self._shared_state.smoke:
            report = apply_smoke_tag(report)
        report, _ = self._mask_secrets(report, None)
        self._persist_report(report)
        event = {
            "type": "synthesis_complete",
            "data": {
                "report": report,
                "verdict": None,
                "timestamp": datetime.now().isoformat(),
            },
        }
        _log_event(event)
        yield event

        final_progress = await self._shared_state.get_progress()
        flow_tree = await self._shared_state.get_flow_tree()
        self._write_summary(final_progress, status="preflight_failed")
        event = {
            "type": "exploration_complete",
            "data": {
                "exploration_id": self._shared_state.exploration_id,
                "target_url": target_url,
                "issues_found": 0,
                "issues": [],
                "flows_explored": 0,
                "total_flows": 0,
                "flow_tree": flow_tree,
                "total_actions": 0,
                "duration_seconds": duration_seconds,
                "final_progress": final_progress,
                "synthesis_report": report,
                "preflight_failed": True,
            },
        }
        _log_event(event)
        yield event

    async def _build_salvage_events(self, target_url: str, error: str) -> list[dict]:
        """Build synthesis/completion events from state collected before a fatal error.

        Returns an empty list when nothing was explored yet (e.g. a startup
        failure before any worker ran) — in that case there is nothing to
        report and the fatal error event alone should surface the failure.
        """
        if not self._shared_state:
            return []

        all_issues = await self._shared_state.get_all_issues()
        if self._shared_state.total_actions == 0 and not all_issues:
            return []

        events = await self._finalize_rechecks()
        all_flows = await self._shared_state.get_all_flows()
        flow_tree = await self._shared_state.get_flow_tree()

        # generate_report falls back to a simple non-AI report internally if
        # the AI call fails, so this only raises on unexpected state errors.
        report = None
        verdict = None
        try:
            synthesis = SynthesisAgent(self.ai)
            report, verdict = await synthesis.generate_report_with_verdict(
                self._shared_state
            )
        except Exception as synth_error:
            logger.warning(f"Salvage synthesis failed: {synth_error}")

        if report:
            report = (
                "> **Warning:** QA Bot encountered an error during this run and "
                "exploration ended early. The results below may be incomplete.\n"
                ">\n"
                f"> `{error[:300]}`\n\n"
            ) + report
            report, verdict = self._mask_secrets(report, verdict)
            self._persist_report(report)
            # CLI captures the report from synthesis_complete
            events.append({
                "type": "synthesis_complete",
                "data": {
                    "report": report,
                    "verdict": verdict,
                    "timestamp": datetime.now().isoformat()
                }
            })

        # Read progress AFTER synthesis so the salvaged completion event
        # carries the synthesis call's token usage and cost (this feeds the
        # CLI's estimated_cost_usd and the action's cost-usd output),
        # matching the normal-completion path's ordering.
        final_progress = await self._shared_state.get_progress()

        # SSE consumers stop at exploration_complete, so embed the error here
        events.append({
            "type": "exploration_complete",
            "data": {
                "exploration_id": self._shared_state.exploration_id,
                "target_url": target_url,
                "error": error,
                "issues_found": len(all_issues),
                "issues": [issue.to_dict() for issue in all_issues],
                "flows_explored": _count_flows_explored(all_flows),
                "total_flows": len(all_flows),
                "flow_tree": flow_tree,
                "total_actions": self._shared_state.total_actions,
                "duration_seconds": (datetime.now() - self._shared_state.start_time).total_seconds(),
                "final_progress": final_progress,
                "synthesis_report": report,
                "recheck_summary": self._shared_state.recheck_summary(),
            }
        })
        return events

    async def _finalize_rechecks(self) -> list[dict]:
        """`recheck_result` events for re-checks settled at the end of the run."""
        return [
            {"type": "recheck_result", "flow_id": data.get("flow_id"), "data": data}
            for data in await self._shared_state.finalize_rechecks()
        ]

    def _mask_secrets(
        self, report: Optional[str], verdict: Optional[dict]
    ) -> tuple[Optional[str], Optional[dict]]:
        """Mask known secret values in the report and the verdict before
        ``synthesis_complete`` carries them out. The report becomes the PR
        comment, the action's ``report`` output and the CI log; the
        verdict's titles become the ``::error::`` line. Until now only
        report.md / summary.json were masked (by the chat logger), and a run
        can be handed a live API key as a credential."""
        secrets = self._shared_state.get_secret_values() if self._shared_state else ()
        return redact_values(report, secrets), redact_obj(verdict, secrets)

    def _persist_report(self, report: Optional[str]):
        """Persist the synthesis report to report.md (best-effort)."""
        if report and self._shared_state and self._shared_state.chat_logger:
            self._shared_state.chat_logger.write_report(report)

    def _write_summary(
        self,
        progress: dict,
        status: str = "completed",
        error: Optional[str] = None,
        issues: Optional[list[dict]] = None,
        verdict: Optional[dict] = None,
    ):
        """Write summary.json to the log directory for monitoring.

        ``verdict`` is the curated synthesis verdict (see
        ``synthesis.parse_verdict``); its critical count/titles are recorded
        next to the raw issue list so the two can be compared per run.
        """
        if not self._shared_state or not self._shared_state.chat_logger:
            return

        # The provider API's own reported charge (OpenRouter's usage.cost);
        # None for Claude, whose cost is only the token-priced estimate.
        charged_cost_usd = getattr(self.ai, "charged_cost_usd", None)
        summary = {
            "exploration_id": progress.get("exploration_id", ""),
            "url": progress.get("target_url", ""),
            "goal": self._shared_state.goal or "",
            "mode": "smoke" if self._shared_state.smoke else "full",
            "started": self._shared_state.start_time.isoformat(),
            "duration_seconds": round(progress.get("elapsed_seconds", 0), 1),
            "model": self._shared_state.model,
            "tokens": progress.get("token_breakdown", {}),
            "estimated_cost_usd": round(progress.get("cost_usd", 0), 4),
            "charged_cost_usd": (
                round(charged_cost_usd, 4)
                if isinstance(charged_cost_usd, (int, float)) else None
            ),
            "flows": {
                "completed": progress.get("flows", {}).get("completed", 0),
                "total": progress.get("flows", {}).get("total", 0),
            },
            "issues_found": progress.get("issues_found", 0),
            "issues": issues if issues is not None else [],
            # Curated (post-synthesis) criticals — what the deploy gate
            # fails on when present; None when no AI verdict was available
            # (the gate then falls back to the raw criticals in "issues").
            "curated_critical_count": verdict["critical_count"] if verdict else None,
            "curated_critical_titles": (
                [c["title"] for c in verdict["critical"]] if verdict else None
            ),
            # Run-over-run labelling from the verdict (None when no previous
            # report was supplied or the verdict carried no counts).
            "new_findings_count": verdict.get("new") if verdict else None,
            "recurring_findings_count": verdict.get("recurring") if verdict else None,
            # Independent re-check of criticals: per-issue outcome counts
            # (None when no critical was a candidate this run)
            "recheck": self._shared_state.recheck_summary(),
            "total_actions": progress.get("total_actions", 0),
            "status": status,
        }
        if error is not None:
            summary["error"] = error

        self._shared_state.chat_logger.write_summary(summary)
        self._last_summary_status = status

    async def _write_provisional_summary(self, status: str, error: Optional[str] = None):
        """Best-effort summary.json for a run that has not reached its outcome yet.

        Used for the "running" / "synthesizing" checkpoints and the
        "interrupted" record written from the finally block, so it must never
        raise — it runs on the cancellation path.
        """
        if not self._shared_state or not self._shared_state.chat_logger:
            return
        try:
            progress = await self._shared_state.get_progress()
            issues = await self._shared_state.get_all_issues()
            self._write_summary(
                progress,
                status=status,
                error=error,
                issues=[issue.to_dict() for issue in issues],
            )
        except Exception:
            logger.warning(f"Failed to write {status} summary.json", exc_info=True)

    async def _stop_supervisor(self):
        """Stop the supervisor and await its task (idempotent)."""
        if self._supervisor:
            self._supervisor.stop()
        if self._supervisor_task and not self._supervisor_task.done():
            self._supervisor_task.cancel()
            try:
                await asyncio.wait_for(self._supervisor_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        self._supervisor_task = None

    async def _cleanup(self):
        """Clean up all resources."""
        await self._stop_supervisor()
        self._supervisor = None

        # Cancel any remaining worker tasks (with timeout to prevent hanging)
        remaining = [t for t in self._active_worker_tasks if not t.done()]
        for task in remaining:
            task.cancel()
        if remaining:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*remaining, return_exceptions=True),
                    timeout=10.0
                )
            except asyncio.TimeoutError:
                logger.warning("Cleanup: workers did not exit after cancellation")

        self._active_worker_tasks.clear()

        # Stop browser pool (with timeout — Playwright cleanup can hang
        # if there are stuck browser operations)
        if self._browser_pool:
            try:
                await asyncio.wait_for(self._browser_pool.stop(), timeout=10.0)
            except asyncio.TimeoutError:
                logger.warning("Cleanup: browser pool stop timed out")
            except Exception as e:
                logger.warning(f"Cleanup: browser pool stop error: {e}")
            self._browser_pool = None

        # Close chat logger to prevent resource leaks
        if self._shared_state and self._shared_state.chat_logger:
            self._shared_state.chat_logger.close()

        # Release heavy data structures inside shared state so the GC can
        # reclaim memory without waiting for the full object graph to die.
        if self._shared_state:
            await self._shared_state.cleanup()

        # Force a GC cycle to collect any circular references promptly.
        gc.collect()

    def request_stop(self):
        """Request exploration to stop gracefully."""
        if self._shared_state:
            self._shared_state.stop_requested = True
            # If paused, wake up wait_for_resume() so coordinator can check stop_requested
            if self._shared_state.is_paused():
                self._shared_state._resume_event.set()
