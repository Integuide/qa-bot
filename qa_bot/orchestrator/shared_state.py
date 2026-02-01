"""Shared state for coordinating flow-based parallel exploration."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, TYPE_CHECKING
from urllib.parse import urlparse
import uuid

from qa_bot.agent.state import Issue
from qa_bot.config import calculate_cost
from qa_bot.orchestrator.flow import FlowTask, FlowCheckpoint, FlowPath, FlowStatus, FlowExplorationData


@dataclass
class UserDataField:
    """Definition of a single field in a user data request."""
    key: str              # Internal key for storage
    label: str            # Human-readable label for UI
    field_type: str = "text"  # "text", "password", "email", "tel"
    placeholder: str = ""
    required: bool = True
    description: str = ""


@dataclass
class UserDataRequest:
    """A request for user data from a worker."""
    request_id: str
    request_name: str         # AI-chosen name (e.g., "site_login", "email_code")
    description: str          # Why the data is needed
    fields: list[UserDataField]
    flow_id: str
    worker_id: str
    checkpoint_id: Optional[str] = None
    requested_at: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "request_id": self.request_id,
            "request_name": self.request_name,
            "description": self.description,
            "fields": [
                {
                    "key": f.key,
                    "label": f.label,
                    "type": f.field_type,
                    "placeholder": f.placeholder,
                    "required": f.required,
                    "description": f.description,
                }
                for f in self.fields
            ],
            "flow_id": self.flow_id,
            "worker_id": self.worker_id,
            "checkpoint_id": self.checkpoint_id,
            "requested_at": self.requested_at,
        }

if TYPE_CHECKING:
    from qa_bot.services.email_service import EmailService
    from qa_bot.utils.chat_logger import ChatLogger


def extract_domain(url: str) -> str:
    """Extract domain from a URL."""
    try:
        parsed = urlparse(url)
        return parsed.netloc or parsed.path.split('/')[0]
    except Exception:
        return url


def format_user_data_for_prompt(
    data: dict[str, str],
    fields: list[UserDataField] | None = None
) -> list[str]:
    """
    Format user data for inclusion in conversation history, masking sensitive values.

    Args:
        data: The key-value data dict
        fields: Optional field definitions to determine which fields are passwords

    Returns:
        List of formatted lines like "- key: value" with passwords masked
    """
    # Build a set of password field keys
    password_keys = set()
    if fields:
        for field in fields:
            if field.field_type == "password":
                password_keys.add(field.key)

    # Also use heuristics for keys that look sensitive (fallback if no fields provided)
    sensitive_patterns = (
        "password", "secret", "token", "api_key", "apikey", "credential",
        "auth", "bearer", "private_key", "privatekey",
    )

    lines = []
    for key, value in data.items():
        # Check if this is a password field or looks sensitive
        is_sensitive = key in password_keys or any(
            pattern in key.lower() for pattern in sensitive_patterns
        )

        if is_sensitive and value:
            # Mask the value, showing only length hint
            masked = f"[MASKED - {len(value)} chars]"
            lines.append(f"- {key}: {masked}")
        else:
            lines.append(f"- {key}: {value}")

    return lines


@dataclass
class SharedFlowState:
    """
    Thread-safe shared state for coordinating flow-based parallel exploration.

    All workers share this state to:
    - Claim flows for exploration
    - Create and claim checkpoints for branching
    - Report issues and track progress

    Key features:
    - Flows are discovered dynamically (not from link scanning)
    - Checkpoints preserve browser + conversation state for forking
    - Each flow is independent (isolated browser context)
    - Flow deduplication via supervisor skip marking
    """
    exploration_id: str
    target_url: str
    target_domain: str
    goal: str

    # Configuration
    max_branches_per_flow: int = 10  # Limit branching to prevent explosion
    max_duration_minutes: int = 30  # 30 minutes default
    max_cost_usd: float = 5.0  # Max cost limit in USD
    model: str = "claude-haiku-4-5"  # Model for cost calculation

    # State tracking
    start_time: datetime = field(default_factory=datetime.now)
    total_actions: int = 0
    # Token tracking by type for accurate cost calculation
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    stop_requested: bool = False

    @property
    def total_tokens_used(self) -> int:
        """Total tokens used (for backward compatibility)."""
        return self.input_tokens + self.output_tokens + self.cache_read_tokens + self.cache_creation_tokens

    @property
    def total_cost_usd(self) -> float:
        """Calculate total cost based on token types and model pricing."""
        return calculate_cost(
            model=self.model,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cache_read_tokens=self.cache_read_tokens,
            cache_creation_tokens=self.cache_creation_tokens,
        )

    # Flow management (protected by _lock)
    _flow_registry: dict[str, FlowExplorationData] = field(default_factory=dict)  # flow_id -> data
    _checkpoint_registry: dict[str, FlowCheckpoint] = field(default_factory=dict)  # checkpoint_id -> checkpoint
    _task_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    _pause_checkpoints: list[FlowTask] = field(default_factory=list)  # High-priority queue for pause-resumed flows

    # Supervisor deduplication
    _skip_set: set[str] = field(default_factory=set)  # flow_ids marked as skip
    _skip_reasons: dict[str, str] = field(default_factory=dict)  # flow_id -> reason

    # Results aggregation
    _all_issues: list[Issue] = field(default_factory=list)

    # Lock for thread-safe operations
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    # Worker tracking
    _active_workers: int = 0

    # Supervisor communication
    _blocked_workers: dict = field(default_factory=dict)  # worker_id -> {"flow_id": str, "reason": str}
    _worker_messages: dict = field(default_factory=dict)  # worker_id -> message
    _unblock_events: dict = field(default_factory=dict)  # worker_id -> asyncio.Event
    _unblock_responses: dict = field(default_factory=dict)  # worker_id -> response dict
    _supervisor_trigger: asyncio.Event = field(default_factory=asyncio.Event)

    # Credential management (legacy - kept for backwards compatibility)
    _credentials: dict = field(default_factory=dict)  # key -> value (e.g., USERNAME -> user@example.com)
    _pending_credential_requests: list = field(default_factory=list)  # Active credential requests from workers

    # User data management (new flexible system)
    _user_data: dict = field(default_factory=dict)  # request_name -> {key: value, ...}
    _pending_data_requests: list = field(default_factory=list)  # list[UserDataRequest]

    # Approval management for irreversible actions
    _pending_approval_requests: dict = field(default_factory=dict)  # flow_id -> {reason, action_description, worker_id, checkpoint_id}
    _approval_responses: dict = field(default_factory=dict)  # flow_id -> {approved: bool, always: bool}
    _always_approved_patterns: set = field(default_factory=set)  # Action patterns that user said "always approve"
    skip_permissions: bool = False  # If True, auto-approve all irreversible actions (dangerous!)

    # Chat logger (set by coordinator if logging enabled)
    chat_logger: Optional["ChatLogger"] = None

    # Browser monitoring aggregates
    _console_error_count: int = 0
    _network_failure_count: int = 0
    _dialog_count: int = 0

    # Email service (set by coordinator if Mailslurp configured)
    email_service: Optional["EmailService"] = None

    # Pause/resume state
    paused: bool = False
    paused_at: Optional[datetime] = None
    pause_reason: str = ""
    _resume_event: asyncio.Event = field(default_factory=asyncio.Event)

    @classmethod
    def create(
        cls,
        target_url: str,
        goal: str,
        max_branches_per_flow: int = 10,
        max_duration_minutes: int = 30,
        max_cost_usd: float = 5.0,
        model: str = "claude-haiku-4-5",
        skip_permissions: bool = False,
    ) -> "SharedFlowState":
        """Create a new shared flow state with initial root flow."""
        exploration_id = str(uuid.uuid4())
        target_domain = extract_domain(target_url)

        state = cls(
            exploration_id=exploration_id,
            target_url=target_url,
            target_domain=target_domain,
            goal=goal,
            max_branches_per_flow=max_branches_per_flow,
            max_duration_minutes=max_duration_minutes,
            max_cost_usd=max_cost_usd,
            model=model,
            skip_permissions=skip_permissions,
        )

        # Create and queue root flow
        root_flow = FlowTask.create_root(
            start_url=target_url,
            goal=goal,
            flow_name="Root"
        )

        # Initialize flow data for root
        state._flow_registry[root_flow.flow_id] = FlowExplorationData(
            flow_id=root_flow.flow_id,
            flow_name=root_flow.flow_name,
            flow_path=root_flow.flow_path,
            status=FlowStatus.PENDING,
            is_first_worker=root_flow.is_first_worker,  # Propagate from FlowTask
        )

        state._task_queue.put_nowait(root_flow)

        return state

    # -------------------------------------------------------------------------
    # Flow Task Management
    # -------------------------------------------------------------------------

    async def claim_pending_flow(self) -> Optional[FlowTask]:
        """
        Claim the next pending flow task (non-blocking).

        Used by coordinator to get flows for spawning workers.
        Prioritizes pause checkpoints (from interrupted workers) over regular queue.
        Returns None if no pending flows available.
        """
        async with self._lock:
            # First check for pause checkpoints (highest priority)
            if self._pause_checkpoints:
                task = self._pause_checkpoints.pop(0)
                if task.flow_id in self._flow_registry:
                    self._flow_registry[task.flow_id].status = FlowStatus.PENDING
                return task

            # Then check regular queue
            try:
                task = self._task_queue.get_nowait()
                return task
            except asyncio.QueueEmpty:
                return None

    async def return_flow_to_pending(self, task: FlowTask):
        """
        Return a flow task to the pending queue.

        Used when a worker slot isn't available and we need to
        put the flow back for later processing.
        """
        await self._task_queue.put(task)

    def has_pending_flows(self) -> bool:
        """Check if there are any pending flows in the queue or pause checkpoints."""
        return bool(self._pause_checkpoints) or not self._task_queue.empty()

    # -------------------------------------------------------------------------
    # Checkpoint Management
    # -------------------------------------------------------------------------

    async def add_checkpoint(
        self,
        checkpoint: FlowCheckpoint,
        goal: str,
        queue_task: bool = True
    ) -> Optional[FlowTask | FlowCheckpoint]:
        """
        Add a checkpoint and optionally create a flow task for it.

        Called by workers when they discover a branch point and want to
        save alternate paths for other workers.

        Args:
            checkpoint: The checkpoint to save
            goal: The exploration goal
            queue_task: If True (default), create and queue a flow task.
                       If False, only save the checkpoint (used for credential/approval
                       resumption where the task is created later by provide_credentials/approval).

        Returns:
            - FlowTask if checkpoint saved and task queued
            - FlowCheckpoint if checkpoint saved but not queued (queue_task=False)
            - None if checkpoint was rejected (duplicate, branch limit, etc.)
        """
        async with self._lock:
            # Check if we already have this checkpoint
            if checkpoint.checkpoint_id in self._checkpoint_registry:
                return None

            # Check for duplicate flow name (case-insensitive)
            normalized_name = checkpoint.branch_name.lower().strip()
            for flow_data in self._flow_registry.values():
                if flow_data.flow_name.lower().strip() == normalized_name:
                    # Flow with this name already exists - skip duplicate
                    return None

            # Check branch limit (prevent infinite branching)
            parent_branches = sum(
                1 for cp in self._checkpoint_registry.values()
                if cp.parent_flow_id == checkpoint.parent_flow_id
            )
            if parent_branches >= self.max_branches_per_flow:
                return None

            # Save checkpoint
            self._checkpoint_registry[checkpoint.checkpoint_id] = checkpoint

            # If not queueing (credential/approval resume), just save the checkpoint
            # Return checkpoint to indicate it was saved (task will be created later)
            if not queue_task:
                return checkpoint

            # Create flow task from checkpoint
            flow_task = FlowTask.create_from_checkpoint(checkpoint, goal)

            # Initialize flow data
            self._flow_registry[flow_task.flow_id] = FlowExplorationData(
                flow_id=flow_task.flow_id,
                flow_name=flow_task.flow_name,
                flow_path=flow_task.flow_path,
                status=FlowStatus.PENDING,
                parent_flow_id=checkpoint.parent_flow_id
            )

            # Track as child of parent flow
            if checkpoint.parent_flow_id in self._flow_registry:
                self._flow_registry[checkpoint.parent_flow_id].child_flow_ids.append(flow_task.flow_id)

            # Queue the task
            await self._task_queue.put(flow_task)

            return flow_task

    async def get_checkpoint(self, checkpoint_id: str) -> Optional[FlowCheckpoint]:
        """Get a checkpoint by ID."""
        async with self._lock:
            return self._checkpoint_registry.get(checkpoint_id)

    async def claim_checkpoint(self, checkpoint_id: str, worker_id: str) -> bool:
        """Mark a checkpoint as claimed by a worker."""
        async with self._lock:
            if checkpoint_id not in self._checkpoint_registry:
                return False

            checkpoint = self._checkpoint_registry[checkpoint_id]
            if checkpoint.claimed:
                return False

            checkpoint.claimed = True
            checkpoint.claimed_by = worker_id
            checkpoint.claimed_at = datetime.now()
            return True

    async def add_pause_checkpoint(self, checkpoint: "FlowCheckpoint", goal: str) -> Optional["FlowTask"]:
        """Add checkpoint created during pause. Gets highest priority (150) on resume."""
        async with self._lock:
            self._checkpoint_registry[checkpoint.checkpoint_id] = checkpoint
            flow_task = FlowTask.create_from_checkpoint(checkpoint, goal)
            flow_task.priority = 150  # Higher than root (100)

            # Propagate is_first_worker from parent flow
            parent_is_first_worker = False
            if checkpoint.parent_flow_id and checkpoint.parent_flow_id in self._flow_registry:
                parent_is_first_worker = self._flow_registry[checkpoint.parent_flow_id].is_first_worker
            flow_task.is_first_worker = parent_is_first_worker

            self._flow_registry[flow_task.flow_id] = FlowExplorationData(
                flow_id=flow_task.flow_id,
                flow_name=flow_task.flow_name,
                flow_path=flow_task.flow_path,
                status=FlowStatus.BLOCKED_FOR_PAUSE,
                parent_flow_id=checkpoint.parent_flow_id,
                is_first_worker=parent_is_first_worker  # Propagate first worker flag
            )
            self._pause_checkpoints.append(flow_task)
            return flow_task

    async def block_flow_for_pause(self, flow_id: str, reason: str):
        """Mark a flow as blocked waiting for resume.

        Also clears the worker_id since the worker is no longer exploring this flow.
        This prevents duplicate worker displays when the flow resumes with a new flow_id.
        """
        async with self._lock:
            if flow_id in self._flow_registry:
                self._flow_registry[flow_id].status = FlowStatus.BLOCKED_FOR_PAUSE
                self._flow_registry[flow_id].completion_reason = f"Paused: {reason}"
                self._flow_registry[flow_id].worker_id = None  # Worker no longer on this flow

    def get_pause_checkpoint_count(self) -> int:
        """Get the number of pause checkpoints waiting for resume."""
        return len(self._pause_checkpoints)

    def get_pause_checkpoint_info(self) -> list[dict]:
        """Get info about pause checkpoints for event emission."""
        result = []
        for task in self._pause_checkpoints:
            checkpoint = self._checkpoint_registry.get(task.checkpoint_id)
            if checkpoint:
                result.append({
                    "flow_id": task.flow_id,
                    "flow_name": task.flow_name,
                    "checkpoint_id": task.checkpoint_id,
                    "url": checkpoint.current_url,
                    "created_by_worker": checkpoint.created_by_worker,
                    "parent_flow_id": checkpoint.parent_flow_id  # Original flow being paused
                })
        return result

    # -------------------------------------------------------------------------
    # Flow Completion
    # -------------------------------------------------------------------------

    async def start_flow_exploration(
        self,
        flow_task: FlowTask,
        worker_id: str
    ) -> FlowExplorationData:
        """
        Mark a flow as being explored and return its data.

        Called by worker when it starts exploring a flow.
        """
        async with self._lock:
            if flow_task.flow_id not in self._flow_registry:
                # Create if not exists (shouldn't happen normally)
                self._flow_registry[flow_task.flow_id] = FlowExplorationData(
                    flow_id=flow_task.flow_id,
                    flow_name=flow_task.flow_name,
                    flow_path=flow_task.flow_path,
                    parent_flow_id=flow_task.parent_flow_id
                )

            flow_data = self._flow_registry[flow_task.flow_id]
            flow_data.status = FlowStatus.EXPLORING
            flow_data.worker_id = worker_id
            flow_data.started_at = datetime.now()

            return flow_data

    async def complete_flow(
        self,
        flow_id: str,
        completion_reason: str
    ):
        """Mark a flow as complete."""
        async with self._lock:
            if flow_id in self._flow_registry:
                flow_data = self._flow_registry[flow_id]
                flow_data.status = FlowStatus.COMPLETED
                flow_data.completed_at = datetime.now()
                flow_data.completion_reason = completion_reason

    async def fail_flow(self, flow_id: str, error: str = ""):
        """Mark a flow as failed."""
        async with self._lock:
            if flow_id in self._flow_registry:
                flow_data = self._flow_registry[flow_id]
                flow_data.status = FlowStatus.FAILED
                flow_data.completed_at = datetime.now()
                flow_data.completion_reason = f"Failed: {error}"

    # -------------------------------------------------------------------------
    # Recording Events
    # -------------------------------------------------------------------------

    async def record_action_for_flow(self, flow_id: str, action: dict):
        """Record an action taken during a flow."""
        async with self._lock:
            if flow_id in self._flow_registry:
                self._flow_registry[flow_id].actions.append(action)

    async def record_issue_for_flow(self, flow_id: str, issue: dict):
        """Record an issue found during a flow."""
        async with self._lock:
            if flow_id in self._flow_registry:
                self._flow_registry[flow_id].issues.append(issue)

    async def record_thinking_for_flow(self, flow_id: str, thinking: str):
        """Record AI thinking for a flow."""
        async with self._lock:
            if flow_id in self._flow_registry:
                self._flow_registry[flow_id].thinking_history.append(thinking)

    async def record_url_visited(self, flow_id: str, url: str):
        """Record a URL visited during a flow."""
        async with self._lock:
            if flow_id in self._flow_registry:
                if url not in self._flow_registry[flow_id].urls_visited:
                    self._flow_registry[flow_id].urls_visited.append(url)

    async def add_issue(self, issue: Issue) -> bool:
        """Add an issue (with deduplication)."""
        async with self._lock:
            for existing in self._all_issues:
                if (existing.description == issue.description and
                    existing.url == issue.url):
                    return False
            self._all_issues.append(issue)
            return True

    async def increment_actions(self, count: int = 1):
        """Increment total action counter."""
        async with self._lock:
            self.total_actions += count

    # -------------------------------------------------------------------------
    # Supervisor Deduplication
    # -------------------------------------------------------------------------

    async def mark_flow_skip(self, flow_id: str, reason: str) -> bool:
        """
        Mark a flow as skipped (duplicate) by supervisor.

        Returns True if successful.
        """
        async with self._lock:
            if flow_id not in self._flow_registry:
                return False

            self._skip_set.add(flow_id)
            self._skip_reasons[flow_id] = reason

            flow_data = self._flow_registry[flow_id]
            flow_data.status = FlowStatus.SKIPPED
            flow_data.skip_reason = reason

            return True

    async def is_flow_skipped(self, flow_id: str) -> bool:
        """Check if a flow has been marked as skip."""
        async with self._lock:
            return flow_id in self._skip_set

    async def is_flow_skipped_by_name(self, flow_name: str) -> bool:
        """
        Check if a flow with similar name has been marked as skip.

        Used for early deduplication before creating checkpoints.
        """
        async with self._lock:
            # Simple name matching - could be made smarter
            for flow_id in self._skip_set:
                if flow_id in self._flow_registry:
                    if self._flow_registry[flow_id].flow_name.lower() == flow_name.lower():
                        return True
            return False

    async def flow_exists_by_name(self, flow_name: str) -> bool:
        """
        Check if a flow with similar name already exists (pending, exploring, or completed).

        Used to prevent duplicate flow creation.
        """
        async with self._lock:
            normalized_name = flow_name.lower().strip()
            for flow_data in self._flow_registry.values():
                if flow_data.flow_name.lower().strip() == normalized_name:
                    return True
            return False

    async def get_pending_flows(self) -> list[FlowExplorationData]:
        """Get all pending flows (for supervisor review)."""
        async with self._lock:
            return [
                data for data in self._flow_registry.values()
                if data.status == FlowStatus.PENDING
            ]

    async def get_all_flows(self) -> list[FlowExplorationData]:
        """Get all flows."""
        async with self._lock:
            return list(self._flow_registry.values())

    # -------------------------------------------------------------------------
    # Worker Management
    # -------------------------------------------------------------------------

    async def increment_active_workers(self):
        """Increment active worker count when a worker starts."""
        async with self._lock:
            self._active_workers += 1

    async def decrement_active_workers(self):
        """Decrement active worker count when a worker stops."""
        async with self._lock:
            self._active_workers -= 1

    # -------------------------------------------------------------------------
    # State Queries
    # -------------------------------------------------------------------------

    # Reserve 10% of cost budget for synthesis report generation
    SYNTHESIS_COST_RESERVE_RATIO = 0.10

    def should_stop(self) -> bool:
        """Check if exploration should stop due to any limit.

        When using cost limits, reserves a portion of the budget for the synthesis
        agent to generate the final report. This ensures we don't exhaust the budget
        during exploration and have nothing left for report generation.
        """
        if self.stop_requested:
            return True

        # Check duration limit
        elapsed_minutes = (datetime.now() - self.start_time).total_seconds() / 60
        if elapsed_minutes >= self.max_duration_minutes:
            return True

        # Check cost limit
        # Reserve some budget for synthesis report generation
        exploration_budget = self.max_cost_usd * (1 - self.SYNTHESIS_COST_RESERVE_RATIO)
        if self.total_cost_usd >= exploration_budget:
            return True

        return False

    def get_stop_reason(self) -> str:
        """Get the reason for stopping exploration.

        Returns a human-readable string describing why exploration stopped.
        Should be called when should_stop() returns True.
        """
        if self.stop_requested:
            return "Stop requested"

        # Check cost limit
        exploration_budget = self.max_cost_usd * (1 - self.SYNTHESIS_COST_RESERVE_RATIO)
        if self.total_cost_usd >= exploration_budget:
            return f"Cost limit reached (${self.total_cost_usd:.2f} of ${self.max_cost_usd:.2f} budget)"

        elapsed_minutes = (datetime.now() - self.start_time).total_seconds() / 60
        if elapsed_minutes >= self.max_duration_minutes:
            return "Time limit reached"

        return "Unknown"

    # -------------------------------------------------------------------------
    # Pause/Resume for Limit Continuation
    # -------------------------------------------------------------------------

    async def pause(self, reason: str):
        """
        Pause exploration when a limit is reached.

        Called by coordinator when cost or time limit is hit.
        Sets paused state and clears the resume event so wait_for_resume blocks.
        """
        async with self._lock:
            self.paused = True
            self.paused_at = datetime.now()
            self.pause_reason = reason
            self._resume_event.clear()

    async def resume(
        self,
        new_max_cost_usd: Optional[float] = None,
        new_max_duration_minutes: Optional[int] = None
    ):
        """
        Resume exploration after user increases limits.

        Updates the limits and signals the coordinator to continue.
        """
        async with self._lock:
            if new_max_cost_usd is not None:
                self.max_cost_usd = new_max_cost_usd
            if new_max_duration_minutes is not None:
                self.max_duration_minutes = new_max_duration_minutes

            self.paused = False
            self.paused_at = None
            self.pause_reason = ""
            self._resume_event.set()

    def is_paused(self) -> bool:
        """Check if exploration is currently paused."""
        return self.paused

    def is_pause_cancellation(self) -> bool:
        """Check if current stop is for pause (should checkpoint) vs user stop (no checkpoint).

        Returns True if workers should save checkpoints before exiting.
        Returns False if user explicitly requested stop (no checkpoint needed).
        """
        if self.paused:
            return True
        if not self.should_stop():
            return False
        return not self.stop_requested

    async def wait_for_resume(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for resume signal.

        Returns True if resumed, False if timed out.
        Timeout of None means wait indefinitely.
        """
        try:
            if timeout is not None:
                await asyncio.wait_for(self._resume_event.wait(), timeout=timeout)
            else:
                await self._resume_event.wait()
            return True
        except asyncio.TimeoutError:
            return False

    async def add_token_usage(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_creation_tokens: int = 0,
    ):
        """Add token usage by type for accurate cost calculation."""
        async with self._lock:
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.cache_read_tokens += cache_read_tokens
            self.cache_creation_tokens += cache_creation_tokens

    # -------------------------------------------------------------------------
    # Browser Monitoring Aggregates
    # -------------------------------------------------------------------------

    async def record_console_error(self):
        """Record a console error for aggregate tracking."""
        async with self._lock:
            self._console_error_count += 1

    async def record_network_failure(self):
        """Record a network failure for aggregate tracking."""
        async with self._lock:
            self._network_failure_count += 1

    async def record_dialog(self):
        """Record a dialog event."""
        async with self._lock:
            self._dialog_count += 1

    def is_complete(self) -> bool:
        """Check if exploration is complete (no active workers, no pending flows, no blocked flows)."""
        if not self._task_queue.empty() or self._active_workers > 0:
            return False
        # Don't complete if there are pause checkpoints waiting
        if self._pause_checkpoints:
            return False
        # Don't complete if there are flows blocked waiting for credentials, approval, or pause
        blocked_flows = sum(
            1 for f in self._flow_registry.values()
            if f.status in (FlowStatus.BLOCKED_FOR_CREDENTIALS, FlowStatus.BLOCKED_FOR_APPROVAL, FlowStatus.BLOCKED_FOR_PAUSE)
        )
        return blocked_flows == 0

    async def get_progress(self) -> dict:
        """Get current progress snapshot."""
        async with self._lock:
            completed = sum(1 for f in self._flow_registry.values() if f.status == FlowStatus.COMPLETED)
            pending = sum(1 for f in self._flow_registry.values() if f.status == FlowStatus.PENDING)
            exploring = sum(1 for f in self._flow_registry.values() if f.status == FlowStatus.EXPLORING)
            failed = sum(1 for f in self._flow_registry.values() if f.status == FlowStatus.FAILED)
            skipped = sum(1 for f in self._flow_registry.values() if f.status == FlowStatus.SKIPPED)
            blocked_for_credentials = sum(1 for f in self._flow_registry.values() if f.status == FlowStatus.BLOCKED_FOR_CREDENTIALS)
            blocked_for_approval = sum(1 for f in self._flow_registry.values() if f.status == FlowStatus.BLOCKED_FOR_APPROVAL)
            blocked_for_pause = sum(1 for f in self._flow_registry.values() if f.status == FlowStatus.BLOCKED_FOR_PAUSE)

            # Calculate exploration budget (reserves portion for synthesis)
            exploration_budget = None
            if self.max_cost_usd is not None:
                exploration_budget = self.max_cost_usd * (1 - self.SYNTHESIS_COST_RESERVE_RATIO)

            return {
                "exploration_id": self.exploration_id,
                "target_url": self.target_url,
                "elapsed_seconds": (datetime.now() - self.start_time).total_seconds(),
                "total_actions": self.total_actions,
                "tokens_used": self.total_tokens_used,
                "cost_usd": self.total_cost_usd,
                "max_cost_usd": self.max_cost_usd,
                "exploration_budget_usd": exploration_budget,  # Budget minus synthesis reserve
                "token_breakdown": {
                    "input": self.input_tokens,
                    "output": self.output_tokens,
                    "cache_read": self.cache_read_tokens,
                    "cache_creation": self.cache_creation_tokens,
                },
                "flows": {
                    "total": len(self._flow_registry),
                    "completed": completed,
                    "pending": pending,
                    "exploring": exploring,
                    "failed": failed,
                    "skipped": skipped,
                    "blocked_for_credentials": blocked_for_credentials,
                    "blocked_for_approval": blocked_for_approval,
                    "blocked_for_pause": blocked_for_pause
                },
                "checkpoints": len(self._checkpoint_registry),
                "issues_found": len(self._all_issues),
                "active_workers": self._active_workers,
                "queue_depth": self._task_queue.qsize(),
                "monitoring": {
                    "console_errors": self._console_error_count,
                    "network_failures": self._network_failure_count,
                    "dialogs_handled": self._dialog_count
                },
                "paused": self.paused,
                "paused_at": self.paused_at.isoformat() if self.paused_at else None,
                "pause_reason": self.pause_reason,
                "max_duration_minutes": self.max_duration_minutes
            }

    async def get_all_issues(self) -> list[Issue]:
        """Get all discovered issues."""
        async with self._lock:
            return list(self._all_issues)

    async def get_flow_data(self, flow_id: str) -> Optional[FlowExplorationData]:
        """Get exploration data for a specific flow."""
        async with self._lock:
            return self._flow_registry.get(flow_id)

    async def get_flow_tree(self) -> list[dict]:
        """
        Get the flow tree structure for UI visualization.

        Returns a list of root flows, each with nested children.
        """
        async with self._lock:
            def build_tree(flow_id: str) -> dict:
                flow_data = self._flow_registry.get(flow_id)
                if not flow_data:
                    return {}

                children = [
                    build_tree(child_id)
                    for child_id in flow_data.child_flow_ids
                ]

                return {
                    "flow_id": flow_data.flow_id,
                    "flow_name": flow_data.flow_name,
                    "flow_path": flow_data.flow_path.to_list(),
                    "status": flow_data.status.value,
                    "worker_id": flow_data.worker_id,
                    "action_count": len(flow_data.actions),
                    "issue_count": len(flow_data.issues),
                    "children": children
                }

            # Find root flows (no parent)
            roots = [
                flow_id for flow_id, data in self._flow_registry.items()
                if data.parent_flow_id is None
            ]

            return [build_tree(root_id) for root_id in roots]

    # -------------------------------------------------------------------------
    # Supervisor-Worker Communication
    # -------------------------------------------------------------------------

    async def block_worker(self, worker_id: str, flow_id: str, reason: str):
        """Mark a worker as blocked awaiting supervisor help."""
        async with self._lock:
            self._blocked_workers[worker_id] = {
                "flow_id": flow_id,
                "reason": reason,
                "blocked_at": datetime.now().isoformat()
            }
            # Create event for this worker to wait on
            self._unblock_events[worker_id] = asyncio.Event()
            # Trigger supervisor
            self._supervisor_trigger.set()

    async def wait_for_unblock(self, worker_id: str, timeout: float = 300) -> Optional[dict]:
        """Wait for supervisor to unblock this worker. Returns response or None on timeout."""
        event = self._unblock_events.get(worker_id)
        if not event:
            return None

        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
            response = self._unblock_responses.pop(worker_id, None)
            self._unblock_events.pop(worker_id, None)
            return response
        except asyncio.TimeoutError:
            return None

    async def unblock_worker(self, worker_id: str, response: dict):
        """Supervisor unblocks a worker with a response."""
        async with self._lock:
            self._blocked_workers.pop(worker_id, None)
            self._unblock_responses[worker_id] = response
            event = self._unblock_events.get(worker_id)
            if event:
                event.set()

    async def send_message_to_worker(self, worker_id: str, message: str):
        """Supervisor sends a message to a specific worker."""
        async with self._lock:
            self._worker_messages[worker_id] = message

    async def get_worker_message(self, worker_id: str) -> Optional[str]:
        """Worker checks for supervisor message (consumes it)."""
        async with self._lock:
            return self._worker_messages.pop(worker_id, None)

    async def stop_worker(self, worker_id: str, reason: str):
        """Supervisor requests a worker to stop."""
        await self.unblock_worker(worker_id, {"action": "stop", "reason": reason})

    async def get_blocked_workers(self) -> list[dict]:
        """Get list of blocked workers for supervisor review."""
        async with self._lock:
            return [
                {"worker_id": wid, **info}
                for wid, info in self._blocked_workers.items()
            ]

    async def wait_for_supervisor_trigger(self, timeout: float = 5.0) -> bool:
        """Wait for supervisor trigger event."""
        try:
            await asyncio.wait_for(self._supervisor_trigger.wait(), timeout=timeout)
            self._supervisor_trigger.clear()
            return True
        except asyncio.TimeoutError:
            return False

    def trigger_supervisor(self):
        """Trigger supervisor to run (called on flow status changes)."""
        self._supervisor_trigger.set()

    # -------------------------------------------------------------------------
    # Add Flow Task (without checkpoint)
    # -------------------------------------------------------------------------

    async def add_flow_task(
        self,
        flow_name: str,
        start_url: str,
        goal: str,
        parent_flow_id: Optional[str] = None
    ) -> Optional[FlowTask]:
        """
        Add a new flow task without checkpoint (fresh browser state).

        Returns None if a flow with the same name already exists (duplicate prevention).
        """
        async with self._lock:
            # Check for duplicate flow name (case-insensitive)
            normalized_name = flow_name.lower().strip()
            for flow_data in self._flow_registry.values():
                if flow_data.flow_name.lower().strip() == normalized_name:
                    # Flow with this name already exists - skip duplicate
                    return None

            # Create flow path
            if parent_flow_id and parent_flow_id in self._flow_registry:
                parent_data = self._flow_registry[parent_flow_id]
                flow_path = parent_data.flow_path.extend(flow_name)
            else:
                flow_path = FlowPath([flow_name])

            # Create flow task
            flow_id = str(uuid.uuid4())
            flow_task = FlowTask(
                flow_id=flow_id,
                flow_name=flow_name,
                flow_path=flow_path,
                start_url=start_url,
                goal=goal,
                is_root=False,
                parent_flow_id=parent_flow_id,
                checkpoint_id=None
            )

            # Initialize flow data
            self._flow_registry[flow_id] = FlowExplorationData(
                flow_id=flow_id,
                flow_name=flow_name,
                flow_path=flow_path,
                status=FlowStatus.PENDING,
                parent_flow_id=parent_flow_id
            )

            # Track as child of parent
            if parent_flow_id and parent_flow_id in self._flow_registry:
                self._flow_registry[parent_flow_id].child_flow_ids.append(flow_id)

            # Queue the task
            await self._task_queue.put(flow_task)

            # Trigger supervisor
            self._supervisor_trigger.set()

            return flow_task

    # -------------------------------------------------------------------------
    # Credential Management
    # -------------------------------------------------------------------------

    async def set_credentials(self, credentials: dict[str, str]):
        """
        Store credentials for use during exploration.

        Can be called at start (pre-provided) or mid-exploration (user-provided).
        Merges with existing credentials (new values override old).
        """
        async with self._lock:
            self._credentials.update(credentials)

    async def get_credentials(self) -> dict[str, str]:
        """Get all stored credentials."""
        async with self._lock:
            return dict(self._credentials)

    async def get_credential(self, key: str) -> Optional[str]:
        """Get a single credential value by key."""
        async with self._lock:
            return self._credentials.get(key)

    async def has_credentials(self) -> bool:
        """Check if any credentials are available."""
        async with self._lock:
            return len(self._credentials) > 0

    async def request_credentials(
        self,
        worker_id: str,
        flow_id: str,
        needed: list[str],
        reason: str,
        checkpoint_id: Optional[str] = None
    ):
        """
        Record a credential request from a worker.

        This is non-blocking - the supervisor will emit an event to the UI
        and immediately unblock the worker to continue exploring.
        When credentials are provided, a new flow will be created.
        """
        async with self._lock:
            request = {
                "worker_id": worker_id,
                "flow_id": flow_id,
                "needed": needed,
                "reason": reason,
                "checkpoint_id": checkpoint_id,
                "requested_at": datetime.now().isoformat()
            }
            self._pending_credential_requests.append(request)

    async def get_pending_credential_requests(self) -> list[dict]:
        """Get all pending credential requests."""
        async with self._lock:
            return list(self._pending_credential_requests)

    async def clear_credential_requests(self):
        """Clear all pending credential requests (after credentials provided)."""
        async with self._lock:
            self._pending_credential_requests.clear()

    async def provide_credentials(self, credentials: dict[str, str]) -> Optional[dict]:
        """
        User provides credentials via UI/API.

        Stores credentials and creates a retry flow if there were pending requests.
        Returns dict with new flow info if created, or None.
        """
        async with self._lock:
            # Store the credentials
            self._credentials.update(credentials)

            # If there were pending requests, create a retry flow
            if self._pending_credential_requests:
                # Get the first pending request for context
                request = self._pending_credential_requests[0]
                original_flow_id = request.get("flow_id")
                checkpoint_id = request.get("checkpoint_id")

                # Clear pending requests
                self._pending_credential_requests.clear()

                # Get original flow data for better naming and first_worker flag
                original_flow_name = ""
                original_flow_path = FlowPath()
                original_is_first_worker = False
                if original_flow_id and original_flow_id in self._flow_registry:
                    original_flow_data = self._flow_registry[original_flow_id]
                    original_flow_name = original_flow_data.flow_name
                    original_flow_path = original_flow_data.flow_path
                    original_is_first_worker = original_flow_data.is_first_worker

                # Create a new flow to retry with credentials
                flow_id = str(uuid.uuid4())
                flow_name = f"Resume: {original_flow_name}" if original_flow_name else "Resume with credentials"
                flow_path = original_flow_path if original_flow_path.steps else FlowPath([flow_name])

                flow_task = FlowTask(
                    flow_id=flow_id,
                    flow_name=flow_name,
                    flow_path=flow_path,
                    start_url=self.target_url,
                    goal=f"Continue exploring with provided credentials. Original reason for blocking: {request.get('reason', 'credentials needed')}",
                    is_root=False,
                    parent_flow_id=original_flow_id,
                    checkpoint_id=checkpoint_id,
                    is_first_worker=original_is_first_worker,  # Propagate first worker flag
                )

                # Initialize flow data
                self._flow_registry[flow_id] = FlowExplorationData(
                    flow_id=flow_id,
                    flow_name=flow_name,
                    flow_path=flow_path,
                    status=FlowStatus.PENDING,
                    parent_flow_id=original_flow_id,
                    is_first_worker=original_is_first_worker,  # Propagate first worker flag
                )

                # Mark original blocked flow as resumed (continuing via new child flow)
                if original_flow_id and original_flow_id in self._flow_registry:
                    original_flow = self._flow_registry[original_flow_id]
                    if original_flow.status == FlowStatus.BLOCKED_FOR_CREDENTIALS:
                        original_flow.status = FlowStatus.RESUMED
                        original_flow.completion_reason = f"Resumed via {flow_name} after credentials provided"
                        original_flow.completed_at = datetime.now()

                # Queue the task
                await self._task_queue.put(flow_task)

                # Trigger supervisor
                self._supervisor_trigger.set()

                return {
                    "flow_id": flow_id,
                    "flow_name": flow_name,
                    "flow_path": [str(s) for s in flow_path.steps] if flow_path.steps else [flow_name],
                    "parent_flow_id": original_flow_id,
                    "status": "pending"
                }

            return None

    async def set_credential_request_checkpoint(
        self,
        flow_id: str,
        checkpoint_id: str
    ):
        """
        Link a checkpoint ID to the most recent credential request for a flow.

        Called by worker after creating a checkpoint when told to checkpoint_and_exit.
        This allows provide_credentials() to later create a flow that resumes from this checkpoint.
        """
        async with self._lock:
            # Find and update the credential request for this flow
            for request in self._pending_credential_requests:
                if request.get("flow_id") == flow_id:
                    request["checkpoint_id"] = checkpoint_id
                    break

    async def block_flow_for_credentials(self, flow_id: str, reason: str):
        """
        Mark a flow as blocked waiting for credentials.

        The flow remains in a special BLOCKED_FOR_CREDENTIALS status until
        credentials are provided and a new flow is created to resume.
        Also clears the worker_id since the worker is no longer exploring this flow.
        """
        async with self._lock:
            if flow_id in self._flow_registry:
                flow_data = self._flow_registry[flow_id]
                flow_data.status = FlowStatus.BLOCKED_FOR_CREDENTIALS
                flow_data.completion_reason = f"Blocked for credentials: {reason}"

                # Decrement active worker count and clear worker_id
                if flow_data.worker_id:
                    self._active_workers = max(0, self._active_workers - 1)
                    flow_data.worker_id = None  # Worker no longer on this flow

    # -------------------------------------------------------------------------
    # User Data Management (new flexible system)
    # -------------------------------------------------------------------------

    async def add_data_request(self, request: "UserDataRequest") -> bool:
        """
        Add a user data request from a worker.

        The request will be emitted to the UI for the user to fill in.
        Returns True if this is a new request_name, False if a request with
        the same name already exists (the worker's request is still added
        so it can be resumed when data is provided).
        """
        async with self._lock:
            # Check if there's already a pending request with this name
            # This helps the supervisor know whether to emit an event to the UI
            is_first_request = not any(
                r.request_name == request.request_name
                for r in self._pending_data_requests
            )

            # Always add the request so this worker can be resumed
            self._pending_data_requests.append(request)
            # Trigger supervisor to emit the event
            self._supervisor_trigger.set()
            return is_first_request

    async def get_pending_data_requests(self) -> list["UserDataRequest"]:
        """Get all pending data requests."""
        async with self._lock:
            return list(self._pending_data_requests)

    async def get_data_request_by_worker(self, worker_id: str) -> Optional["UserDataRequest"]:
        """Get the pending data request for a specific worker, if any."""
        async with self._lock:
            for request in self._pending_data_requests:
                if request.worker_id == worker_id:
                    return request
            return None

    async def get_data_request_by_name(self, request_name: str) -> Optional["UserDataRequest"]:
        """Get a pending data request by its name."""
        async with self._lock:
            for request in self._pending_data_requests:
                if request.request_name == request_name:
                    return request
            return None

    async def provide_user_data(self, request_name: str, data: dict[str, str]) -> list[dict]:
        """
        User provides data for a request.

        Stores the data and creates resume flows for all matching requests.
        Returns list of new flow info dicts created.
        """
        async with self._lock:
            # Store the data
            self._user_data[request_name] = data

            # Find all pending requests with this name and create resume flows
            matching_requests = [
                r for r in self._pending_data_requests
                if r.request_name == request_name
            ]

            new_flows = []
            for request in matching_requests:
                # Remove from pending
                self._pending_data_requests.remove(request)

                # Get original flow data for naming and first_worker flag
                original_flow_id = request.flow_id
                original_flow_name = ""
                original_flow_path = FlowPath()
                original_is_first_worker = False
                if original_flow_id in self._flow_registry:
                    original_flow_data = self._flow_registry[original_flow_id]
                    original_flow_name = original_flow_data.flow_name
                    original_flow_path = original_flow_data.flow_path
                    original_is_first_worker = original_flow_data.is_first_worker

                # Create a new flow to resume with data
                flow_id = str(uuid.uuid4())
                flow_name = f"Resume: {original_flow_name}" if original_flow_name else f"Resume with {request_name}"
                flow_path = original_flow_path if original_flow_path.steps else FlowPath([flow_name])

                flow_task = FlowTask(
                    flow_id=flow_id,
                    flow_name=flow_name,
                    flow_path=flow_path,
                    start_url=self.target_url,
                    goal=f"Continue exploring with provided data. Original reason: {request.description}",
                    is_root=False,
                    parent_flow_id=original_flow_id,
                    checkpoint_id=request.checkpoint_id,
                    is_first_worker=original_is_first_worker,  # Propagate first worker flag
                )

                # Initialize flow data
                self._flow_registry[flow_id] = FlowExplorationData(
                    flow_id=flow_id,
                    flow_name=flow_name,
                    flow_path=flow_path,
                    status=FlowStatus.PENDING,
                    parent_flow_id=original_flow_id,
                    is_first_worker=original_is_first_worker,  # Propagate first worker flag
                )

                # Mark original blocked flow as resumed
                if original_flow_id in self._flow_registry:
                    original_flow = self._flow_registry[original_flow_id]
                    if original_flow.status == FlowStatus.BLOCKED_FOR_CREDENTIALS:
                        original_flow.status = FlowStatus.RESUMED
                        original_flow.completion_reason = f"Resumed via {flow_name} after data provided"
                        original_flow.completed_at = datetime.now()

                # Queue the task
                await self._task_queue.put(flow_task)

                new_flows.append({
                    "flow_id": flow_id,
                    "flow_name": flow_name,
                    "flow_path": [str(s) for s in flow_path.steps] if flow_path.steps else [flow_name],
                    "parent_flow_id": original_flow_id,
                    "status": "pending"
                })

            # Trigger supervisor
            if new_flows:
                self._supervisor_trigger.set()

            return new_flows

    async def get_user_data(self, request_name: str) -> Optional[dict[str, str]]:
        """Get stored user data by request name."""
        async with self._lock:
            return self._user_data.get(request_name)

    async def get_all_user_data(self) -> dict[str, dict[str, str]]:
        """Get all stored user data."""
        async with self._lock:
            return dict(self._user_data)

    async def set_data_request_checkpoint(self, request_name: str, checkpoint_id: str):
        """
        Link a checkpoint ID to a pending data request.

        Called by worker after creating a checkpoint when told to checkpoint_and_exit.
        """
        async with self._lock:
            for request in self._pending_data_requests:
                if request.request_name == request_name:
                    request.checkpoint_id = checkpoint_id
                    break

    # -------------------------------------------------------------------------
    # Approval Management (for irreversible actions)
    # -------------------------------------------------------------------------

    def _extract_approval_pattern(self, action_description: str) -> str:
        """
        Extract a normalized pattern from an action description for "always approve" matching.

        Extracts the action type (delete, publish, send, purchase) and normalizes it.
        Uses word boundary matching to avoid false positives (e.g., "unpublished" shouldn't match "publish").
        """
        import re
        action_lower = action_description.lower()
        # Extract key action verbs using word boundaries to avoid substring matches
        # e.g., "unpublished" should NOT match "publish"
        action_verbs = ['delete', 'remove', 'publish', 'send', 'purchase', 'pay', 'subscribe', 'cancel', 'unsubscribe', 'upgrade', 'downgrade']
        for verb in action_verbs:
            # Use word boundary regex to match whole words only
            if re.search(rf'\b{verb}\b', action_lower):
                return verb
        # Fallback to first few words
        words = action_lower.split()[:3]
        return ' '.join(words) if words else action_lower[:20]

    async def check_always_approved(self, action_description: str) -> bool:
        """
        Check if an action matches a pattern the user said "always approve".

        Returns True if this action should be auto-approved.
        """
        async with self._lock:
            if self.skip_permissions:
                return True
            pattern = self._extract_approval_pattern(action_description)
            return pattern in self._always_approved_patterns

    async def request_approval(
        self,
        worker_id: str,
        flow_id: str,
        reason: str,
        action_description: str,
        checkpoint_id: Optional[str] = None
    ) -> dict:
        """
        Request user approval for an irreversible action.

        If skip_permissions is True, auto-approves immediately.
        If action matches an "always approve" pattern, auto-approves.
        Otherwise, records the request for UI to handle.

        Returns:
            {"auto_approved": True} if auto-approved
            {"pending": True} if waiting for user
        """
        # Check for auto-approval conditions
        if self.skip_permissions:
            return {"auto_approved": True}

        if await self.check_always_approved(action_description):
            return {"auto_approved": True}

        # Record the approval request
        async with self._lock:
            self._pending_approval_requests[flow_id] = {
                "worker_id": worker_id,
                "flow_id": flow_id,
                "reason": reason,
                "action_description": action_description,
                "checkpoint_id": checkpoint_id,
                "requested_at": datetime.now().isoformat()
            }

        return {"pending": True}

    async def set_approval_request_checkpoint(self, flow_id: str, checkpoint_id: str):
        """
        Link a checkpoint ID to an approval request for a flow.

        Called by worker after creating a checkpoint when told to checkpoint_and_exit.
        """
        async with self._lock:
            if flow_id in self._pending_approval_requests:
                self._pending_approval_requests[flow_id]["checkpoint_id"] = checkpoint_id

    async def get_pending_approval_requests(self) -> list[dict]:
        """Get all pending approval requests."""
        async with self._lock:
            return list(self._pending_approval_requests.values())

    async def provide_approval(
        self,
        flow_id: str,
        approved: bool,
        always: bool = False
    ) -> Optional[dict]:
        """
        User responds to an approval request.

        Args:
            flow_id: The flow that requested approval
            approved: Whether to approve the action
            always: If True and approved, add pattern to always-approve list

        Returns:
            Dict with new flow info if created, None otherwise
        """
        async with self._lock:
            if flow_id not in self._pending_approval_requests:
                return None

            request = self._pending_approval_requests[flow_id]
            action_description = request.get("action_description", "")
            checkpoint_id = request.get("checkpoint_id")
            original_flow_id = flow_id

            # Store the response
            self._approval_responses[flow_id] = {
                "approved": approved,
                "always": always,
                "responded_at": datetime.now().isoformat()
            }

            # If "always approve" was selected and approved, add pattern
            if approved and always:
                pattern = self._extract_approval_pattern(action_description)
                self._always_approved_patterns.add(pattern)

            # Remove from pending
            del self._pending_approval_requests[flow_id]

            # Get original flow data for naming and first_worker flag
            original_flow_name = ""
            original_flow_path = FlowPath()
            original_is_first_worker = False
            if original_flow_id in self._flow_registry:
                original_flow_data = self._flow_registry[original_flow_id]
                original_flow_name = original_flow_data.flow_name
                original_flow_path = original_flow_data.flow_path
                original_is_first_worker = original_flow_data.is_first_worker

            # Create a new flow to resume with the approval result
            new_flow_id = str(uuid.uuid4())
            approval_status = "approved" if approved else "denied"
            flow_name = f"Resume ({approval_status}): {original_flow_name}" if original_flow_name else f"Resume after approval ({approval_status})"
            flow_path = original_flow_path if original_flow_path.steps else FlowPath([flow_name])

            goal_suffix = "Proceed with the action." if approved else "Skip the action and continue testing other aspects."
            flow_task = FlowTask(
                flow_id=new_flow_id,
                flow_name=flow_name,
                flow_path=flow_path,
                start_url=self.target_url,
                goal=f"Continue exploring after user {approval_status} the action. {goal_suffix}",
                is_root=False,
                parent_flow_id=original_flow_id,
                checkpoint_id=checkpoint_id,
                is_first_worker=original_is_first_worker,  # Propagate first worker flag
            )

            # Initialize flow data
            self._flow_registry[new_flow_id] = FlowExplorationData(
                flow_id=new_flow_id,
                flow_name=flow_name,
                flow_path=flow_path,
                status=FlowStatus.PENDING,
                parent_flow_id=original_flow_id,
                is_first_worker=original_is_first_worker,  # Propagate first worker flag
            )

            # Mark original blocked flow as resumed (continuing via new child flow)
            if original_flow_id in self._flow_registry:
                original_flow = self._flow_registry[original_flow_id]
                if original_flow.status == FlowStatus.BLOCKED_FOR_APPROVAL:
                    original_flow.status = FlowStatus.RESUMED
                    original_flow.completion_reason = f"Resumed via {flow_name} after user {approval_status}"
                    original_flow.completed_at = datetime.now()

            # Queue the task
            await self._task_queue.put(flow_task)

            # Trigger supervisor
            self._supervisor_trigger.set()

            return {
                "flow_id": new_flow_id,
                "flow_name": flow_name,
                "flow_path": [str(s) for s in flow_path.steps] if flow_path.steps else [flow_name],
                "parent_flow_id": original_flow_id,
                "status": "pending"
            }

    async def get_approval_result(self, flow_id: str) -> Optional[dict]:
        """
        Get the approval result for a flow (used when resuming from checkpoint).

        Returns None if no approval result exists for this flow.
        The result includes:
        - approved: bool
        - always: bool (if it was "always approve")
        """
        async with self._lock:
            # Check by flow_id directly
            if flow_id in self._approval_responses:
                return self._approval_responses[flow_id]

            # Check if this is a resume flow by looking at parent
            if flow_id in self._flow_registry:
                parent_id = self._flow_registry[flow_id].parent_flow_id
                if parent_id and parent_id in self._approval_responses:
                    return self._approval_responses[parent_id]

            return None

    async def block_flow_for_approval(self, flow_id: str, reason: str):
        """
        Mark a flow as blocked waiting for user approval of an irreversible action.

        Similar to block_flow_for_credentials but for approval requests.
        Also clears the worker_id since the worker is no longer exploring this flow.
        """
        async with self._lock:
            if flow_id in self._flow_registry:
                flow_data = self._flow_registry[flow_id]
                flow_data.status = FlowStatus.BLOCKED_FOR_APPROVAL
                flow_data.completion_reason = f"Blocked for approval: {reason}"

                # Decrement active worker count and clear worker_id
                if flow_data.worker_id:
                    self._active_workers = max(0, self._active_workers - 1)
                    flow_data.worker_id = None  # Worker no longer on this flow
