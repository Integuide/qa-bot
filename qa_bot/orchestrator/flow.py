"""Flow-based exploration data structures.

This module defines the core data structures for flow-based QA exploration,
where the unit of work is a "user flow" (e.g., login, signup, checkout) rather
than a single URL.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import uuid


class FlowStatus(str, Enum):
    """Status of a flow in the exploration."""
    PENDING = "pending"          # Checkpoint exists, not yet claimed
    EXPLORING = "exploring"      # Currently being explored by a worker
    COMPLETED = "completed"      # AI determined flow is complete
    FAILED = "failed"            # Exploration failed (error)
    SKIPPED = "skipped"          # Supervisor marked as duplicate/skip
    BLOCKED_FOR_CREDENTIALS = "blocked_for_credentials"  # Waiting for user credentials
    BLOCKED_FOR_APPROVAL = "blocked_for_approval"  # Waiting for user approval of irreversible action
    BLOCKED_FOR_PAUSE = "blocked_for_pause"  # Waiting for exploration resume
    RESUMED = "resumed"          # Blocked flow resumed via new child flow


@dataclass
class FlowPath:
    """
    Breadcrumb trail showing how we reached this point in the flow tree.

    Example: ["Home", "Login", "Forgot Password"]
    """
    steps: list[str] = field(default_factory=list)

    def extend(self, step: str) -> "FlowPath":
        """Create a new path with an additional step."""
        return FlowPath(steps=self.steps + [step])

    def as_string(self, separator: str = " -> ") -> str:
        """Convert to human-readable string."""
        return separator.join(self.steps) if self.steps else "Root"

    def __len__(self) -> int:
        return len(self.steps)

    def to_list(self) -> list[str]:
        """Convert to plain list for serialization."""
        return list(self.steps)

    @classmethod
    def from_list(cls, steps: list[str]) -> "FlowPath":
        """Create from a list of steps."""
        return cls(steps=list(steps))


@dataclass
class FlowCheckpoint:
    """
    Saved state for forking a flow at a branch point.

    When an agent discovers multiple possible paths (e.g., "Login" vs "Signup"),
    it creates checkpoints for the paths it's NOT taking. Other workers can then
    pick up these checkpoints and continue from that exact point.

    Contains everything needed to restore the exploration state:
    - Browser state (cookies, localStorage, sessionStorage)
    - Conversation history (so the AI has context)
    - Flow name/description (so the AI knows what to explore)
    """
    checkpoint_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_flow_id: str = ""

    # Browser state (from Playwright context.storage_state())
    browser_storage_state: dict = field(default_factory=dict)
    current_url: str = ""

    # Conversation history for AI context inheritance
    # List of API-format message dicts: [{"role": "user"|"assistant", "content": ...}]
    conversation_history: list[dict] = field(default_factory=list)

    # Flow context
    flow_path: FlowPath = field(default_factory=FlowPath)
    branch_name: str = ""           # Human-readable name for this branch

    # Metadata
    created_by_worker: str = ""
    created_at: datetime = field(default_factory=datetime.now)

    # Claim status
    claimed: bool = False
    claimed_by: Optional[str] = None
    claimed_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "parent_flow_id": self.parent_flow_id,
            "browser_storage_state": self.browser_storage_state,
            "current_url": self.current_url,
            "conversation_history": self.conversation_history,
            "flow_path": self.flow_path.to_list(),
            "branch_name": self.branch_name,
            "created_by_worker": self.created_by_worker,
            "created_at": self.created_at.isoformat(),
            "claimed": self.claimed,
            "claimed_by": self.claimed_by,
            "claimed_at": self.claimed_at.isoformat() if self.claimed_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FlowCheckpoint":
        """Create from dictionary."""
        return cls(
            checkpoint_id=data["checkpoint_id"],
            parent_flow_id=data["parent_flow_id"],
            browser_storage_state=data["browser_storage_state"],
            current_url=data["current_url"],
            conversation_history=data["conversation_history"],
            flow_path=FlowPath.from_list(data["flow_path"]),
            branch_name=data["branch_name"],
            created_by_worker=data["created_by_worker"],
            created_at=datetime.fromisoformat(data["created_at"]),
            claimed=data["claimed"],
            claimed_by=data["claimed_by"],
            claimed_at=datetime.fromisoformat(data["claimed_at"]) if data["claimed_at"] else None,
        )


@dataclass
class FlowTask:
    """
    Unit of work for flow-based exploration.

    This represents either:
    1. A root flow - starting fresh from a URL
    2. A continuation flow - picking up from a checkpoint

    Flows are about USER JOURNEYS, not individual pages.
    A "Login Flow" might touch multiple pages but represents one coherent test path.
    """
    flow_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    flow_name: str = ""             # Human-readable name (e.g., "Login Flow")
    flow_path: FlowPath = field(default_factory=FlowPath)

    # Root or continuation?
    is_root: bool = True
    checkpoint_id: Optional[str] = None  # Present if is_root=False

    # For root flows only
    start_url: Optional[str] = None
    goal: str = ""

    # Hierarchy
    parent_flow_id: Optional[str] = None

    # Metadata
    priority: int = 0               # Higher = more important
    created_at: datetime = field(default_factory=datetime.now)

    # First worker flag - the first worker only discovers flows, doesn't explore them
    # This flag is propagated through resume flows so the first worker retains its behavior
    is_first_worker: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "flow_id": self.flow_id,
            "flow_name": self.flow_name,
            "flow_path": self.flow_path.to_list(),
            "is_root": self.is_root,
            "checkpoint_id": self.checkpoint_id,
            "start_url": self.start_url,
            "goal": self.goal,
            "parent_flow_id": self.parent_flow_id,
            "priority": self.priority,
            "created_at": self.created_at.isoformat(),
            "is_first_worker": self.is_first_worker,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FlowTask":
        """Create from dictionary."""
        return cls(
            flow_id=data["flow_id"],
            flow_name=data["flow_name"],
            flow_path=FlowPath.from_list(data["flow_path"]),
            is_root=data["is_root"],
            checkpoint_id=data.get("checkpoint_id"),
            start_url=data.get("start_url"),
            goal=data.get("goal", ""),
            parent_flow_id=data.get("parent_flow_id"),
            priority=data.get("priority", 0),
            created_at=datetime.fromisoformat(data["created_at"]),
            is_first_worker=data.get("is_first_worker", False),
        )

    @classmethod
    def create_root(cls, start_url: str, goal: str, flow_name: str = "Root") -> "FlowTask":
        """Create a root flow task.

        The root flow is always the first worker, which only discovers flows
        without exploring them. This flag is propagated through resume flows.
        """
        return cls(
            flow_name=flow_name,
            flow_path=FlowPath(steps=[flow_name]),
            is_root=True,
            start_url=start_url,
            goal=goal,
            priority=100,  # Root flows get high priority
            is_first_worker=True,  # Root flow is always first worker
        )

    @classmethod
    def create_from_checkpoint(
        cls,
        checkpoint: FlowCheckpoint,
        goal: str,
    ) -> "FlowTask":
        """Create a continuation flow task from a checkpoint."""
        return cls(
            flow_name=checkpoint.branch_name,
            flow_path=checkpoint.flow_path,
            is_root=False,
            checkpoint_id=checkpoint.checkpoint_id,
            goal=goal,
            parent_flow_id=checkpoint.parent_flow_id,
            priority=max(0, 100 - len(checkpoint.flow_path) * 10),  # Lower priority for deeper flows
        )


@dataclass
class FlowExplorationData:
    """
    All exploration data for a single flow.

    This is the per-flow equivalent of URLExplorationData,
    tracking everything that happened during a flow's exploration.
    """
    flow_id: str
    flow_name: str
    flow_path: FlowPath
    status: FlowStatus = FlowStatus.PENDING
    parent_flow_id: Optional[str] = None
    checkpoint_id: Optional[str] = None

    # Worker assignment
    worker_id: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    completion_reason: Optional[str] = None  # Why AI marked it complete

    # First worker flag - propagated to resume flows
    is_first_worker: bool = False

    # Exploration data
    actions: list[dict] = field(default_factory=list)
    issues: list[dict] = field(default_factory=list)
    thinking_history: list[str] = field(default_factory=list)

    # URLs visited during this flow
    urls_visited: list[str] = field(default_factory=list)

    # Child flows (branches discovered)
    child_flow_ids: list[str] = field(default_factory=list)

    # Skip reason (if status == SKIPPED)
    skip_reason: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "flow_id": self.flow_id,
            "flow_name": self.flow_name,
            "flow_path": self.flow_path.to_list(),
            "status": self.status.value,
            "parent_flow_id": self.parent_flow_id,
            "checkpoint_id": self.checkpoint_id,
            "worker_id": self.worker_id,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "completion_reason": self.completion_reason,
            "is_first_worker": self.is_first_worker,
            "actions": self.actions,
            "issues": self.issues,
            "thinking_history": self.thinking_history,
            "urls_visited": self.urls_visited,
            "child_flow_ids": self.child_flow_ids,
            "skip_reason": self.skip_reason,
        }
