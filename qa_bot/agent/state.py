from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Issue:
    """Represents a discovered issue during QA testing."""
    description: str
    severity: str  # critical, major, minor, cosmetic
    url: str
    timestamp: datetime = field(default_factory=datetime.now)
    action_context: str = ""
    screenshot_path: str = ""  # Relative path to screenshot file on disk
    # Attribution: which worker/flow filed the issue, on which turn (the
    # worker's 0-based action count — matches the `TURN n` header in the
    # worker's chat transcript and the turnNNN.png filename), and from which
    # detector. Lets a false critical in the report or summary.json be traced
    # straight to the model's reasoning turn instead of grepped for.
    worker_id: str = ""
    flow_id: str = ""
    flow_name: str = ""
    turn: Optional[int] = None
    source: str = "ai"  # "ai" | "console" | "network" | "dialog"

    def to_dict(self) -> dict:
        result = {
            "description": self.description,
            "severity": self.severity,
            "url": self.url,
            "timestamp": self.timestamp.isoformat(),
            "context": self.action_context,
            "source": self.source,
        }
        if self.screenshot_path:
            result["screenshot_path"] = self.screenshot_path
        if self.worker_id:
            result["worker_id"] = self.worker_id
        if self.flow_id:
            result["flow_id"] = self.flow_id
        if self.flow_name:
            result["flow_name"] = self.flow_name
        if self.turn is not None:
            result["turn"] = self.turn
        return result


@dataclass
class ActionRecord:
    """Record of an action taken during the session."""
    action_type: str
    reasoning: str
    url: str
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = True
    error_message: Optional[str] = None
    post_action_url: Optional[str] = None

    def to_dict(self) -> dict:
        result = {
            "action_type": self.action_type,
            "reasoning": self.reasoning,
            "url": self.url,
            "timestamp": self.timestamp.isoformat(),
            "success": self.success,
            "error_message": self.error_message,
        }
        if self.post_action_url and self.post_action_url != self.url:
            result["post_action_url"] = self.post_action_url
            result["navigated"] = True
        return result
