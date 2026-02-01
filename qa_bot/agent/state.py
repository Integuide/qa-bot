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
    screenshot_base64: str = ""  # Base64-encoded PNG screenshot when issue detected

    def to_dict(self) -> dict:
        result = {
            "description": self.description,
            "severity": self.severity,
            "url": self.url,
            "timestamp": self.timestamp.isoformat(),
            "context": self.action_context
        }
        if self.screenshot_base64:
            result["screenshot"] = self.screenshot_base64
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
