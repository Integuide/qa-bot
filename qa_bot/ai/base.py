from abc import ABC, abstractmethod
from typing import AsyncGenerator, Literal, Optional, Union
from pydantic import BaseModel


class AgentAction(BaseModel):
    """Structured action from AI for browser interaction.

    Action format matches Claude for Chrome MCP tools exactly.
    See: https://github.com/anthropics/anthropic-quickstarts/tree/main/computer-use-demo
    """

    action_type: Literal[
        # Browser actions - Chrome MCP computer tool
        "left_click",      # Primary click
        "right_click",     # Context menu
        "double_click",    # Double click
        "triple_click",    # Triple click (select paragraph)
        "hover",           # Mouse hover
        "type",            # Type text character by character
        "scroll",          # Scroll page
        "scroll_to",       # Scroll element into view
        "key",             # Press keyboard key
        "left_click_drag", # Drag from start to end coordinate
        "screenshot",      # Take screenshot
        "zoom",            # Zoom into region for inspection
        "wait",            # Wait/pause
        # Chrome MCP separate tools
        "form_input",      # Set form value (form_input tool)
        "navigate",        # Go to URL (navigate tool)
        "resize",          # Resize viewport (resize_window tool)
        # QA-specific actions (extensions)
        "done",            # Mark flow complete
        "block",           # Request supervisor help
        "add_flow",        # Create new flow to test
        "report_issue",    # Report QA issue found
        "close_popup",     # Close active popup window
        "request_data",    # Request data from user (credentials, codes, etc.)
    ]
    reasoning: str = ""

    # Element targeting - ref format: "ref_1", "ref_2", etc.
    ref: Optional[str] = None
    element: Optional[str] = None  # Human-readable element description

    # Coordinate-based actions
    coordinate: Optional[tuple[int, int]] = None  # [x, y] endpoint
    start_coordinate: Optional[tuple[int, int]] = None  # [x, y] for left_click_drag

    # For zoom action
    region: Optional[tuple[int, int, int, int]] = None  # [x0, y0, x1, y1] rectangle

    # For type action
    text: Optional[str] = None

    # For form_input action - flexible value type
    value: Optional[Union[str, bool, int, float]] = None

    # For scroll action
    scroll_direction: Optional[Literal["up", "down", "left", "right"]] = None
    scroll_amount: Optional[int] = None  # Scroll ticks (1-10)

    # For key action
    key: Optional[str] = None  # Key name or space-separated keys
    modifiers: Optional[list[str]] = None  # ["ctrl", "shift", "alt", "cmd"]

    # For wait action
    duration: Optional[float] = None  # Seconds to wait (max 30)

    # For navigate action
    url: Optional[str] = None

    # For resize action
    width: Optional[int] = None
    height: Optional[int] = None

    # For screenshot action
    full_page: Optional[bool] = None  # Capture full scrollable page

    # For done/block actions
    reason: Optional[str] = None

    # For add_flow action
    flow_name: Optional[str] = None
    flow_description: Optional[str] = None
    keep_state: bool = False

    # For report_issue action
    issue_description: Optional[str] = None
    severity: Optional[Literal["critical", "major", "minor", "cosmetic"]] = None

    # For request_data action
    request_name: Optional[str] = None  # AI-chosen name for this data request
    request_description: Optional[str] = None  # Why the data is needed
    request_fields: Optional[list[dict]] = None  # List of field definitions


class WorkerActionResponse(BaseModel):
    """Response from AI containing a structured action to execute."""

    action: AgentAction
    thinking: str = ""


class AIProvider(ABC):
    """Abstract base class for AI vision providers."""

    @abstractmethod
    async def analyze_for_worker_stream(
        self,
        screenshot_bytes: bytes,
        ref_list: str,
        current_url: str,
        flow_name: str,
        flow_goal: str,
        action_history: list[dict],
        conversation_history: list[dict] = None,
        prior_context: str = "",
        additional_context: str = "",
        is_first_worker: bool = False,
        worker_number: int = 0,
        flow_description: str = "",
        parent_flow_name: str = "",
        target_domain: str = "",
        viewport_width: int = 1280,
        viewport_height: int = 720,
        credentials: dict[str, str] = None,
        user_data: dict[str, dict[str, str]] = None,
    ) -> AsyncGenerator[dict, None]:
        """
        Stream AI analysis for worker-based exploration.

        The AI receives a screenshot (for visual analysis) and a text list
        of interactive elements with ref numbers, and outputs a structured
        JSON action to execute.

        Yields events:
            {"type": "thinking_start"}
            {"type": "thinking_delta", "text": "..."}
            {"type": "thinking_complete", "text": "full thinking"}
            {"type": "complete", "action": AgentAction, "thinking": str,
             "assistant_content": list[dict]}
        """
        pass

    @abstractmethod
    async def analyze_for_supervisor(
        self,
        active_workers: list[dict],
        blocked_workers: list[dict],
        pending_flows: list[dict],
        completed_flows: list[dict],
        issues: list[dict],
    ) -> dict:
        """
        Call AI for supervisor decisions.

        Returns parsed action dict.
        """
        pass

    @abstractmethod
    async def generate_synthesis_report(
        self,
        target_url: str,
        duration: str,
        flows_tested: int,
        issues: list[dict],
        completed_flows: list[dict],
    ) -> dict:
        """
        Generate final QA synthesis report.

        Returns dict with:
            - report: markdown-formatted report string
            - input_tokens: tokens used for input
            - output_tokens: tokens used for output
        """
        pass
