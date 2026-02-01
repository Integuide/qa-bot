import asyncio
from dataclasses import dataclass, field
from datetime import datetime
import logging
from typing import Optional, Any
import re
import time

from playwright.async_api import Page, ConsoleMessage, Request, Response, Dialog


logger = logging.getLogger(__name__)


@dataclass
class CapturedConsoleMessage:
    """A captured console message from the browser."""

    level: str  # "error", "warning", "info", "debug", "log"
    text: str
    url: str
    timestamp: str  # ISO format
    location: str  # e.g., "app.js:42:10"


@dataclass
class CapturedNetworkRequest:
    """A captured network request."""

    url: str
    method: str
    status: int  # 0 if no response
    status_text: str
    resource_type: str  # "xhr", "fetch", "document", "script", etc.
    timestamp: str
    duration_ms: float
    failed: bool
    failure_reason: str  # Empty if successful


@dataclass
class CapturedDialog:
    """A captured JavaScript dialog."""

    dialog_type: str  # "alert", "confirm", "prompt", "beforeunload"
    message: str
    default_value: str  # For prompt dialogs
    url: str
    timestamp: str
    was_accepted: bool
    response_value: str = ""  # What was entered for prompts


@dataclass
class CapturedHttpAuthChallenge:
    """A captured HTTP Basic/Digest Auth challenge (401 response)."""

    url: str
    auth_type: str  # "Basic", "Digest", etc.
    realm: str  # The realm from WWW-Authenticate header
    timestamp: str


@dataclass
class CapturedPopup:
    """A captured popup window opened via window.open()."""

    url: str
    opener_url: str  # URL of the page that opened the popup
    timestamp: str
    is_closed: bool = False


class BrowserController:
    """
    Wrapper around Playwright Page for flow exploration workers.

    Provides a simplified interface for the most common operations.
    Workers get a page from the BrowserPool and wrap it in this controller.

    Includes browser monitoring features:
    - Console message capture (errors, warnings, etc.)
    - Network request tracking (failed requests, 4xx/5xx)
    - Dialog handling (auto-accept with recording)
    """

    def __init__(self, page: Page):
        self._page = page
        self._console_messages: list[CapturedConsoleMessage] = []
        self._network_requests: list[CapturedNetworkRequest] = []
        self._pending_requests: dict[str, dict] = {}  # url -> start info
        self._dialogs: list[CapturedDialog] = []
        self._dialog_handler: str = "auto"  # "auto", "accept", "dismiss"
        self._http_auth_challenges: list[CapturedHttpAuthChallenge] = []
        self._popups: list[CapturedPopup] = []
        self._popup_pages: list[Page] = []  # Track actual popup Page objects
        self._active_popup: Page | None = None  # Currently active popup (if any)
        self._setup_listeners()

    def _setup_listeners(self):
        """Set up event listeners for console, network, dialogs, and popups."""
        if not self._page:
            return  # No page, no listeners to set up
        self._setup_console_listener()
        self._setup_network_listeners()
        self._setup_dialog_listener()
        self._setup_popup_listener()

    def _setup_console_listener(self):
        """Set up console message listener on the page."""

        def on_console(msg: ConsoleMessage):
            location_str = ""
            if msg.location:
                loc = msg.location
                location_str = f"{loc.get('url', '')}:{loc.get('lineNumber', '')}:{loc.get('columnNumber', '')}"

            self._console_messages.append(
                CapturedConsoleMessage(
                    level=msg.type,  # "error", "warning", "info", etc.
                    text=msg.text,
                    url=self._page.url if self._page else "",
                    timestamp=datetime.now().isoformat(),
                    location=location_str,
                )
            )

        self._page.on("console", on_console)

    def _setup_network_listeners(self):
        """Set up network request/response listeners."""

        def on_request(request: Request):
            self._pending_requests[request.url] = {
                "method": request.method,
                "resource_type": request.resource_type,
                "start_time": time.time(),
            }

        def on_response(response: Response):
            url = response.url
            if url in self._pending_requests:
                start_info = self._pending_requests.pop(url)
                duration = (time.time() - start_info["start_time"]) * 1000

                self._network_requests.append(
                    CapturedNetworkRequest(
                        url=url,
                        method=start_info["method"],
                        status=response.status,
                        status_text=response.status_text,
                        resource_type=start_info["resource_type"],
                        timestamp=datetime.now().isoformat(),
                        duration_ms=duration,
                        failed=response.status >= 400,
                        failure_reason="" if response.status < 400 else f"HTTP {response.status}",
                    )
                )

                # Capture HTTP 401 auth challenges (deduplicate by URL)
                if response.status == 401:
                    # Check if we already recorded a challenge for this URL
                    existing_urls = {c.url for c in self._http_auth_challenges}
                    if url not in existing_urls:
                        try:
                            # Get WWW-Authenticate header (sync access via all_headers())
                            headers = response.headers
                            www_auth = headers.get("www-authenticate", "")
                            if www_auth:
                                # Parse auth type and realm from header
                                # Format: "Basic realm="Site Name"" or "Digest realm=..."
                                auth_type = www_auth.split()[0] if www_auth else "Unknown"
                                realm = ""
                                if 'realm="' in www_auth:
                                    realm = www_auth.split('realm="')[1].split('"')[0]
                                elif "realm=" in www_auth:
                                    realm = www_auth.split("realm=")[1].split(",")[0].strip()

                                self._http_auth_challenges.append(
                                    CapturedHttpAuthChallenge(
                                        url=url,
                                        auth_type=auth_type,
                                        realm=realm,
                                        timestamp=datetime.now().isoformat(),
                                    )
                                )
                        except Exception as e:
                            # Log the parse error for debugging
                            logger.debug(f"Failed to parse WWW-Authenticate header: {e}")
                            # If header parsing fails, still record the challenge
                            self._http_auth_challenges.append(
                                CapturedHttpAuthChallenge(
                                    url=url,
                                    auth_type="Unknown",
                                    realm="",
                                    timestamp=datetime.now().isoformat(),
                                )
                            )

        def on_request_failed(request: Request):
            url = request.url
            if url in self._pending_requests:
                start_info = self._pending_requests.pop(url)
                duration = (time.time() - start_info["start_time"]) * 1000

                failure_text = ""
                if request.failure:
                    failure_text = request.failure

                self._network_requests.append(
                    CapturedNetworkRequest(
                        url=url,
                        method=start_info["method"],
                        status=0,
                        status_text="",
                        resource_type=start_info["resource_type"],
                        timestamp=datetime.now().isoformat(),
                        duration_ms=duration,
                        failed=True,
                        failure_reason=failure_text or "Request failed",
                    )
                )

        self._page.on("request", on_request)
        self._page.on("response", on_response)
        self._page.on("requestfailed", on_request_failed)

    def _setup_dialog_listener(self):
        """Set up dialog auto-handling and recording."""

        async def on_dialog(dialog: Dialog):
            captured = CapturedDialog(
                dialog_type=dialog.type,
                message=dialog.message,
                default_value=dialog.default_value or "",
                url=self._page.url if self._page else "",
                timestamp=datetime.now().isoformat(),
                was_accepted=True,  # Will be updated based on action
                response_value="",
            )

            if self._dialog_handler == "auto":
                # Auto behavior: accept most dialogs, dismiss beforeunload
                if dialog.type == "alert":
                    await dialog.accept()
                    captured.was_accepted = True
                elif dialog.type == "confirm":
                    await dialog.accept()
                    captured.was_accepted = True
                elif dialog.type == "prompt":
                    response_val = dialog.default_value or "test"
                    await dialog.accept(response_val)
                    captured.was_accepted = True
                    captured.response_value = response_val
                elif dialog.type == "beforeunload":
                    await dialog.dismiss()
                    captured.was_accepted = False
            elif self._dialog_handler == "accept":
                await dialog.accept()
                captured.was_accepted = True
            else:  # dismiss
                await dialog.dismiss()
                captured.was_accepted = False

            self._dialogs.append(captured)

        self._page.on("dialog", on_dialog)

    def _setup_popup_listener(self):
        """Set up popup window listener."""

        def on_popup(popup: Page):
            """Handle new popup windows opened via window.open()."""
            popup_url = popup.url or "about:blank"
            popup_record = CapturedPopup(
                url=popup_url,
                opener_url=self._page.url if self._page else "",
                timestamp=datetime.now().isoformat(),
                is_closed=False,
            )
            self._popups.append(popup_record)
            self._popup_pages.append(popup)
            # Auto-switch to popup when it opens
            self._active_popup = popup

            # Set up monitoring listeners on the popup page
            # This ensures console/network/dialog events in popups are captured
            self._setup_popup_monitoring(popup)

            # Set up close listener to track when popup is closed
            # Use reference matching to find the correct popup record
            def on_popup_close():
                # Find and mark this specific popup as closed by matching URL and timestamp
                for p in self._popups:
                    if p.url == popup_url and p.timestamp == popup_record.timestamp and not p.is_closed:
                        p.is_closed = True
                        break
                # Remove from active pages
                if popup in self._popup_pages:
                    self._popup_pages.remove(popup)
                # If active popup was closed, switch back to main
                if self._active_popup == popup:
                    self._active_popup = None

            popup.on("close", on_popup_close)

        self._page.on("popup", on_popup)

    def _setup_popup_monitoring(self, popup: Page):
        """Set up console/network/dialog monitoring on a popup page."""
        # Console listener for popup
        def on_console(msg: ConsoleMessage):
            location_str = ""
            if msg.location:
                loc = msg.location
                location_str = f"{loc.get('url', '')}:{loc.get('lineNumber', '')}:{loc.get('columnNumber', '')}"

            self._console_messages.append(
                CapturedConsoleMessage(
                    level=msg.type,
                    text=msg.text,
                    url=popup.url if popup else "",
                    timestamp=datetime.now().isoformat(),
                    location=location_str,
                )
            )

        popup.on("console", on_console)

        # Network listeners for popup
        def on_request(request: Request):
            self._pending_requests[request.url] = {
                "method": request.method,
                "resource_type": request.resource_type,
                "start_time": time.time(),
            }

        def on_response(response: Response):
            url = response.url
            if url in self._pending_requests:
                start_info = self._pending_requests.pop(url)
                duration = (time.time() - start_info["start_time"]) * 1000

                self._network_requests.append(
                    CapturedNetworkRequest(
                        url=url,
                        method=start_info["method"],
                        status=response.status,
                        status_text=response.status_text,
                        resource_type=start_info["resource_type"],
                        timestamp=datetime.now().isoformat(),
                        duration_ms=duration,
                        failed=response.status >= 400,
                        failure_reason="" if response.status < 400 else f"HTTP {response.status}",
                    )
                )

        def on_request_failed(request: Request):
            url = request.url
            if url in self._pending_requests:
                start_info = self._pending_requests.pop(url)
                duration = (time.time() - start_info["start_time"]) * 1000

                failure_text = ""
                if request.failure:
                    failure_text = request.failure

                self._network_requests.append(
                    CapturedNetworkRequest(
                        url=url,
                        method=start_info["method"],
                        status=0,
                        status_text="",
                        resource_type=start_info["resource_type"],
                        timestamp=datetime.now().isoformat(),
                        duration_ms=duration,
                        failed=True,
                        failure_reason=failure_text or "Request failed",
                    )
                )

        popup.on("request", on_request)
        popup.on("response", on_response)
        popup.on("requestfailed", on_request_failed)

        # Dialog listener for popup
        async def on_dialog(dialog: Dialog):
            captured = CapturedDialog(
                dialog_type=dialog.type,
                message=dialog.message,
                default_value=dialog.default_value or "",
                url=popup.url if popup else "",
                timestamp=datetime.now().isoformat(),
                was_accepted=True,
                response_value="",
            )

            if self._dialog_handler == "auto":
                if dialog.type == "alert":
                    await dialog.accept()
                    captured.was_accepted = True
                elif dialog.type == "confirm":
                    await dialog.accept()
                    captured.was_accepted = True
                elif dialog.type == "prompt":
                    response_val = dialog.default_value or "test"
                    await dialog.accept(response_val)
                    captured.was_accepted = True
                    captured.response_value = response_val
                elif dialog.type == "beforeunload":
                    await dialog.dismiss()
                    captured.was_accepted = False
            elif self._dialog_handler == "accept":
                await dialog.accept()
                captured.was_accepted = True
            else:
                await dialog.dismiss()
                captured.was_accepted = False

            self._dialogs.append(captured)

        popup.on("dialog", on_dialog)

    @classmethod
    def from_page(cls, page: Page) -> "BrowserController":
        """Create a controller from an existing page."""
        return cls(page)

    # ===== Console Monitoring =====

    def get_console_messages(
        self, level: str | None = None, clear: bool = False
    ) -> list[CapturedConsoleMessage]:
        """
        Get captured console messages.

        Args:
            level: Filter by level ("error", "warning", etc.) or None for all
            clear: If True, clear messages after returning

        Returns:
            List of captured messages
        """
        if level:
            msgs = [m for m in self._console_messages if m.level == level]
        else:
            msgs = list(self._console_messages)

        if clear:
            self._console_messages.clear()
        return msgs

    def get_console_errors(self) -> list[CapturedConsoleMessage]:
        """Get only error-level console messages."""
        return [m for m in self._console_messages if m.level == "error"]

    def clear_console_messages(self):
        """Clear all captured console messages."""
        self._console_messages.clear()

    # ===== Network Monitoring =====

    def get_failed_requests(self, clear: bool = False) -> list[CapturedNetworkRequest]:
        """Get all failed network requests (4xx, 5xx, or connection failures)."""
        failed = [r for r in self._network_requests if r.failed]
        if clear:
            self._network_requests = [r for r in self._network_requests if not r.failed]
        return failed

    def get_network_requests(
        self,
        include_static: bool = False,
        min_status: int | None = None,
        clear: bool = False,
    ) -> list[CapturedNetworkRequest]:
        """
        Get captured network requests.

        Args:
            include_static: Include static resources (images, fonts, css, scripts)
            min_status: Only include requests with status >= this value
            clear: Clear requests after returning

        Returns:
            List of captured requests
        """
        static_types = {"image", "font", "stylesheet", "script"}

        requests = self._network_requests
        if not include_static:
            requests = [r for r in requests if r.resource_type not in static_types]
        if min_status:
            requests = [r for r in requests if r.status >= min_status]

        if clear:
            self._network_requests.clear()
        return list(requests)

    def clear_network_requests(self):
        """Clear all captured network requests."""
        self._network_requests.clear()
        self._pending_requests.clear()

    # ===== Dialog Handling =====

    def get_dialogs(self, clear: bool = False) -> list[CapturedDialog]:
        """Get all captured dialogs."""
        dialogs = list(self._dialogs)
        if clear:
            self._dialogs.clear()
        return dialogs

    def set_dialog_handler(self, mode: str):
        """
        Set dialog handling mode.

        Args:
            mode: "auto" (smart defaults), "accept" (accept all), "dismiss" (dismiss all)
        """
        if mode not in ("auto", "accept", "dismiss"):
            raise ValueError(f"Invalid dialog handler mode: {mode}")
        self._dialog_handler = mode

    def clear_dialogs(self):
        """Clear captured dialogs."""
        self._dialogs.clear()

    # ===== HTTP Auth Monitoring =====

    def get_http_auth_challenges(self, clear: bool = False) -> list[CapturedHttpAuthChallenge]:
        """
        Get captured HTTP Basic/Digest Auth challenges (401 responses).

        Args:
            clear: If True, clear challenges after returning

        Returns:
            List of captured auth challenges
        """
        challenges = list(self._http_auth_challenges)
        if clear:
            self._http_auth_challenges.clear()
        return challenges

    def clear_http_auth_challenges(self):
        """Clear captured HTTP auth challenges."""
        self._http_auth_challenges.clear()

    # ===== Popup Handling =====

    @property
    def active_page(self) -> Page:
        """
        Get the currently active page (popup or main).

        Returns the active popup if one is open, otherwise returns the main page.
        All screenshot and action methods should use this property to interact
        with the correct page.
        """
        if self._active_popup is not None and not self._active_popup.is_closed():
            return self._active_popup
        return self._page

    @property
    def has_active_popup(self) -> bool:
        """Check if there's an active popup window."""
        return self._active_popup is not None and not self._active_popup.is_closed()

    def get_popups(self, clear: bool = False) -> list[CapturedPopup]:
        """
        Get captured popup windows.

        Args:
            clear: If True, clear popups after returning

        Returns:
            List of captured popups
        """
        popups = list(self._popups)
        if clear:
            self._popups.clear()
        return popups

    async def close_active_popup(self, timeout: float = 5000) -> bool:
        """
        Close the active popup window and return focus to main page.

        Args:
            timeout: Maximum time in milliseconds to wait for popup to close

        Returns:
            True if a popup was closed, False if no active popup
        """
        if self._active_popup is not None and not self._active_popup.is_closed():
            try:
                await asyncio.wait_for(
                    self._active_popup.close(),
                    timeout=timeout / 1000  # Convert ms to seconds
                )
            except asyncio.TimeoutError:
                logger.warning(f"Timeout closing popup after {timeout}ms")
            self._active_popup = None
            return True
        return False

    def switch_to_main_page(self) -> None:
        """
        Switch focus back to the main page, ignoring any open popups.

        The popup will still exist but screenshot/action methods will
        target the main page.
        """
        self._active_popup = None

    def switch_to_popup(self, index: int = -1) -> bool:
        """
        Switch focus to a popup window.

        Args:
            index: Index of popup to switch to (-1 for most recent)

        Returns:
            True if switched successfully, False if no popup at index
        """
        if not self._popup_pages:
            return False

        try:
            popup = self._popup_pages[index]
            if not popup.is_closed():
                self._active_popup = popup
                return True
        except IndexError:
            pass
        return False

    def clear_popups(self):
        """Clear captured popup records (does not close them)."""
        self._popups.clear()

    async def screenshot(self, full_page: bool = False) -> bytes:
        """
        Take a screenshot of the current viewport (popup or main page).

        Args:
            full_page: If True, capture the entire scrollable page

        Returns:
            PNG image bytes
        """
        page = self.active_page
        if not page:
            raise RuntimeError("No page available")

        return await page.screenshot(type="png", full_page=full_page)

    @property
    def current_url(self) -> str:
        """Get the current page URL (popup or main page)."""
        page = self.active_page
        if page:
            return page.url
        return ""

    @property
    def page(self) -> Page:
        """Get the underlying main Playwright page (not popup)."""
        return self._page

    async def accessibility_snapshot(self) -> str:
        """
        Get the accessibility tree snapshot as YAML-like text.

        Uses Playwright's aria_snapshot() which provides a structured
        representation of the accessibility tree that maps directly to
        Playwright's getByRole() and other locator APIs.

        Returns:
            YAML-formatted accessibility tree string
        """
        page = self.active_page
        if not page:
            raise RuntimeError("No page available")

        # Use new Playwright API (1.49+)
        snapshot = await page.locator(":root").aria_snapshot()
        return snapshot or ""

    async def get_viewport_size(self) -> dict[str, int]:
        """Get current viewport dimensions (of active page)."""
        page = self.active_page
        size = page.viewport_size if page else None
        if size:
            return {"width": size["width"], "height": size["height"]}
        return {"width": 1280, "height": 720}

    # ===== Ref-Based Element Targeting =====
    # Ref format matches Claude for Chrome MCP: "ref_1", "ref_2", etc.

    def _extract_ref_number(self, ref: str) -> str:
        """
        Extract the numeric part from a ref string for DOM lookup.

        Args:
            ref: Ref string in format "ref_5"

        Returns:
            The numeric part as string, e.g., "5"
        """
        if ref.startswith("ref_"):
            return ref[4:]  # Remove "ref_" prefix
        return ref

    async def inject_refs(self) -> dict[str, str]:
        """
        Inject data-qabot-ref attributes into interactive elements (on active page).

        Returns:
            Dictionary mapping ref strings ("ref_1", "ref_2", ...) to element descriptions.
            Format aligned with Claude for Chrome MCP.
        """
        page = self.active_page
        if not page:
            raise RuntimeError("No page available")

        script = """
        () => {
            const refMap = {};
            let refCounter = 1;

            // Select interactive elements
            const selectors = [
                'a', 'button', 'input', 'select', 'textarea',
                '[role="button"]', '[role="link"]', '[role="menuitem"]',
                '[role="tab"]', '[role="checkbox"]', '[role="radio"]',
                '[role="option"]', '[role="switch"]', '[role="textbox"]',
                '[onclick]', '[tabindex]:not([tabindex="-1"])'
            ];

            const elements = document.querySelectorAll(selectors.join(','));

            elements.forEach(el => {
                // Skip hidden elements
                const rect = el.getBoundingClientRect();
                if (rect.width === 0 && rect.height === 0) return;
                if (getComputedStyle(el).display === 'none') return;
                if (getComputedStyle(el).visibility === 'hidden') return;

                const ref = refCounter++;
                el.setAttribute('data-qabot-ref', ref.toString());

                // Build description
                const role = el.getAttribute('role') || el.tagName.toLowerCase();
                const name = el.getAttribute('aria-label') ||
                            el.textContent?.trim().slice(0, 50) ||
                            el.getAttribute('placeholder') ||
                            el.getAttribute('name') || '';

                // Use string key format: "ref_1", "ref_2", etc.
                refMap['ref_' + ref] = role + ': ' + name.replace(/\\s+/g, ' ');
            });

            return refMap;
        }
        """
        return await page.evaluate(script)

    async def accessibility_snapshot_with_refs(self) -> tuple[str, dict[str, str]]:
        """
        Get accessibility tree with ref strings for element targeting (on active page).

        Injects ref attributes into interactive elements, then takes the
        accessibility snapshot. Refs can be used for precise element targeting.

        Returns:
            Tuple of (YAML snapshot, ref_map: {"ref_1": description, ...})
        """
        page = self.active_page
        if not page:
            raise RuntimeError("No page available")

        # Inject refs into interactive elements
        ref_map = await self.inject_refs()

        # Get the accessibility snapshot
        snapshot = await page.locator(":root").aria_snapshot()

        return snapshot or "", ref_map

    async def get_element_by_ref(self, ref: str):
        """
        Get a Playwright locator by ref string (on active page).

        Args:
            ref: The ref string (e.g., "ref_5")

        Returns:
            Playwright Locator for the element
        """
        page = self.active_page
        if not page:
            raise RuntimeError("No page available")

        ref_num = self._extract_ref_number(ref)
        return page.locator(f'[data-qabot-ref="{ref_num}"]')

    async def click_ref(self, ref: str) -> None:
        """
        Click an element by its ref (left click).

        Args:
            ref: The ref string (e.g., "ref_5")
        """
        locator = await self.get_element_by_ref(ref)
        await locator.click()

    async def right_click_ref(self, ref: str) -> None:
        """
        Right-click an element by its ref (context menu).

        Args:
            ref: The ref string (e.g., "ref_5")
        """
        locator = await self.get_element_by_ref(ref)
        await locator.click(button="right")

    async def double_click_ref(self, ref: str) -> None:
        """
        Double-click an element by its ref.

        Args:
            ref: The ref string (e.g., "ref_5")
        """
        locator = await self.get_element_by_ref(ref)
        await locator.dblclick()

    async def triple_click_ref(self, ref: str) -> None:
        """
        Triple-click an element by its ref (select paragraph).

        Args:
            ref: The ref string (e.g., "ref_5")
        """
        locator = await self.get_element_by_ref(ref)
        await locator.click(click_count=3)

    async def hover_ref(self, ref: str) -> None:
        """
        Hover over an element by its ref.

        Args:
            ref: The ref string (e.g., "ref_5")
        """
        locator = await self.get_element_by_ref(ref)
        await locator.hover()

    async def scroll_to_ref(self, ref: str) -> None:
        """
        Scroll an element into view by its ref.

        Args:
            ref: The ref string (e.g., "ref_5")
        """
        locator = await self.get_element_by_ref(ref)
        await locator.scroll_into_view_if_needed()

    async def fill_ref(self, ref: str, text: str) -> None:
        """
        Fill text into an element by ref (clears existing content first).

        Args:
            ref: The ref string (e.g., "ref_5")
            text: Text to fill
        """
        locator = await self.get_element_by_ref(ref)
        await locator.fill(text)

    def _parse_and_format_date(self, value: str, target_format: str) -> str:
        """
        Parse a date string in various formats and convert to target format.

        Supports parsing:
        - YYYY-MM-DD (ISO format)
        - DD/MM/YYYY (European)
        - MM/DD/YYYY (US)
        - DD-MM-YYYY
        - MM-DD-YYYY
        - DDMMYYYY (no separators)
        - MMDDYYYY (no separators)

        Args:
            value: Date string to parse
            target_format: strftime format string for output

        Returns:
            Formatted date string, or original value if parsing fails
        """
        value = str(value).strip()

        # First try ISO format - unambiguous
        if re.match(r"^\d{4}-\d{2}-\d{2}$", value):
            try:
                dt = datetime.strptime(value, "%Y-%m-%d")
                return dt.strftime(target_format)
            except ValueError:
                pass

        # For ambiguous formats, try DD/MM/YYYY first (more common globally),
        # then fall back to MM/DD/YYYY
        ambiguous_attempts = [
            # DD/MM/YYYY style
            ("%d/%m/%Y", r"^\d{2}/\d{2}/\d{4}$"),
            ("%m/%d/%Y", r"^\d{2}/\d{2}/\d{4}$"),
            # DD-MM-YYYY style
            ("%d-%m-%Y", r"^\d{2}-\d{2}-\d{4}$"),
            ("%m-%d-%Y", r"^\d{2}-\d{2}-\d{4}$"),
            # No separators
            ("%d%m%Y", r"^\d{8}$"),
            ("%m%d%Y", r"^\d{8}$"),
        ]

        for fmt, pattern in ambiguous_attempts:
            if re.match(pattern, value):
                try:
                    dt = datetime.strptime(value, fmt)
                    # Sanity check: day should be 1-31, month 1-12
                    if 1 <= dt.day <= 31 and 1 <= dt.month <= 12:
                        return dt.strftime(target_format)
                except ValueError:
                    continue

        # If no format matched, return original value
        return value

    def _parse_and_format_time(self, value: str) -> str:
        """
        Parse a time string and convert to HH:MM format for HTML5 time inputs.

        Args:
            value: Time string to parse

        Returns:
            Formatted time string (HH:MM), or original value if parsing fails
        """
        value = str(value).strip()

        # Already in correct format
        if re.match(r"^\d{2}:\d{2}(:\d{2})?$", value):
            return value[:5]  # Return HH:MM portion

        # Try parsing various formats
        time_formats = [
            "%H:%M:%S",
            "%H:%M",
            "%I:%M %p",  # 12-hour with AM/PM
            "%I:%M%p",
            "%H%M",  # No separator
        ]

        for fmt in time_formats:
            try:
                dt = datetime.strptime(value, fmt)
                return dt.strftime("%H:%M")
            except ValueError:
                continue

        return value

    async def set_form_value(self, ref: str, value: str | bool | int | float) -> None:
        """
        Set form element value by ref. Handles different input types.

        For checkboxes: value should be bool (True to check, False to uncheck)
        For selects: value should be the option value or text
        For date inputs: value can be various formats (DD/MM/YYYY, YYYY-MM-DD, etc.)
        For time inputs: value can be HH:MM or HH:MM:SS
        For text inputs: value should be string

        Args:
            ref: The ref string (e.g., "ref_5")
            value: Value to set (type depends on element type)
        """
        locator = await self.get_element_by_ref(ref)

        # Get element tag name to determine how to set value
        tag_name = await locator.evaluate("el => el.tagName.toLowerCase()")
        input_type = await locator.evaluate("el => el.type || ''")

        if tag_name == "select":
            # For select elements, use select_option
            await locator.select_option(str(value))
        elif input_type in ("checkbox", "radio"):
            # For checkboxes/radios, check or uncheck based on bool value
            is_checked = await locator.is_checked()
            should_check = bool(value)
            if should_check and not is_checked:
                await locator.check()
            elif not should_check and is_checked:
                await locator.uncheck()
        elif input_type == "date":
            # HTML5 date inputs require YYYY-MM-DD format
            formatted_value = self._parse_and_format_date(str(value), "%Y-%m-%d")
            await locator.fill(formatted_value)
        elif input_type == "time":
            # HTML5 time inputs require HH:MM format
            formatted_value = self._parse_and_format_time(str(value))
            await locator.fill(formatted_value)
        elif input_type == "datetime-local":
            # HTML5 datetime-local inputs require YYYY-MM-DDTHH:MM format
            # Try to parse date and time parts
            str_value = str(value)
            if "T" in str_value:
                # Already has T separator
                parts = str_value.split("T")
                date_part = self._parse_and_format_date(parts[0], "%Y-%m-%d")
                time_part = self._parse_and_format_time(parts[1]) if len(parts) > 1 else "00:00"
            elif " " in str_value:
                # Space separator
                parts = str_value.split(" ", 1)
                date_part = self._parse_and_format_date(parts[0], "%Y-%m-%d")
                time_part = self._parse_and_format_time(parts[1]) if len(parts) > 1 else "00:00"
            else:
                # Just a date
                date_part = self._parse_and_format_date(str_value, "%Y-%m-%d")
                time_part = "00:00"
            formatted_value = f"{date_part}T{time_part}"
            await locator.fill(formatted_value)
        elif input_type == "month":
            # HTML5 month inputs require YYYY-MM format
            formatted_value = self._parse_and_format_date(str(value), "%Y-%m")
            await locator.fill(formatted_value)
        elif input_type == "week":
            # HTML5 week inputs require YYYY-Www format (e.g., 2024-W01)
            # This is complex to parse, so just pass through if already formatted
            str_value = str(value)
            if "-W" in str_value.upper():
                await locator.fill(str_value)
            else:
                # Try to parse as a date and convert to week
                formatted_date = self._parse_and_format_date(str_value, "%Y-%m-%d")
                try:
                    dt = datetime.strptime(formatted_date, "%Y-%m-%d")
                    iso_cal = dt.isocalendar()
                    formatted_value = f"{iso_cal.year}-W{iso_cal.week:02d}"
                    await locator.fill(formatted_value)
                except ValueError:
                    await locator.fill(str_value)
        else:
            # For text inputs, textareas, etc., use fill
            await locator.fill(str(value))

    async def drag(
        self,
        start: tuple[int, int],
        end: tuple[int, int]
    ) -> None:
        """
        Drag from start coordinate to end coordinate (on active page).

        Args:
            start: Starting [x, y] coordinates
            end: Ending [x, y] coordinates
        """
        page = self.active_page
        if not page:
            raise RuntimeError("No page available")

        await page.mouse.move(start[0], start[1])
        await page.mouse.down()
        await page.mouse.move(end[0], end[1])
        await page.mouse.up()

    async def get_ref_list(self) -> str:
        """
        Get a text list of interactive elements with ref strings.

        Injects refs into the page and returns a formatted text list
        suitable for passing to the AI alongside a screenshot.

        Returns:
            Formatted string like:
            Interactive elements:
              ref_1: button "Sign Up"
              ref_2: link "Login"
              ref_3: textbox "Email"
        """
        ref_map = await self.inject_refs()
        if not ref_map:
            return "No interactive elements found on this page."

        lines = ["Interactive elements:"]
        # Sort by numeric part of ref string
        sorted_refs = sorted(ref_map.items(), key=lambda x: int(x[0].replace("ref_", "")))
        for ref, desc in sorted_refs:
            # Clean up description - remove extra whitespace
            clean_desc = " ".join(desc.split())
            lines.append(f"  {ref}: {clean_desc}")
        return "\n".join(lines)
