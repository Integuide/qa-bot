import asyncio
from dataclasses import dataclass, field
from datetime import datetime
import logging
from typing import Optional, Any
import re
import time
from urllib.parse import urlparse

from playwright.async_api import Page, Frame, ConsoleMessage, Request, Response, Dialog, Error as PlaywrightError


logger = logging.getLogger(__name__)

# page.evaluate() has no timeout of its own, so a blocked JS main thread or a
# navigation tearing down the execution context mid-call would hang the caller
# indefinitely. Bound ref injection and let it retry once.
REF_INJECTION_TIMEOUT_S = 15.0

# Subframe ref injection is best-effort: a wedged third-party iframe (ads,
# analytics) must not stall the whole turn, so subframes are injected
# concurrently, each with a short budget, and failures are skipped rather
# than retried. Worst-case added latency is one timeout, not one per frame.
SUBFRAME_REF_INJECTION_TIMEOUT_S = 5.0
MAX_REF_SUBFRAMES = 20

# Concurrent subframes can't share a live counter, so each gets a disjoint
# numbering range: frame i starts at the next multiple of the stride after
# the main frame's refs, plus i * stride. Ref numbers stay unique but are
# no longer contiguous across frames (refs are opaque to the AI anyway).
SUBFRAME_REF_STRIDE = 1000

# Cap on main-frame refs. Every ref line costs prompt tokens on every turn,
# so an uncapped list on a link-farm page (portals, mega-footers, admin
# tables) can dwarf the rest of the prompt. Elements beyond the cap are
# reported as a count in get_ref_list() so partial coverage stays visible.
MAIN_FRAME_MAX_REFS = 500


DUPLICATE_REF_PREFIX = "[duplicate-ref] "

# _pending_requests pruning: entries whose response/failure event never fired
# (page closed mid-flight) are dropped once the dict grows past the prune
# size and they are older than the max age.
_PENDING_REQUESTS_PRUNE_SIZE = 512
_PENDING_REQUESTS_MAX_AGE_S = 120.0


class StaleRefError(Exception):
    """Raised when a ref matches zero DOM elements (ref map is stale).

    Refs are injected once per turn; if the page re-rendered since the last
    screenshot the attribute is gone and Playwright's auto-wait would burn
    the full default timeout on a locator that can never match. Failing
    fast lets the AI re-observe the page on its next turn instead.
    """

    def __init__(self, ref: str):
        self.ref = ref
        super().__init__(
            f"Element {ref} no longer exists — the page changed since the "
            f"last screenshot. Observe the new page state and pick a fresh "
            f"ref from the updated element list."
        )


class DuplicateRefError(Exception):
    """Raised when a ref matches multiple DOM elements.

    Attributes:
        ref: The original ref string (e.g. "ref_5").
        descriptions: List of human-readable descriptions for each duplicate,
            including position and parent context so the AI can disambiguate.
    """

    def __init__(self, ref: str, descriptions: list[str]):
        self.ref = ref
        self.descriptions = descriptions
        summary = "; ".join(f"({i+1}) {d}" for i, d in enumerate(descriptions))
        super().__init__(
            f"{DUPLICATE_REF_PREFIX}{ref} matches {len(descriptions)} elements: "
            f"{summary}. "
            f"Look at the screenshot and element list to pick the correct one."
        )


# Browser-initiated abort reasons that are not real failures.
# These occur during normal navigation, ad-blocking, or page unload.
# Shared with qa_bot.orchestrator.worker.classify_network_failure (Layer 2).
BENIGN_FAILURE_REASONS = frozenset({
    "net::ERR_ABORTED",
    "net::ERR_BLOCKED_BY_CLIENT",
})

# Resource types where aborts are almost always harmless
# (tracking pixels, analytics beacons, prefetched assets, etc.)
_NOISE_RESOURCE_TYPES = frozenset({
    "ping", "beacon", "cspviolationreport", "preflight",
})


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


@dataclass
class CapturedDownload:
    """A captured file download (clicked link or direct download URL).

    Downloads are otherwise invisible to the AI: the screenshot doesn't
    change, so a working export/PDF link looks like a dead button.
    """

    url: str  # URL the file was downloaded from
    suggested_filename: str  # Filename suggested by the server/browser
    page_url: str  # URL of the page that triggered the download
    timestamp: str


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
        # Keyed by the Request object, not URL: two concurrent requests to
        # the same URL (polling, retried fetches, popup pages sharing these
        # listeners) must not clobber each other's start info.
        self._pending_requests: dict[Request, dict] = {}
        self._dialogs: list[CapturedDialog] = []
        self._dialog_handler: str = "auto"  # "auto", "accept", "dismiss"
        self._http_auth_challenges: list[CapturedHttpAuthChallenge] = []
        self._popups: list[CapturedPopup] = []
        self._popup_pages: list[Page] = []  # Track actual popup Page objects
        self._active_popup: Page | None = None  # Currently active popup (if any)
        self._pinned_page: Page | None = None  # Turn pin (see pin_active_page)
        self._downloads: list[CapturedDownload] = []
        # Refs injected into subframes (iframes) can't be found via
        # page.locator(), which never crosses frame boundaries — remember
        # which frame each subframe ref lives in. Main-frame refs are not
        # recorded here (absence means "look on the page itself").
        self._ref_frames: dict[str, Frame] = {}
        # Coverage gaps from the last injection, surfaced in get_ref_list()
        # so the AI knows the list is partial instead of reading it as "the
        # whole page": subframes that couldn't be scanned (over the frame
        # cap, or evaluation failed/timed out) and elements beyond a frame's
        # ref cap.
        self._unscanned_subframes = 0
        self._unlisted_elements = 0
        self._setup_listeners()

    def _setup_listeners(self):
        """Set up event listeners for console, network, dialogs, downloads, and popups."""
        if not self._page:
            return  # No page, no listeners to set up
        self._setup_console_listener(self._page)
        self._setup_network_listeners(self._page)
        self._setup_dialog_listener(self._page)
        self._setup_download_listener(self._page)
        self._setup_popup_listener(self._page)

    def _setup_console_listener(self, page: Page):
        """Set up console message listener on a page."""

        def on_console(msg: ConsoleMessage):
            location_str = ""
            if msg.location:
                loc = msg.location
                location_str = f"{loc.get('url', '')}:{loc.get('lineNumber', '')}:{loc.get('columnNumber', '')}"

            self._console_messages.append(
                CapturedConsoleMessage(
                    level=msg.type,  # "error", "warning", "info", etc.
                    text=msg.text,
                    url=page.url if page else "",
                    timestamp=datetime.now().isoformat(),
                    location=location_str,
                )
            )

        page.on("console", on_console)

    def _setup_network_listeners(self, page: Page):
        """Set up network request/response listeners on a page."""

        def on_request(request: Request):
            # Requests that never get a response/failure event (page closed
            # mid-flight) would accumulate forever now that keys are unique
            # per request — prune stale entries once the dict grows.
            if len(self._pending_requests) > _PENDING_REQUESTS_PRUNE_SIZE:
                cutoff = time.time() - _PENDING_REQUESTS_MAX_AGE_S
                for req in [
                    r for r, info in self._pending_requests.items()
                    if info["start_time"] < cutoff
                ]:
                    del self._pending_requests[req]

            self._pending_requests[request] = {
                "method": request.method,
                "resource_type": request.resource_type,
                "start_time": time.time(),
            }

        def on_response(response: Response):
            url = response.url
            start_info = self._pending_requests.pop(response.request, None)
            if start_info is not None:
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
            start_info = self._pending_requests.pop(request, None)
            if start_info is not None:
                duration = (time.time() - start_info["start_time"]) * 1000

                failure_text = ""
                if request.failure:
                    failure_text = request.failure

                resource_type = start_info["resource_type"]

                # Layer 1: Skip benign aborts for noise resource types entirely.
                # Layer 2 (classify_network_failure in worker.py) handles the
                # remaining cases for non-noise resource types.
                if failure_text in BENIGN_FAILURE_REASONS and resource_type in _NOISE_RESOURCE_TYPES:
                    return

                self._network_requests.append(
                    CapturedNetworkRequest(
                        url=url,
                        method=start_info["method"],
                        status=0,
                        status_text="",
                        resource_type=resource_type,
                        timestamp=datetime.now().isoformat(),
                        duration_ms=duration,
                        failed=True,
                        failure_reason=failure_text or "Request failed",
                    )
                )

        page.on("request", on_request)
        page.on("response", on_response)
        page.on("requestfailed", on_request_failed)

    def _setup_dialog_listener(self, page: Page):
        """Set up dialog auto-handling and recording on a page."""

        async def on_dialog(dialog: Dialog):
            captured = CapturedDialog(
                dialog_type=dialog.type,
                message=dialog.message,
                default_value=dialog.default_value or "",
                url=page.url if page else "",
                timestamp=datetime.now().isoformat(),
                was_accepted=True,  # Will be updated based on action
                response_value="",
            )

            if self._dialog_handler == "auto":
                # Auto behavior: accept all dialogs, including beforeunload.
                # Accepting beforeunload means "leave the page" — dismissing
                # would cancel every navigation away from a page with an
                # unsaved-changes guard, trapping the bot there for the rest
                # of the flow. Exploration contexts are throwaway, and the
                # dialog is still recorded so the report can mention the
                # unsaved-changes prompt.
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
                    await dialog.accept()
                    captured.was_accepted = True
            elif self._dialog_handler == "accept":
                await dialog.accept()
                captured.was_accepted = True
            else:  # dismiss
                await dialog.dismiss()
                captured.was_accepted = False

            self._dialogs.append(captured)

        page.on("dialog", on_dialog)

    def _setup_download_listener(self, page: Page):
        """Set up file-download recording on a page.

        Downloads are auto-accepted by Playwright (saved to a temp dir);
        this listener only records that they happened so the AI learns the
        click/navigation worked instead of treating it as a dead button.
        """

        def on_download(download):
            self._downloads.append(
                CapturedDownload(
                    url=download.url,
                    suggested_filename=download.suggested_filename or "",
                    page_url=page.url if page else "",
                    timestamp=datetime.now().isoformat(),
                )
            )

        page.on("download", on_download)

    def _setup_popup_listener(self, page: Page):
        """Set up popup window listener on a page.

        Registered on the main page and on every popup page, so
        popup-from-popup chains are tracked too.
        """

        def on_popup(popup: Page):
            """Handle new popup windows opened via window.open()."""
            popup_url = popup.url or "about:blank"
            popup_record = CapturedPopup(
                url=popup_url,
                # The opener is the page this listener was registered on
                # (which may itself be a popup), not necessarily the main page.
                opener_url=page.url if page else "",
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

        page.on("popup", on_popup)

    def _setup_popup_monitoring(self, popup: Page):
        """Set up console/network/dialog/download/popup monitoring on a popup page.

        Reuses the same listener setup methods as the main page. The popup
        listener is included so popup-from-popup chains stay visible.
        """
        self._setup_console_listener(popup)
        self._setup_network_listeners(popup)
        self._setup_dialog_listener(popup)
        self._setup_download_listener(popup)
        self._setup_popup_listener(popup)

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
            if clear:
                self._console_messages = [m for m in self._console_messages if m.level != level]
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
            # Only clear the requests that matched our filters
            returned_set = set(id(r) for r in requests)
            self._network_requests = [r for r in self._network_requests if id(r) not in returned_set]
        return list(requests)

    def clear_network_requests(self):
        """Clear all captured (completed) network requests to free memory.

        Does not clear _pending_requests so in-flight requests can still
        be tracked to completion.
        """
        self._network_requests.clear()

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

        Returns the pinned page while a worker turn is in progress (see
        pin_active_page), else the active popup if one is open, else the
        main page. All screenshot and action methods use this property to
        interact with the correct page.
        """
        if self._pinned_page is not None and not self._pinned_page.is_closed():
            return self._pinned_page
        if self._active_popup is not None and not self._active_popup.is_closed():
            return self._active_popup
        return self._page

    def pin_active_page(self) -> Page:
        """
        Freeze active_page at its current value for the duration of a
        worker turn.

        The popup listener flips the active page the instant a popup opens.
        Without pinning, a popup arriving mid-turn can split the AI's view:
        screenshot of one page, ref list of another, and the chosen action
        executed on a third. Pinning makes the whole turn (observe → decide
        → act) target one page; the popup becomes active on the next turn's
        pin. A pinned page that closes mid-turn falls back to the normal
        popup/main resolution.

        Returns the page that was pinned.
        """
        self._pinned_page = None  # Resolve from live popup/main state
        self._pinned_page = self.active_page
        return self._pinned_page

    def unpin_active_page(self) -> None:
        """Release the turn pin; active_page follows popups again."""
        self._pinned_page = None

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
                self._active_popup = None
                return True
            except asyncio.TimeoutError:
                logger.warning(f"Timeout closing popup after {timeout}ms")
                # Don't clear reference - popup may still be open
                return False
        return False

    def switch_to_main_page(self) -> None:
        """
        Switch focus back to the main page, ignoring any open popups.

        The popup will still exist but screenshot/action methods will
        target the main page.
        """
        self._active_popup = None
        # A deliberate switch (AI action) takes effect immediately, even
        # inside a pinned turn — only the popup listener's auto-switch is
        # deferred by the pin.
        if self._pinned_page is not None:
            self._pinned_page = self._page

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
                # Deliberate switch (AI action) overrides the turn pin
                if self._pinned_page is not None:
                    self._pinned_page = popup
                return True
        except IndexError:
            pass
        return False

    def clear_popups(self):
        """Clear captured popup records (does not close them)."""
        self._popups.clear()

    # ===== Download Handling =====

    def get_downloads(self, clear: bool = False) -> list[CapturedDownload]:
        """
        Get captured file downloads.

        Args:
            clear: If True, clear downloads after returning

        Returns:
            List of captured downloads
        """
        downloads = list(self._downloads)
        if clear:
            self._downloads.clear()
        return downloads

    def clear_downloads(self):
        """Clear captured download records."""
        self._downloads.clear()

    async def screenshot(self, full_page: bool = False, timeout_ms: int | None = None) -> bytes:
        """
        Take a screenshot of the current viewport (popup or main page).

        Args:
            full_page: If True, capture the entire scrollable page
            timeout_ms: Explicit Playwright timeout in milliseconds. When
                None, the context's default action timeout applies.

        Returns:
            PNG image bytes
        """
        page = self.active_page
        if not page:
            raise RuntimeError("No page available")

        if timeout_ms is not None:
            return await page.screenshot(type="png", full_page=full_page, timeout=timeout_ms)
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

    # Runs inside each frame. Walks the document plus every open shadow
    # root (closed shadow roots are unreachable by design), numbering
    # elements from startCounter so refs stay unique across frames.
    # maxRefs bounds the count so a frame can't overflow its numbering range
    # (and caps prompt-token cost); the skipped count keeps the truncation
    # visible to the caller.
    _INJECT_REFS_SCRIPT = """
    ({startCounter, maxRefs}) => {
        // Select interactive elements
        const selectors = [
            'a', 'button', 'input', 'select', 'textarea', 'summary',
            '[role="button"]', '[role="link"]', '[role="menuitem"]',
            '[role="tab"]', '[role="checkbox"]', '[role="radio"]',
            '[role="option"]', '[role="switch"]', '[role="textbox"]',
            '[onclick]', '[tabindex]:not([tabindex="-1"])'
        ].join(',');

        // Clean up refs from previous injection to avoid duplicates,
        // in the document and in every open shadow root
        const removeOldRefs = (root) => {
            root.querySelectorAll('[data-qabot-ref]').forEach(el => {
                el.removeAttribute('data-qabot-ref');
            });
            root.querySelectorAll('*').forEach(el => {
                if (el.shadowRoot) removeOldRefs(el.shadowRoot);
            });
        };
        removeOldRefs(document);

        const refMap = {};
        let refCounter = startCounter;
        let skipped = 0;

        const visit = (root) => {
            root.querySelectorAll(selectors).forEach(el => {
                // Skip hidden elements. Either dimension being zero means
                // Playwright has no clickable point — a zero-width-but-tall
                // element would burn the full click timeout.
                const rect = el.getBoundingClientRect();
                if (rect.width === 0 || rect.height === 0) return;
                if (getComputedStyle(el).display === 'none') return;
                if (getComputedStyle(el).visibility === 'hidden') return;

                if (refCounter - startCounter >= maxRefs) {
                    skipped++;
                    return;
                }

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
            root.querySelectorAll('*').forEach(el => {
                if (el.shadowRoot) visit(el.shadowRoot);
            });
        };
        visit(document);

        return {refMap, skipped};
    }
    """

    async def inject_refs(self) -> dict[str, str]:
        """
        Inject data-qabot-ref attributes into interactive elements (on active page).

        Covers the main frame, open shadow roots, and iframe subframes
        (including cross-origin ones such as Stripe/PayPal card fields —
        Playwright can evaluate in any frame regardless of origin).

        Returns:
            Dictionary mapping ref strings ("ref_1", "ref_2", ...) to element
            descriptions. Subframe refs are prefixed with "[iframe: <host>]".
            Format aligned with Claude for Chrome MCP.
        """
        page = self.active_page
        if not page:
            raise RuntimeError("No page available")

        self._ref_frames = {}
        self._unscanned_subframes = 0
        self._unlisted_elements = 0

        # Main frame keeps the retry semantics: a destroyed execution context
        # almost always means a navigation just committed; a short
        # domcontentloaded wait lets the new document settle before the retry.
        ref_map: dict[str, str] = {}
        for attempt in range(2):
            try:
                result = await asyncio.wait_for(
                    page.evaluate(
                        self._INJECT_REFS_SCRIPT,
                        {"startCounter": 1, "maxRefs": MAIN_FRAME_MAX_REFS},
                    ),
                    timeout=REF_INJECTION_TIMEOUT_S,
                )
                ref_map = result["refMap"]
                self._unlisted_elements += result["skipped"]
                break
            except (asyncio.TimeoutError, PlaywrightError) as e:
                if attempt == 0:
                    logger.warning(
                        f"Ref injection failed ({type(e).__name__}); waiting for "
                        f"the page to settle and retrying once"
                    )
                    try:
                        await page.wait_for_load_state(
                            "domcontentloaded",
                            timeout=REF_INJECTION_TIMEOUT_S * 1000,
                        )
                    except Exception:
                        pass  # Best-effort settle; the retry may still succeed
                    continue
                raise RuntimeError(
                    f"Ref injection failed after retry: {e}"
                ) from e

        subframes = [
            f for f in page.frames
            if f is not page.main_frame and not f.is_detached()
        ]
        if len(subframes) > MAX_REF_SUBFRAMES:
            self._unscanned_subframes += len(subframes) - MAX_REF_SUBFRAMES
            logger.warning(
                "Page has %d subframes; only injecting refs into the first %d",
                len(subframes),
                MAX_REF_SUBFRAMES,
            )
            subframes = subframes[:MAX_REF_SUBFRAMES]

        if not subframes:
            return ref_map

        # Inject into all subframes concurrently so one wedged iframe costs
        # a single timeout, not one per frame, and healthy frames still get
        # refs. Each frame numbers from its own disjoint range (see
        # SUBFRAME_REF_STRIDE); base starts past the main frame's refs.
        base = (len(ref_map) // SUBFRAME_REF_STRIDE + 1) * SUBFRAME_REF_STRIDE

        async def inject_into(frame: Frame, start: int) -> dict | None:
            try:
                return await asyncio.wait_for(
                    frame.evaluate(
                        self._INJECT_REFS_SCRIPT,
                        {"startCounter": start,
                         "maxRefs": SUBFRAME_REF_STRIDE - 1},
                    ),
                    timeout=SUBFRAME_REF_INJECTION_TIMEOUT_S,
                )
            except (asyncio.TimeoutError, PlaywrightError) as e:
                # Best-effort: a wedged/mid-navigation iframe must not
                # stall the turn or hide the rest of the page.
                logger.debug(
                    "Skipping ref injection in subframe %s: %s", frame.url, e
                )
                return None

        results = await asyncio.gather(
            *(inject_into(f, base + i * SUBFRAME_REF_STRIDE)
              for i, f in enumerate(subframes))
        )

        for frame, result in zip(subframes, results):
            if result is None:
                self._unscanned_subframes += 1
                continue
            self._unlisted_elements += result["skipped"]
            label = self._frame_label(frame)
            for ref, desc in result["refMap"].items():
                ref_map[ref] = f"[iframe: {label}] {desc}" if label else desc
                self._ref_frames[self._extract_ref_number(ref)] = frame

        return ref_map

    @staticmethod
    def _frame_label(frame: Frame) -> str:
        """Short human-readable identifier for a subframe (host, else name)."""
        try:
            host = urlparse(frame.url).netloc
        except (ValueError, AttributeError):
            host = ""
        return host or frame.name or ""

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

        If exactly one element matches, returns it directly. If multiple
        elements share the same ref (page cloned elements after injection),
        tries to narrow to a single visible element. If that still leaves
        ambiguity, raises DuplicateRefError with descriptions of each
        duplicate so the AI can choose on its next turn.

        Args:
            ref: The ref string (e.g., "ref_5")

        Returns:
            Playwright Locator for the element

        Raises:
            StaleRefError: When no element matches (page changed since the
                ref map was injected). Raised immediately instead of letting
                Playwright auto-wait its full timeout on a hopeless locator.
            DuplicateRefError: When multiple elements match and can't be
                auto-resolved to a single visible one.
        """
        page = self.active_page
        if not page:
            raise RuntimeError("No page available")

        ref_num = self._extract_ref_number(ref)

        # Subframe refs must be looked up in their own frame — page.locator()
        # pierces open shadow roots but never crosses iframe boundaries.
        frame = self._ref_frames.get(ref_num)
        if frame is not None:
            if frame.is_detached():
                logger.warning(
                    "Ref %s lives in a detached iframe — ref map is stale", ref
                )
                raise StaleRefError(ref)
            locator = frame.locator(f'[data-qabot-ref="{ref_num}"]')
        else:
            locator = page.locator(f'[data-qabot-ref="{ref_num}"]')

        count = await locator.count()
        if count > 1:
            # Multiple elements share this ref (page cloned elements after
            # injection, e.g. sticky header, modal overlay, responsive layout).
            # Try to narrow to the single visible element.
            visible = locator.locator("visible=true")
            visible_count = await visible.count()
            if visible_count == 1:
                return visible

            # Can't auto-resolve — gather descriptions so the AI can choose.
            descriptions = await self._describe_duplicate_elements(
                locator, count
            )
            raise DuplicateRefError(ref, descriptions)

        if count == 0:
            logger.warning("Ref %s matched no elements — ref map is stale", ref)
            raise StaleRefError(ref)

        return locator

    _MAX_DUPLICATE_DESCRIPTIONS = 5

    async def _describe_duplicate_elements(
        self, locator, count: int
    ) -> list[str]:
        """Build human-readable descriptions for each element matched by a
        duplicate-ref locator, including tag, text, position and parent context.
        """
        descriptions: list[str] = []
        for i in range(min(count, self._MAX_DUPLICATE_DESCRIPTIONS)):
            el = locator.nth(i)
            try:
                info = await el.evaluate("""el => {
                    const rect = el.getBoundingClientRect();
                    const tag = el.tagName.toLowerCase();
                    const role = el.getAttribute('role') || '';
                    const text = (el.textContent || '').trim().slice(0, 60);
                    const ariaLabel = el.getAttribute('aria-label') || '';
                    const visible = rect.width > 0 && rect.height > 0
                        && getComputedStyle(el).visibility !== 'hidden';

                    // Walk up to find a landmark parent for context
                    let parent = el.parentElement;
                    let parentDesc = '';
                    for (let j = 0; j < 5 && parent; j++) {
                        const pTag = parent.tagName.toLowerCase();
                        const pRole = parent.getAttribute('role') || '';
                        if (['header','footer','nav','main','aside','dialog',
                             'form','section'].includes(pTag) || pRole) {
                            parentDesc = pRole ? pRole : pTag;
                            const pLabel = parent.getAttribute('aria-label') || '';
                            if (pLabel) parentDesc += ' "' + pLabel + '"';
                            break;
                        }
                        parent = parent.parentElement;
                    }

                    return {tag, role, text, ariaLabel, parentDesc, visible,
                            x: Math.round(rect.x), y: Math.round(rect.y)};
                }""")
                label = info.get("ariaLabel") or info.get("text", "")
                if len(label) > 50:
                    label = label[:47] + "..."
                role = info.get("role") or info.get("tag", "?")
                vis = "visible" if info.get("visible") else "hidden"
                parent = info.get("parentDesc", "")
                parent_str = f" in <{parent}>" if parent else ""
                desc = (
                    f'{role} "{label}" at y={info.get("y", "?")}'
                    f"{parent_str} ({vis})"
                )
            except Exception:
                desc = f"element #{i + 1} (could not inspect)"
            descriptions.append(desc)
        return descriptions

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

        # Partial coverage must be visible to the AI: without these notes, a
        # payment widget in an unscanned iframe reads as "not on the page".
        note = ""
        if self._unscanned_subframes:
            note += (
                f"\n  (note: {self._unscanned_subframes} iframe(s) could not "
                f"be scanned; elements inside them have no refs)"
            )
        if self._unlisted_elements:
            note += (
                f"\n  (note: {self._unlisted_elements} additional interactive "
                f"element(s) exceeded the ref limit and are not listed)"
            )

        if not ref_map:
            return "No interactive elements found on this page." + note

        lines = ["Interactive elements:"]
        # Sort by numeric part of ref string
        sorted_refs = sorted(ref_map.items(), key=lambda x: int(x[0].replace("ref_", "")))
        for ref, desc in sorted_refs:
            # Clean up description - remove extra whitespace
            clean_desc = " ".join(desc.split())
            lines.append(f"  {ref}: {clean_desc}")
        return "\n".join(lines) + note
