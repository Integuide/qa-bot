"""Browser pool for managing isolated browser contexts per flow."""

import asyncio
import base64
import logging
from typing import Optional
from urllib.parse import urlparse

from playwright.async_api import async_playwright, Browser, BrowserContext, Playwright

from qa_bot.config import IGNORE_HTTPS_ERRORS


logger = logging.getLogger(__name__)


# Permissions to auto-grant to all browser contexts
# This prevents permission prompts from blocking automation
AUTO_GRANT_PERMISSIONS = [
    "geolocation",
    "notifications",
    "camera",
    "microphone",
    "clipboard-read",
    "clipboard-write",
]


class BrowserPool:
    """
    Manages browser resources for flow-based exploration.

    Each flow gets its own isolated browser context, which means:
    - No state leakage between flows
    - Independent cookies/localStorage per flow
    - Checkpoints can be restored with full browser state
    """

    def __init__(
        self,
        max_pages: int = 10,
        headless: bool = True,
        viewport_width: int = 1280,
        viewport_height: int = 720
    ):
        self.max_pages = max_pages
        self.headless = headless
        self.viewport = {"width": viewport_width, "height": viewport_height}

        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._lock = asyncio.Lock()
        self._started = False

    async def start(self):
        """Initialize Playwright and launch browser."""
        if self._started:
            return

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=self.headless)
        self._started = True

    async def stop(self):
        """Close browser and stop Playwright."""
        if not self._started:
            return

        if self._browser:
            await self._browser.close()
            self._browser = None

        if self._playwright:
            await self._playwright.stop()
            self._playwright = None

        self._started = False

    async def create_isolated_context(
        self,
        storage_state: Optional[dict] = None
    ) -> BrowserContext:
        """
        Create a new isolated browser context for a flow.

        Each flow gets its own context to prevent cross-flow state
        contamination (e.g., one flow logging out affecting another).

        Args:
            storage_state: Optional storage state to restore (from checkpoint).
                          If None, creates a fresh context.

        Returns:
            A new BrowserContext that is independent of other contexts.
        """
        if not self._started:
            raise RuntimeError("BrowserPool not started. Call start() first.")

        if IGNORE_HTTPS_ERRORS:
            logger.info("Creating browser context with HTTPS error bypass enabled")

        context = await self._browser.new_context(
            viewport=self.viewport,
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 QABot/1.0",
            storage_state=storage_state if storage_state else None,
            # Auto-grant permissions to prevent prompts from blocking automation
            permissions=AUTO_GRANT_PERMISSIONS,
            # Bypass HTTPS/certificate errors (self-signed certs, expired certs, etc.)
            # Controlled by IGNORE_HTTPS_ERRORS config flag (default: false)
            ignore_https_errors=IGNORE_HTTPS_ERRORS,
        )

        return context

    async def snapshot_storage_state(self, context: BrowserContext) -> dict:
        """
        Capture the current storage state of a browser context.

        This includes cookies, localStorage, and sessionStorage.
        Used for creating checkpoints at branch points.

        Args:
            context: The browser context to snapshot

        Returns:
            Storage state dict that can be passed to create_isolated_context()
        """
        return await context.storage_state()

    @property
    def browser(self) -> Optional[Browser]:
        """Get the browser instance."""
        return self._browser

    async def apply_http_auth(
        self,
        context: BrowserContext,
        username: str,
        password: str,
        target_url: str,
    ) -> None:
        """
        Apply HTTP Basic Auth credentials to a browser context.

        Uses route interception to add Authorization header to requests
        matching the target domain. This prevents credential leakage to
        third-party resources.

        Args:
            context: The browser context to apply auth to
            username: HTTP Basic Auth username
            password: HTTP Basic Auth password
            target_url: The URL requiring auth (credentials scoped to this domain)
        """
        # Encode credentials as base64
        credentials = f"{username}:{password}"
        encoded = base64.b64encode(credentials.encode()).decode()
        auth_header = f"Basic {encoded}"

        # Parse target URL to scope auth to this domain only
        parsed = urlparse(target_url)
        # Match scheme://host/** to avoid credential leakage to third parties
        route_pattern = f"{parsed.scheme}://{parsed.netloc}/**"

        logger.debug(f"Applying HTTP auth to route pattern: {route_pattern}")

        async def add_auth_header(route):
            """Route handler that adds Authorization header to matching requests."""
            headers = {**route.request.headers, "Authorization": auth_header}
            await route.continue_(headers=headers)

        # Apply only to requests matching the target domain
        await context.route(route_pattern, add_auth_header)
