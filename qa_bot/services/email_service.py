"""Testmail.app email service for testing email verification flows."""

import asyncio
import re
import uuid
from dataclasses import dataclass
from typing import Optional

import httpx


@dataclass
class Email:
    """Represents an email received from Testmail.app."""

    id: str
    to: str
    from_addr: str
    subject: str
    body_text: str
    body_html: str
    received_at: str


@dataclass
class EmailInbox:
    """Represents a Testmail.app inbox (namespace + tag)."""

    email_address: str
    inbox_id: str  # The tag portion, used for querying


class EmailService:
    """Testmail.app email service for testing email verification flows.

    Usage:
        service = EmailService(api_key="your-api-key", namespace="your-namespace")
        inbox = await service.create_inbox()
        # Use inbox.email_address for signup
        email = await service.wait_for_email(inbox, timeout_seconds=60)
        links = service.extract_links(email)
        code = service.extract_code(email)
        # No need to delete inbox - tags are disposable
    """

    API_BASE = "https://api.testmail.app/api/json"

    def __init__(self, api_key: str, namespace: str):
        """Initialize the email service with Testmail.app credentials.

        Args:
            api_key: Your Testmail.app API key from the developer console
            namespace: Your Testmail.app namespace (e.g., "abc123")
        """
        self.api_key = api_key
        self.namespace = namespace
        self._client = httpx.AsyncClient(timeout=120.0)

    async def create_inbox(self, prefix: str = "qa-test") -> EmailInbox:
        """Create a new test email inbox using a unique tag.

        Args:
            prefix: Prefix for the tag (default: "qa-test")

        Returns:
            EmailInbox with the generated email address and tag
        """
        # Generate unique tag: prefix-uuid
        tag = f"{prefix}-{uuid.uuid4().hex[:8]}"
        email_address = f"{self.namespace}.{tag}@inbox.testmail.app"

        return EmailInbox(email_address=email_address, inbox_id=tag)

    async def wait_for_email(
        self,
        inbox: EmailInbox,
        timeout_seconds: int = 60,
        subject_contains: Optional[str] = None,
    ) -> Email:
        """Wait for an email to arrive in the inbox.

        Uses Testmail.app's livequery feature to wait for new emails.

        Args:
            inbox: The inbox to check
            timeout_seconds: How long to wait for an email (default 60s)
            subject_contains: Optional filter - only return email if subject contains this

        Returns:
            The received Email

        Raises:
            TimeoutError: If no email arrives within the timeout
            ValueError: If subject filter doesn't match
        """
        params = {
            "apikey": self.api_key,
            "namespace": self.namespace,
            "tag": inbox.inbox_id,
            "livequery": "true",
            "limit": 1,
        }

        try:
            response = await asyncio.wait_for(
                self._client.get(self.API_BASE, params=params),
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()

            if data.get("result") != "success":
                raise ValueError(f"API error: {data.get('message', 'Unknown error')}")

            emails = data.get("emails", [])
            if not emails:
                raise TimeoutError(f"No email received within {timeout_seconds} seconds")

            email_data = emails[0]

            # Check subject filter if specified
            subject = email_data.get("subject", "")
            if subject_contains and subject_contains.lower() not in subject.lower():
                raise ValueError(
                    f"Email subject '{subject}' does not contain '{subject_contains}'"
                )

            return Email(
                id=email_data.get("id", ""),
                to=f"{self.namespace}.{inbox.inbox_id}@inbox.testmail.app",
                from_addr=email_data.get("from", ""),
                subject=subject,
                body_text=email_data.get("text", ""),
                body_html=email_data.get("html", ""),
                received_at=str(email_data.get("timestamp", "")),
            )

        except asyncio.TimeoutError:
            raise TimeoutError(f"No email received within {timeout_seconds} seconds")
        except httpx.HTTPStatusError as e:
            raise ValueError(f"API request failed: {e}")

    def extract_links(self, email: Email) -> list[str]:
        """Extract all HTTP/HTTPS links from email body.

        Args:
            email: The email to extract links from

        Returns:
            List of URLs found in the email
        """
        # Use HTML body if available, fall back to text
        body = email.body_html or email.body_text
        pattern = r'https?://[^\s<>"\'\])]+'
        links = re.findall(pattern, body)
        # Clean up any trailing punctuation that might have been captured
        cleaned_links = []
        for link in links:
            # Remove trailing punctuation that's likely not part of the URL
            link = link.rstrip(".,;:!?)")
            if link:
                cleaned_links.append(link)
        return cleaned_links

    def extract_code(
        self, email: Email, pattern: str = r"\b\d{6}\b"
    ) -> Optional[str]:
        """Extract a verification code from email body.

        Args:
            email: The email to extract code from
            pattern: Regex pattern for the code (default: 6 digits)

        Returns:
            The extracted code or None if not found
        """
        match = re.search(pattern, email.body_text)
        return match.group(0) if match else None

    async def delete_inbox(self, inbox: EmailInbox) -> None:
        """No-op for Testmail.app - tags are disposable and don't need cleanup.

        Args:
            inbox: The inbox (ignored)
        """
        # Testmail.app uses tag-based addressing - no explicit cleanup needed
        pass

    async def close(self) -> None:
        """Close the HTTP client and clean up resources."""
        if self._client:
            await self._client.aclose()
            self._client = None
