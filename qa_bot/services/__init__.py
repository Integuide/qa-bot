"""Services module for QA Bot external integrations."""

from .email_service import Email, EmailInbox, EmailService

__all__ = ["EmailService", "EmailInbox", "Email"]
