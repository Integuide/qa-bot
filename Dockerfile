# QA Bot GitHub Action - Docker Container
#
# Based on Microsoft's Playwright Python image which includes:
# - Python 3.x
# - Playwright with browser dependencies
# - Ubuntu Jammy base

FROM mcr.microsoft.com/playwright/python:v1.49.1-jammy

LABEL org.opencontainers.image.title="QA Bot"
LABEL org.opencontainers.image.description="AI-powered website QA testing using Claude"
LABEL org.opencontainers.image.source="https://github.com/integuide/qa-bot"

WORKDIR /app

# Install additional tools needed for the entrypoint
RUN apt-get update && apt-get install -y \
    git \
    curl \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers (Chromium only for CI efficiency)
RUN playwright install chromium

# Copy application code (qa_bot package and action files)
COPY qa_bot/ qa_bot/

# Environment variables for CI mode.
# LOG_CHAT_HISTORY must be true for issue screenshots to be persisted (the
# chat logger owns the log directory); LOG_SCREENSHOTS=false keeps noisy
# per-turn screenshots disabled. LOG_DIR points at a container-private path:
# GitHub runs Docker actions with the host-mounted runner workspace as the
# working directory, and chat logs (which can contain page content and typed
# credentials) must never land there. Only the screenshots/ subdirectory is
# exported by entrypoint.sh — chat logs never leave the container.
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV BROWSER_HEADLESS=true
ENV LOG_CHAT_HISTORY=true
ENV LOG_SCREENSHOTS=false
ENV LOG_DIR=/tmp/qa-bot-logs
ENV ENVIRONMENT=ci

# Make entrypoint executable
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
