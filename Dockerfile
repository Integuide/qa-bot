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

# Environment variables for CI mode
ENV PYTHONUNBUFFERED=1
ENV BROWSER_HEADLESS=true
ENV LOG_CHAT_HISTORY=false
ENV LOG_SCREENSHOTS=false
ENV ENVIRONMENT=ci

# Make entrypoint executable
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
