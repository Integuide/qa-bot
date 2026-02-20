import logging
import os
import re
import subprocess
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def _get_port_from_branch() -> int:
    """
    Detect port from git branch name.
    Branch pattern: *_claude_N -> Port 81N0
    Examples:
      - 0105_claude_0 -> 8100
      - 0105_claude_2 -> 8120
      - master -> 8101 (default)
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            branch = result.stdout.strip()
            match = re.search(r"_claude_(\d+)$", branch)
            if match:
                claude_num = int(match.group(1))
                return 8100 + (claude_num * 10)
    except Exception:
        pass
    return 8101  # Default port for master/other branches


# AI Provider Configuration
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Model ID constants (single source of truth for model identifiers)
MODEL_HAIKU = "claude-haiku-4-5"
MODEL_SONNET = "claude-sonnet-4-6"
MODEL_OPUS = "claude-opus-4-6"
DEFAULT_MODEL = MODEL_HAIKU

AI_MODEL = os.getenv("AI_MODEL", DEFAULT_MODEL)

# Model pricing per million tokens (USD)
# Used for cost tracking and Max Cost -> Max Tokens conversion
# Source: https://platform.claude.com/docs/en/about-claude/pricing (Feb 2026)
MODEL_PRICING: dict[str, dict[str, float]] = {
    MODEL_HAIKU: {
        "input": 1.0,           # $1/MTok for input tokens
        "output": 5.0,          # $5/MTok for output tokens
        "cache_read": 0.10,     # $0.10/MTok (10% of input)
        "cache_creation": 1.25,    # $1.25/MTok (125% of input)
        "estimated_blended": 2.2,  # For max cost conversion (~70% input, ~30% output)
    },
    MODEL_SONNET: {
        "input": 3.0,           # $3/MTok for input tokens
        "output": 15.0,         # $15/MTok for output tokens
        "cache_read": 0.30,     # $0.30/MTok (10% of input)
        "cache_creation": 3.75,    # $3.75/MTok (125% of input)
        "estimated_blended": 6.6,  # For max cost conversion (~70% input, ~30% output)
    },
    MODEL_OPUS: {
        "input": 5.0,           # $5/MTok for input tokens
        "output": 25.0,         # $25/MTok for output tokens
        "cache_read": 0.50,     # $0.50/MTok (10% of input)
        "cache_creation": 6.25,    # $6.25/MTok (125% of input)
        "estimated_blended": 11.0,  # For max cost conversion (~70% input, ~30% output)
    },
}

# Default pricing for unknown models (uses haiku rates)
DEFAULT_MODEL_PRICING = MODEL_PRICING[DEFAULT_MODEL]


def get_model_pricing(model: str) -> dict[str, float]:
    """Get pricing for a model, with fallback to default."""
    if model not in MODEL_PRICING:
        logger.warning(f"Unknown model '{model}', using default (haiku) pricing for cost calculation")
    return MODEL_PRICING.get(model, DEFAULT_MODEL_PRICING)


def calculate_cost(
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
) -> float:
    """Calculate cost in USD for given token counts and model."""
    pricing = get_model_pricing(model)
    cost = (
        (input_tokens / 1_000_000) * pricing["input"]
        + (output_tokens / 1_000_000) * pricing["output"]
        + (cache_read_tokens / 1_000_000) * pricing["cache_read"]
        + (cache_creation_tokens / 1_000_000) * pricing["cache_creation"]
    )
    return cost


def max_cost_to_tokens(model: str, max_cost_usd: float) -> int:
    """Convert a max cost budget to estimated max tokens using blended rate."""
    pricing = get_model_pricing(model)
    blended_rate = pricing["estimated_blended"]
    # cost = tokens / 1M * rate => tokens = cost * 1M / rate
    return int(max_cost_usd * 1_000_000 / blended_rate)


# Browser Configuration
BROWSER_HEADLESS = os.getenv("BROWSER_HEADLESS", "true").lower() == "true"
VIEWPORT_WIDTH = int(os.getenv("VIEWPORT_WIDTH", "1280"))
VIEWPORT_HEIGHT = int(os.getenv("VIEWPORT_HEIGHT", "720"))
# Bypass HTTPS certificate errors (self-signed, expired, wrong host)
# WARNING: Enabling this silently accepts invalid certificates. Use only for testing.
IGNORE_HTTPS_ERRORS = os.getenv("IGNORE_HTTPS_ERRORS", "false").lower() == "true"

# Warn at module load if HTTPS errors are being bypassed
if IGNORE_HTTPS_ERRORS:
    _env = os.getenv("ENVIRONMENT", "development")
    if _env == "production":
        logger.warning(
            "SECURITY WARNING: IGNORE_HTTPS_ERRORS=true in production! "
            "This bypasses certificate validation and is a security risk."
        )
    else:
        logger.warning(
            "IGNORE_HTTPS_ERRORS=true - bypassing HTTPS certificate validation. "
            "Only use this for testing internal/staging sites with self-signed certs."
        )

# Flow Exploration Settings
_max_agents_raw = int(os.getenv("MAX_AGENTS", "3"))
MAX_AGENTS = max(1, min(20, _max_agents_raw))  # Clamp to 1-20 range
MAX_DURATION_MINUTES = int(os.getenv("MAX_DURATION_MINUTES", "30"))  # 30 minutes default
MAX_CONCURRENT_API_CALLS = int(os.getenv("MAX_CONCURRENT_API_CALLS", "2"))  # Limit concurrent Anthropic API calls
MAX_COST_CAP_USD = 200.0  # Maximum allowed cost per run (safety cap)

# Production Safeguards
MAX_CONCURRENT_EXPLORATIONS = int(os.getenv("MAX_CONCURRENT_EXPLORATIONS", "3"))  # Max parallel exploration sessions
QUEUE_TIMEOUT_SECONDS = int(os.getenv("QUEUE_TIMEOUT_SECONDS", "600"))  # 10 min timeout for queued requests
CLEANUP_INTERVAL_SECONDS = int(os.getenv("CLEANUP_INTERVAL_SECONDS", "300"))  # 5 min background cleanup interval

# Logging Configuration
LOG_CHAT_HISTORY = os.getenv("LOG_CHAT_HISTORY", "true").lower() == "true"
LOG_DIR = os.getenv("LOG_DIR", "./logs")
LOG_SCREENSHOTS = os.getenv("LOG_SCREENSHOTS", "auto").lower()  # "auto", "true", "false"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()  # DEBUG, INFO, WARNING, ERROR
LOG_MAX_RUNS = int(os.getenv("LOG_MAX_RUNS", "10"))  # Keep only last N exploration logs

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
# Port is auto-detected from branch name if not explicitly set
# Branch *_claude_N -> Port 81N0 (e.g., 0105_claude_0 -> 8100)
API_PORT = int(os.getenv("API_PORT", str(_get_port_from_branch())))
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

# Email Testing Service (Testmail.app) - for testing email verification flows
# Get API key and namespace at https://testmail.app
TESTMAIL_API_KEY = os.getenv("TESTMAIL_API_KEY", "")
TESTMAIL_NAMESPACE = os.getenv("TESTMAIL_NAMESPACE", "")


def is_api_key_required() -> bool:
    """Return True if user must provide their own Anthropic API key.

    In production, users must always provide their own key.
    In development, the server's key is used by default but users can override.
    """
    return ENVIRONMENT != "development" or not ANTHROPIC_API_KEY
