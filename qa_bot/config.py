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
MODEL_SONNET = "claude-sonnet-5"
MODEL_OPUS = "claude-opus-4-8"
# Selectable via --model / AI_MODEL / the manual workflow, never defaults.
MODEL_OPUS_5_5 = "claude-opus-5-5"
MODEL_FABLE = "claude-fable-5-1"
MODEL_FABLE_5 = "claude-fable-5"
DEFAULT_MODEL = MODEL_SONNET

AI_MODEL = os.getenv("AI_MODEL", DEFAULT_MODEL)

# OpenRouter: any "vendor/slug" model id (a Claude id never has a slash) runs
# on OpenRouterProvider instead of Claude — CLI / GitHub Action only, never a
# default. Screenshots, page text and test credentials in prompts then go to
# OpenRouter and the model vendor, not Anthropic.
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_GPT6_LUNA = "openai/gpt-6-luna"
# Reasoning effort per role ("minimal"/"low"/"medium"/"high"/"xhigh"; empty =
# the model's default). Workers default to high to match Sonnet 5's default
# adaptive effort, so a Luna-vs-Sonnet A/B compares like with like.
OPENROUTER_WORKER_EFFORT = os.getenv("OPENROUTER_WORKER_EFFORT", "high").strip()
OPENROUTER_SUPERVISOR_EFFORT = os.getenv("OPENROUTER_SUPERVISOR_EFFORT", "low").strip()
OPENROUTER_SYNTHESIS_EFFORT = os.getenv("OPENROUTER_SYNTHESIS_EFFORT", "low").strip()
# Per-request timeout. Calls are non-streaming, so this bounds a whole reply.
OPENROUTER_TIMEOUT_SECONDS = float(os.getenv("OPENROUTER_TIMEOUT_SECONDS", "300"))


def is_openrouter_model(model: str) -> bool:
    """True for OpenRouter "vendor/slug" ids (openai/gpt-6-luna)."""
    return "/" in (model or "")

# Model pricing per million tokens (USD)
# Used for cost tracking and Max Cost -> Max Tokens conversion
# Source: https://platform.claude.com/docs/en/about-claude/pricing (Jul 2026;
# Sonnet 5 re-checked 2026-09-23: "The $2/$10 per million input/output token
# pricing for Claude Sonnet 5, announced at launch as introductory pricing
# through August 31, 2026, is now the standard price. The previously scheduled
# increase to $3/$15 ... will not occur."). Note: Sonnet 5's tokenizer emits
# ~30% more tokens for the same text than Haiku/Sonnet 4.6, so a given dollar cap
# buys less exploration than the raw per-token rate ratio suggests.
MODEL_PRICING: dict[str, dict[str, float]] = {
    MODEL_HAIKU: {
        "input": 1.0,           # $1/MTok for input tokens
        "output": 5.0,          # $5/MTok for output tokens
        "cache_read": 0.10,     # $0.10/MTok (10% of input)
        "cache_creation": 1.25,    # $1.25/MTok (125% of input)
        "estimated_blended": 2.2,  # For max cost conversion (~70% input, ~30% output)
    },
    MODEL_SONNET: {
        "input": 2.0,           # $2/MTok for input tokens
        "output": 10.0,         # $10/MTok for output tokens
        "cache_read": 0.20,     # $0.20/MTok (10% of input)
        "cache_creation": 2.50,    # $2.50/MTok (125% of input)
        "estimated_blended": 4.4,  # For max cost conversion (~70% input, ~30% output)
    },
    MODEL_OPUS: {
        "input": 5.0,           # $5/MTok for input tokens
        "output": 25.0,         # $25/MTok for output tokens
        "cache_read": 0.50,     # $0.50/MTok (10% of input)
        "cache_creation": 6.25,    # $6.25/MTok (125% of input)
        "estimated_blended": 11.0,  # For max cost conversion (~70% input, ~30% output)
    },
    MODEL_OPUS_5_5: {
        "input": 4.0,           # $4/MTok for input tokens
        "output": 20.0,         # $20/MTok for output tokens
        "cache_read": 0.20,     # $0.20/MTok (5% of input)
        "cache_creation": 5.0,     # $5/MTok (125% of input)
        "estimated_blended": 8.8,  # For max cost conversion (~70% input, ~30% output)
    },
    MODEL_FABLE: {
        "input": 10.0,          # $10/MTok for input tokens
        "output": 50.0,         # $50/MTok for output tokens
        "cache_read": 0.25,     # $0.25/MTok (2.5% of input)
        "cache_creation": 12.5,    # $12.50/MTok (125% of input)
        "estimated_blended": 22.0,  # For max cost conversion (~70% input, ~30% output)
    },
    MODEL_FABLE_5: {
        "input": 10.0,          # $10/MTok for input tokens
        "output": 50.0,         # $50/MTok for output tokens
        "cache_read": 1.0,      # $1/MTok (10% of input)
        "cache_creation": 12.5,    # $12.50/MTok (125% of input)
        "estimated_blended": 22.0,  # For max cost conversion (~70% input, ~30% output)
    },
    # GPT-6 Luna via OpenRouter (openrouter.ai/api/v1/models, read 2026-09-23).
    # Without this row the unknown-model fallback prices Luna at the per-field
    # maximum and the cost cap trips ~100x early. Prompts over 272K tokens bill
    # 2x input / 1.5x output; worker requests stay far below that, so it is not
    # modelled. OpenRouter's own usage.cost is reported as charged_cost_usd.
    MODEL_GPT6_LUNA: {
        "input": 0.10,          # $0.10/MTok for input tokens
        "output": 0.50,         # $0.50/MTok for output tokens (reasoning included)
        "cache_read": 0.01,     # $0.01/MTok (10% of input)
        "cache_creation": 0.125,   # $0.125/MTok (125% of input)
        "estimated_blended": 0.22,  # For max cost conversion (~70% input, ~30% output)
    },
}

# Fallback pricing for unknown models: the per-FIELD maximum over every known
# row, not any one model's rates. An unpriced model must tighten the cost cap,
# never loosen it — and no single row is the most expensive on every field
# (Fable 5.1 has the top input/output but a 0.025x cache read), so picking a
# model rots the moment a row is added.
DEFAULT_MODEL_PRICING: dict[str, float] = {
    field: max(rates[field] for rates in MODEL_PRICING.values())
    for field in next(iter(MODEL_PRICING.values()))
}


def get_model_pricing(model: str) -> dict[str, float]:
    """Get pricing for a model, with fallback to the most expensive known rates."""
    if model not in MODEL_PRICING:
        logger.warning(
            f"Unknown model '{model}', using the most-expensive known rates "
            f"for cost calculation so the cost cap stays conservative"
        )
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
# Past worker turns go into conversation history as a compact stub (turn, URL,
# one-off context) instead of the full action prompt — see HISTORY_TURN_STUB in
# orchestrator/worker.py. HISTORY_COMPACTION=false stores full prompts (A/B).
HISTORY_COMPACTION = os.getenv("HISTORY_COMPACTION", "true").lower() == "true"


def recheck_criticals_default(interactive: bool) -> bool:
    """Whether CRITICAL findings get an independent re-check this run.

    ``RECHECK_CRITICALS`` = ``true`` / ``false`` forces it; unset or
    ``auto`` means on for non-interactive runs (CLI/CI — a false critical
    fails the deploy gate with no human in the loop) and off for the web UI
    (a human is watching and judges the finding). Read at call time so a
    workflow step's ``env:`` reaches the action's container.
    """
    value = os.getenv("RECHECK_CRITICALS", "auto").strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    return not interactive

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
# /api/flow/start checks the API key against Anthropic before a run starts so
# a revoked/typo'd key fails in <1s with a clear message. Set to true only
# where no real key exists (the e2e suite runs the server with a dummy key).
SKIP_API_KEY_VALIDATION = os.getenv("SKIP_API_KEY_VALIDATION", "false").lower() == "true"

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
