#!/usr/bin/env python3
"""CLI entry point for QA Bot - runs exploration without web server.

This provides a command-line interface to the same core exploration logic
used by the web UI. Results can be output as markdown or JSON.

Usage:
    python cli.py https://example.com
    python cli.py https://example.com --goal "Test checkout flow" --max-duration 10
    python cli.py https://example.com --output json --output-file results.json
    python cli.py https://example.com --log-level full  # Show all activity

    # With credentials from file
    python cli.py https://example.com --credentials .env.test

    # With inline credentials
    python cli.py https://example.com --credential "USERNAME=user" --credential "PASSWORD=pass"
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from qa_bot.config import LOG_LEVEL, MAX_CONCURRENT_API_CALLS

# Configure logging with level from environment
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

from qa_bot.ai.claude_provider import ClaudeProvider
from qa_bot.orchestrator.coordinator import FlowExplorationOrchestrator


LogLevel = Literal["quiet", "summary", "full"]


def parse_credentials_file(file_path: str) -> dict[str, str]:
    """
    Parse credentials from a file in env format (KEY=value).

    Lines starting with # are ignored.
    Empty lines are ignored.
    """
    credentials = {}
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Credentials file not found: {file_path}")

    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            # Parse KEY=value
            if '=' not in line:
                print(f"Warning: Skipping invalid line {line_num} in {file_path}: missing '='", file=sys.stderr)
                continue
            key, _, value = line.partition('=')
            key = key.strip()
            value = value.strip()
            # Remove quotes if present
            if (value.startswith('"') and value.endswith('"')) or \
               (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]
            credentials[key] = value

    return credentials


def parse_inline_credential(credential_str: str) -> tuple[str, str]:
    """
    Parse a single inline credential in KEY=value format.

    Returns (key, value) tuple.
    """
    if '=' not in credential_str:
        raise ValueError(f"Invalid credential format: {credential_str}. Expected KEY=value")
    key, _, value = credential_str.partition('=')
    return key.strip(), value.strip()


def format_event_for_log(event: dict, log_level: LogLevel) -> str | None:
    """
    Format an event for CLI logging based on log level.

    Args:
        event: The event dict from the orchestrator
        log_level: One of 'quiet', 'summary', or 'full'

    Returns:
        Formatted string to print, or None to skip
    """
    event_type = event.get("type", "")
    data = event.get("data", {})

    if log_level == "quiet":
        return None

    # Summary level: flow lifecycle and issues only
    if log_level == "summary":
        if event_type == "flow_started":
            flow_name = data.get("flow_name", "Unknown")
            return f"[Flow Started] {flow_name}"
        elif event_type == "flow_completed":
            flow_name = data.get("flow_name", "Unknown")
            reason = data.get("reason", "")
            return f"[Flow Completed] {flow_name}" + (f" - {reason}" if reason else "")
        elif event_type == "flow_failed":
            flow_name = data.get("flow_name", "Unknown")
            error = data.get("error", "Unknown error")
            return f"[Flow Failed] {flow_name}: {error}"
        elif event_type == "issue":
            severity = data.get("severity", "unknown")
            title = data.get("title", "Unknown issue")
            return f"[Issue] [{severity}] {title}"
        elif event_type == "exploration_complete":
            flows = data.get("flows_explored", 0)
            issues = data.get("issues_found", 0)
            duration = data.get("duration_seconds", 0)
            return f"[Complete] {flows} flows explored, {issues} issues found ({duration:.1f}s)"
        return None

    # Full level: all events (similar to web UI activity log)
    if log_level == "full":
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Lifecycle events
        if event_type == "exploration_started":
            url = data.get("target_url", "")
            return f"[{timestamp}] Exploration started: {url}"
        elif event_type == "exploration_complete":
            flows = data.get("flows_explored", 0)
            issues = data.get("issues_found", 0)
            duration = data.get("duration_seconds", 0)
            return f"[{timestamp}] Exploration complete: {flows} flows, {issues} issues ({duration:.1f}s)"
        elif event_type == "stopping":
            return f"[{timestamp}] Stopping exploration..."

        # Flow events
        elif event_type == "flow_started":
            flow_name = data.get("flow_name", "Unknown")
            worker_id = data.get("worker_id", "")
            return f"[{timestamp}] [Flow] Started: {flow_name} (worker: {worker_id})"
        elif event_type == "flow_completed":
            flow_name = data.get("flow_name", "Unknown")
            reason = data.get("reason", "")
            return f"[{timestamp}] [Flow] Completed: {flow_name}" + (f" - {reason}" if reason else "")
        elif event_type == "flow_failed":
            flow_name = data.get("flow_name", "Unknown")
            error = data.get("error", "Unknown error")
            return f"[{timestamp}] [Flow] Failed: {flow_name}: {error}"
        elif event_type == "flow_skipped_by_supervisor":
            flow_name = data.get("flow_name", "Unknown")
            reason = data.get("reason", "")
            return f"[{timestamp}] [Flow] Skipped: {flow_name}" + (f" - {reason}" if reason else "")

        # Worker events
        elif event_type == "worker_started":
            worker_id = data.get("worker_id", "")
            return f"[{timestamp}] [Worker] Started: {worker_id}"
        elif event_type == "worker_stopped":
            worker_id = data.get("worker_id", "")
            return f"[{timestamp}] [Worker] Stopped: {worker_id}"
        elif event_type == "worker_error":
            worker_id = data.get("worker_id", "")
            error = data.get("error", "Unknown error")
            return f"[{timestamp}] [Worker] Error ({worker_id}): {error}"

        # Action events
        elif event_type == "action":
            action_type = data.get("action_type", "unknown")
            description = data.get("description", "")
            worker_id = data.get("worker_id", "")
            if action_type in ("left_click", "right_click", "double_click", "triple_click"):
                element = data.get("element", "unknown element")
                ref = data.get("ref", "")
                click_name = action_type.replace("_", " ").title()
                return f"[{timestamp}] [Action] {click_name}: {element}" + (f" ({ref})" if ref else "")
            elif action_type == "hover":
                element = data.get("element", "unknown element")
                return f"[{timestamp}] [Action] Hover: {element}"
            elif action_type == "type":
                element = data.get("element", "unknown element")
                text = data.get("text", "")[:30]  # Truncate long text
                return f"[{timestamp}] [Action] Type: '{text}' into {element}"
            elif action_type == "form_input":
                element = data.get("element", "unknown element")
                value = data.get("value", "")
                return f"[{timestamp}] [Action] Form input: {value} into {element}"
            elif action_type == "scroll":
                direction = data.get("scroll_direction", "down")
                return f"[{timestamp}] [Action] Scroll {direction}"
            elif action_type == "scroll_to":
                element = data.get("element", "unknown element")
                return f"[{timestamp}] [Action] Scroll to: {element}"
            elif action_type == "navigate":
                url = data.get("url", "")
                return f"[{timestamp}] [Action] Navigate: {url}"
            elif action_type == "wait":
                duration = data.get("duration", 0)
                return f"[{timestamp}] [Action] Wait {duration}s"
            elif action_type == "key":
                key = data.get("key", "")
                modifiers = data.get("modifiers", [])
                key_str = "+".join(modifiers + [key]) if modifiers else key
                return f"[{timestamp}] [Action] Press key: {key_str}"
            elif action_type == "left_click_drag":
                start = data.get("start_coordinate", [])
                end = data.get("coordinate", [])
                return f"[{timestamp}] [Action] Drag: {start} → {end}"
            elif action_type == "screenshot":
                full_page = data.get("full_page", False)
                return f"[{timestamp}] [Action] Screenshot" + (" (full page)" if full_page else "")
            elif action_type == "zoom":
                region = data.get("region", [])
                return f"[{timestamp}] [Action] Zoom: {region}"
            elif action_type == "resize":
                width = data.get("width", 0)
                height = data.get("height", 0)
                return f"[{timestamp}] [Action] Resize: {width}x{height}"
            elif action_type == "add_flow":
                flow_name = data.get("flow_name", "")
                return f"[{timestamp}] [Action] Add flow: {flow_name}"
            elif action_type == "done":
                reason = data.get("reason", "")
                return f"[{timestamp}] [Action] Done: {reason}"
            elif action_type == "block":
                reason = data.get("reason", "")
                return f"[{timestamp}] [Action] Blocked: {reason}"
            else:
                return f"[{timestamp}] [Action] {action_type}: {description}"

        # Issue events
        elif event_type == "issue":
            severity = data.get("severity", "unknown")
            title = data.get("title", "Unknown issue")
            description = data.get("description", "")[:100]  # Truncate
            return f"[{timestamp}] [Issue] [{severity.upper()}] {title}" + (f"\n           {description}" if description else "")

        # Checkpoint events
        elif event_type == "checkpoint_created":
            flow_name = data.get("flow_name", "Unknown")
            return f"[{timestamp}] [Checkpoint] Created for: {flow_name}"
        elif event_type == "checkpoint_claimed":
            flow_name = data.get("flow_name", "Unknown")
            return f"[{timestamp}] [Checkpoint] Claimed: {flow_name}"

        # Supervisor events
        elif event_type == "supervisor_started":
            return f"[{timestamp}] [Supervisor] Started"
        elif event_type == "supervisor_stopped":
            return f"[{timestamp}] [Supervisor] Stopped"
        elif event_type == "supervisor_action":
            action = data.get("action", "unknown")
            target = data.get("target", "")
            return f"[{timestamp}] [Supervisor] Action: {action}" + (f" -> {target}" if target else "")
        elif event_type == "supervisor_reviewing_flows":
            count = data.get("pending_count", 0)
            return f"[{timestamp}] [Supervisor] Reviewing {count} pending flows"
        elif event_type == "supervisor_reviewing_block":
            worker_id = data.get("worker_id", "")
            return f"[{timestamp}] [Supervisor] Reviewing blocked worker: {worker_id}"

        # Approval events
        elif event_type == "approval_request":
            action_desc = data.get("action_description", "Unknown action")
            return f"[{timestamp}] [Approval] Action requires approval: {action_desc}"
        elif event_type == "flow_blocked_for_approval":
            reason = data.get("reason", "")
            return f"[{timestamp}] [Approval] Flow blocked for approval: {reason}"

        # AI thinking events (condensed for CLI)
        elif event_type == "ai_thinking_start":
            worker_id = data.get("worker_id", "")
            return f"[{timestamp}] [AI] Thinking... ({worker_id})"
        elif event_type == "ai_thinking_complete":
            # Skip - we already showed start
            return None
        elif event_type == "ai_thinking_delta":
            # Skip streaming deltas in CLI (too verbose)
            return None

        # AI error events
        elif event_type == "ai_error":
            error = data.get("error", "Unknown error")[:100]
            return f"[{timestamp}] [AI Error] {error}"

        # Screenshot events
        elif event_type == "screenshot":
            url = data.get("url", "")[:60]
            index = data.get("index", 0)
            return f"[{timestamp}] [Screenshot] #{index} captured: {url}"

        # Synthesis events
        elif event_type == "synthesis_started":
            return f"[{timestamp}] [Synthesis] Generating report..."
        elif event_type == "synthesis_complete":
            return f"[{timestamp}] [Synthesis] Report complete"
        elif event_type == "synthesis_error":
            error = data.get("error", "Unknown error")
            return f"[{timestamp}] [Synthesis] Error: {error}"

        # Progress events
        elif event_type == "progress":
            active = data.get("active_workers", 0)
            pending = data.get("pending_flows", 0)
            completed = data.get("completed_flows", 0)
            tokens = data.get("tokens_used", 0)
            return f"[{timestamp}] [Progress] Workers: {active} active | Flows: {completed} done, {pending} pending | Tokens: {tokens:,}"

        # Error events
        elif event_type == "error":
            message = data.get("message", "Unknown error")
            fatal = data.get("fatal", False)
            prefix = "FATAL" if fatal else "Error"
            return f"[{timestamp}] [{prefix}] {message}"

        # Unknown events - show in debug
        else:
            return f"[{timestamp}] [{event_type}] {json.dumps(data)[:100]}"

    return None


async def run_exploration(
    url: str,
    goal: str,
    max_agents: int,
    max_duration: int,
    api_key: str,
    model: str,
    headless: bool = True,
    log_level: LogLevel = "quiet",
    credentials: Optional[dict[str, str]] = None,
    max_cost_usd: float = 5.0,
    skip_permissions: bool = False
) -> dict:
    """
    Run QA exploration and return results.

    Args:
        url: Target URL to test
        goal: Testing goal/focus
        max_agents: Maximum parallel agents
        max_duration: Maximum duration in minutes
        api_key: Anthropic API key
        model: Claude model to use
        headless: Run browser in headless mode
        log_level: One of 'quiet', 'summary', or 'full'
        credentials: Optional dict of credentials (e.g., {"USERNAME": "user", "PASSWORD": "pass"})
        max_cost_usd: Maximum cost in USD (default: $5.00)
        skip_permissions: If True, auto-approve all irreversible actions (dangerous!)

    Returns:
        dict with report, issues, and metadata
    """
    ai_provider = ClaudeProvider(
        api_key=api_key,
        model=model,
        max_concurrent_calls=MAX_CONCURRENT_API_CALLS
    )
    orchestrator = FlowExplorationOrchestrator(
        ai_provider=ai_provider,
        max_agents=max_agents,
        headless=headless,
        credentials=credentials,
        skip_permissions=skip_permissions
    )

    events = []
    report = None
    issues = []
    flows_explored = 0
    duration_seconds = 0

    if log_level != "quiet":
        print(f"Starting exploration of {url}", file=sys.stderr)
        print(f"Goal: {goal}", file=sys.stderr)
        limit_info = f"Max agents: {max_agents}, Max duration: {max_duration} min, Max cost: ${max_cost_usd:.2f}"
        print(limit_info, file=sys.stderr)
        print("-" * 70, file=sys.stderr)

    async for event in orchestrator.run_exploration(
        target_url=url,
        goal=goal,
        max_agents=max_agents,
        max_duration_minutes=max_duration,
        max_cost_usd=max_cost_usd
    ):
        events.append(event)
        event_type = event.get("type", "")

        # Capture synthesis report
        if event_type == "synthesis_complete":
            report = event.get("data", {}).get("report", "")

        # Capture issues
        if event_type == "issue":
            issues.append(event.get("data", {}))

        # Capture final summary
        if event_type == "exploration_complete":
            data = event.get("data", {})
            flows_explored = data.get("flows_explored", 0)
            duration_seconds = data.get("duration_seconds", 0)

        # Log event based on log level
        log_line = format_event_for_log(event, log_level)
        if log_line:
            print(log_line, file=sys.stderr)

    if log_level != "quiet":
        print("-" * 70, file=sys.stderr)

    return {
        "report": report or "No report generated",
        "issues": issues,
        "issues_found": len(issues),
        "flows_explored": flows_explored,
        "duration_seconds": round(duration_seconds, 2),
        "target_url": url,
        "goal": goal,
        "timestamp": datetime.now().isoformat()
    }


def main():
    parser = argparse.ArgumentParser(
        description="QA Bot - AI-powered website testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage (quiet, just outputs report)
    python cli.py https://example.com

    # With custom goal
    python cli.py https://example.com --goal "Test the checkout flow"

    # Quick exploration with fewer agents
    python cli.py https://example.com --max-agents 2 --max-duration 5

    # JSON output to file
    python cli.py https://example.com --output json --output-file results.json

    # Show flow-level progress (summary)
    python cli.py https://example.com --log-level summary

    # Show full activity log (like web UI)
    python cli.py https://example.com --log-level full

    # Shorthand for summary logging
    python cli.py https://example.com -v

    # Shorthand for full logging
    python cli.py https://example.com -vv
        """
    )

    parser.add_argument(
        "url",
        help="Target URL to test"
    )
    parser.add_argument(
        "--goal", "-g",
        default="Explore user flows and find bugs, broken elements, or usability problems.",
        help="Testing goal/focus (default: general exploration)"
    )
    parser.add_argument(
        "--max-agents", "-a",
        type=int,
        default=3,
        help="Maximum parallel agents (default: 3)"
    )
    parser.add_argument(
        "--max-cost",
        type=float,
        default=5.0,
        help="Maximum cost in USD (default: $5.00)"
    )
    parser.add_argument(
        "--max-duration", "-d",
        type=int,
        default=30,
        help="Maximum duration in minutes (default: 30)"
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)"
    )
    parser.add_argument(
        "--model",
        default="claude-haiku-4-5",
        help="Claude model to use (default: claude-haiku-4-5)"
    )
    parser.add_argument(
        "--testmail-api-key",
        default=None,
        help="Testmail.app API key for email verification testing (or set TESTMAIL_API_KEY env var)"
    )
    parser.add_argument(
        "--testmail-namespace",
        default=None,
        help="Testmail.app namespace (or set TESTMAIL_NAMESPACE env var)"
    )
    parser.add_argument(
        "--output", "-o",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format (default: markdown)"
    )
    parser.add_argument(
        "--output-file", "-f",
        help="Write output to file instead of stdout"
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Run browser in headed mode (visible, for debugging)"
    )
    parser.add_argument(
        "--log-level", "-l",
        choices=["quiet", "summary", "full"],
        default="quiet",
        help="Log verbosity: quiet (default), summary (flow events), full (all activity)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Increase log verbosity (-v for summary, -vv for full)"
    )
    parser.add_argument(
        "--credentials", "-c",
        help="Path to credentials file (env format: KEY=value per line)"
    )
    parser.add_argument(
        "--credential",
        action="append",
        dest="inline_credentials",
        metavar="KEY=VALUE",
        help="Inline credential (can be specified multiple times)"
    )
    parser.add_argument(
        "--dangerously-skip-permissions",
        action="store_true",
        help="Auto-approve all irreversible actions (payments, deletions, etc.). Use with caution."
    )

    args = parser.parse_args()

    # Determine log level from --log-level or -v flags
    if args.log_level != "quiet":
        log_level = args.log_level
    elif args.verbose >= 2:
        log_level = "full"
    elif args.verbose == 1:
        log_level = "summary"
    else:
        log_level = "quiet"

    # Get API key from arg or environment
    api_key = args.api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set", file=sys.stderr)
        print("Set it via --api-key or ANTHROPIC_API_KEY environment variable", file=sys.stderr)
        sys.exit(1)

    # Parse credentials
    credentials = {}
    if args.credentials:
        try:
            credentials.update(parse_credentials_file(args.credentials))
            if log_level != "quiet":
                print(f"Loaded {len(credentials)} credentials from {args.credentials}", file=sys.stderr)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error parsing credentials file: {e}", file=sys.stderr)
            sys.exit(1)

    if args.inline_credentials:
        for cred_str in args.inline_credentials:
            try:
                key, value = parse_inline_credential(cred_str)
                credentials[key] = value
            except ValueError as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)
        if log_level != "quiet":
            print(f"Added {len(args.inline_credentials)} inline credentials", file=sys.stderr)

    # Set Testmail.app credentials if provided (coordinator reads from env)
    testmail_key = args.testmail_api_key or os.getenv("TESTMAIL_API_KEY")
    testmail_namespace = args.testmail_namespace or os.getenv("TESTMAIL_NAMESPACE")
    if testmail_key and testmail_namespace:
        os.environ["TESTMAIL_API_KEY"] = testmail_key
        os.environ["TESTMAIL_NAMESPACE"] = testmail_namespace
        if log_level != "quiet":
            print("Email testing enabled (Testmail.app)", file=sys.stderr)
    elif testmail_key or testmail_namespace:
        print("Warning: Both --testmail-api-key and --testmail-namespace are required for email testing", file=sys.stderr)

    # Run exploration
    try:
        result = asyncio.run(run_exploration(
            url=args.url,
            goal=args.goal,
            max_agents=args.max_agents,
            max_duration=args.max_duration,
            api_key=api_key,
            model=args.model,
            headless=not args.headed,
            log_level=log_level,
            credentials=credentials if credentials else None,
            max_cost_usd=args.max_cost,
            skip_permissions=args.dangerously_skip_permissions
        ))
    except KeyboardInterrupt:
        print("\nExploration cancelled", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Error during exploration: {e}", file=sys.stderr)
        sys.exit(1)

    # Format output
    if args.output == "json":
        output = json.dumps(result, indent=2, default=str)
    else:
        output = result["report"]

    # Write output
    if args.output_file:
        with open(args.output_file, "w") as f:
            f.write(output)
        if log_level != "quiet":
            print(f"Output written to {args.output_file}", file=sys.stderr)
    else:
        print(output)

    # Exit with error code if critical issues found
    critical_issues = [
        i for i in result["issues"]
        if i.get("severity", "").lower() == "critical"
    ]
    if critical_issues:
        if log_level != "quiet":
            print(f"Found {len(critical_issues)} critical issues", file=sys.stderr)
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
