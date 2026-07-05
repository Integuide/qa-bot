#!/bin/bash
# QA Bot GitHub Action Entrypoint
#
# Runs the QA bot CLI and handles GitHub-specific integrations:
# - Parses JSON results
# - Sets GitHub Action outputs
# - Posts PR comments
# - Sets exit code based on findings

set -e

echo "::group::QA Bot Configuration"
echo "URL: $INPUT_URL"
echo "Goal: $INPUT_GOAL"
echo "Max Agents: $INPUT_MAX_AGENTS"
echo "Max Cost: \$$INPUT_MAX_COST"
echo "Max Duration: $INPUT_MAX_DURATION minutes"
echo "Model: $INPUT_MODEL"
echo "Post Comment: $INPUT_POST_COMMENT"
echo "Fail on Critical: $INPUT_FAIL_ON_CRITICAL"
echo "Fail on Zero Flows: ${INPUT_FAIL_ON_ZERO_FLOWS:-true}"
if [ -n "$INPUT_CREDENTIALS" ]; then
    echo "Credentials: [provided - $(echo "$INPUT_CREDENTIALS" | grep -c '=') key(s)]"
else
    echo "Credentials: [not provided]"
fi
if [ -n "$TESTMAIL_API_KEY" ] && [ -n "$TESTMAIL_NAMESPACE" ]; then
    echo "Email Testing: Configured (Testmail.app) - experimental, email-reading actions not yet active"
else
    echo "Email Testing: Disabled (no Testmail.app credentials)"
fi
if [ "$INPUT_DANGEROUSLY_SKIP_PERMISSIONS" = "true" ]; then
    echo "Skip Permissions: ENABLED (auto-approving all irreversible actions — default in CI)"
else
    echo "Skip Permissions: Disabled (will pause for approval — may cause timeouts in CI)"
fi
echo "::endgroup::"

# Handle credentials
CREDS_ARGS=""
if [ -n "$INPUT_CREDENTIALS" ]; then
    # Write credentials to temp file (secure - only accessible by this process)
    CREDS_FILE=$(mktemp)
    echo "$INPUT_CREDENTIALS" > "$CREDS_FILE"
    CREDS_ARGS="--credentials $CREDS_FILE"
fi

# Handle skip permissions flag
SKIP_PERMISSIONS_ARG=""
if [ "$INPUT_DANGEROUSLY_SKIP_PERMISSIONS" = "true" ]; then
    SKIP_PERMISSIONS_ARG="--dangerously-skip-permissions"
fi

# Run QA bot and capture output. A non-zero exit here is not necessarily
# fatal (the CLI exits 1 when critical issues are found but still writes
# results), so capture the exit code and decide based on the result file.
echo "::group::Running QA Bot Exploration"
CLI_EXIT=0
# shellcheck disable=SC2086
python -m qa_bot.cli "$INPUT_URL" \
    --goal "$INPUT_GOAL" \
    --max-agents "$INPUT_MAX_AGENTS" \
    --max-cost "$INPUT_MAX_COST" \
    --max-duration "$INPUT_MAX_DURATION" \
    --model "$INPUT_MODEL" \
    --output json \
    --output-file /tmp/qa-result.json \
    --log-level full \
    $CREDS_ARGS \
    $SKIP_PERMISSIONS_ARG || CLI_EXIT=$?
echo "::endgroup::"

# Clean up credentials file
if [ -n "$CREDS_FILE" ] && [ -f "$CREDS_FILE" ]; then
    rm -f "$CREDS_FILE"
fi

# Export issue screenshots so the workflow can upload them as artifacts.
# Standalone Docker action: the working directory is the runner workspace, so
# qa-bot-screenshots/ is visible to later steps. Mono wrapper: QA_ARTIFACTS_DIR
# points at a mounted volume. Only screenshots are exported — chat logs (page
# content) deliberately stay behind.
SCREENSHOT_COUNT=0
ARTIFACTS_DIR="${QA_ARTIFACTS_DIR:-${GITHUB_WORKSPACE:+$GITHUB_WORKSPACE/qa-bot-screenshots}}"
if [ -n "$ARTIFACTS_DIR" ]; then
    mkdir -p "$ARTIFACTS_DIR"
    # Clear stale screenshots from previous runs (persistent workspaces on
    # self-hosted runners, repeated invocations in one job) so the artifact
    # and the count below only reflect this run — even when the CLI crashed
    # before creating a log directory. The clear recurses just like the
    # count below, so nested stale PNGs can't survive the clear and inflate
    # SCREENSHOT_COUNT.
    find "$ARTIFACTS_DIR" -name '*.png' -delete 2>/dev/null || true
    if [ -d "${LOG_DIR:-logs}" ]; then
        find "${LOG_DIR:-logs}" -path '*/screenshots/*.png' -exec cp {} "$ARTIFACTS_DIR/" \; || true
    fi
    SCREENSHOT_COUNT=$(find "$ARTIFACTS_DIR" -name '*.png' | wc -l)
    echo "Exported $SCREENSHOT_COUNT issue screenshot(s) to $ARTIFACTS_DIR"
fi

# Check that the result file exists and is valid JSON. A runner timeout or
# mid-write crash can leave a truncated file; under `set -e` a later jq parse
# failure would kill the script silently, so treat it as a failed run here.
if ! jq empty /tmp/qa-result.json >/dev/null 2>&1; then
    echo "::error::QA Bot failed to produce usable results (CLI exit code: $CLI_EXIT)"

    # Close the loop on the PR instead of failing silently: requesters are
    # told results will be posted, so post a short failure notice when we
    # have a token and a PR context.
    if [ "$INPUT_POST_COMMENT" = "true" ] && [ -n "$GITHUB_TOKEN" ] && [ -f "$GITHUB_EVENT_PATH" ]; then
        PR_NUMBER=$(jq -r '.pull_request.number // .issue.number // empty' "$GITHUB_EVENT_PATH")

        if [ -n "$PR_NUMBER" ] && [ "$PR_NUMBER" != "null" ]; then
            RUN_LOGS="the workflow run logs"
            if [ -n "$GITHUB_RUN_ID" ] && [ -n "$GITHUB_REPOSITORY" ]; then
                RUN_LOGS="[the workflow run logs](${GITHUB_SERVER_URL:-https://github.com}/$GITHUB_REPOSITORY/actions/runs/$GITHUB_RUN_ID)"
            fi

            FAILURE_BODY=$(cat <<FAILURE_EOF
## QA Bot Report

**QA Bot failed before producing usable results** (exit code $CLI_EXIT). No exploration report is available.

Common causes: unreachable target URL, API authentication or credit errors. See $RUN_LOGS for details.

<sub>Generated by [QA Bot](https://github.com/integuide/qa-bot) using Claude AI</sub>
FAILURE_EOF
)

            HTTP_STATUS=$(curl -s -o /tmp/comment-response.json -w "%{http_code}" -X POST \
                -H "Authorization: token $GITHUB_TOKEN" \
                -H "Accept: application/vnd.github.v3+json" \
                "https://api.github.com/repos/$GITHUB_REPOSITORY/issues/$PR_NUMBER/comments" \
                -d "$(jq -n --arg body "$FAILURE_BODY" '{body: $body}')")

            if [ "$HTTP_STATUS" = "201" ]; then
                echo "Failure notice posted to PR #$PR_NUMBER"
            else
                echo "::warning::Failed to post failure notice (HTTP $HTTP_STATUS)"
            fi
        fi
    fi
    exit 1
fi

# Parse results
REPORT=$(jq -r '.report // "No report generated"' /tmp/qa-result.json)
ISSUES_COUNT=$(jq -r '.issues_found // 0' /tmp/qa-result.json)
FLOWS_EXPLORED=$(jq -r '.flows_explored // 0' /tmp/qa-result.json)
# Guard the numeric comparisons below — a malformed value would kill the
# script under set -e.
if ! [[ "$FLOWS_EXPLORED" =~ ^[0-9]+$ ]]; then
    echo "::warning::Non-numeric flows_explored in result file; treating as 0"
    FLOWS_EXPLORED=0
fi
DURATION=$(jq -r '.duration_seconds // 0' /tmp/qa-result.json)
COST_USD=$(jq -r '.estimated_cost_usd // 0' /tmp/qa-result.json)
# Never write unvalidated result-file content to GITHUB_OUTPUT: a malformed
# estimated_cost_usd (e.g. a string with embedded newlines) could inject
# extra output lines. Require a plain non-negative decimal, else report 0.
if ! [[ "$COST_USD" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "::warning::Non-numeric estimated_cost_usd in result file; reporting cost-usd=0"
    COST_USD=0
fi
# printf -v avoids command substitution: on a non-numeric value, plain
# $(printf ...) would capture printf's best-effort output AND run the
# fallback, yielding garbage like '0.00abc'. With -v the fallback assignment
# cleanly overwrites the partial conversion.
printf -v COST_DISPLAY '%.2f' "$COST_USD" 2>/dev/null || COST_DISPLAY="$COST_USD"

# Count critical issues
CRITICAL_COUNT=$(jq -r '[.issues // [] | .[] | select(.severity == "critical" or .severity == "Critical")] | length' /tmp/qa-result.json)

echo "::group::Results Summary"
echo "Flows explored: $FLOWS_EXPLORED"
echo "Issues found: $ISSUES_COUNT"
echo "Critical issues: $CRITICAL_COUNT"
echo "Duration: ${DURATION}s"
echo "Estimated cost: \$${COST_DISPLAY}"
echo "::endgroup::"

# Set GitHub Action outputs
# Using the recommended approach for multiline strings, with a random heredoc
# delimiter (per GitHub's hardening guidance): the report is LLM output
# influenced by the tested site, so it must not be able to terminate the
# heredoc early and inject extra outputs.
REPORT_DELIM="qabot_$(dd if=/dev/urandom bs=15 count=1 status=none | base64)"
{
    echo "report<<$REPORT_DELIM"
    echo "$REPORT"
    echo "$REPORT_DELIM"
} >> "$GITHUB_OUTPUT"

echo "issues-count=$ISSUES_COUNT" >> "$GITHUB_OUTPUT"
echo "critical-issues=$CRITICAL_COUNT" >> "$GITHUB_OUTPUT"
echo "flows-explored=$FLOWS_EXPLORED" >> "$GITHUB_OUTPUT"
echo "cost-usd=$COST_USD" >> "$GITHUB_OUTPUT"

# Post PR comment if requested
if [ "$INPUT_POST_COMMENT" = "true" ]; then
    echo "::group::Posting PR Comment"

    if [ -z "$GITHUB_TOKEN" ]; then
        echo "::warning::post-comment is enabled but no github-token was provided. Add 'github-token: \${{ github.token }}' to your workflow's 'with:' block."
    # Check if we have the event file and a PR context
    elif [ -f "$GITHUB_EVENT_PATH" ]; then
        # Try to get PR number from various event types
        PR_NUMBER=$(jq -r '
            .pull_request.number //
            .issue.number //
            empty
        ' "$GITHUB_EVENT_PATH")

        if [ -n "$PR_NUMBER" ] && [ "$PR_NUMBER" != "null" ]; then
            echo "Posting comment to PR #$PR_NUMBER..."

            # Point reviewers at the visual evidence when screenshots exist
            SCREENSHOTS_NOTE=""
            if [ "$SCREENSHOT_COUNT" -gt 0 ]; then
                RUN_REF="this workflow run"
                if [ -n "$GITHUB_RUN_ID" ] && [ -n "$GITHUB_REPOSITORY" ]; then
                    RUN_REF="the [workflow run](${GITHUB_SERVER_URL:-https://github.com}/$GITHUB_REPOSITORY/actions/runs/$GITHUB_RUN_ID)"
                fi
                SCREENSHOTS_NOTE="**Screenshots:** $SCREENSHOT_COUNT issue screenshot(s) captured — download the \`qa-bot-screenshots\` artifact from $RUN_REF (uploaded if your workflow includes the upload-artifact step from the QA Bot examples)."
            fi

            # Build comment body with proper escaping
            COMMENT_BODY=$(cat <<COMMENT_EOF
## QA Bot Report

$REPORT

---
**Run stats:** Explored $FLOWS_EXPLORED user flows | $ISSUES_COUNT raw issue(s) detected during exploration ($CRITICAL_COUNT critical) | Duration: ${DURATION}s | Cost: \$${COST_DISPLAY}
$SCREENSHOTS_NOTE

<sub>Note: "raw issues detected" is the count of observations workers flagged during exploration, before the report above deduplicates, filters, and curates them — so it may differ from the issue count in the report. The report is the source of truth for findings.</sub>

<sub>Generated by [QA Bot](https://github.com/integuide/qa-bot) using Claude AI</sub>
COMMENT_EOF
)

            # Post comment using GitHub API
            HTTP_STATUS=$(curl -s -o /tmp/comment-response.json -w "%{http_code}" -X POST \
                -H "Authorization: token $GITHUB_TOKEN" \
                -H "Accept: application/vnd.github.v3+json" \
                "https://api.github.com/repos/$GITHUB_REPOSITORY/issues/$PR_NUMBER/comments" \
                -d "$(jq -n --arg body "$COMMENT_BODY" '{body: $body}')")

            if [ "$HTTP_STATUS" = "201" ]; then
                COMMENT_URL=$(jq -r '.html_url' /tmp/comment-response.json)
                echo "Comment posted successfully: $COMMENT_URL"
            else
                echo "::warning::Failed to post comment (HTTP $HTTP_STATUS)"
                cat /tmp/comment-response.json
            fi
        else
            echo "No PR number found in event context, skipping comment"
        fi
    else
        echo "No GitHub event file found, skipping comment"
    fi
    echo "::endgroup::"
fi

# Determine exit code
EXIT_CODE=0

if [ "$INPUT_FAIL_ON_CRITICAL" = "true" ] && [ "$CRITICAL_COUNT" -gt 0 ]; then
    echo "::error::Found $CRITICAL_COUNT critical issues"
    EXIT_CODE=1
fi

# A run that tested nothing must not read as a green check: zero explored
# flows means the deploy was NOT verified (missing credentials, unreachable
# target, ...). The posted report names the blockers.
if [ "${INPUT_FAIL_ON_ZERO_FLOWS:-true}" = "true" ] && [ "$FLOWS_EXPLORED" -eq 0 ]; then
    echo "::error::QA Bot completed without testing any flows — the target was NOT verified. See the report for blockers."
    EXIT_CODE=1
fi

# Print report to logs for visibility
echo "::group::Full QA Report"
echo "$REPORT"
echo "::endgroup::"

exit $EXIT_CODE
