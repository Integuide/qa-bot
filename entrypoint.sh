#!/bin/bash
# QA Bot GitHub Action Entrypoint
#
# Runs the QA bot CLI and handles GitHub-specific integrations:
# - Parses JSON results
# - Sets GitHub Action outputs
# - Posts PR comments
# - Sets exit code based on findings

set -e

# Run mode: `full` (default) or `smoke`. Smoke maps to the CLI's --smoke:
# preflight probe, one turn-capped worker opening each top-level nav link
# once, no forking, report tagged SMOKE TEST ONLY. Anything else is a typo
# in the workflow — fail loudly rather than silently running a full paid
# exploration (or silently skipping the smoke semantics).
INPUT_MODE="${INPUT_MODE:-full}"
case "$INPUT_MODE" in
    full|smoke) ;;
    *)
        echo "::error::Invalid mode '$INPUT_MODE' — expected 'full' or 'smoke'"
        exit 1
        ;;
esac
SMOKE_TAG="SMOKE TEST ONLY — not a regression test"

# Hidden markers identify QA Bot comments on re-runs: the report comment and
# the failure notice both carry QABOT_COMMENT_MARKER so a later run updates
# (or supersedes) whichever is newest instead of stacking a new one.
QABOT_COMMENT_MARKER="<!-- qa-bot-report -->"
QABOT_SUPERSEDED_MARKER="<!-- qa-bot-report:superseded -->"

# Post (or update) the QA Bot comment on PR $PR_NUMBER with body $QABOT_BODY.
# comment-mode=update (default): PATCH the newest earlier QA Bot comment and
# collapse any OTHER earlier ones (a failure notice left standing next to an
# old report, for example) so exactly one current comment remains.
# comment-mode=new: POST a fresh comment and collapse every earlier one.
# Listing failures fall through to posting a fresh comment.
upsert_qabot_comment() {
    local COMMENTS_API="https://api.github.com/repos/$GITHUB_REPOSITORY/issues"
    local PRIOR_COMMENT_IDS="" PAGE=1 LIST_STATUS PAGE_IDS LATEST_COMMENT_ID HTTP_STATUS
    local POSTED="" UPDATED_ID="" OLD_ID OLD_STATUS SUPERSEDED_BODY
    while [ "$PAGE" -le 10 ]; do
        LIST_STATUS=$(curl -s -o /tmp/comments-page.json -w "%{http_code}" \
            -H "Authorization: token $GITHUB_TOKEN" \
            -H "Accept: application/vnd.github.v3+json" \
            "$COMMENTS_API/$PR_NUMBER/comments?per_page=100&page=$PAGE")
        if [ "$LIST_STATUS" != "200" ] || ! jq -e 'type == "array"' /tmp/comments-page.json >/dev/null 2>&1; then
            echo "::warning::Could not list existing PR comments (HTTP $LIST_STATUS); posting a new comment"
            break
        fi
        PAGE_IDS=$(jq -r --arg m "$QABOT_COMMENT_MARKER" --arg s "$QABOT_SUPERSEDED_MARKER" \
            '[.[] | select((.body // "" | contains($m)) and (.body // "" | contains($s) | not)) | .id | tostring] | join(" ")' \
            /tmp/comments-page.json)
        PRIOR_COMMENT_IDS="$PRIOR_COMMENT_IDS${PAGE_IDS:+ $PAGE_IDS}"
        [ "$(jq 'length' /tmp/comments-page.json)" -lt 100 ] && break
        PAGE=$((PAGE + 1))
        if [ "$PAGE" -gt 10 ]; then
            echo "::warning::PR has more than 1000 comments; an older QA Bot report may not be updated"
        fi
    done
    # Comments list oldest-first, so the last id is the newest QA Bot comment.
    LATEST_COMMENT_ID="${PRIOR_COMMENT_IDS##* }"

    if [ "${INPUT_COMMENT_MODE:-update}" != "new" ] && [ -n "$LATEST_COMMENT_ID" ]; then
        HTTP_STATUS=$(curl -s -o /tmp/comment-response.json -w "%{http_code}" -X PATCH \
            -H "Authorization: token $GITHUB_TOKEN" \
            -H "Accept: application/vnd.github.v3+json" \
            "$COMMENTS_API/comments/$LATEST_COMMENT_ID" \
            -d "$(jq -n --arg body "$QABOT_BODY" '{body: $body}')")
        if [ "$HTTP_STATUS" = "200" ]; then
            POSTED=1
            UPDATED_ID="$LATEST_COMMENT_ID"
            echo "Updated existing comment: $(jq -r '.html_url' /tmp/comment-response.json)"
        else
            echo "::warning::Failed to update existing comment $LATEST_COMMENT_ID (HTTP $HTTP_STATUS); posting a new comment"
            cat /tmp/comment-response.json
        fi
    fi

    if [ -z "$POSTED" ]; then
        HTTP_STATUS=$(curl -s -o /tmp/comment-response.json -w "%{http_code}" -X POST \
            -H "Authorization: token $GITHUB_TOKEN" \
            -H "Accept: application/vnd.github.v3+json" \
            "$COMMENTS_API/$PR_NUMBER/comments" \
            -d "$(jq -n --arg body "$QABOT_BODY" '{body: $body}')")
        if [ "$HTTP_STATUS" = "201" ]; then
            POSTED=1
            echo "Comment posted successfully: $(jq -r '.html_url' /tmp/comment-response.json)"
        else
            echo "::warning::Failed to post comment (HTTP $HTTP_STATUS)"
            cat /tmp/comment-response.json
            return 1
        fi
    fi

    # The current comment is up: collapse every other earlier QA Bot comment
    # (body kept verbatim inside <details>) so only one reads as current.
    for OLD_ID in $PRIOR_COMMENT_IDS; do
        [ "$OLD_ID" = "$UPDATED_ID" ] && continue
        OLD_STATUS=$(curl -s -o /tmp/comment-old.json -w "%{http_code}" \
            -H "Authorization: token $GITHUB_TOKEN" \
            -H "Accept: application/vnd.github.v3+json" \
            "$COMMENTS_API/comments/$OLD_ID")
        if [ "$OLD_STATUS" != "200" ]; then
            echo "::warning::Could not read earlier report comment $OLD_ID (HTTP $OLD_STATUS); leaving it as is"
            continue
        fi
        SUPERSEDED_BODY=$(jq -r --arg s "$QABOT_SUPERSEDED_MARKER" \
            '$s + "\n<details><summary>Superseded QA Bot report (" + ((.created_at // "") | split("T")[0]) + ")</summary>\n\n" + (.body // "") + "\n\n</details>"' \
            /tmp/comment-old.json)
        OLD_STATUS=$(curl -s -o /tmp/comment-old.json -w "%{http_code}" -X PATCH \
            -H "Authorization: token $GITHUB_TOKEN" \
            -H "Accept: application/vnd.github.v3+json" \
            "$COMMENTS_API/comments/$OLD_ID" \
            -d "$(jq -n --arg body "$SUPERSEDED_BODY" '{body: $body}')")
        if [ "$OLD_STATUS" = "200" ]; then
            echo "Collapsed superseded report comment $OLD_ID"
        else
            echo "::warning::Failed to collapse earlier report comment $OLD_ID (HTTP $OLD_STATUS)"
        fi
    done
    return 0
}

# Close the loop on the PR when the run ends without a report: requesters are
# told results will be posted, so post a short notice (via the same upsert, so
# it replaces a stale report and is replaced by the next one) when there is a
# token and a PR context; otherwise do nothing. $1 is the markdown saying what
# went wrong. Callers exit non-zero afterwards.
post_failure_notice() {
    local RUN_LOGS="the workflow run logs"
    if [ "$INPUT_POST_COMMENT" != "true" ] || [ -z "$GITHUB_TOKEN" ] || [ ! -f "$GITHUB_EVENT_PATH" ]; then
        return 0
    fi
    PR_NUMBER=$(jq -r '.pull_request.number // .issue.number // empty' "$GITHUB_EVENT_PATH")
    if [ -z "$PR_NUMBER" ] || [ "$PR_NUMBER" = "null" ]; then
        return 0
    fi
    if [ -n "$GITHUB_RUN_ID" ] && [ -n "$GITHUB_REPOSITORY" ]; then
        RUN_LOGS="[the workflow run logs](${GITHUB_SERVER_URL:-https://github.com}/$GITHUB_REPOSITORY/actions/runs/$GITHUB_RUN_ID)"
    fi
    QABOT_BODY=$(cat <<FAILURE_EOF
## QA Bot Report

$1 See $RUN_LOGS for details.

<sub>Generated by [QA Bot](https://github.com/integuide/qa-bot) using ${MODEL_CREDIT:-Claude AI}</sub>
$QABOT_COMMENT_MARKER
FAILURE_EOF
)
    if upsert_qabot_comment; then
        echo "Failure notice posted to PR #$PR_NUMBER"
    else
        echo "::warning::Failed to post failure notice"
    fi
}
# --- end PR comment helpers

# Provider follows the model id: an OpenRouter "vendor/slug" id (a Claude id
# never has a slash) needs openrouter-api-key, anything else
# anthropic-api-key. Check the one this run needs before any browser or AI
# spend, so a missing secret fails here with its input's name instead of as
# an aborted run — and say so on the PR: Docker actions don't enforce
# `required: true`, so a missing or misnamed secret arrives as an empty input
# and this is the only place that notices.
case "$INPUT_MODEL" in
    */*)
        PROVIDER="OpenRouter"
        PROVIDER_KEY_VAR="OPENROUTER_API_KEY"
        PROVIDER_KEY_INPUT="openrouter-api-key"
        MODEL_CREDIT="$INPUT_MODEL via OpenRouter"
        ;;
    *)
        PROVIDER="Anthropic"
        PROVIDER_KEY_VAR="ANTHROPIC_API_KEY"
        PROVIDER_KEY_INPUT="anthropic-api-key"
        MODEL_CREDIT="Claude AI"
        ;;
esac
if [ -z "${!PROVIDER_KEY_VAR}" ]; then
    echo "::error::Model '$INPUT_MODEL' runs on $PROVIDER, but the $PROVIDER_KEY_INPUT input is empty — pass it from a secret (e.g. $PROVIDER_KEY_INPUT: \${{ secrets.$PROVIDER_KEY_VAR }})"
    post_failure_notice "**QA Bot did not run:** model \`$INPUT_MODEL\` runs on $PROVIDER, but the \`$PROVIDER_KEY_INPUT\` input is empty — a missing or misnamed secret passes an empty string. Pass it from a secret, e.g. \`$PROVIDER_KEY_INPUT: \${{ secrets.$PROVIDER_KEY_VAR }}\`. Nothing was tested."
    exit 1
fi

echo "::group::QA Bot Configuration"
echo "URL: $INPUT_URL"
echo "Mode: $INPUT_MODE"
if [ "$INPUT_MODE" = "smoke" ]; then
    echo "Goal: (fixed smoke goal — the goal input is ignored in smoke mode)"
else
    # Model-written from the PR description: one line, so no line of it can
    # start with "::" and run as a workflow command
    echo "Goal: $(printf '%s' "$INPUT_GOAL" | tr '\r\n' '  ')"
fi
# Empty limit inputs mean "the CLI's default for this mode" (3 agents /
# $5.00 / 30 min for full, 1 / $0.75 / 5 min for smoke).
echo "Max Agents: ${INPUT_MAX_AGENTS:-(mode default)}"
if [ -n "$INPUT_MAX_COST" ]; then echo "Max Cost: \$$INPUT_MAX_COST"; else echo "Max Cost: (mode default)"; fi
echo "Max Duration: ${INPUT_MAX_DURATION:-(mode default)}${INPUT_MAX_DURATION:+ minutes}"
echo "Model: $INPUT_MODEL"
if [ "$PROVIDER" = "OpenRouter" ]; then
    echo "Provider: OpenRouter (screenshots, page text and test credentials go to OpenRouter and the model vendor, not Anthropic)"
else
    echo "Provider: Anthropic"
fi
case "${INPUT_COMMENT_MODE:-update}" in
    update|new) ;;
    *)
        echo "::error::Invalid comment-mode '$INPUT_COMMENT_MODE' (expected 'update' or 'new')"
        exit 1
        ;;
esac
echo "Post Comment: $INPUT_POST_COMMENT (mode: ${INPUT_COMMENT_MODE:-update})"
echo "Fail on Critical: $INPUT_FAIL_ON_CRITICAL"
echo "Fail on Zero Flows: ${INPUT_FAIL_ON_ZERO_FLOWS:-true}"
if [ -n "$INPUT_CREDENTIALS" ]; then
    echo "Credentials: [provided - $(echo "$INPUT_CREDENTIALS" | grep -c '=') key(s)]"
else
    echo "Credentials: [not provided]"
fi
if [ -n "$INPUT_KNOWN_ISSUES" ]; then
    # printf '%s' (no added newline) + grep -c '' counts every line exactly:
    # `grep -c .` missed blank lines; `echo | wc -l` overcounted the YAML
    # block scalar's trailing newline.
    echo "Known Issues: [provided - $(printf '%s' "$INPUT_KNOWN_ISSUES" | grep -c '') line(s)]"
else
    echo "Known Issues: [not provided]"
fi
if [ -n "$INPUT_PREVIOUS_REPORT" ]; then
    echo "Previous Report: [provided - $(printf '%s' "$INPUT_PREVIOUS_REPORT" | grep -c '') line(s) — findings will be labelled NEW / recurring]"
else
    echo "Previous Report: [not provided — first run for this target, or no artifact found]"
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

# Handle known issues (multiline free text — an array keeps the value intact
# through the unquoted-args expansion below)
KNOWN_ISSUES_ARGS=()
if [ -n "$INPUT_KNOWN_ISSUES" ]; then
    KNOWN_ISSUES_ARGS=(--known-issues "$INPUT_KNOWN_ISSUES")
fi

# Previous run's report (path in the workspace, or the markdown inline — the
# CLI tells them apart). Labels findings NEW / recurring; deliberately NOT
# merged into --known-issues, which would demote last run's regressions.
PREVIOUS_REPORT_ARGS=()
if [ -n "$INPUT_PREVIOUS_REPORT" ]; then
    PREVIOUS_REPORT_ARGS=(--previous-report "$INPUT_PREVIOUS_REPORT")
fi

# Mode and limits. Limit flags are only passed when set so the CLI applies
# its per-mode defaults (see the configuration group above).
MODE_ARGS=()
if [ "$INPUT_MODE" = "smoke" ]; then
    MODE_ARGS=(--smoke)
fi
LIMIT_ARGS=()
if [ -n "$INPUT_MAX_AGENTS" ]; then
    LIMIT_ARGS+=(--max-agents "$INPUT_MAX_AGENTS")
fi
if [ -n "$INPUT_MAX_COST" ]; then
    LIMIT_ARGS+=(--max-cost "$INPUT_MAX_COST")
fi
if [ -n "$INPUT_MAX_DURATION" ]; then
    LIMIT_ARGS+=(--max-duration "$INPUT_MAX_DURATION")
fi

# Run QA bot and capture output. A non-zero exit here is not necessarily
# fatal (the CLI exits 1 when critical issues are found but still writes
# results), so capture the exit code and decide based on the result file.
echo "::group::Running QA Bot Exploration"
# The CLI's log (event lines, Python logging at INFO) carries model output
# from site-influenced text; a line of it starting with "::" would run as a
# workflow command (::add-mask::, ::stop-commands::, a fake ::error::). The
# CLI flattens what it formats itself, but not every logger call can be
# trusted to: pause command processing for the whole run with a random
# token, resumed right after. (The CLI's own ::warning:: line is re-emitted
# by this script from the result file below.)
CLI_STOP_TOKEN="qabot_$(dd if=/dev/urandom bs=15 count=1 status=none | base64 | tr -dc 'A-Za-z0-9')"
echo "::stop-commands::$CLI_STOP_TOKEN"
CLI_EXIT=0
# shellcheck disable=SC2086
python -m qa_bot.cli "$INPUT_URL" \
    --goal "$INPUT_GOAL" \
    "${MODE_ARGS[@]}" \
    "${LIMIT_ARGS[@]}" \
    --model "$INPUT_MODEL" \
    --output json \
    --output-file /tmp/qa-result.json \
    --log-level full \
    "${KNOWN_ISSUES_ARGS[@]}" \
    "${PREVIOUS_REPORT_ARGS[@]}" \
    $CREDS_ARGS \
    $SKIP_PERMISSIONS_ARG || CLI_EXIT=$?
echo "::$CLI_STOP_TOKEN::"
echo "::endgroup::"

# Clean up credentials file
if [ -n "$CREDS_FILE" ] && [ -f "$CREDS_FILE" ]; then
    rm -f "$CREDS_FILE"
fi

# Export issue screenshots, report.md and summary.json so the workflow can
# upload them as artifacts. Standalone Docker action: the working directory
# is the runner workspace, so qa-bot-screenshots/ is visible to later steps.
# Mono wrapper: QA_ARTIFACTS_DIR points at a mounted volume. report.md and
# summary.json are credential-masked by the chat logger (report.md by key
# pattern and by value, summary.json by value) and are what the next run
# reads back as `previous-report`; chat logs (page content) deliberately
# stay behind.
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
    # report.md / summary.json: clear stale copies first, then export this
    # run's (the newest run directory wins if the log dir somehow has several).
    rm -f "$ARTIFACTS_DIR/report.md" "$ARTIFACTS_DIR/summary.json"
    if [ -d "${LOG_DIR:-logs}" ]; then
        for RUN_FILE in report.md summary.json; do
            LATEST_RUN_FILE=$(find "${LOG_DIR:-logs}" -mindepth 2 -maxdepth 2 -name "$RUN_FILE" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
            if [ -n "$LATEST_RUN_FILE" ]; then
                cp "$LATEST_RUN_FILE" "$ARTIFACTS_DIR/$RUN_FILE" || true
            fi
        done
    fi
    SCREENSHOT_COUNT=$(find "$ARTIFACTS_DIR" -name '*.png' | wc -l)
    echo "Exported $SCREENSHOT_COUNT issue screenshot(s) to $ARTIFACTS_DIR"
fi

# Check that the result file exists and is valid JSON. A runner timeout or
# mid-write crash can leave a truncated file; under `set -e` a later jq parse
# failure would kill the script silently, so treat it as a failed run here.
if ! jq empty /tmp/qa-result.json >/dev/null 2>&1; then
    echo "::error::QA Bot failed to produce usable results (CLI exit code: $CLI_EXIT)"
    post_failure_notice "**QA Bot failed before producing usable results** (exit code $CLI_EXIT). No exploration report is available.

Common causes: unreachable target URL, API authentication or credit errors."
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
# OpenRouter's reported charge (usage.cost — the real bill); null for Claude,
# whose cost is only the token-priced estimate above. Same validation.
CHARGED_COST_USD=$(jq -r '.charged_cost_usd // empty' /tmp/qa-result.json)
CHARGED_DISPLAY=""
if [[ "$CHARGED_COST_USD" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    printf -v CHARGED_DISPLAY '%.2f' "$CHARGED_COST_USD" 2>/dev/null || CHARGED_DISPLAY="$CHARGED_COST_USD"
else
    CHARGED_COST_USD=""
fi

# Count critical issues (RAW worker-reported, pre-synthesis). Kept as the
# `critical-issues` output for compatibility; it only gates when synthesis
# produced no curated verdict.
CRITICAL_COUNT=$(jq -r '[.issues // [] | .[] | select(.severity == "critical" or .severity == "Critical")] | length' /tmp/qa-result.json)
# Curated critical count from the synthesis report's qa-verdict block. Empty
# when the CLI had no AI verdict (fallback / NOT TESTED report, unparseable
# block, older CLI). The fail-on-critical gate below uses THIS count when it
# is present — the report is the source of truth — and the raw count only
# as a fallback, so the red check and the posted report tell the same story.
CURATED_CRITICAL_COUNT=$(jq -r '.curated_critical_count // empty' /tmp/qa-result.json)
if ! [[ "$CURATED_CRITICAL_COUNT" =~ ^[0-9]+$ ]]; then
    CURATED_CRITICAL_COUNT=""
fi
if [ -n "$CURATED_CRITICAL_COUNT" ]; then
    GATE_COUNT="$CURATED_CRITICAL_COUNT"
    GATE_DISPLAY="$CURATED_CRITICAL_COUNT critical after curation ($CRITICAL_COUNT raw worker finding(s))"
    if [ "$CURATED_CRITICAL_COUNT" -lt "$CRITICAL_COUNT" ]; then
        # The curated verdict is AI-written from site-influenced text, so a
        # downgrade must never be silent: say it where the check is read.
        echo "::warning::Synthesis downgraded $((CRITICAL_COUNT - CURATED_CRITICAL_COUNT)) of $CRITICAL_COUNT raw critical finding(s); review the report's Likely False Positives / Known Issues sections before trusting a green check"
    elif [ "$CURATED_CRITICAL_COUNT" -gt "$CRITICAL_COUNT" ]; then
        # Same check as cli.curated_excess_warning, whose line printed as
        # plain text inside the CLI's stop-commands window above
        echo "::warning::The curated report lists $CURATED_CRITICAL_COUNT critical finding(s) but only $CRITICAL_COUNT worker finding(s) are still critical after the severity guard and the independent re-check; check that no Critical Issues entry is a finding the re-check did not reproduce (tagged UNCONFIRMED) under a new title"
    fi
    # Titles of the curated criticals, for the ::error:: line. LLM output:
    # flatten to one line so it can't smuggle in extra workflow commands.
    CURATED_TITLES=$(jq -r '[.curated_critical_titles // [] | .[] | select(type == "string") | gsub("[\\r\\n]+"; " ")] | join(" | ") | .[0:600]' /tmp/qa-result.json 2>/dev/null || true)
else
    GATE_COUNT="$CRITICAL_COUNT"
    GATE_DISPLAY="$CRITICAL_COUNT critical (raw worker findings; synthesis produced no curated verdict)"
    CURATED_TITLES=""
fi

# Run-over-run labels from the verdict (Report Quality Standard 11): set
# only when a previous report was supplied AND synthesis produced a verdict
# whose counts fit the findings it listed (synthesis.py drops the rest).
# Labels only — the gate above is untouched by them.
NEW_COUNT=$(jq -r '.new_findings_count // empty' /tmp/qa-result.json)
RECURRING_COUNT=$(jq -r '.recurring_findings_count // empty' /tmp/qa-result.json)
if ! [[ "$NEW_COUNT" =~ ^[0-9]+$ ]] || ! [[ "$RECURRING_COUNT" =~ ^[0-9]+$ ]]; then
    NEW_COUNT=""
    RECURRING_COUNT=""
fi
SINCE_PREVIOUS_NOTE=""
if [ -n "$NEW_COUNT" ]; then
    SINCE_PREVIOUS_NOTE="**Since previous run:** $NEW_COUNT new, $RECURRING_COUNT recurring (labels only — severity is unchanged; previous findings not seen this time are listed under \"Not Observed This Run\" in the report, never as fixed)"
elif [ -n "$INPUT_PREVIOUS_REPORT" ]; then
    SINCE_PREVIOUS_NOTE="**Since previous run:** a previous report was supplied but findings were not labelled new/recurring (synthesis produced no curated verdict, or its counts were missing or did not match the findings it listed)"
fi

# Independent re-check of criticals (qa_bot/orchestrator/recheck.py): per-issue
# outcome counts, absent when no critical was a candidate. The gate counts
# above already reflect it (a not-reproduced critical arrives as major); this
# only says so where the check is read. Numbers only, or no note at all.
RECHECK_NOTE=""
RECHECK_LOG="n/a (no critical finding was a re-check candidate)"
RECHECK_COUNTS=$(jq -r 'if (.recheck_summary | type) == "object" then [.recheck_summary.not_reproduced, .recheck_summary.reproduced, .recheck_summary.inconclusive, .recheck_summary.not_rechecked] | map(tostring) | join(" ") else empty end' /tmp/qa-result.json 2>/dev/null || true)
if [[ "$RECHECK_COUNTS" =~ ^([0-9]+)\ ([0-9]+)\ ([0-9]+)\ ([0-9]+)$ ]]; then
    RECHECK_LOG="${BASH_REMATCH[1]} not reproduced (demoted to major), ${BASH_REMATCH[2]} reproduced, ${BASH_REMATCH[3]} inconclusive, ${BASH_REMATCH[4]} not re-checked"
    RECHECK_NOTE="**Independent re-check of criticals:** ${BASH_REMATCH[1]} not reproduced (demoted to major, tagged UNCONFIRMED — these cannot fail the gate), ${BASH_REMATCH[2]} reproduced, ${BASH_REMATCH[3]} inconclusive and ${BASH_REMATCH[4]} not re-checked (both kept critical)"
fi

echo "::group::Results Summary"
echo "Mode: $INPUT_MODE"
echo "Flows explored: $FLOWS_EXPLORED"
echo "Issues found: $ISSUES_COUNT"
echo "Critical issues (raw worker findings): $CRITICAL_COUNT"
echo "Critical issues (after curation): ${CURATED_CRITICAL_COUNT:-n/a (no curated verdict from synthesis)}"
echo "Gate: $GATE_DISPLAY"
echo "Independent re-check: $RECHECK_LOG"
SINCE_PREVIOUS="n/a (no previous report, no curated verdict, or inconsistent counts dropped)"
[ -n "$NEW_COUNT" ] && SINCE_PREVIOUS="$NEW_COUNT new, $RECURRING_COUNT recurring"
echo "Since previous run: $SINCE_PREVIOUS"
echo "Duration: ${DURATION}s"
echo "Estimated cost: \$${COST_DISPLAY}"
if [ -n "$CHARGED_COST_USD" ]; then
    echo "Charged cost (OpenRouter): \$${CHARGED_COST_USD}"
fi
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
echo "curated-critical-issues=$CURATED_CRITICAL_COUNT" >> "$GITHUB_OUTPUT"
echo "flows-explored=$FLOWS_EXPLORED" >> "$GITHUB_OUTPUT"
echo "cost-usd=$COST_USD" >> "$GITHUB_OUTPUT"
echo "mode=$INPUT_MODE" >> "$GITHUB_OUTPUT"
echo "new-findings=$NEW_COUNT" >> "$GITHUB_OUTPUT"
echo "recurring-findings=$RECURRING_COUNT" >> "$GITHUB_OUTPUT"

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

            # OpenRouter reports its real charge: show it next to the estimate.
            COST_NOTE="Cost: \$${COST_DISPLAY}"
            if [ -n "$CHARGED_DISPLAY" ]; then
                COST_NOTE="Cost: \$${CHARGED_DISPLAY} charged by OpenRouter (token estimate \$${COST_DISPLAY})"
            fi

            # Smoke runs are tagged in the title AND the footer so a smoke
            # pass can't be mistaken for regression coverage at a glance.
            COMMENT_TITLE="## QA Bot Report"
            MODE_NOTE=""
            if [ "$INPUT_MODE" = "smoke" ]; then
                COMMENT_TITLE="## QA Bot Report — $SMOKE_TAG"
                MODE_NOTE="**Mode:** $SMOKE_TAG (one worker opened each top-level navigation link once; nothing else was exercised)"
            fi

            # Build comment body with proper escaping
            COMMENT_BODY=$(cat <<COMMENT_EOF
$QABOT_COMMENT_MARKER
$COMMENT_TITLE

$REPORT

---
**Run stats:** Explored $FLOWS_EXPLORED user flows | $ISSUES_COUNT raw issue(s) detected during exploration ($CRITICAL_COUNT critical) | Duration: ${DURATION}s | $COST_NOTE
**Gate:** $GATE_DISPLAY
$SINCE_PREVIOUS_NOTE
$RECHECK_NOTE
$MODE_NOTE
$SCREENSHOTS_NOTE

<sub>Note: "raw issues detected" is the count of observations workers flagged during exploration, before the report above deduplicates, filters, and curates them — so it may differ from the issue count in the report. The report is the source of truth for findings.</sub>

<sub>Generated by [QA Bot](https://github.com/integuide/qa-bot) using ${MODEL_CREDIT:-Claude AI}</sub>
COMMENT_EOF
)

            # Re-runs (fix, @QABot again) must not stack stale full reports:
            # upsert_qabot_comment replaces the newest earlier QA Bot comment
            # (comment-mode=update) or posts a fresh one and collapses the
            # earlier ones (comment-mode=new).
            QABOT_BODY="$COMMENT_BODY"
            upsert_qabot_comment || true
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

# Gate on the CURATED count when synthesis produced a verdict (the report is
# the source of truth), else on the raw worker count (see GATE_COUNT above).
if [ "$INPUT_FAIL_ON_CRITICAL" = "true" ] && [ "$GATE_COUNT" -gt 0 ]; then
    if [ -n "$CURATED_CRITICAL_COUNT" ]; then
        echo "::error::Found $GATE_COUNT critical issue(s) after curation${CURATED_TITLES:+: $CURATED_TITLES} ($CRITICAL_COUNT raw worker finding(s))"
    else
        echo "::error::Found $CRITICAL_COUNT critical issue(s) (raw worker findings; synthesis produced no curated verdict)"
    fi
    EXIT_CODE=1
fi

# A run that tested nothing must not read as a green check: zero explored
# flows means the deploy was NOT verified (missing credentials, unreachable
# target, ...). The posted report names the blockers.
if [ "${INPUT_FAIL_ON_ZERO_FLOWS:-true}" = "true" ] && [ "$FLOWS_EXPLORED" -eq 0 ]; then
    echo "::error::QA Bot completed without testing any flows — the target was NOT verified. See the report for blockers."
    EXIT_CODE=1
fi

# Print report to logs for visibility. LLM output from site-influenced text:
# no line of it may run as a workflow command, so command processing is
# paused around it with a random token (GitHub's documented pattern).
echo "::group::Full QA Report"
REPORT_STOP_TOKEN="qabot_$(dd if=/dev/urandom bs=15 count=1 status=none | base64 | tr -dc 'A-Za-z0-9')"
echo "::stop-commands::$REPORT_STOP_TOKEN"
echo "$REPORT"
echo "::$REPORT_STOP_TOKEN::"
echo "::endgroup::"

exit $EXIT_CODE
