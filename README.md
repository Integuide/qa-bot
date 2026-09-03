# QA Bot

AI-powered website QA testing using Claude. Automatically explores user flows and finds bugs, broken elements, or usability problems.

## How It Works

QA Bot uses Claude AI to explore your website like a real user would:

1. **Discovers flows** - Finds signup, login, checkout, and other user journeys
2. **Tests interactions** - Clicks buttons, fills forms, navigates pages
3. **Identifies issues** - Reports bugs, broken elements, and UX problems
4. **Generates report** - Posts findings as a PR comment

## Quick Start

```yaml
- uses: Integuide/qa-bot@main
  with:
    url: 'https://your-staging-site.com'
    anthropic-api-key: ${{ secrets.ANTHROPIC_API_KEY }}
    github-token: ${{ github.token }}  # Required to post the report as a PR comment
```

To post the report as a PR comment, the job also needs `permissions: pull-requests: write` (see the full example below).

## Inputs

| Input | Required | Default | Description |
|-------|----------|---------|-------------|
| `url` | Yes | - | Target URL to test |
| `anthropic-api-key` | Yes | - | Anthropic API key |
| `goal` | No | Explore flows and find bugs | Testing focus |
| `known-issues` | No | - | Already-acknowledged issues / environment caveats (free text, one per line). Matches aren't re-investigated and appear only as one-line "observed again" notes in the report, not as new findings |
| `previous-report` | No | - | The previous run's report for the same target (path to its `report.md`, or the markdown inline). Findings are labelled **NEW** / **recurring (seen in previous run)** and previous findings not observed are listed once (never as "fixed"); the PR footer prints "N new, M recurring". Labels only — severity never changes; use `known-issues` to acknowledge a finding |
| `mode` | No | `full` | `full` explores user flows against the goal; `smoke` runs one turn-capped worker that opens each top-level navigation link once (no sign-up/login/forms, no flow forking, `goal` ignored) and tags the report and PR comment "SMOKE TEST ONLY — not a regression test". For deploys that don't change the surface under test |
| `max-agents` | No | `3` (smoke: `1`) | Parallel AI agents (1-10). Empty = the default for the mode |
| `max-cost` | No | `5.0` (smoke: `0.75`) | Maximum cost in USD. Empty = the default for the mode |
| `max-duration` | No | `30` (smoke: `5`) | Maximum duration in minutes. Empty = the default for the mode |
| `model` | No | `claude-sonnet-5` | Claude model |
| `post-comment` | No | `true` | Post results as PR comment |
| `comment-mode` | No | `update` | On re-runs, `update` edits the earlier QA Bot report comment in place (one current report per PR); `new` posts a fresh comment and collapses the earlier one under a "Superseded QA Bot report" summary |
| `github-token` | No | - | GitHub token for PR comments. Pass `${{ github.token }}` when `post-comment` is `true` |
| `fail-on-critical` | No | `true` | Fail workflow on critical issues (curated report count when available, raw worker count otherwise) |
| `fail-on-zero-flows` | No | `true` | Fail the workflow when no flows were tested at all (missing credentials, unreachable target) so an untested deploy can't read as a green check |
| `credentials` | No | - | Test credentials (see below) |
| `testmail-api-key` | No | - | Testmail.app API key for email flows (experimental — not yet active) |
| `testmail-namespace` | No | - | Testmail.app namespace (experimental — not yet active) |
| `dangerously-skip-permissions` | No | `true` | Auto-approve destructive actions (enabled by default in CI) |

## Outputs

| Output | Description |
|--------|-------------|
| `report` | Markdown QA report with findings |
| `issues-count` | Total number of issues found (raw, pre-synthesis) |
| `critical-issues` | Number of critical issues — raw worker-reported count (pre-curation). Gates `fail-on-critical` only when `curated-critical-issues` is empty |
| `curated-critical-issues` | Critical issues in the synthesized report after curation — drives `fail-on-critical` whenever present (empty when no AI verdict was available; the raw count gates then) |
| `flows-explored` | Number of user flows tested to completion (the initial flow-enumeration step is not counted) |
| `cost-usd` | Estimated Claude API cost of the run in USD |
| `mode` | The run mode that produced the report: `full` or `smoke` |
| `new-findings` / `recurring-findings` | Findings labelled NEW / recurring against `previous-report` (empty when none was supplied or no curated verdict) |

## Issue Screenshots and Run Files

When the bot captures visual evidence for an issue, it exports the PNGs to a
`qa-bot-screenshots/` directory in the workspace, together with the run's
credential-masked `report.md` and `summary.json`. Upload `report.md` under a
stable per-target artifact name and pass its text back as `previous-report`
on the next run to get NEW / recurring labels (the Integuide monorepo
workflows show the wiring). Add an
`actions/upload-artifact` step after the QA Bot step (shown in the examples
below) to publish them as a workflow artifact — the PR comment points
reviewers at that artifact whenever screenshots were captured, and the report
cites each finding's screenshot by filename (e.g. `issue_003.png`) so it can
be found in the artifact.

> **Warning:** Issue screenshots may capture authenticated pages — anything
> visible after the bot logs in with the credentials you provide (account
> dashboards, profile details, etc.). Workflow artifacts are downloadable by
> anyone with read access to the repository, so omit the upload step if that
> audience should not see those pages.

## Examples

### Run on Pull Requests

Test your staging environment whenever a PR is opened:

```yaml
name: QA Bot
on:
  pull_request:
    types: [opened, synchronize]

jobs:
  qa:
    runs-on: ubuntu-latest
    permissions:
      pull-requests: write  # Allows QA Bot to post the report as a PR comment
    steps:
      - uses: Integuide/qa-bot@main
        with:
          url: 'https://staging.example.com'
          anthropic-api-key: ${{ secrets.ANTHROPIC_API_KEY }}
          github-token: ${{ github.token }}

      # Publish issue screenshots as an artifact (referenced by the PR comment)
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: qa-bot-screenshots
          path: qa-bot-screenshots/
          if-no-files-found: ignore
```

### Manual Trigger with Custom URL

Run on-demand with a configurable target:

```yaml
name: QA Bot (Manual)
on:
  workflow_dispatch:
    inputs:
      url:
        description: 'URL to test'
        required: true
        default: 'https://staging.example.com'
      goal:
        description: 'What to test'
        required: false

jobs:
  qa:
    runs-on: ubuntu-latest
    steps:
      - uses: Integuide/qa-bot@main
        with:
          url: ${{ github.event.inputs.url }}
          goal: ${{ github.event.inputs.goal }}
          anthropic-api-key: ${{ secrets.ANTHROPIC_API_KEY }}
          github-token: ${{ github.token }}
```

### With Test Credentials

Provide credentials for authenticated flows:

```yaml
- uses: Integuide/qa-bot@main
  with:
    url: 'https://staging.example.com'
    anthropic-api-key: ${{ secrets.ANTHROPIC_API_KEY }}
    github-token: ${{ github.token }}
    credentials: |
      TEST_USER_EMAIL=${{ secrets.TEST_USER_EMAIL }}
      TEST_USER_PASSWORD=${{ secrets.TEST_USER_PASSWORD }}
```

### Gate Production Deploys

Block production deployment if critical issues are found:

```yaml
name: Deploy to Production
on:
  push:
    branches: [main]

jobs:
  qa:
    runs-on: ubuntu-latest
    steps:
      - uses: Integuide/qa-bot@main
        with:
          url: 'https://staging.example.com'
          anthropic-api-key: ${{ secrets.ANTHROPIC_API_KEY }}
          fail-on-critical: 'true'

  deploy:
    needs: qa
    runs-on: ubuntu-latest
    steps:
      - run: echo "Deploying to production..."
```

### Email Verification Testing (experimental — not yet active)

Inputs for testing email verification flows with [Testmail.app](https://testmail.app) are accepted, but the email-reading actions are **not yet integrated** into the exploration engine. Today the bot cannot open verification emails — it reports email verification steps as untestable and continues with other flows. Once the integration lands, configuration will look like:

```yaml
- uses: Integuide/qa-bot@main
  with:
    url: 'https://staging.example.com'
    anthropic-api-key: ${{ secrets.ANTHROPIC_API_KEY }}
    github-token: ${{ github.token }}
    goal: 'Test the full signup flow including email verification'
    testmail-api-key: ${{ secrets.TESTMAIL_API_KEY }}
    testmail-namespace: ${{ secrets.TESTMAIL_NAMESPACE }}
```

## Cost Management

QA Bot tracks API costs in real-time and stops when limits are reached:

- **Default limit:** $5.00 per run
- **Sonnet 5 model (default):** ~$2.00-8.00 per typical run
- **Haiku model (budget):** ~$0.50-2.00 per typical run
- **Opus 4.8 model (premium):** ~$4.00-15.00 per typical run

Adjust with `max-cost` and `max-duration` inputs.

When the budget runs out mid-run, flows are cut off in queue order — so the
bot schedules flows that match the `goal` text first and tells workers to
wrap up early as the limit approaches. A specific `goal` (naming the features
to verify) therefore buys much better coverage per dollar than a generic one.
If a run's report says goal-critical flows were still untested at the cost
limit, raise `max-cost` for that workflow.

## Issue Severity

Issues are categorized by severity:

| Severity | Description | Example |
|----------|-------------|---------|
| **Critical** | Blocks core functionality | Signup fails, checkout broken |
| **High** | Major feature broken | Search returns wrong results |
| **Medium** | Noticeable problems | Form validation missing |
| **Low** | Minor issues | Typos, styling glitches |

## Requirements

- Anthropic API key ([get one here](https://console.anthropic.com/))
- Publicly accessible URL (or use a tunnel for localhost)

**Re-running on the same PR?** The report comment is updated in place by default (it carries a hidden `<!-- qa-bot-report -->` marker), so a fix-and-rerun cycle never leaves a stack of stale reports. Set `comment-mode: new` to keep each run as its own comment — earlier reports are then collapsed under a "Superseded QA Bot report" summary. A run that fails before producing results always posts a separate, short failure notice.

**Report not appearing as a PR comment?** Make sure the workflow passes `github-token: ${{ github.token }}` to the action and the job grants `permissions: pull-requests: write` — without both, the run completes but the comment is silently skipped.

## License

MIT
