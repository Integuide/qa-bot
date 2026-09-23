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
| `anthropic-api-key` | For Claude models | - | Anthropic API key (every model id without a slash, including the default) |
| `openrouter-api-key` | For OpenRouter models | - | OpenRouter API key, only for a `vendor/slug` `model` such as `openai/gpt-6-luna` — see [Non-Claude models](#non-claude-models-openrouter) |
| `goal` | No | Explore flows and find bugs | Testing focus |
| `known-issues` | No | - | Already-acknowledged issues / environment caveats (free text, one per line). Matches aren't re-investigated and appear only as one-line "observed again" notes in the report, not as new findings |
| `previous-report` | No | - | The previous run's report for the same target (path to its `report.md`, or the markdown inline). Findings are labelled **NEW** / **recurring (seen in previous run)** and previous findings not observed are listed once (never as "fixed"); the PR footer prints "N new, M recurring". Labels only — severity never changes; use `known-issues` to acknowledge a finding |
| `mode` | No | `full` | `full` explores user flows against the goal; `smoke` runs one turn-capped worker that opens each top-level navigation link once (no sign-up/login/forms, no flow forking, `goal` ignored) and tags the report and PR comment "SMOKE TEST ONLY — not a regression test". For deploys that don't change the surface under test |
| `max-agents` | No | `3` (smoke: `1`) | Parallel AI agents (1-10). Empty = the default for the mode |
| `max-cost` | No | `5.0` (smoke: `0.75`) | Maximum cost in USD. Empty = the default for the mode |
| `max-duration` | No | `30` (smoke: `5`) | Maximum duration in minutes. Empty = the default for the mode |
| `model` | No | `claude-sonnet-5` | A Claude model, or an OpenRouter `vendor/slug` id such as `openai/gpt-6-luna` |
| `post-comment` | No | `true` | Post results as PR comment |
| `comment-mode` | No | `update` | On re-runs, `update` edits the earlier QA Bot report comment in place (one current report per PR); `new` posts a fresh comment and collapses the earlier one under a "Superseded QA Bot report" summary |
| `github-token` | No | - | GitHub token for PR comments. Pass `${{ github.token }}` when `post-comment` is `true` |
| `fail-on-critical` | No | `true` | Fail workflow on critical issues (curated report count when available, raw worker count otherwise). Every critical is independently re-checked first — see [Issue Severity](#issue-severity) |
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
| `cost-usd` | Estimated model API cost of the run in USD (token-priced; an OpenRouter run's PR footer also shows the charge OpenRouter reported) |
| `mode` | The run mode that produced the report: `full` or `smoke` |
| `new-findings` / `recurring-findings` | Findings labelled NEW / recurring against `previous-report` (empty when none was supplied, no curated verdict, or the counts exceeded the findings listed) |

## Issue Screenshots and Run Files

When the bot captures visual evidence for an issue, it exports the PNGs to a
`qa-bot-screenshots/` directory in the workspace, together with the run's
credential-masked `report.md` and `summary.json`. Upload `report.md` under a
stable per-target artifact name and pass its text back as `previous-report`
on the next run to get NEW / recurring labels (the [example workflows](#example-workflows)
show the wiring). Add an
`actions/upload-artifact` step named `qa-bot-screenshots` after the QA Bot step (shown in the examples
below) to publish them as a workflow artifact — the PR comment points
reviewers at that artifact whenever screenshots were captured, and the report
cites each finding's screenshot by filename (e.g. `issue_003.png`) so it can
be found in the artifact.

> **Warning:** Issue screenshots may capture authenticated pages — anything
> visible after the bot logs in with the credentials you provide (account
> dashboards, profile details, etc.). Workflow artifacts are downloadable by
> anyone with read access to the repository, so omit the upload step if that
> audience should not see those pages.

## Example Workflows

Start from the complete workflows in [`examples/`](examples/) — they are kept
in step with the workflows that gate Integuide's own deploys, and are what
most setups should copy rather than the minimal snippets below:

| File | Use |
|------|-----|
| [`qa-staging-to-prod.yml`](examples/qa-staging-to-prod.yml) | Deploy gate on staging → production PRs (and `@QABot` comments on them). `mode: full` with a goal generated from the PR description when the deploy touches the app (`FULL_MODE_PATHS`), `mode: smoke` otherwise; both gates on |
| [`qa-on-demand.yml`](examples/qa-on-demand.yml) | Report-only run when a maintainer starts a PR comment with `@QABot [url]`. Skips deploy PRs the gate already answers; `fail-on-zero-flows: 'false'` |
| [`qa-manual.yml`](examples/qa-manual.yml) | `workflow_dispatch` with URL, goal, mode, limits and model inputs |

All three carry the practices the snippets leave out:

- **Run-over-run memory** — `known-issues` from a checked-in
  `.qa-known-issues.md` (one issue per line, `#` comments), and
  `previous-report` from the newest `qa-bot-report-*` artifact, which every run
  uploads (`report.md` + `summary.json`) under a stable per-host (and per-mode)
  name. This needs `permissions: actions: read`. The two are never merged: a
  previous report pasted into known-issues demotes a regression found the run
  before.
- **Uploads** of the `qa-bot-screenshots` artifact the PR comment points to,
  and of the report for the next run.
- **Job-level `concurrency`** — a workflow-level group is evaluated before the
  job's `if`, so any PR comment would cancel an in-flight run.
- **Credentials scoped to the target** — the on-demand and manual examples pass
  `QA_BOT_CREDENTIALS` (and known issues) only when the URL is on the
  `QA_BOT_STAGING_URL` host, since the bot types credential values into the
  tested site. "On the host" means the host the browser will actually open:
  the examples parse the URL with Python's `urlsplit` and give no credentials
  to anything but a plain `http(s)://host[:port]` (no userinfo, backslash or
  %-escape), because cutting the URL at the first `:` or `/` reads
  `http://staging.example.com:80@evil.example/` as your staging host. If you
  extend the check to more hosts, compare that parsed host, never a
  substring of the raw URL.
- **Goal prompts kept out of the jq program** — prose lives in a shell
  variable, so an apostrophe can't break the single-quoted jq filter and kill
  the step before the fallback goal engages.

### Run on Pull Requests

Minimal: test your staging environment whenever a PR is opened:

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
      - uses: actions/upload-artifact@v7
        if: always()
        with:
          name: qa-bot-screenshots
          path: qa-bot-screenshots/
          if-no-files-found: ignore
```

### Manual Trigger with Custom URL

Minimal version of [`qa-manual.yml`](examples/qa-manual.yml):

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
          url: ${{ inputs.url }}
          goal: ${{ inputs.goal }}
          anthropic-api-key: ${{ secrets.ANTHROPIC_API_KEY }}
          post-comment: 'false'  # No PR to comment on
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
- **Sonnet 5 model (default):** ~$1.50-7.00 per typical full run (smoke runs ~$0.25)
- **Haiku model (budget):** ~$0.50-2.00 per typical run
- **Opus 4.8 model (premium):** ~$4.00-15.00 per typical run

Adjust with `max-cost` and `max-duration` inputs.

When the budget runs out mid-run, flows are cut off in queue order — so the
bot schedules flows that match the `goal` text first and tells workers to
wrap up early as the limit approaches. A specific `goal` (naming the features
to verify) therefore buys much better coverage per dollar than a generic one.
If a run's report says goal-critical flows were still untested at the cost
limit, raise `max-cost` for that workflow.

## Non-Claude models (OpenRouter)

Set `model` to an OpenRouter `vendor/slug` id and pass `openrouter-api-key`
instead of `anthropic-api-key`:

```yaml
- uses: Integuide/qa-bot@main
  with:
    url: 'https://your-staging-site.com'
    model: openai/gpt-6-luna
    openrouter-api-key: ${{ secrets.OPENROUTER_API_KEY }}
    github-token: ${{ github.token }}
```

The same prompts run through OpenRouter's chat API. The action fails fast
if the chosen model's key is missing, and the PR footer shows the charge
OpenRouter reported next to the token estimate. **Screenshots, page text
and test credentials go to OpenRouter and the model vendor, not
Anthropic.**

## Issue Severity

Issues are categorized by severity:

| Severity | Description | Example |
|----------|-------------|---------|
| **Critical** | Blocks core functionality | Signup fails, checkout broken |
| **High** | Major feature broken | Search returns wrong results |
| **Medium** | Noticeable problems | Form validation missing |
| **Low** | Minor issues | Typos, styling glitches |

Before a critical can fail the check, a separate verifier re-tests it during
the run in a fresh browser. If it does not reproduce, the finding is demoted
to major and tagged `[UNCONFIRMED — independent re-check did not reproduce]`
(still reported, never gating); reproduced, inconclusive and not-re-checked
criticals stay critical. The PR footer's **Independent re-check of
criticals** line gives the counts. Set `RECHECK_CRITICALS: 'false'` in the
step's `env:` to turn it off.

## Requirements

- Anthropic API key ([get one here](https://console.anthropic.com/)), or an OpenRouter key for an OpenRouter `model`
- Publicly accessible URL (or use a tunnel for localhost)

**Re-running on the same PR?** The report comment is updated in place by default (it carries a hidden `<!-- qa-bot-report -->` marker), so a fix-and-rerun cycle never leaves a stack of stale reports. Set `comment-mode: new` to keep each run as its own comment — earlier reports are then collapsed under a "Superseded QA Bot report" summary. A run that fails before producing results always posts a separate, short failure notice.

**Report not appearing as a PR comment?** Make sure the workflow passes `github-token: ${{ github.token }}` to the action and the job grants `permissions: pull-requests: write` — without both, the run completes but the comment is silently skipped.

## License

MIT
