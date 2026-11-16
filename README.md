# QA Bot

AI-powered website QA testing — GPT-6 Luna via OpenRouter, or Claude. Automatically explores user flows and finds bugs, broken elements, or usability problems.

## How It Works

QA Bot uses an AI model (GPT-6 Luna with an OpenRouter key, Claude with an Anthropic key) to explore your website like a real user would:

1. **Discovers flows** - Finds signup, login, checkout, and other user journeys
2. **Tests interactions** - Clicks buttons, fills forms, navigates pages
3. **Identifies issues** - Reports bugs, broken elements, and UX problems
4. **Generates report** - Posts findings as a PR comment

## Quick Start

```yaml
- uses: Integuide/qa-bot@main
  with:
    url: 'https://your-staging-site.com'
    openrouter-api-key: ${{ secrets.QA_BOT_OPENROUTER_API_KEY }}  # runs GPT-6 Luna via OpenRouter/OpenAI (recommended)
    # or: anthropic-api-key: ${{ secrets.ANTHROPIC_API_KEY }}  # runs Claude Sonnet 5.5
    github-token: ${{ github.token }}  # Required to post the report as a PR comment
```

To post the report as a PR comment, the job also needs `permissions: pull-requests: write` (see the full example below).

## Inputs

| Input | Required | Default | Description |
|-------|----------|---------|-------------|
| `url` | Yes | - | Target URL to test |
| `openrouter-api-key` | One key required | - | OpenRouter API key — recommended: with no `model` set the run uses `openai/gpt-6-luna`; also needed for any other `vendor/slug` model. See [Model and keys](#model-and-keys) |
| `anthropic-api-key` | One key required | - | Anthropic API key, for Claude models (every model id without a slash); with no `model` and no `openrouter-api-key` the run uses `claude-sonnet-5-5` |
| `goal` | No | Explore flows and find bugs | Testing focus |
| `known-issues` | No | - | Already-acknowledged issues / environment caveats (free text, one per line). Matches aren't re-investigated and appear only as one-line "observed again" notes in the report, not as new findings |
| `previous-report` | No | - | The previous run's report for the same target (path to its `report.md`, or the markdown inline). Findings are labelled **NEW** / **recurring (seen in previous run)** and previous findings not observed are listed once (never as "fixed"); the PR footer prints "N new, M recurring". Labels only — severity never changes; use `known-issues` to acknowledge a finding |
| `mode` | No | `full` | `full` explores user flows against the goal; `smoke` runs one turn-capped worker that opens each top-level navigation link once (no sign-up/login/forms, no flow forking, `goal` ignored) and tags the report and PR comment "SMOKE TEST ONLY — not a regression test". For deploys that don't change the surface under test |
| `max-agents` | No | `3` (smoke: `1`) | Parallel AI agents (1-20; 1-10 recommended). Empty = the default for the mode |
| `max-cost` | No | `5.0` (smoke: `0.75`) | Maximum cost in USD. Empty = the default for the mode |
| `max-duration` | No | `30` (smoke: `5`) | Maximum duration in minutes. Empty = the default for the mode |
| `model` | No | (empty) | Empty = `openai/gpt-6-luna` when `openrouter-api-key` is set, else `claude-sonnet-5-5`. Or a Claude model / an OpenRouter `vendor/slug` id, which always wins |
| `post-comment` | No | `true` | Post results as PR comment |
| `comment-mode` | No | `update` | On re-runs, `update` edits the earlier QA Bot report comment in place (one current report per PR); `new` posts a fresh comment and collapses the earlier one under a "Superseded QA Bot report" summary |
| `github-token` | No | - | GitHub token for PR comments. Pass `${{ github.token }}` when `post-comment` is `true` |
| `fail-on-critical` | No | `true` | Fail workflow on critical issues (curated report count when available, raw worker count otherwise). Every critical is independently re-checked first — see [Issue Severity](#issue-severity) |
| `fail-on-zero-flows` | No | `true` | Fail the workflow when no flows were tested at all (missing credentials, unreachable target) so an untested deploy can't read as a green check |
| `credentials` | No | - | Test credentials (see below) |
| `testmail-api-key` | No | - | Testmail.app API key for email flows (experimental — not yet active) |
| `testmail-namespace` | No | - | Testmail.app namespace (experimental — not yet active) |
| `dangerously-skip-permissions` | No | `true` | Auto-approve destructive actions (enabled by default in CI) |

The four on/off inputs (`post-comment`, `fail-on-critical`, `fail-on-zero-flows`, `dangerously-skip-permissions`) take `true` or `false` in any case. Any other value (`yes`, `0`, `off`) fails the step with an error naming the input, so a misspelt `fail-on-critical` can't quietly switch the gate off. `known-issues` is always text: a value that names a file is not read.

A run that ended on a fatal error (a rejected API key, no credit, a crash) always fails, whatever `fail-on-critical` and `fail-on-zero-flows` say. So does a run that produced no report: there is nothing to base a verdict on. Any findings it made before the abort are still posted, marked **Run aborted**.

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
          openrouter-api-key: ${{ secrets.QA_BOT_OPENROUTER_API_KEY }}  # or anthropic-api-key for Claude
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
          openrouter-api-key: ${{ secrets.QA_BOT_OPENROUTER_API_KEY }}  # or anthropic-api-key for Claude
          post-comment: 'false'  # No PR to comment on
```

### With Test Credentials

Provide credentials for authenticated flows:

```yaml
- uses: Integuide/qa-bot@main
  with:
    url: 'https://staging.example.com'
    openrouter-api-key: ${{ secrets.QA_BOT_OPENROUTER_API_KEY }}  # or anthropic-api-key for Claude
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
          openrouter-api-key: ${{ secrets.QA_BOT_OPENROUTER_API_KEY }}  # or anthropic-api-key for Claude
          fail-on-critical: 'true'    # the default, shown for clarity
          fail-on-zero-flows: 'true'  # the default: a run that tested nothing fails too
          post-comment: 'false'       # a push has no PR to comment on

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
    openrouter-api-key: ${{ secrets.QA_BOT_OPENROUTER_API_KEY }}  # or anthropic-api-key for Claude
    github-token: ${{ github.token }}
    goal: 'Test the full signup flow including email verification'
    testmail-api-key: ${{ secrets.TESTMAIL_API_KEY }}
    testmail-namespace: ${{ secrets.TESTMAIL_NAMESPACE }}
```

## Command Line

The same tester runs from a terminal, with no GitHub involved. It needs
Python 3.10 or newer.

```bash
python -m venv .venv && source .venv/bin/activate
pip install git+https://github.com/Integuide/qa-bot
playwright install chromium

export OPENROUTER_API_KEY=...   # GPT-6 Luna; or ANTHROPIC_API_KEY for Claude
qa-bot https://staging.example.com --goal "Test the checkout flow"
qa-bot --help                   # every option
qa-bot --version
```

It is installed from this repository and is not on PyPI, so never
`pip install qa-bot`. Add a [release tag](https://github.com/Integuide/qa-bot/tags) from
`v0.2.0` on to the URL (`...qa-bot@v0.2.0`) to install that release. From a clone, `pip install .` does the
same, and `python -m qa_bot.cli` runs it without installing
(`pip install -r requirements.txt` first).

The report is printed to stdout (`--output json --output-file result.json`
writes the full result instead). Stderr carries a `Model:` line saying which
model ran and why, then warnings and errors; `-v` adds flow progress and `-vv`
every action. Run logs, the report and issue screenshots are also kept under
`./logs/<run id>/` (the newest 10 runs).

The exit code is the deploy gate, so `qa-bot URL && deploy` only
deploys a site that was tested and came back clean:

| Exit code | Meaning |
|-----------|---------|
| `0` | The site was tested and no critical finding gates |
| `1` | Critical findings (the report's curated count when it has a verdict, else the raw worker count) |
| `2` | No verdict: the run aborted (rejected key, no credit, a crash), nothing was tested (unreachable target, no user flow completed, no report), or a configuration error (missing key, bad credentials file, a URL or limit that cannot work) |
| `130` | Interrupted (Ctrl-C) |

`--allow-zero-flows` accepts a run in which no flow was tested (a report-only
run). With `--output json`, the result's `gate` key carries the same verdict
and its reasons.

The Action's inputs have matching flags: `--smoke`, `--max-agents`,
`--max-cost`, `--max-duration`, `--model`, `--credentials creds.env` (or
`--credential KEY=VALUE`), `--previous-report report.md`, and
`--dangerously-skip-permissions`, which is **off** by default here (the Action
turns it on because nobody can answer an approval request in CI). Known issues
come as text (`--known-issues "one per line"`) or from a file
(`--known-issues-file .qa-known-issues.md`; blank lines and `#` lines are
skipped). `--previous-report report.md --output-file report.md` keeps one
file as the run-over-run memory: the old report is read before it is
replaced.

## Cost Management

QA Bot tracks API costs in real-time and stops when limits are reached:

- **Default limit:** $5.00 per full run, $0.75 per smoke run

| Model | When it runs | Cost |
|-------|--------------|------|
| **GPT-6 Luna** (`openai/gpt-6-luna`) | The default with an OpenRouter key | $0.10 / $0.50 per million input / output tokens. One measured comparison: the same 8-minute run cost $0.15 on Luna and $1.89 on Claude Sonnet 5, about 13x less. No typical full-run range has been measured yet |
| **Claude Sonnet 5.5** (`claude-sonnet-5-5`) | The default with only an Anthropic key | About $1.50-7.00 per typical full run (smoke runs about $0.25) |
| **Claude Haiku 5.5** (`claude-haiku-5-5`) | When named as `model` | The same per-token rates as GPT-6 Luna; not yet measured on a full run, and a lower bound for very long flows (see [Model and keys](#model-and-keys)) |
| **Claude Opus 4.8** (`claude-opus-4-8`) | When named as `model` | About $4.00-15.00 per typical full run |

Adjust with `max-cost` and `max-duration` inputs.

When the budget runs out mid-run, flows are cut off in queue order — so the
bot schedules flows that match the `goal` text first and tells workers to
wrap up early as the limit approaches. A specific `goal` (naming the features
to verify) therefore buys much better coverage per dollar than a generic one.
If a run's report says goal-critical flows were still untested at the cost
limit, raise `max-cost` for that workflow.

## Model and keys

With `model` empty (the default), QA Bot runs **GPT-6 Luna**
(`openai/gpt-6-luna`, via OpenRouter) whenever `openrouter-api-key` is set —
about 13x cheaper than Claude Sonnet 5 on the same run, with somewhat
shallower coverage — and **Claude Sonnet 5.5** otherwise, so a workflow that
passes only `anthropic-api-key` keeps running Claude. Name a `model` to pick
one yourself. The job log says which model ran and why.

The action fails fast if the resolved model's key is missing, and for an
OpenRouter run the PR footer shows the charge OpenRouter reported next to
the token estimate. **With an OpenRouter key, screenshots, page text and
test credentials go to OpenRouter and the model vendor (OpenAI), not
Anthropic.**

**Earlier reasoning on newer Claude models:** after each turn the worker
replaces the prompt it sent with a short summary, and Claude's
preserved-thinking check rejects a replayed thinking block once the
conversation before it has changed (HTTP 400, "bound to a different
conversation"). Claude Fable 5.2 applies the check to every account; Haiku
5.5, Sonnet 5.5, Opus 5.5 and Fable 5.1 apply it to Anthropic accounts
created on or after 2026-08-31. Where it applies, the bot sends earlier
turns without their thinking blocks: from the start on Fable 5.2, and
otherwise after the first rejected request of a run, which is retried at
once and logged as a warning. The model then works from each turn's
screenshot and the Recent Actions recap, without its own earlier reasoning.
Older accounts on the other models are unchanged.

Haiku 5.5's cost figure is a lower bound for very long flows: a call whose
prompt passes 100K tokens bills at $0.50/$2.50 per MTok (5x), which the
estimate does not model. The run logs one warning when that happens and
records the largest prompt as `max_call_prompt_tokens` in `summary.json`.

## Issue Severity

Issues are categorized by severity:

| Severity | Description | Example |
|----------|-------------|---------|
| **Critical** | Blocks core functionality, a security issue, or data loss. The only level that fails the check | Signup fails, checkout broken |
| **Major** | A feature is broken, or a significant UX problem | Search returns wrong results |
| **Minor** | Small bugs and inconsistencies | A missing validation message |
| **Cosmetic** | Visual styling only | Misaligned buttons, typos |

Before a critical can fail the check, a separate verifier re-tests it during
the run in a fresh browser. If it does not reproduce, the finding is demoted
to major and tagged `[UNCONFIRMED — independent re-check did not reproduce]`
(still reported, never gating); reproduced, inconclusive and not-re-checked
criticals stay critical. The PR footer's **Independent re-check of
criticals** line gives the counts. Set `RECHECK_CRITICALS: 'false'` in the
step's `env:` to turn it off.

## Versions and Pinning

`Integuide/qa-bot@main` always runs the newest QA Bot: the tester's prompts,
default model and gate logic can change between two of your runs. That suits a
report-only check. For a deploy gate you may prefer to choose when it changes:
pin a release and move the pin when you want the newer one.

Since `v0.1.0`, every change to this repository is a release with a tag
(`v0.2.0`, `v0.3.0`, ...; see [tags](https://github.com/Integuide/qa-bot/tags)). Pin the tag's
commit and keep the tag in a comment:

```yaml
- uses: Integuide/qa-bot@<full commit sha>  # v0.2.0
```

With that comment, Dependabot's `github-actions` updates open a pull request
when a newer release exists, so the tester changes when you merge it. `main`
cannot be force-pushed or deleted, so a pinned commit stays valid. The run's
log starts with the version (`QA Bot 0.2.0: starting ...`).

## Requirements

- An OpenRouter API key ([get one here](https://openrouter.ai/settings/keys)) or an Anthropic API key ([get one here](https://console.anthropic.com/))
- Publicly accessible URL (or use a tunnel for localhost)

**Re-running on the same PR?** The report comment is updated in place by default (it carries a hidden `<!-- qa-bot-report -->` marker), so a fix-and-rerun cycle never leaves a stack of stale reports. Set `comment-mode: new` to keep each run as its own comment — earlier reports are then collapsed under a "Superseded QA Bot report" summary. A run that fails before producing results posts a short failure notice the same way: in `update` mode it replaces the earlier report comment, so the PR never shows a stale report as current.

**Very long report?** GitHub caps a comment at 65,536 characters. A report over 58,000 bytes is cut at a line boundary in the comment, with a note linking the workflow run; the `report` output, the job log and the `report.md` artifact keep the whole report.

**Report not appearing as a PR comment?** Make sure the workflow passes `github-token: ${{ github.token }}` to the action and the job grants `permissions: pull-requests: write` — without both, the run completes but the comment is silently skipped.

## Contributing and Security

This repository is generated from Integuide's private monorepo on every
release, and each sync replaces its whole contents. A pull request merged
here would be overwritten by the next one, so please open an
[issue](https://github.com/Integuide/qa-bot/issues) instead: bug reports,
false positives the tester should not have raised, and feature requests are
all welcome there.

QA Bot is given an API key and, if you pass them, test-account credentials,
and it drives a browser against a site whose content it cannot trust. If you
find a way for a tested page to make it leak a credential or act outside the
target site, please report it privately through the repository's
**Security** tab ("Report a vulnerability") instead of a public issue. If
that button is not shown, open an issue that asks for a private contact and
leaves the details out.

## License

MIT
