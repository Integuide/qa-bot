# QA Bot

AI-powered website QA testing using Claude. Automatically explores user flows and finds bugs, broken elements, or usability problems.

[![GitHub Marketplace](https://img.shields.io/badge/Marketplace-QA%20Bot-blue?logo=github)](https://github.com/marketplace/actions/qa-bot)

## How It Works

QA Bot uses Claude AI to explore your website like a real user would:

1. **Discovers flows** - Finds signup, login, checkout, and other user journeys
2. **Tests interactions** - Clicks buttons, fills forms, navigates pages
3. **Identifies issues** - Reports bugs, broken elements, and UX problems
4. **Generates report** - Posts findings as a PR comment

## Quick Start

```yaml
- uses: Integuide/qa-bot@v1
  with:
    url: 'https://your-staging-site.com'
    anthropic-api-key: ${{ secrets.ANTHROPIC_API_KEY }}
```

## Inputs

| Input | Required | Default | Description |
|-------|----------|---------|-------------|
| `url` | Yes | - | Target URL to test |
| `anthropic-api-key` | Yes | - | Anthropic API key |
| `goal` | No | Explore flows and find bugs | Testing focus |
| `max-agents` | No | `3` | Parallel AI agents (1-10) |
| `max-cost` | No | `5.0` | Maximum cost in USD |
| `max-duration` | No | `30` | Maximum duration in minutes |
| `model` | No | `claude-haiku-4-5` | Claude model |
| `post-comment` | No | `true` | Post results as PR comment |
| `fail-on-critical` | No | `true` | Fail workflow on critical issues |
| `credentials` | No | - | Test credentials (see below) |
| `testmail-api-key` | No | - | Testmail.app API key for email flows |
| `testmail-namespace` | No | - | Testmail.app namespace |
| `dangerously-skip-permissions` | No | `false` | Auto-approve destructive actions (use with caution) |

## Outputs

| Output | Description |
|--------|-------------|
| `report` | Markdown QA report with findings |
| `issues-count` | Total number of issues found |
| `critical-issues` | Number of critical issues |
| `flows-explored` | Number of user flows tested |

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
    steps:
      - uses: Integuide/qa-bot@v1
        with:
          url: 'https://staging.example.com'
          anthropic-api-key: ${{ secrets.ANTHROPIC_API_KEY }}
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
      - uses: Integuide/qa-bot@v1
        with:
          url: ${{ github.event.inputs.url }}
          goal: ${{ github.event.inputs.goal }}
          anthropic-api-key: ${{ secrets.ANTHROPIC_API_KEY }}
```

### With Test Credentials

Provide credentials for authenticated flows:

```yaml
- uses: Integuide/qa-bot@v1
  with:
    url: 'https://staging.example.com'
    anthropic-api-key: ${{ secrets.ANTHROPIC_API_KEY }}
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
      - uses: Integuide/qa-bot@v1
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

### Email Verification Testing

Test signup flows with email verification using [Testmail.app](https://testmail.app):

```yaml
- uses: Integuide/qa-bot@v1
  with:
    url: 'https://staging.example.com'
    anthropic-api-key: ${{ secrets.ANTHROPIC_API_KEY }}
    goal: 'Test the full signup flow including email verification'
    testmail-api-key: ${{ secrets.TESTMAIL_API_KEY }}
    testmail-namespace: ${{ secrets.TESTMAIL_NAMESPACE }}
```

## Cost Management

QA Bot tracks API costs in real-time and stops when limits are reached:

- **Default limit:** $5.00 per run
- **Haiku model:** ~$0.50-2.00 per typical run
- **Sonnet model:** ~$2.00-8.00 per typical run

Adjust with `max-cost` and `max-duration` inputs.

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

## License

MIT
