# Model CICD

Continuous model discovery, sandboxed benchmarking, and human-approved
promotion — for every place your application calls an LLM.

Model CICD does not try to find "the best model." It finds the best model
**for one specific job in your application**, using that job's own test
cases and its own quality bar, and it never changes what your app uses in
production until a human explicitly approves the change.

```python
from modelcicd.resolver import resolve

model = resolve("support_bot_reply", fallback="gpt-4o-mini")
# call your existing LLM client with `model`, exactly as before
```

That's the entire integration surface. Everything else in this repo exists
to answer one question well: which model id should that line return?

---

## Why this exists

- **New models ship constantly, and prices move.** Re-checking the field by
  hand does not scale past one or two use cases.
- **A generic leaderboard lies to you.** A model that tops a public
  benchmark can still write unusable output for a task nobody else tested it
  on. "Correctness" means something different for a support bot than for a
  codegen assistant.
- **An LLM grading its own family's writing is not a benchmark.** Judging
  needs to be blind and done by a model that isn't also a candidate.
- **A model swap in production is a real decision.** It should require the
  same kind of sign-off a dependency upgrade would get — not happen silently
  because a scheduled job found a slightly higher number.

Model CICD is built around those four constraints. It's a flat-file,
self-hosted, single-tenant tool — no server, no database, no signup — meant
to be dropped into an existing repository.

---

## How it works

```
 catalogue.py          guardrails.py         sandbox.py        judge.py        rank.py         state.py
┌───────────────┐    ┌────────────────┐    ┌────────────┐    ┌───────────┐   ┌───────────┐   ┌──────────────┐
│ poll every     │ →  │ filter by      │ →  │ call every  │ →  │ blind,    │ → │ leaderboard│ → │ approve()    │
│ model + price  │    │ price/context/ │    │ candidate   │    │ pinned    │   │ per price  │   │ is the ONLY  │
│ (free, no LLM  │    │ JSON-mode/     │    │ with your   │    │ judge     │   │ tier, ties │   │ thing that   │
│ call)          │    │ modality       │    │ test cases  │    │ scores    │   │ within     │   │ changes what │
│                │    │ (free)         │    │ ($)         │    │ each      │   │ noise      │   │ resolve()    │
│                │    │                │    │             │    │ answer($) │   │            │   │ returns      │
└───────────────┘    └────────────────┘    └────────────┘    └───────────┘   └───────────┘   └──────────────┘
```

1. **Catalogue** — polls OpenRouter's model list (id, price, context length,
   JSON-mode support). Pure `GET` request; no model is ever called here.
2. **Guardrails** — filters that catalogue against one use case's price
   ceiling, minimum context, JSON-mode requirement, and modality. Every
   rejection is named with the exact rule and value that failed it. Still
   free.
3. **Sandbox** — runs every surviving candidate against that use case's own
   test cases, one call per test case. A model that fails is recorded with
   its error, not silently dropped from the leaderboard.
4. **Judge** — a separate, pinned model scores every answer against the use
   case's own rubric, blind (one answer at a time, never told which model
   wrote it). It refuses to run if the judge is also a candidate.
5. **Rank** — builds a leaderboard **within each price tier** (free /
   paid-low / paid-mid / paid-high), because a global ranking always
   concludes the expensive model won. Scores within a small margin are
   shown as a tie, not a winner — one sample per test case is confident at
   the extremes, not enough to crown a narrow "best."
6. **State + notify** — a candidate that clearly beats the currently
   approved model is recorded as **pending** and, if configured, emailed to
   a human. Nothing in production changes yet.
7. **Approve** — the one action that promotes a model. Only after this does
   `resolve()` return the new id.

---

## Quickstart

```bash
# 1. Install
python -m venv .venv
.venv\Scripts\activate            # Windows
source .venv/bin/activate         # macOS/Linux
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# edit .env and set OPENROUTER_API_KEY

# 3. Write a use case (or copy the example under examples/prep_material/)
python -m modelcicd.cli init --out use_cases/my_feature.yaml

# 4. See what candidates survive your guardrails — free, no model calls
python -m modelcicd.cli shortlist --use-case use_cases/my_feature.yaml

# 5. Run the bench — this is the only command that spends money,
#    and it shows the cost and asks before it does
python -m modelcicd.cli run --use-case use_cases/my_feature.yaml

# 6. Check what's approved vs. pending
python -m modelcicd.cli status --use-case use_cases/my_feature.yaml

# 7. Promote a candidate once you're satisfied
python -m modelcicd.cli approve --use-case use_cases/my_feature.yaml
```

---

## CLI reference

| Command | Cost | What it does |
|---|---|---|
| `init` | free | Writes a starting `use_case.yaml` template |
| `catalogue` | free | Polls OpenRouter and saves `out/catalogue.json` |
| `shortlist --use-case <path>` | free | Applies one use case's guardrails to the catalogue |
| `run --use-case <path>` | **$** | Sandboxes + judges + ranks the shortlist |
| `status [--use-case <path> \| --use-case-name <name>]` | free | Shows the approved model and any pending candidate |
| `approve [--use-case <path> \| --use-case-name <name>] [--model <id>]` | free | Promotes a model — the only thing that changes `resolve()` |

Useful `run` flags: `--tier {free,paid-low,paid-mid,paid-high}` to restrict
which price band is benched, `--models a,b,c` to bench an explicit list
instead of the guardrail-filtered shortlist, `--limit N` to cap candidate
count, `--yes` to skip the interactive cost confirmation (for scheduled
jobs).

---

## One use case per feature

Every place your app calls an LLM gets its own `use_case.yaml` — its own
system prompt, test cases, rubric, price ceiling, and notification target.
Nothing else needs a database; the file **is** the onboarding.

```yaml
useCase: support_bot_reply
description: Drafts a first-line reply to an incoming support ticket.

systemPrompt: |
  You are a support agent. Read the ticket and draft a reply. Return
  exactly: {"reply": "the message to send"}

testCases:
  - id: angry_refund_request
    input: "Customer: This is the third time my order arrived broken.
            I want a refund NOW."
    reference: null

rubric:
  - id: tone
    description: Calm, empathetic, never defensive.
    weight: 2
  - id: resolves
    description: Actually proposes a concrete next step, not just an apology.
    weight: 1.5
  - id: format
    description: Returns exactly the one required JSON field.
    weight: 1

guardrails:
  maxPriceIn: 0.50
  maxPriceOut: 3.00
  minContext: 32000
  requireJson: true
  allowFree: true
  tiers: [free, paid-low, paid-mid]

judgeModel: openai/gpt-4o
maxTokens: 700

notify:
  email: null          # or set MODELCICD_NOTIFY_EMAIL in the environment
  minImprovement: 0.20
```

Each use case's state (`state/<name>.json`) and every bench run's full
results (`out/<timestamp>_<name>.json`) are tracked independently — running
or approving one use case never touches another. So a real application ends
up with several use cases, each on its own bench-and-approve cadence:

```
use_cases/
  support_bot_reply.yaml
  ticket_summarizer.yaml
  codegen_assistant.yaml
```

```python
from modelcicd.resolver import resolve

reply_model   = resolve("support_bot_reply",   fallback="gpt-4o-mini")
summary_model = resolve("ticket_summarizer",   fallback="gpt-4o-mini")
codegen_model = resolve("codegen_assistant",   fallback="claude-sonnet")
```

---

## Configuration

Set these in `.env` (see `.env.example`):

| Variable | Required | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | yes, for `run` | Every candidate and the judge are called through OpenRouter |
| `MODELCICD_NOTIFY_EMAIL` | no | Default recipient for pending-candidate emails, if not set per use case |
| `SMTP_HOST` / `SMTP_PORT` / `SMTP_USER` / `SMTP_PASSWORD` / `SMTP_FROM` / `SMTP_STARTTLS` | no | If unset, the CLI prints the approve command instead of emailing it |

API keys never belong in a `use_case.yaml` — that file is meant to be
committed to your repo, and a config with a key in it is a leaked key.

---

## Project layout

```
modelcicd/
  catalogue.py    fetch + normalize the model catalogue (free)
  guardrails.py   filter candidates by price/context/JSON/modality (free)
  config.py       load and validate a use_case.yaml
  client.py       the one place this project calls a model (JSON-mode, retried)
  sandbox.py      run one candidate against a use case's test cases
  judge.py        blind, pinned-judge scoring against the use case's rubric
  bench.py        orchestrates sandbox + judge across all candidates
  rank.py         builds the per-tier leaderboard
  state.py        what's approved per use case, and its history
  resolver.py     resolve(use_case) -> the approved model id (the integration point)
  notify.py       emails a human when a candidate is pending
  cli.py          the `python -m modelcicd.cli ...` entrypoint
  tests/          a dependency-free test module: python -m modelcicd.tests.test_modelcicd
examples/
  prep_material/use_case.yaml   a worked example use case
```

## Testing

```bash
python -m modelcicd.tests.test_modelcicd
```

No `pytest` dependency — a small, self-contained runner that checks the
specific edge cases the design relies on (negative-price sentinels, the
judge never being a candidate, tier ranking, stale-state bugs, and so on).

---

## Design notes worth knowing before you extend this

- **Unknown is not free.** A model with no published price is excluded by
  the cost guardrail, never assumed to cost $0.
- **A logical model keeps its hosting options rather than merging them.**
  The same weights on two platforms can have different quantization,
  context windows, or JSON-mode support — collapsing them into one row
  throws away the thing worth measuring.
- **Ranking is always per price tier**, never global — otherwise the
  expensive model always "wins."
- **One sample per test case** is deliberate: a screening pass over forty
  candidates costs 40 calls this way and 120+ if repeated for statistical
  confidence. Deepen sampling on the shortlist a candidate makes it to,
  never on the whole field.
- **`approve()` is the only function that changes what `resolve()`
  returns.** Every other step in the pipeline only ever proposes.
