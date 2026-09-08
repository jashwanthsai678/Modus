# Design notes

Why Model CICD works the way it does. This is the reasoning behind the
decisions — read it before extending the project, or if you want to
understand why something that looks like an obvious improvement was
deliberately not done.

For usage, see the [README](../README.md). For feature-by-feature reference,
see [GUIDE.md](GUIDE.md).

---

## Who needs it

Any developer or team whose application calls an LLM in one or more places,
and who has ever had to answer (or been unable to answer) questions like:
"why are we using this particular model here," "are we overpaying for it,"
"would a newer or cheaper model do just as well," or "who approved this and
when." If your app has exactly one throwaway LLM call that nobody would
notice if it got slightly worse or slightly more expensive, you probably
don't need this yet. The moment even one LLM call matters enough that its
cost or quality is worth tracking, this is for you.


---

## Why they need it

Because a model choice, left alone, decays:

- **New, cheaper, or better models ship every few weeks.** Without a
  standing process to re-check, you either keep overpaying for a model a
  newer one now beats, or never find out a cheaper option would do the same
  job.
- **A quality regression from a model swap is invisible until a user hits
  it.** Nobody manually re-tests forty candidates by hand every time a new
  model drops — so most teams simply don't, and find out something got worse
  from a support ticket instead of a benchmark.
- **"Which model, exactly?" is usually answered from memory**, not
  evidence — no record of what was tried, what it scored, or why it was
  chosen, so the decision can't be revisited or defended later.
- **A silent model swap in production is a real risk**, on par with an
  unreviewed dependency upgrade — it should never happen without a human
  deciding it should.


---

## What value it adds

- **Lower cost without guessing.** Every candidate is filtered by your
  price ceiling before anything is spent, and ranked within its own price
  tier, so "cheaper" is a fact you can see, not a hope.
- **Quality protected, not just cost.** Nothing gets proposed as a
  replacement without first being scored, blind, against the rubric that
  feature's own owner wrote — a model can't win purely by being cheap.
- **A record that survives memory.** Every run, every candidate, every
  score, and every approval is saved to disk with a timestamp — "why this
  model" is always answerable later, by anyone, not just whoever remembers.
- **A single, low-friction decision point.** A human sees the evidence and
  clicks Approve (CLI or dashboard) — nothing changes silently, and nothing
  requires re-deriving a process from scratch each time.
- **This, repeated, per feature.** The same loop runs independently for
  every place your app calls an LLM, so a five-feature application ends up
  with five independently tracked, independently approved model choices,
  not one blunt global setting.

---


---

## The design constraints behind it

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


---

## Rate-limit visibility — observed, never guessed

Free-tier models (OpenRouter's especially) hit shared rate limits fast in
real use. There's no public catalogue of per-model rate limits to check
that against, so this isn't another guardrail filter — it's evidence,
gathered the same way the score is: by actually calling the candidate.

```yaml
usage:
  callsPerDay: 2000   # optional, per AI feature — the wizard asks for this
```

If a candidate returns 429 (Too Many Requests) during the bench — even a
light one, a handful of test cases — that's recorded as its own outcome,
distinct from a generic failure, and shown on the leaderboard next to your
usage estimate if you gave one:

```
-  vendor/free-model   could not be scored — rate limited during this bench
                        (3 call(s)) — a light test already hit it; your
                        ~2,000/day estimate is unlikely to be sustainable here
```

The candidate still appears on the leaderboard either way — this never
disqualifies anything automatically. It's a warning attached to real
evidence, for the human who's about to approve something.

---


---

## Judge noise, not just candidate noise

Every score here comes from ONE judged sample — cheap by design, but that
means a "tie" between two close candidates could be candidate noise (they're
genuinely similar) or **judge noise** (the judge itself wasn't consistent
about it), and a single sample can't tell those apart. Left alone, a tie
band that only protects against the first kind can make a shaky ranking look
confident for the wrong reason.

So a `run` deepens on exactly the candidates that made a tie-band shortlist:
their already-generated answers (no candidate is re-run) get re-scored by
the judge two more times, and the spread across those looks is shown right
next to the score:

```
*  1  vendor/model-b   4.60  (judge spread ±0.70)
   2  vendor/model-a   4.25
```

A small spread means the judge agreed with itself; a wide one means this
"tie" is judge noise, not a real result — worth reading before approving a
close call. A candidate outside the tie zone was never in question, so it's
never re-judged — same principle as everywhere else here: deepen on the
shortlist, never the whole field.

---


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

Grouped under a project instead, the same independence holds one level up —
one connected application, several use cases, each still benched and
approved on its own:

```
projects/support-portal/
  project.yaml
  use_cases/
    support_bot_reply.yaml
    ticket_summarizer.yaml
  state/
    support_bot_reply.json
    ticket_summarizer.json
  out/
    20260310T090000Z_support_bot_reply.json
```

```python
from modelcicd.resolver import resolve

reply_model   = resolve("support_bot_reply",   fallback="gpt-4o-mini")
summary_model = resolve("ticket_summarizer",   fallback="gpt-4o-mini")
codegen_model = resolve("codegen_assistant",   fallback="claude-sonnet")
```

---


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
  returns.** Every other step in the pipeline only ever proposes — including
  a scheduled run finding a candidate on its own.
- **A live endpoint captures a baseline; it never runs a candidate.** The
  leaderboard's "current (your endpoint)" row exists so a human compares a
  candidate against what they actually have, not just against other
  candidates — but nothing routes a candidate model through someone else's
  application.
- **The CLI and the dashboard write through the same functions, always.**
  `wizard.to_yaml`, `project.create`, and `state.approve` are each called
  from exactly one place in both `cli.py` and `dashboard.py` — never
  reimplemented for the browser — so the two can never quietly drift apart.
- **Approving a model and patching your source are two separate decisions.**
  `state.approve()` never touches a filesystem outside `state/`; a
  `codeTarget` patch always gets its own preview and its own confirmation
  (`apply-code-patch`, or its own button on the use case page) — writing into
  someone else's real application is a bigger, harder-to-reverse action than
  writing a JSON file, and it is never bundled into a click meant to do the
  smaller one.
- **A scan proposes; it never writes.** `scan-repo` reads a connected repo
  with a model to find likely LLM call sites, but the result is only ever a
  pre-fill — a person still confirms (and can edit) the prompt and test
  cases through the same wizard every other AI feature goes through. Like
  `run`, it shows its real cost (one model call per candidate file) before
  spending anything.
- **A provider's price/JSON-mode support is only ever what its own API
  publishes.** Groq's and Fireworks' `/models` endpoints don't guarantee
  either — those fields come back unknown rather than fabricated, and the
  existing "unknown is not free" guardrail excludes such models the same
  way it excludes any other unpriced one. No provider gets a hardcoded price
  table that could go stale or wrong.
- **Rate-limit risk is observed, never fabricated.** There's no public
  catalogue of per-model rate limits the way there is for price, so this
  isn't a guardrail — a 429 during the bench is real evidence, recorded as
  its own outcome and shown next to the use case's own usage estimate. It
  never disqualifies a candidate automatically; it's context for the human
  who's about to approve one.
- **`reject()` is not `approve()`.** It's the only other function allowed to
  clear `pending`, and it's structurally incapable of touching
  `approvedModel` — a dismissal is recorded (`state["rejected"]`), never
  silently discarded, and never promotes anything.
- **A scheduled run notifies once per candidate, not once per check.**
  `state.notifiedModel` tracks what you were last emailed about; the same
  still-pending candidate doesn't re-trigger the email on every scheduled
  run, only a genuinely different candidate does.
- **Judge noise gets the same "deepen only the shortlist" treatment as
  everything else.** Re-scoring is never applied to the whole field — only
  to candidates that already made a tie-band shortlist, and only against
  answers already generated, so it costs extra judge calls and zero extra
  candidate calls.
- **`client.py` stays standalone; it never learns what a "provider" is.**
  It's meant to be dropped into another repo on its own, so it can't import
  `catalogue.py`. Multi-platform routing is `sandbox.py`'s job — it resolves
  a candidate's endpoint and key env var and passes them in as plain
  parameters, which default to exactly OpenRouter, exactly as `call_json`
  always behaved before providers existed.
- **A key is never displayed back once saved.** `secrets.status()` reports
  whether one is set, never its value — the connect form and `/keys` show a
  configured/not-set badge, not the key itself.

