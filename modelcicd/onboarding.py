"""The guided quickstart — ONE copy, shown by `cli.py onboarding` and the
dashboard's home page. Written once here so the CLI and the browser can
never show two different command lists that drift apart from each other.
"""

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5000


def text(*, dashboard_host: str = DEFAULT_HOST, dashboard_port: int = DEFAULT_PORT) -> str:
    return f"""MODEL CICD — quickstart

  1. Connect an application (a project — just a name, nothing is scanned).
     Add --repo-path (or --repo-url to have it cloned) if your app is
     already local and you want the wizard to reference its code, scan it,
     and eventually patch a hardcoded model. --providers picks which
     marketplace(s) to search (default: OpenRouter alone):
       python -m modelcicd.cli project create --name "My App" \\
         --repo-path D:\\repos\\my-app --providers openrouter,groq

  2. Optional — read the connected repo with a model to find likely LLM
     call sites, instead of typing the system prompt in by hand:
       python -m modelcicd.cli scan-repo --project my-app
       python -m modelcicd.cli wizard --project my-app --from-scan 0

  3. Or define an AI feature directly — a guided flow either way, no YAML
     to hand-write:
       python -m modelcicd.cli wizard --project my-app

  4. See what candidates survive its price/quality guardrails (free):
       python -m modelcicd.cli shortlist --use-case projects/my-app/use_cases/<name>.yaml --project my-app

  5. Bench it — the only step that spends money, and it asks first:
       python -m modelcicd.cli run --use-case projects/my-app/use_cases/<name>.yaml --project my-app

  6. See what's approved vs. pending, and approve a winner — or dismiss a
     candidate you looked at and don't want (never changes resolve(), just
     clears the suggestion; it can resurface if a future run finds it again):
       python -m modelcicd.cli status  --use-case-name <name> --project my-app
       python -m modelcicd.cli approve --use-case-name <name> --project my-app
       python -m modelcicd.cli reject  --use-case-name <name> --project my-app

  7. Everything waiting for a decision, across every project at once:
       python -m modelcicd.cli pending

  8. If the feature has a code target (a hardcoded model in your connected
     repo), write the approved model into it — previewed, confirmed:
       python -m modelcicd.cli apply-code-patch --use-case-name <name> --project my-app

  9. Or do all of the above from the browser instead:
       python -m modelcicd.cli ui                        -> http://{dashboard_host}:{dashboard_port}

  10. Let it re-check on its own (only if the AI feature has a `schedule:`
      block — set from the wizard, or by hand in its YAML). A scheduled run
      only re-notifies you when the pending candidate actually CHANGES, not
      every time it checks:
       python -m modelcicd.cli scheduler serve            # leave this running
       python -m modelcicd.cli scheduler run-due          # or trigger it from cron / Task Scheduler

A model only ever reaches production when a human clicks Approve — everything
above step 6 only ever proposes. Step 8 is a further, separate confirmation —
approving never edits your source on its own. Steps 2 and 5 are the only ones
that spend money, and each shows the cost and asks before it does.
"""
