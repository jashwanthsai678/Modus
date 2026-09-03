"""Model CICD — continuous model discovery, sandboxing and human-approved
promotion for one specific place an LLM is called in your application.

    from modelcicd.resolver import resolve
    model = resolve("your_use_case", fallback="gpt-4o-mini")

The rest of this package — catalogue, guardrails, sandbox, judge, rank, state,
notify — is the machinery that decides what `resolve()` returns, run on
whatever schedule you point at `modelcicd.cli run`. See README.md.
"""
