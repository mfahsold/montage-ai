# Code Health Guidelines

This document explains the process for running a dead-code detector (vulture) and how to safely triage and remove unused code.

## Running the scanner

Run locally with:

```bash
make code-health
```

## Triage steps

1. Run `vulture` with a moderate confidence threshold (e.g. 50%).
2. Manually inspect each finding; consider whether code is used dynamically (plugins, optional dependencies).
3. If a finding is safe to remove, open a small, focused PR with a concise rationale and a local CI run (`make ci-local`), and reference the triage issue (e.g., #12).

## CI

We run a non-blocking vulture smoke test in CI and maintain a triage issue to track findings.