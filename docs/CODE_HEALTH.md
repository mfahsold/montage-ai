# Code Health & Dead Code Detection

This document describes the repository's lightweight dead-code detection approach.

- We use `vulture` to scan `src/` for potentially unused code.
- Findings must be triaged manually: some items may be false positives (dynamic usage, optional imports, plugin hooks).

Usage:

- Run locally: `make code-health` (non-blocking; prints results)
- Address high-confidence findings by removing clearly unused imports/vars or by adding explanatory comments/guarding imports.

This PR removes obvious unused imports and parameters and adds a `make code-health` target.

Related: issue #12