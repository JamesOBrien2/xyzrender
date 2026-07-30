# xyzrender project memory

## GitHub workflow

- Upstream repository: `aligfellow/xyzrender`; pull requests target `main`.
- The authenticated account is `JamesOBrien2`, which does not have direct push access to upstream. Push contribution branches to the existing fork `JamesOBrien2/xyzrender`, then open cross-repository PRs against upstream.
- Use `bugfix/<description>` for fixes and `feature/<description>` for features; never create `agent/` branches. The existing TS-bond branch is the explicitly requested exception `feat/ts-bond-gif`.
- Keep commit messages and PR bodies concise, general, and human-sounding. PR bodies should briefly explain what changed and why, without test/validation details unless requested.

## Current GitHub work

- Draft PR #164 (`aligfellow/xyzrender`) is from `JamesOBrien2:feat/ts-bond-gif` and adds manual TS-bond selection for vibration GIFs.
- Issue #163 is fixed locally on `bugfix/issue-163-gif-rot-bool` at commit `1648ec0`; it has not been pushed.
- Issue #162 is fixed locally on `bugfix/issue-162-notebook-svg-ids` at commit `503eeb5`; it has not been pushed.
