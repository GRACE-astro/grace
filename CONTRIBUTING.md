# Contributing to GRACE

Thank you for your interest in contributing to GRACE.

GRACE is maintained by a single principal author who reviews all contributions
and sets project direction. Pull requests are reviewed on a best-effort basis
and may be declined if they don't fit the project's direction or quality bar —
opening an issue first to discuss a non-trivial change is the most efficient
path for both sides.

The full contribution guide lives in
[`doc/contributing/index.rst`](doc/contributing/index.rst). Below is the
short version.

## Workflow

1. **Fork** the repository to your own GitHub account.
2. **Branch** off `main` in your fork, named descriptively:
   ```
   git checkout -b feature/add-xdmf-support
   ```
3. **Commit** your changes with concise, descriptive messages
   (`feat:` / `fix:` / `doc:` / `test:` / `refactor:` prefixes preferred).
4. **Push** the branch to your fork and **open a pull request** against
   `GRACE-astro/grace:main`. Include a clear title, a short description
   of what changes and why, and links to any related issues.
5. **Respond to review comments**; CI must pass and the maintainer must
   approve before the PR is merged.

## Code review

All pull requests are reviewed and merged by the project maintainer.
Reviews focus on:

- correctness and numerical behavior;
- consistency with existing abstractions and naming;
- adequate test coverage for non-trivial logic;
- documentation updates for user-facing changes.

PRs that bundle many unrelated changes are likely to be asked to split. Keep
PRs focused and small.

## Coding standards

- C++: Kokkos conventions; existing surrounding style (indentation, naming);
  Doxygen-style comments on public API.
- Python: PEP 8; existing style in `tools/` and the codegen pipeline.
- New files must carry the GRACE GPL header and an `@author` line.
- New configuration parameters must be added to the appropriate
  `parameters/*.yaml` schema with `type`, `range`, `default`, and a clear
  `description`.

## Documentation

User-visible changes require documentation updates under `doc/`. Examples
should be minimal and reproducible.

## Tests and CI

Run the local test suite before opening a PR:

```
cmake --build build
ctest --test-dir build/test -L fast --output-on-failure
```

CI runs the `fast` lane on every push and PR. Heavier MPI / conservation /
boundary-condition test lanes exist locally and should be run before
opening PRs that touch the relevant subsystems.

## Licensing and attribution

All contributions are accepted under the project's license
(GPL-3.0-or-later). By opening a pull request you confirm that you have
the right to contribute the code under this license. If your contribution
references or adapts external code, cite the source in a comment and
ensure license compatibility.

## Issues

Use GitHub Issues for bug reports and feature requests. For bugs,
include: the reduced configuration, the GRACE git SHA, the compiler /
backend, and the failure mode (stack trace, output, expected vs.
observed).
