.. _grace-contribution-guidelines:

Contribution Guidelines
==========================================

Thank you for your interest in contributing to GRACE! 

Workflow / Branching
------------------------------

- **Protected main branch**: Only the project owner can push directly to `main`. All contributions must go through pull requests (PRs).
- **Feature branches**: Create a branch off `main` for new features, bug fixes, or documentation changes:

  .. code-block:: bash

      git checkout -b feature/add-xdmf-support

- **Pull Requests**: Open a PR targeting `main` when your changes are ready. Include a clear title and description.

Code Review
-----------

- Every PR should be reviewed by at least one collaborator with admin status
- Keep PRs focused and small; it's easier to review and test.
- Address review comments before merging.

Coding Standards
----------------

- Follow existing code style for C++/Python (Kokkos conventions for C++, PEP8 for Python).
- Maintain consistent indentation, naming conventions, and documentation comments.
- Include unit or integration tests for new functionality whenever possible.

Documentation
-------------

- Update the documentation whenever you add new features or modify APIs.
- Follow the project's style for Sphinx/reST.
- Include minimal, reproducible examples for new functions or classes.

Issue Tracking
--------------

- Open issues for bugs, feature requests, or enhancements.
- Reference issues in PRs (e.g., ``Closes #42``) to automatically close them upon merge.

Commit Messages
--------------------------

- Use concise and descriptive commit messages.
- Example style:

  .. code-block::

      feat: add XDMF export support
      fix: handle ghost zone edge cases
      doc: update Python library example

- Consider squashing minor commits in a PR for a cleaner history.

Testing & Continuous Integration
------------------------------------------

- Run all unit tests and code formatting checks locally before opening a PR.
- PRs should pass CI checks (if available) before merging.

Licensing & Attribution
---------------------------------

- All contributions are under the project's license.
- Include references if using external code snippets or algorithms.

Optional Notes
-------------------------

- For trivial typos or documentation fixes, a PR may be optional.
- Even if you're unsure about a change, open a PR — feedback is welcome.

