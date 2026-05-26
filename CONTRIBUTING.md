# Contributing to ``desilike``

Thanks for considering to contribute to ``desilike``. We welcome contributions from all members of the DESI collaboration. Below, we outline the general workflow and coding guidelines.

## Prerequisites

To contribute to ``desilike``, you need a GitHub account and be a member of the [cosmodesi GitHub group](https://github.com/cosmodesi). If you are not already a member of the group, you can reach out to [Johannes U. Lange](mailto:jlange@american.edu) to receive an invitation.

## Workflow

Code contributions to ``desilike`` are submitted via pull requests on GitHub. You do not need to fork the ``cosmodesi/desilike`` repository. Instead, you can create a new branch directly on ``cosmodesi/desilike`` where you place your contributions. Once you are satisfied with your changes, open a pull request against the ``main`` branch. When submitting the pull request, you will be asked a few quick questions about the motivation for the changes, code provenance, and AI use. A package maintainer will then review your proposed changes and, if all looks good, merge your contributions.

## Guidelines

As an open science package, we want to ensure that ``desilike`` is well-tested, documented, reliable, and easy to modify. Thus, we ask that code contributions meet the following basic guidelines. All commands assume that you are in the main ``desilike`` directory.

### Installation

Please check that your modified version of ``desilike`` installs correctly via:

```
pip install -e .
```

In case your modifications require updated dependencies, please list those in the ``pyproject.toml`` file.

### Unit Tests

We use ``pytest`` and GitHub Actions for continuous integration. Once you create a pull request, GitHub Actions will automatically trigger unit tests. Those are required to pass before your contributions can be accepted. To perform the unit tests locally, run:

```
pytest tests
```

If your code contributions add new features, we also encourage you to add unit tests under ``tests`` that verify the new functionality works as expected.

### Docstrings

Please ensure that all public functions you contribute have valid [NumPy-style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html).

### Syntax

Please ensure that the modified code reasonably complies with the PEP8 syntax standard. GitHub Actions uses ``ruff`` to test for this and your code contributions can only be accepted after they comply with PEP8. To run the syntax check locally, run:

```
ruff check desilike --ignore E701,E721,F401,E711,F811,E402,F841,E714,E731,F541,F405,F403,E722,F523,E741,F601,E401,E713,F524
```

Note the long list of ignored errors and warnings. Ideally, your code contributions do not need those exceptions and pass a regular ``ruff check``.

### Generative AI

If you used generative AI to help you draft your code contributions, you must disclose that in the pull request. Additionally, if new code was substantially written by AI, you must describe the steps you have taken to ensure that the code works correctly.

Generally, we discourage community members from contributing code that they could not have written without AI since, in this case, they may not be able to verify its accuracy. On the other hand, using AI to check existing code or to copyedit documentation can be a good idea.
