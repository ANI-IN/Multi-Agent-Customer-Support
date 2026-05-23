# Contributing

Thanks for your interest in improving the Music Store Multi-Agent Support project. This guide explains how to set up a local development environment, the conventions the codebase follows, and what makes a good pull request.

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating you agree to uphold it.

## Reporting Bugs and Requesting Features

- Use the GitHub issue templates under `.github/ISSUE_TEMPLATE/`.
- For bugs, include the exact reproduction steps, the model and provider you used, the relevant log lines (with secrets redacted), and your Python version.
- For security issues, do not open a public issue. See [SECURITY.md](SECURITY.md).

## Local Setup

### Option A: virtualenv

```bash
git clone https://github.com/ANI-IN/Multi-Agent-Customer-Support.git
cd Multi-Agent-Customer-Support
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env and set OPENAI_API_KEY
python app.py
```

Open http://localhost:7860.

### Option B: Docker

```bash
docker build -t music-support .
docker run --rm -p 7860:7860 -e OPENAI_API_KEY=$OPENAI_API_KEY music-support
```

## Running Tests

```bash
pytest tests/ -v
```

The full suite (28 tests) runs against the in-memory Chinook database in under a second and requires no API key.

## Code Style

- Python 3.12 is the supported runtime. Code targets Python 3.12 in `pyproject.toml`.
- Formatting is enforced by `ruff format` (Black-compatible) at a line length of 120.
- Linting is enforced by `ruff` with the rule set defined in `pyproject.toml`.
- Type hints are used at module boundaries. The project does not enforce a strict type checker today; `mypy` may be added later.

Install the developer toolchain locally:

```bash
pip install ruff black pytest pre-commit
pre-commit install
```

Then every commit runs the trailing-whitespace, end-of-file, YAML/TOML/JSON validity, large-file, merge-conflict, line-ending, `ruff`, `ruff-format`, and `gitleaks` hooks.

## Pull Request Conventions

1. Open a draft PR against `main` as soon as you have a meaningful diff. CI must be green before review.
2. Use [Conventional Commits](https://www.conventionalcommits.org/) in the commit subject (`feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, `test:`, `ci:`, `build:`, `perf:`).
3. Keep PRs focused. One logical change per PR is easier to review than a kitchen sink.
4. Update or add tests for behavior changes. New tools require new tests in `tests/test_tools.py` patterns.
5. Update `CHANGELOG.md` under the `[Unreleased]` heading when the change is user-visible.

## What Is In Scope

- Agent graph orchestration: new sub-agents, conditional routing, interrupt patterns.
- Tool functions over the existing Chinook schema (or a new schema if added explicitly).
- Prompt engineering changes that improve grounding, refusal accuracy, or memory hygiene.
- UI ergonomics: clearer status states, better error surfacing, accessibility improvements.
- Documentation, examples, and reproducibility.

## What Is Out of Scope (For Now)

- Replacing LangGraph with a different orchestration framework.
- Swapping Gradio for a different UI without first opening a design discussion.
- Adding paid third-party services as hard dependencies.
- Breaking changes to the public `State` schema without a deprecation path.

## Architecture Pointers

Read these in order to understand the codebase:

1. `src/state.py` for the shared state contract.
2. `src/agents/graph.py` for how the multi-agent graph is assembled.
3. `src/agents/nodes.py` for what each graph node does.
4. `src/agents/prompts.py` for the LLM behavior contracts.
5. `src/tools/music_catalog.py` and `src/tools/invoice.py` for the data surface.
6. `src/ui/app.py` for how streaming and interrupts are surfaced to the user.

A longer walkthrough lives in [docs/architecture.md](docs/architecture.md).

## Releasing

The project is currently single-maintainer and unversioned. When the first tagged release happens, it will follow [Semantic Versioning](https://semver.org/) and the changelog format already seeded in `CHANGELOG.md`.

## Questions

Open a discussion on GitHub or file an issue tagged `question`.
