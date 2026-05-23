# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) for any future tagged releases.

## [Unreleased]

### Added

- `LICENSE` (MIT) at the repository root.
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, and `SECURITY.md` for community and security guidance.
- `pyproject.toml` with `ruff`, `black`, and `pytest` configuration.
- `.pre-commit-config.yaml` chaining trailing-whitespace, end-of-file, YAML/TOML/JSON validation, ruff, ruff-format, and gitleaks secret scanning.
- `.editorconfig` for cross-editor formatting consistency.
- GitHub Actions CI workflow at `.github/workflows/ci.yml` running lint, format check, and the pytest suite.
- Issue and pull request templates under `.github/ISSUE_TEMPLATE/` and `.github/PULL_REQUEST_TEMPLATE.md`.
- `docs/architecture.md` with a system flowchart, sequence diagram, "what lives where" table, trust boundaries, and architectural invariants.
- `docs/getting-started.md` with the local virtualenv and Docker setup paths.
- `docker-compose.yml` for one-command local container runs.
- A rewritten `README.md` covering all 24 sections of the project README spec.

### Changed

- The previous `README.md` has been replaced by a fully expanded version. Git history retains the original.

### Notes

The application code under `src/`, the test suite under `tests/`, and the existing `Dockerfile`, `.dockerignore`, `.gitignore`, and `.env.example` were not modified.
