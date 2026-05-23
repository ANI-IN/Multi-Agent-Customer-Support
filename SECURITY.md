# Security Policy

## Reporting a Vulnerability

If you discover a security issue in this project, please **do not open a public GitHub issue**. Instead, report it privately so it can be triaged and fixed before details are made public.

Preferred channel:

- Open a [private security advisory](https://github.com/ANI-IN/Multi-Agent-Customer-Support/security/advisories/new) on GitHub.

Fallback channel:

- Contact the maintainer (Animesh Kumar) via the contact details on the GitHub profile linked from this repository.

When reporting, please include:

- A clear description of the issue.
- Steps to reproduce, including the model and provider you used.
- The potential impact, in your own words.
- Any suggested fix, if you have one.

You should receive an acknowledgment within seven calendar days. Please give the maintainer reasonable time to investigate and ship a fix before disclosing publicly.

## Supported Versions

The project does not currently publish tagged releases. Security fixes target the `main` branch. When versioned releases begin, the latest minor release will receive security fixes; older releases will be supported on a best-effort basis.

## Scope

In scope:

- The Python code under `src/` and `app.py`.
- The Dockerfile and any deployment manifests in this repository.
- The prompts in `src/agents/prompts.py` insofar as they affect grounding, refusal, or data isolation.

Out of scope:

- Vulnerabilities in third-party services the project integrates with (OpenAI, Hugging Face Spaces, Gradio, LangGraph upstream). Report those to the respective vendors.
- Issues in the Chinook sample dataset itself.
- Self-inflicted misconfiguration (for example, committing a real `.env` file).

## Known Risk Areas

These areas of the code are sensitive and worth careful review for any contributor or auditor.

### 1. Identity verification (`src/agents/nodes.py`)

`verify_info` and `get_customer_id_from_identifier` are the only writers of `customer_id`. Any change that lets a sub-agent or tool set `customer_id` directly would break account isolation. Treat the `customer_id` field as write-once per session by `verify_info` only.

### 2. SQL parameterization (`src/db/database.py`, `src/tools/`)

Every query uses SQLAlchemy `text()` with bound `:name` parameters. Any new tool or query must follow the same pattern. **Never** build a query with f-strings, `+` concatenation, or `.format()` over user-supplied input. The `_safe_int` helper in each tool file guards numeric arguments and should be reused for any new integer parameter.

### 3. Memory writes (`src/agents/nodes.py::create_memory`)

The memory node performs a set union and is documented to never delete. If you change the merge semantics, make sure the empty-result-with-existing-memory case still skips the write. Otherwise an LLM hiccup could erase a user's stored preferences.

### 4. Prompt injection via user messages

The verification path passes user messages to a structured-output LLM call. The structured schema (`UserInput`) restricts what can be extracted to one identifier string, which limits blast radius. Any change that lets the model emit free-form fields back into application state is a security review trigger.

### 5. Outbound network on first run

`src/db/database.py::_load_sql_script` fetches the Chinook SQL from a public GitHub URL on first run if the cached file is missing. The URL is hard-coded and HTTPS, but if you change it, make sure the new source is trustworthy and the download path is still validated.

## Secret Handling

- Real credentials must never be committed. `.gitignore` excludes `.env`, and `.env.example` carries placeholder values only.
- The repository ships no real keys. If you find any committed secret, open a private advisory and rotate the secret immediately.
- Pre-commit runs `gitleaks` (see `.pre-commit-config.yaml`) to catch accidental commits.
- CI runs the same secret scan on every push and pull request.
