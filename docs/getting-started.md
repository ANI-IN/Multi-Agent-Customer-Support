# Getting Started

This guide walks you from a fresh clone to a running chat session in either a Python virtualenv or a Docker container.

## Prerequisites

- **Python 3.12.x.** Python 3.13 is currently blocked because some transitive dependencies still import `audioop`, which was removed in 3.13.
- **Git.**
- **An OpenAI-compatible LLM endpoint.** Any of the following work:
  - OpenAI (`https://api.openai.com/v1`)
  - Groq (`https://api.groq.com/openai/v1`)
  - Together AI (`https://api.together.xyz/v1`)
  - Azure OpenAI (`https://<your-resource>.openai.azure.com/`)
  - A local server such as LM Studio, Ollama, or vLLM exposing the OpenAI protocol.
- **Docker** (optional, only for the container path).

## Option A: Python virtualenv

```bash
git clone https://github.com/ANI-IN/Multi-Agent-Customer-Support.git
cd Multi-Agent-Customer-Support

python3.12 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt

cp .env.example .env
# Open .env and set OPENAI_API_KEY to your real key.

python app.py
```

Then open http://localhost:7860 in a browser.

The first launch downloads the Chinook sample SQL script (~1.5 MB) from GitHub and caches it as `Chinook_Sqlite.sql` at the repo root. Subsequent launches read the cache.

## Option B: Docker

```bash
git clone https://github.com/ANI-IN/Multi-Agent-Customer-Support.git
cd Multi-Agent-Customer-Support

docker build -t music-support .

docker run --rm -p 7860:7860 \
  -e OPENAI_API_KEY=sk-... \
  -e MODEL_NAME=gpt-4o-mini \
  music-support
```

The Gradio app binds to `0.0.0.0:7860` inside the container and the published port `7860` on the host. Open http://localhost:7860.

## Option C: Docker Compose

```bash
cp .env.example .env
# Edit .env, set OPENAI_API_KEY.

docker compose up --build
```

The compose service reads `.env` directly, restarts on failure, and exposes a basic HTTP healthcheck against the Gradio root.

## Configure a Non-OpenAI Provider

Set both `OPENAI_API_BASE` and `OPENAI_API_KEY` in your environment. Examples:

### Groq

```env
OPENAI_API_BASE=https://api.groq.com/openai/v1
OPENAI_API_KEY=gsk_...
MODEL_NAME=llama-3.3-70b-versatile
```

### Together AI

```env
OPENAI_API_BASE=https://api.together.xyz/v1
OPENAI_API_KEY=...
MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct-Turbo
```

### Local server (LM Studio, Ollama, vLLM)

```env
OPENAI_API_BASE=http://localhost:1234/v1
OPENAI_API_KEY=not-needed
MODEL_NAME=your-local-model
```

The supervisor routing relies on the model following structured instructions. Models below ~7B parameters often degrade routing accuracy on mixed queries.

## First Conversation

The agent verifies your identity before exposing any account data. Try this sequence:

```
> My customer ID is 5
> What AC/DC albums do you have?
> What was my most recent purchase?
> I love rock music.
```

The last line is captured as a music preference and persisted across the session via the in-memory long-term store. If you restart the app the preference is lost (the store is in memory by design; see the roadmap in [docs/architecture.md](architecture.md)).

## Running the Tests

```bash
pytest tests/ -v
```

The suite exercises the SQL helpers, identifier normalization, and every tool function against the in-memory Chinook database. It does not call any LLM, so no API key is needed.

## Where to Go Next

- Read [docs/architecture.md](architecture.md) for a system tour with flowcharts, a sequence diagram, and a "what lives where" table.
- Read [SECURITY.md](../SECURITY.md) for the disclosure process and the project's known sensitive surfaces.
- Read [CONTRIBUTING.md](../CONTRIBUTING.md) when you are ready to open a pull request.
