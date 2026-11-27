# AskHR

[![CI](https://github.com/Odilove5/askhr/actions/workflows/ci.yml/badge.svg)](https://github.com/Odilove5/askhr/actions/workflows/ci.yml)

**AskHR** is a bilingual (French/English) HR assistant for school boards. Staff ask questions in plain language ("Mon enfant a un rendez-vous chez le médecin mardi. Est-ce que je peux m'absenter ?"), and AskHR answers from the board's own HR documents (policies, administrative directives, collective agreements, benefits guides), **cites its sources**, and **hands off to the right HR contact** when the documents don't answer the question.

It is built as an agentic, corrective RAG system with contextual retrieval, using LangChain (LCEL) and LangGraph, with local models through Ollama by default.

> **About the documents.** Development uses the HR policies and administrative directives that Conseil scolaire Viamonde publishes on its public website. They are copyrighted, so this repository commits only their titles and links (`corpus/manifest.yaml`). The downloader fetches the PDFs onto your own machine, into `data/`, which is never committed. Tests use made-up text. No employee or personal data is used anywhere.

## Status

🚧 Early development: **M0 complete, M1 in progress.**

| Milestone | Goal | Target |
|---|---|---|
| M0 | Project setup: repo, uv, linting, tests, CI | Nov 30, 2026 ✅ |
| M1 | HR corpus (public Viamonde documents) and ingestion pipeline | Dec 7, 2026 🚧 |
| M2 | Hybrid retrieval (Postgres + pgvector) and relevance grader | Dec 18, 2026 |
| M3 | LangGraph agent: routing, query rewriting, HR handoff | Jan 8, 2027 |
| M4 | Cited, bilingual answers with quote checking | Jan 18, 2027 |
| M5 | Evals (Ragas / LangSmith) and guardrails | Jan 29, 2027 |
| M6 | FastAPI app, chat UI, Docker, deployed in Canada | Feb 12, 2027 |
| M7 | Portfolio: architecture diagram, demo video, lessons learned | Feb 21, 2027 |

M1 so far: website connector, downloader, data model, PDF loader, chunker, settings, model factory, the contextual-notes chain, and a one-command ingest (`askhr.ingest`) with a resumable notes cache. Next: the M1 review and tag v0.2.0.

## How it works (so far)

```
Viamonde website ──connector──▶ corpus/manifest.yaml ──downloader──▶ data/raw/*.pdf
        ──loader──▶ Document (clean pages) ──chunker──▶ Chunks
        ──contextual notes (LCEL: prompt | Ollama | parser)──▶ Chunks with a short note each
```

Each chunk gets a one- or two-sentence French note that says where it sits in its document and adds a few search keywords. At M2, the note is indexed with the chunk so a passage that only says "elle avise son superviseur" can still be found by a search about medical leave.

## Tech stack

- **Python 3.12**, managed with [uv](https://docs.astral.sh/uv/)
- **LangChain** (LCEL chains) now, **LangGraph** for the agent from M3
- **Ollama** for local models (default: Gemma 4 12B); **Claude API** optional for hard questions, **off by default**
- **pydantic-settings** for configuration, **pypdf** for PDFs, **httpx** and **Beautiful Soup** for the website connector
- **ruff** (lint and format), **mypy** (strict), **pytest**, **GitHub Actions** for CI
- Planned: **Postgres 16 + pgvector** (M2), **LangSmith** tracing during development (M3), **FastAPI** and **Docker** (M6)

## Getting started

Requirements: [uv](https://docs.astral.sh/uv/getting-started/installation/), Git, and [Ollama](https://ollama.com) for the steps that call a model.

```bash
git clone https://github.com/Odilove5/askhr.git
cd askhr
uv sync                      # creates .venv and installs everything from uv.lock
cp .env.example .env         # local settings; .env is gitignored
```

Run the same checks as CI:

```bash
uv run ruff check
uv run ruff format --check
uv run mypy src tests
uv run pytest
```

Run the pipeline step by step:

```bash
uv run python -m askhr.connectors.viamonde_site   # list the HR documents -> corpus/manifest.yaml
uv run python -m askhr.connectors.downloader      # download the PDFs -> data/raw/
uv run python -m askhr.loader                     # read and clean the PDFs
uv run python -m askhr.chunker                    # cut them into chunks
ollama pull hf.co/unsloth/gemma-4-12b-it-GGUF:UD-Q5_K_XL
uv run python -m askhr.context 5,202a             # contextual notes for one document
uv run python -m askhr.ingest --dry-run           # check every prompt fits, count the notes to write
uv run python -m askhr.ingest                     # every chunk with its note -> data/chunks.jsonl
```

## Configuration

All settings are environment variables starting with `ASKHR_`, read from the environment or from `.env` (see `.env.example`). The most useful ones:

| Variable | Default | Meaning |
|---|---|---|
| `ASKHR_OLLAMA_MODEL` | Gemma 4 12B (GGUF) | Local model |
| `ASKHR_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server |
| `ASKHR_CLOUD_ENABLED` | `false` | Kill switch for the Claude API |
| `ASKHR_ANTHROPIC_API_KEY` | none | Claude API key, only in `.env`, never committed |

The cloud model **fails closed**: it refuses to run unless the switch is on and a key is set.

## Project structure

```
askhr/
├── .github/workflows/ci.yml       # CI: lint, format, type check, tests
├── corpus/manifest.yaml           # titles and links of the HR documents (no content)
├── src/askhr/
│   ├── connectors/
│   │   ├── viamonde_site.py       # reads the board's document index page
│   │   └── downloader.py          # downloads the PDFs into data/raw/
│   ├── models.py                  # Document, Page, Chunk (frozen dataclasses)
│   ├── text.py                    # text helpers
│   ├── loader.py                  # PDF -> clean pages (headers removed)
│   ├── chunker.py                 # pages -> chunks by section
│   ├── settings.py                # configuration (ASKHR_ variables, .env)
│   ├── llm.py                     # builds the local or cloud chat model
│   ├── context.py                 # contextual-notes chain (LCEL)
│   ├── notes_cache.py             # notes already written (resumable runs)
│   └── ingest.py                  # one command: every chunk with its note
├── tests/                         # one test file per module, fake models only
├── CLAUDE.md                      # guidance for AI coding assistants
├── pyproject.toml                 # metadata, dependencies, tool settings
└── uv.lock                        # exact, locked dependency versions
```

## Author

Built by Odilon ([@Odilove5](https://github.com/Odilove5)).
