# AskHR

[![CI](https://github.com/Odilove5/askhr/actions/workflows/ci.yml/badge.svg)](https://github.com/Odilove5/askhr/actions/workflows/ci.yml)

**AskHR** is a bilingual (French/English) HR assistant for Ontario school boards.
Staff ask questions in plain language ("How many days of parental leave do I get?"),
and AskHR answers from the board's own HR documents (policies, administrative
directives, collective agreements, benefits booklets), **cites its sources**, and
**hands off to the right HR contact** when the documents don't answer the question.

It is built as an agentic, corrective RAG system with contextual retrieval, using
LangChain and LangGraph, with local models through Ollama for the demo.

> All documents used in this repository are fictional. No real employee or board data is included.

## Status

🚧 Early development: **M0 (project setup) complete.**

| Milestone | Goal | Target |
|---|---|---|
| M0 | Project setup: repo, uv, linting, tests, CI | Nov 30, 2026 ✅ |
| M1 | Fictional FR/EN HR corpus and ingestion pipeline | Dec 7, 2026 |
| M2 | Hybrid retrieval and relevance grader | Dec 18, 2026 |
| M3 | LangGraph agent: routing, query rewriting, HR handoff | Jan 8, 2027 |
| M4 | Cited, bilingual answers with quote checking | Jan 18, 2027 |
| M5 | Evals (Ragas / LangSmith) and guardrails | Jan 29, 2027 |
| M6 | FastAPI app, chat UI, Docker, deployed in Canada | Feb 12, 2027 |
| M7 | Portfolio: architecture diagram, demo video, lessons learned | Feb 21, 2027 |

## Tech stack

- **Python 3.12**, managed with [uv](https://docs.astral.sh/uv/)
- **LangChain / LangGraph** for the agent (from M1)
- **Ollama** for local models during development
- **ruff** (lint and format), **mypy** (strict type checking), **pytest** (tests)
- **GitHub Actions** for CI

## Getting started

Requirements: [uv](https://docs.astral.sh/uv/getting-started/installation/) and Git.

```bash
git clone https://github.com/Odilove5/askhr.git
cd askhr
uv sync                      # creates .venv and installs everything from uv.lock
```

Run the same checks as CI:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
uv run pytest
```

## Project structure

```
askhr/
├── .github/workflows/ci.yml   # CI: lint, format, type check, tests
├── src/askhr/                 # the AskHR package
│   ├── __init__.py            # package version
│   └── text.py                # text helpers (whitespace normalization)
├── tests/                     # pytest tests
├── pyproject.toml             # project metadata, dependencies, tool settings
└── uv.lock                    # exact, locked dependency versions
```

## Author

Built by Odilon ([@Odilove5](https://github.com/Odilove5)).
