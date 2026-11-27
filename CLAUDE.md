# CLAUDE.md

Guidance for Claude (and any AI assistant) working in this repository. Keep this file short; detailed plans live outside the repo.

## Project

AskHR is a bilingual (French/English) HR assistant for Ontario school boards. It answers staff questions from the board's own HR documents, cites its sources, and hands off to the right HR contact when it can't answer. Agentic corrective RAG with contextual retrieval, built with LangChain (LCEL) and LangGraph. Local models through Ollama by default.

## How we work: mentor mode

The owner is building AskHR to learn. Act as a mentor, not a code generator:
- Explain concepts and Python idioms, give hints and small snippets, review the owner's code.
- Write full code **only when explicitly asked** ("write the code", "fix the code").
- Work in small steps and wait for "next" before moving on.
- Never assume a first version is right. Check it a second time, on the full real data, before saying it's done.

## Commands

```bash
uv sync                          # install dependencies (Python 3.12)
cp .env.example .env             # first time only; .env is gitignored
uv run pytest                    # tests
uv run ruff check                # lint
uv run ruff format --check       # formatting (drop --check to fix)
uv run mypy src tests            # type check (strict)
```

CI runs the same four checks on every push. All of them must pass before a commit.

Pipeline, in order (each module has a `main()`):

```bash
uv run python -m askhr.connectors.viamonde_site   # list the HR documents -> corpus/manifest.yaml
uv run python -m askhr.connectors.downloader      # download the PDFs -> data/raw/
uv run python -m askhr.loader                     # read the PDFs
uv run python -m askhr.chunker                    # cut into chunks
uv run python -m askhr.context [doc_id]           # contextual notes (needs Ollama running)
```

The planned stack is fixed: Postgres 16 + pgvector for search (M2), LangGraph for the agent (M3), LangSmith for tracing during development. Don't introduce alternatives (for example Chroma) without asking.

## Layout

- `src/askhr/connectors/`: website connector and downloader (plain Python, httpx).
- `src/askhr/models.py`: `Document`, `Page`, `Chunk`, all frozen dataclasses.
- `src/askhr/loader.py`, `text.py`, `chunker.py`: PDF reading, cleaning, chunking.
- `src/askhr/settings.py`: all configuration (pydantic-settings).
- `src/askhr/llm.py`: the only place that builds chat models.
- `src/askhr/context.py`: LCEL chain that writes a contextual note per chunk.
- `tests/`: one test file per module; `tests/fixtures/` holds small saved web pages.

## Rules

**Data and privacy**
- Never commit the downloaded documents or text copied from them (they are copyrighted). `data/` is gitignored; only `corpus/manifest.yaml` (titles and URLs) is committed.
- Tests use made-up text only. A test that needs a real PDF must be skipped when `data/raw/` is missing (as in CI).
- This repository is public: no secrets, no personal data, no internal school-board information.

**Configuration and models**
- Every setting lives in `Settings` and is read from `ASKHR_` environment variables or `.env`. Secrets go only in `.env` (gitignored), never in code, tests or chat.
- Get models only through `get_chat_model()` in `llm.py`.
- The cloud model fails closed: off unless `ASKHR_CLOUD_ENABLED=true` and a key is set. Never add a silent fallback between local and cloud.
- Personal or sensitive questions must always go to the local model (the router is built at M3).
- Don't add a dependency without asking. Check its licence first: no AGPL in a product that will be sold.

**Code style**
- Compose every LLM chain with LCEL (`prompt | model | parser`). Plain Python for everything that doesn't call a model. LangGraph for the agent loop.
- Data types are frozen dataclasses; use `dataclasses.replace` to make changed copies.
- Type hints everywhere (mypy strict), line length 100, small pure functions where possible.
- Raise clear errors instead of guessing or cutting data silently.
- Tests use fake models (`FakeListChatModel`), never a real LLM.
- Code, comments and docs are in English. Prompts for French documents are in French.
- Text processing must respect French typography: keep the space before `:`, `;`, `?` and `!`.

**Prompts**
- Bump `PROMPT_VERSION` in `context.py` whenever its prompt changes; the ingest cache (M1 step 6c) uses it to know which notes to redo.
- Every prompt rule should come from a problem seen in a real run. After a prompt change, re-run on the largest document and read the outputs.
- A prompt rule is a request, not a guarantee: back important rules with a check in code (as `clean_note` does for page numbers).
- Never let a prompt be cut silently: estimate its size first and raise if it doesn't fit the window.

## Workflow for each step

1. The owner writes `docs/steps/<step>/intent.md`.
2. We agree on `spec.md` (including **done when**), then `plan.md`. Commit each one.
3. Build in small steps, following the plan; update `plan.md` if the code has to differ.
4. Verify: the four checks, then a second check on the full real data.
5. Fill in the Result section of `plan.md`.

Templates are in `docs/steps/README.md`.

**Bugs:** first commit a test that fails because of the bug, then fix the code without changing that test.

**Learning from mistakes:** any mistake made twice gets a rule in this file.

**Hooks** (`.claude/settings.json`) enforce the rules that must always hold: edits to `data/`, `.env` and `uv.lock` are blocked, text that looks like a secret is blocked, and every edited Python file is formatted and linted. Don't work around a blocked edit; ask the owner.

## Commits

Small commits with conventional messages: `feat:`, `fix:`, `docs:`, `test:`, `chore:`. Run the four checks first. The owner reviews and commits.
