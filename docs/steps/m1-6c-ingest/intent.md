# Intent: M1 step 6c, one-command ingest

> Draft written by Claude from our discussions. Owner: edit anything that isn't what you want.

Problem: Each pipeline step runs separately, `context.py` handles one document at a time, and nothing is saved. Writing all the notes takes about 40 minutes, and today a crash at minute 39 would lose everything.

Outcome: One command turns the 37 HR documents into a single file of chunks, each with its contextual note, ready for M2 to embed and index. A second run with nothing changed is instant and gives the same file.

Users: M2 (embeddings and search) reads the file. The owner reads it in step 6d to review chunk quality.

Constraints: Runs on the owner's Mac with Ollama only (no cloud). The output stays in `data/` and is never committed. No silent damage: a cut, empty or misplaced note must be caught, not stored.

Open questions: What to do with a note cut at the length cap? Stop at the first error or skip the document? Where does the cache live?
