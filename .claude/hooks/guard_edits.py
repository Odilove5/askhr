"""PreToolUse hook: refuse actions that break AskHR's must-hold rules.

Claude Code sends the planned tool call as JSON on stdin. Exit code 2 blocks it and
shows our message to Claude; exit code 0 lets it through.

Edit / Write / MultiEdit are blocked when they:
- touch a protected file: anything under data/ (copyrighted documents), .env (secrets)
  or uv.lock (changed only by `uv add` / `uv lock`);
- add text that looks like a secret (API keys, private keys).

Bash commands are blocked when they:
- mention .env at all (reading it would put secrets in the conversation);
- write to data/ or uv.lock with a shell tool (>, >>, tee, rm, mv, cp, sed -i, truncate).
  Running the project's own code (for example `uv run python -m askhr.ingest`, which
  writes data/chunks.jsonl) is allowed.
- add text that looks like a secret.

The Bash checks are best effort: a determined command can always hide a write. The
`permissions.deny` rules in settings.json are a second layer.
"""

import json
import os
import re
import sys
from pathlib import PurePath

PROTECTED_DIRS = ("data",)
PROTECTED_FILES = (".env", "uv.lock")

SECRET_PATTERNS = [
    re.compile(r"sk-ant-[A-Za-z0-9_-]{40,}"),  # Anthropic API key (test fakes are shorter)
    re.compile(r"sk-[A-Za-z0-9]{32,}"),  # OpenAI-style key
    re.compile(r"ghp_[A-Za-z0-9]{30,}"),  # GitHub token
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
]

# ".env" as a file name, but not ".env.example" or ".envrc".
ENV_IN_COMMAND = re.compile(r"(?<![\w.-])\.env(?![\w.-])")
# A shell write (redirect or file tool) followed, in the same command, by a protected path.
_WRITE = r"(?:>>?|\btee\b|\brm\b|\bmv\b|\bcp\b|\bsed\s+-i|\btruncate\b)"
_TARGET = r"(?<![\w./-])(?:\./)?(?:data\b|uv\.lock\b)"
SHELL_WRITE = re.compile(_WRITE + r"[^|;&]*?" + _TARGET)


def protected_reason(file_path: str, project_dir: str) -> str | None:
    path = PurePath(file_path)
    try:
        relative = path.relative_to(project_dir) if path.is_absolute() else path
    except ValueError:
        return None  # outside the project: not our business
    parts = relative.parts
    if parts and parts[0] in PROTECTED_DIRS:
        return f"{relative} is under {parts[0]}/, which holds copyrighted documents."
    if str(relative) in PROTECTED_FILES:
        return f"{relative} is protected (secrets or lock file). Change it by hand."
    return None


def secret_reason(text: str) -> str | None:
    for pattern in SECRET_PATTERNS:
        if pattern.search(text):
            return "the new text looks like it contains a secret (API key or private key)."
    return None


def command_reason(command: str) -> str | None:
    if ENV_IN_COMMAND.search(command):
        return ".env holds secrets: shell commands may not read or change it."
    if SHELL_WRITE.search(command):
        return "shell commands may not write to data/ or uv.lock."
    return secret_reason(command)


def edit_reason(tool_input: dict[str, object], project_dir: str) -> str | None:
    texts = [str(tool_input.get(key, "")) for key in ("content", "new_string")]
    edits = tool_input.get("edits", [])
    if isinstance(edits, list):
        texts += [str(edit.get("new_string", "")) for edit in edits if isinstance(edit, dict)]
    file_path = str(tool_input.get("file_path", ""))
    return protected_reason(file_path, project_dir) or secret_reason("\n".join(texts))


def main() -> int:
    event = json.load(sys.stdin)
    tool_input = event.get("tool_input") or {}
    project_dir = os.environ.get("CLAUDE_PROJECT_DIR") or event.get("cwd", "")
    if event.get("tool_name") == "Bash":
        reason = command_reason(str(tool_input.get("command", "")))
    else:
        reason = edit_reason(tool_input, project_dir)
    if reason:
        print(f"Blocked by AskHR guard: {reason} See CLAUDE.md > Rules.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
