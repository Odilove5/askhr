"""PreToolUse hook: refuse edits that break AskHR's must-hold rules.

Claude Code sends the planned edit as JSON on stdin. Exit code 2 blocks the edit and
shows our message to Claude; exit code 0 lets it through.

Blocked:
- files we must never write: anything under data/ (copyrighted documents), .env
  (secrets) and uv.lock (changed only by `uv add` / `uv lock`);
- text that looks like a secret (API keys, private keys).
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


def main() -> int:
    event = json.load(sys.stdin)
    tool_input = event.get("tool_input", {})
    file_path = tool_input.get("file_path", "")
    project_dir = os.environ.get("CLAUDE_PROJECT_DIR") or event.get("cwd", "")
    texts = [str(tool_input.get(key, "")) for key in ("content", "new_string")]
    texts += [str(edit.get("new_string", "")) for edit in tool_input.get("edits", [])]
    new_text = "\n".join(texts)

    reason = protected_reason(file_path, project_dir) or secret_reason(new_text)
    if reason:
        print(f"Blocked by AskHR guard: {reason} See CLAUDE.md > Rules.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
