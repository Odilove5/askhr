"""PostToolUse hook: format and lint a Python file right after Claude edits it.

Runs ruff from the project's virtual environment on that one file (fast). Remaining
lint problems are sent back to Claude (exit code 2) so it fixes them now, not at commit.
The full checks (mypy, pytest) stay at commit time and in CI.
"""

import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    event = json.load(sys.stdin)
    file_path = event.get("tool_input", {}).get("file_path", "")
    if not file_path.endswith(".py") or not Path(file_path).exists():
        return 0
    project_dir = os.environ.get("CLAUDE_PROJECT_DIR") or event.get("cwd", ".")
    ruff = Path(project_dir) / ".venv" / "bin" / "ruff"
    if not ruff.exists():
        return 0  # no venv yet: nothing to do
    # Fix lint first (it can remove lines), then format the result.
    check = subprocess.run(
        [ruff, "check", "--fix", "--quiet", file_path], capture_output=True, text=True
    )
    subprocess.run([ruff, "format", "--quiet", file_path], check=False)
    if check.returncode != 0:
        print(f"ruff found problems in {file_path}:\n{check.stdout}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
