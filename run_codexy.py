"""Task Scheduler entry point for the Codex-backed Codexy instance.

Same rationale as run_bender.py -- loads `.env.codexy` into the process
environment manually (pydantic-settings has no env_file configured)
before launching `python -m bender`, and logs output to disk since a
scheduled task has no attached console.
"""

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
ENV_FILE = HERE / ".env.codexy"
LOG_FILE = HERE / "codexy_boot.log"


def load_env(path: Path) -> dict[str, str]:
    env = os.environ.copy()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip()
    # Python block-buffers stdout when it isn't a TTY (i.e. always, here --
    # this is redirected to LOG_FILE). A force-killed process (Stop-Process
    # -Force, no flush) loses whatever log lines were still sitting in that
    # buffer -- confirmed missing exactly the entries needed to debug a
    # stuck thread. Unbuffered trades a little throughput for logs that are
    # actually on disk by the time something goes wrong.
    env["PYTHONUNBUFFERED"] = "1"
    return env


def main() -> None:
    env = load_env(ENV_FILE)
    with open(LOG_FILE, "a", encoding="utf-8") as log:
        subprocess.run(
            [sys.executable, "-m", "bender"],
            cwd=HERE,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )


if __name__ == "__main__":
    main()
