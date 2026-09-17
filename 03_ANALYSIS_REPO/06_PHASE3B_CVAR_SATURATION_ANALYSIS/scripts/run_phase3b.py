from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from phase3b_core import run_phase3b


if __name__ == "__main__":
    run_phase3b()
