from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.phase3_core import run_phase3


if __name__ == "__main__":
    run_phase3()
