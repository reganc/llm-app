"""Test bootstrap for the llm-app gateway.

`api/config.py` creates DATA_DIR at import time and defaults it to `/app/data`,
which only exists inside the container. Point it at a throwaway directory before
any app module is imported, and put `api/` on the path (app modules import each
other flat, e.g. `from config import CFG`).
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_DIR", tempfile.mkdtemp(prefix="llm-app-tests-"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "api"))
