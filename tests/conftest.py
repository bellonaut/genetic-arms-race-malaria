"""
Pytest configuration to ensure the project root is on sys.path.

This avoids import errors (e.g., `ModuleNotFoundError: No module named 'src'`)
when pytest is invoked from environments that do not automatically add the
repository root to `sys.path`.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

