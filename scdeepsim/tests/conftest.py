from __future__ import annotations

import sys
from pathlib import Path


repository_root = Path(__file__).resolve().parents[2]
if str(repository_root) not in sys.path:
    sys.path.insert(0, str(repository_root))
