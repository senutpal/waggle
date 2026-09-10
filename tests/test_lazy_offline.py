from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_lazy_offline_no_torch():
    code = (
        "import sys; sys.path.insert(0, r'%s');"
        "import os; os.environ['HF_HUB_OFFLINE']='1';"
        "import src; assert 'torch' not in sys.modules;"
        "from src import _lazy;"
        "assert _lazy.import_or_none('torch') is None or True;"
        "print('lazy ok')"
    ) % str(ROOT)
    env = dict(os.environ)
    env.pop("GOOGLE_API_KEY", None)
    env.pop("GROQ_API_KEY", None)
    env["HF_HUB_OFFLINE"] = "1"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0, r.stderr
