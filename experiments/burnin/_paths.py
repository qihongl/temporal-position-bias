"""Locate this experiment and reuse the repository's original source modules."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
if not (REPO_ROOT / "src" / "model.py").is_file():
    raise RuntimeError("Place this experiment in <repository>/experiments/burnin/")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
