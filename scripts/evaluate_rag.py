from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag.evaluation import evaluate_rag


if __name__ == "__main__":
    print(json.dumps(evaluate_rag(), indent=2))
