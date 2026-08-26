from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.enrichment.entity_resolution import build_entity_resolution_artifacts


if __name__ == "__main__":
    print(json.dumps(build_entity_resolution_artifacts(), indent=2))
