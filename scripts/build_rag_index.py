from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag.index import RagIndexConfig, build_rag_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cached hybrid RAG index over CFPB complaint narratives.")
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--dense-components", type=int, default=128)
    args = parser.parse_args()
    metrics = build_rag_index(config=RagIndexConfig(max_docs=args.max_docs, dense_components=args.dense_components))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
