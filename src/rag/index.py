from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer

from src.config import COMPLAINTS_PARQUET, RAG_INDEX_PATH, RAG_METADATA_PATH, ensure_directories
from src.nlp.text import narrative_frame

INDEX_COLUMNS = [
    "complaint_id",
    "date_received",
    "company",
    "product",
    "sub_product",
    "issue",
    "sub_issue",
    "state",
    "clean_narrative",
]


@dataclass
class RagIndexConfig:
    max_docs: int | None = None
    min_chars: int = 80
    dense_components: int = 128
    max_features: int = 50000


def _prepare_index_frame(complaints: pd.DataFrame, config: RagIndexConfig) -> pd.DataFrame:
    docs = narrative_frame(complaints, min_chars=config.min_chars).copy()
    docs["date_received"] = pd.to_datetime(docs["date_received"], errors="coerce")
    docs = docs.sort_values("date_received", ascending=False)
    if config.max_docs:
        docs = docs.head(config.max_docs)
    for col in INDEX_COLUMNS:
        if col not in docs.columns:
            docs[col] = pd.NA
    docs = docs[INDEX_COLUMNS].reset_index(drop=True)
    docs["row_id"] = np.arange(len(docs))
    return docs


def build_rag_index(
    complaints_path: Path = COMPLAINTS_PARQUET,
    index_path: Path = RAG_INDEX_PATH,
    metadata_path: Path = RAG_METADATA_PATH,
    config: RagIndexConfig | None = None,
) -> dict[str, object]:
    ensure_directories()
    config = config or RagIndexConfig()
    complaints = pd.read_parquet(complaints_path)
    docs = _prepare_index_frame(complaints, config)
    if docs.empty:
        raise ValueError("No complaint narratives available for RAG indexing.")

    lexical_vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=config.max_features,
        min_df=2 if len(docs) >= 50 else 1,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    lexical_matrix = lexical_vectorizer.fit_transform(docs["clean_narrative"])

    dense_components = min(config.dense_components, max(2, lexical_matrix.shape[1] - 1), max(2, lexical_matrix.shape[0] - 1))
    dense_model = TruncatedSVD(n_components=dense_components, random_state=42)
    dense_normalizer = Normalizer(copy=False)
    dense_matrix = dense_normalizer.fit_transform(dense_model.fit_transform(lexical_matrix))

    index = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "embedding_model": f"local_lsa_tfidf_svd_{dense_components}",
        "retrieval_methods": ["tfidf_lexical", "lsa_dense", "metadata_filter", "rank_fusion"],
        "lexical_vectorizer": lexical_vectorizer,
        "lexical_matrix": lexical_matrix,
        "dense_model": dense_model,
        "dense_normalizer": dense_normalizer,
        "dense_matrix": dense_matrix,
        "metadata": docs.drop(columns=["clean_narrative"]),
        "narratives": docs["clean_narrative"].tolist(),
    }
    joblib.dump(index, index_path)
    docs.to_parquet(metadata_path, index=False)
    metrics = {
        "indexed_narratives": int(len(docs)),
        "embedding_model": index["embedding_model"],
        "retrieval_methods": index["retrieval_methods"],
        "index_path": str(index_path),
        "metadata_path": str(metadata_path),
        "created_at_utc": index["created_at_utc"],
    }
    (index_path.with_suffix(".json")).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def load_rag_index(index_path: Path = RAG_INDEX_PATH) -> dict:
    if not index_path.exists():
        raise FileNotFoundError(f"Missing RAG index at {index_path}. Run `python scripts/build_rag_index.py`.")
    return joblib.load(index_path)
