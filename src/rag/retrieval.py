from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.config import RAG_INDEX_PATH
from src.rag.index import load_rag_index


@dataclass
class RagFilters:
    company: str | None = None
    product: str | None = None
    issue: str | None = None
    state: str | None = None
    date_start: str | None = None
    date_end: str | None = None


@dataclass
class RetrievalResult:
    query: str
    filters: RagFilters
    evidence: pd.DataFrame
    trace: dict[str, object] = field(default_factory=dict)


def _metadata_mask(metadata: pd.DataFrame, filters: RagFilters) -> pd.Series:
    mask = pd.Series(True, index=metadata.index)
    for col in ["company", "product", "issue", "state"]:
        value = getattr(filters, col)
        if value:
            mask &= metadata[col].astype(str).str.lower().eq(str(value).lower())
    dates = pd.to_datetime(metadata["date_received"], errors="coerce")
    if filters.date_start:
        mask &= dates >= pd.Timestamp(filters.date_start)
    if filters.date_end:
        mask &= dates <= pd.Timestamp(filters.date_end)
    return mask


def _minmax(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    lo = np.nanmin(values)
    hi = np.nanmax(values)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return np.zeros_like(values, dtype=float)
    return (values - lo) / (hi - lo)


def hybrid_retrieve(
    query: str,
    filters: RagFilters | None = None,
    top_k: int = 8,
    lexical_weight: float = 0.52,
    dense_weight: float = 0.48,
    index_path=RAG_INDEX_PATH,
) -> RetrievalResult:
    filters = filters or RagFilters()
    index = load_rag_index(index_path)
    metadata: pd.DataFrame = index["metadata"].copy()
    narratives = index["narratives"]
    mask = _metadata_mask(metadata, filters)
    candidate_idx = np.flatnonzero(mask.to_numpy())
    if candidate_idx.size == 0:
        return RetrievalResult(query, filters, pd.DataFrame(), {"candidate_count": 0, "reason": "metadata_filter_returned_no_rows"})

    q_lexical = index["lexical_vectorizer"].transform([query])
    lexical_scores = cosine_similarity(q_lexical, index["lexical_matrix"][candidate_idx]).ravel()
    q_dense = index["dense_normalizer"].transform(index["dense_model"].transform(q_lexical))
    dense_scores = cosine_similarity(q_dense, index["dense_matrix"][candidate_idx]).ravel()
    fused = lexical_weight * _minmax(lexical_scores) + dense_weight * _minmax(dense_scores)

    order = np.argsort(fused)[::-1][:top_k]
    selected_idx = candidate_idx[order]
    evidence = metadata.iloc[selected_idx].copy().reset_index(drop=True)
    evidence["clean_narrative"] = [narratives[i] for i in selected_idx]
    evidence["lexical_score"] = lexical_scores[order]
    evidence["dense_score"] = dense_scores[order]
    evidence["hybrid_score"] = fused[order]
    evidence["citation"] = evidence["complaint_id"].map(lambda cid: f"[CFPB Complaint {int(cid)}]" if pd.notna(cid) else "[CFPB Complaint missing]")
    return RetrievalResult(
        query,
        filters,
        evidence,
        {
            "candidate_count": int(candidate_idx.size),
            "returned_count": int(len(evidence)),
            "lexical_weight": lexical_weight,
            "dense_weight": dense_weight,
            "embedding_model": index.get("embedding_model"),
        },
    )
