from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.nlp.text import narrative_frame


def retrieve_evidence(df: pd.DataFrame, question: str, top_k: int = 8) -> pd.DataFrame:
    data = narrative_frame(df).reset_index(drop=True)
    if data.empty:
        return pd.DataFrame()
    min_df = 1 if len(data) < 50 else 2
    vectorizer = TfidfVectorizer(stop_words="english", max_features=40000, ngram_range=(1, 2), min_df=min_df)
    matrix = vectorizer.fit_transform(data["clean_narrative"])
    q_vec = vectorizer.transform([question])
    scores = cosine_similarity(q_vec, matrix).ravel()
    top_idx = scores.argsort()[-top_k:][::-1]
    cols = ["complaint_id", "date_received", "company", "product", "issue", "clean_narrative"]
    result = data.iloc[top_idx][cols].copy()
    result["retrieval_score"] = scores[top_idx]
    return result
