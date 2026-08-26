from __future__ import annotations

import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from src.nlp.text import narrative_frame


def discover_topics(df: pd.DataFrame, n_topics: int = 12, max_docs: int = 50000, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    narratives = narrative_frame(df).head(max_docs).reset_index(drop=True)
    if len(narratives) < max(20, n_topics * 3):
        return pd.DataFrame(), narratives.assign(topic_id=pd.NA, topic_score=pd.NA)
    vectorizer = TfidfVectorizer(stop_words="english", max_features=20000, min_df=5, max_df=0.75, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(narratives["clean_narrative"])
    model = NMF(n_components=n_topics, random_state=random_state, init="nndsvda", max_iter=400)
    doc_topic = model.fit_transform(matrix)
    terms = vectorizer.get_feature_names_out()
    narratives["topic_id"] = doc_topic.argmax(axis=1)
    narratives["topic_score"] = doc_topic.max(axis=1)
    rows = []
    for topic_id, weights in enumerate(model.components_):
        top_terms = [terms[idx] for idx in weights.argsort()[-12:][::-1]]
        subset = narratives[narratives["topic_id"] == topic_id]
        rep_cols = ["complaint_id", "date_received", "company", "product", "issue", "clean_narrative"]
        reps = subset.sort_values("topic_score", ascending=False).head(3)[rep_cols].to_dict("records")
        rows.append(
            {
                "topic_id": topic_id,
                "topic_name": "; ".join(top_terms[:4]),
                "top_terms": ", ".join(top_terms),
                "volume": int(len(subset)),
                "product_distribution": subset["product"].value_counts().head(5).to_dict(),
                "company_distribution": subset["company"].value_counts().head(5).to_dict(),
                "representative_complaints": reps,
            }
        )
    return pd.DataFrame(rows), narratives
