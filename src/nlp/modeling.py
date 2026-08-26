from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import MODEL_REPORT_PATH
from src.nlp.text import narrative_frame


def train_issue_classifier(df: pd.DataFrame, target_col: str = "product", max_docs: int = 60000, output_path: Path = MODEL_REPORT_PATH) -> dict:
    data = narrative_frame(df).dropna(subset=[target_col]).head(max_docs)
    class_counts = data[target_col].value_counts()
    eligible_classes = class_counts[class_counts >= 50].index
    data = data[data[target_col].isin(eligible_classes)]
    if len(data) < 500 or data[target_col].nunique() < 2:
        report = {"status": "skipped", "reason": "Not enough narrative records or target diversity."}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    x_train, x_test, y_train, y_test = train_test_split(
        data["clean_narrative"],
        data[target_col],
        test_size=0.25,
        random_state=42,
        stratify=data[target_col],
    )
    baseline = DummyClassifier(strategy="most_frequent").fit(x_train, y_train)
    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(stop_words="english", max_features=50000, ngram_range=(1, 2), min_df=3)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=1)),
        ]
    ).fit(x_train, y_train)
    baseline_pred = baseline.predict(x_test)
    model_pred = model.predict(x_test)
    report = {
        "status": "trained",
        "target": target_col,
        "records": int(len(data)),
        "classes": int(data[target_col].nunique()),
        "baseline_macro_f1": float(f1_score(y_test, baseline_pred, average="macro")),
        "model_macro_f1": float(f1_score(y_test, model_pred, average="macro")),
        "classification_report": classification_report(y_test, model_pred, output_dict=True),
        "business_metric_note": "Macro F1 weights product classes equally, which matters when routing less frequent complaint types.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
