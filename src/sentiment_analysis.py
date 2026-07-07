"""Sentiment analysis of responses using Cardiff Twitter RoBERTa."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from transformers import pipeline


@dataclass
class SentimentScores:
    positive: float
    negative: float
    neutral: float
    label: str


class SentimentAnalyzer:
    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        self.model_name = model_name
        self.classifier = pipeline(
            task="sentiment-analysis",
            model=model_name,
            top_k=None,
        )

    def analyze(self, text: str) -> SentimentScores:
        # RoBERTa has a 512-token input limit, so long responses are truncated.
        results = self.classifier(text, truncation=True, max_length=512)[0]
        scores = {item["label"].lower(): item["score"] for item in results}
        label = max(scores, key=scores.get)
        return SentimentScores(
            positive=scores.get("positive", 0.0),
            negative=scores.get("negative", 0.0),
            neutral=scores.get("neutral", 0.0),
            label=label,
        )


def run_analysis():
    project_root = Path(__file__).resolve().parent.parent
    responses_path = project_root / "data" / "responses.csv"
    output_path = project_root / "data" / "sentiment_results.csv"

    print(f"Loading responses from {responses_path}")
    df = pd.read_csv(responses_path)
    print(f"Loaded {len(df)} responses")

    successful = df[df["success"] == True].copy()
    print(f"Filtering to {len(successful)} successful responses")

    print("Loading sentiment model (Cardiff Twitter RoBERTa)...")
    analyzer = SentimentAnalyzer()
    print("Model loaded.")

    print("Running sentiment analysis on all responses...")
    positives = []
    negatives = []
    neutrals = []
    labels = []
    for i, row in successful.iterrows():
        text = str(row["response_text"])
        scores = analyzer.analyze(text)
        positives.append(scores.positive)
        negatives.append(scores.negative)
        neutrals.append(scores.neutral)
        labels.append(scores.label)
        print(f"  [{i + 1}/{len(successful)}] {row['model_name']} - {row['prompt_id']}: {scores.label}")

    successful["sentiment_positive"] = positives
    successful["sentiment_negative"] = negatives
    successful["sentiment_neutral"] = neutrals
    successful["sentiment_label"] = labels

    successful.to_csv(output_path, index=False)
    print(f"\nSaved results to {output_path}")
    print(f"Sentiment distribution:")
    print(successful["sentiment_label"].value_counts())


if __name__ == "__main__":
    run_analysis()