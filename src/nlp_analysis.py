import pandas as pd
from transformers import pipeline

"""NLP analysis utilities for topic and sentiment metadata generation."""
_sentiment_pipeline = None


def get_sentiment_pipeline():
    """Load the model once, reuse it - same principle as your embedding model."""

    global _sentiment_pipeline

    if _sentiment_pipeline is None:
        _sentiment_pipeline = pipeline(task="text-classification",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )

    return _sentiment_pipeline
def add_sentiment(chunk_DF, batch_size=16):
    """
    Adds two columns to chunk_DF:
      - sentiment       : 'POSITIVE' or 'NEGATIVE'
      - sentiment_score : model's confidence (0-1)
    """
    print("\n ---- SENTIMENT ANALYSIS -----")

    if chunk_DF.empty:
        chunk_DF["sentiment"] = []
        chunk_DF["sentiment_score"] = []
        return chunk_DF

    sentiment_pipeline = get_sentiment_pipeline()
    texts = chunk_DF["text"].tolist()

    # DistilBERT has a 512-token limit - truncate defensively so long
    # chunks don't crash the pipeline (an "edge case" you're asked to handle)
    truncated_texts = [t[:512] for t in texts]

    results = sentiment_pipeline(truncated_texts, batch_size=batch_size, truncation=True)

    chunk_DF["sentiment"] = [r["label"] for r in results]
    chunk_DF["sentiment_score"] = [round(r["score"], 4) for r in results]

    print(chunk_DF["sentiment"].value_counts())
    return chunk_DF


def validate_edge_cases(chunk_DF, min_length=20, low_confidence_threshold=0.6):
    """
    Flags rows that need manual review:
      - very short chunks (risky for both topic modeling and sentiment)
      - low-confidence sentiment predictions (model is unsure - likely jargon/ambiguous text)
    """
    print("\n ---- EDGE CASE VALIDATION -----")
    report = {}

    short_docs = chunk_DF[chunk_DF["text"].str.len() < min_length]
    report["short_docs_count"] = len(short_docs)
    print(f"Short chunks (<{min_length} chars): {len(short_docs)}")

    if "topic_id" in chunk_DF.columns:
        outliers = chunk_DF[chunk_DF["topic_id"] == -1]
        report["topic_outliers_count"] = len(outliers)
        print(f"Topic outliers (-1): {len(outliers)}")

    if "sentiment_score" in chunk_DF.columns:
        low_conf = chunk_DF[chunk_DF["sentiment_score"] < low_confidence_threshold]
        report["low_confidence_sentiment_count"] = len(low_conf)
        print(f"Low-confidence sentiment (<{low_confidence_threshold}): {len(low_conf)}")

    return report