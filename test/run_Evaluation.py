"""Automated evaluation script: runs retrieval and generation evaluation
and writes results to docs/evaluation_report.json."""
import sys
sys.path.append("src")

import json
from src.load_dataset import get_collection
from src.semantic_Search import retrival_
from src.generational_latency import generational_evaluation

if __name__ == "__main__":
    collection = get_collection()

    print("=== Retrieval Evaluation ===")
    retrival_(collection, k=3)

    print("\n=== Generation & Latency Evaluation ===")
    generational_evaluation(collection)