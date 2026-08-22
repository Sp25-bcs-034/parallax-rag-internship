import sys
sys.path.append("src")
from src.nlp_analysis import get_sentiment_pipeline

# Hand-labeled by YOU, reading 15-20 real examples from your dataset
manual_test_set = [
    {"text": "The team celebrated a stunning victory in the championship final.", "expected": "POSITIVE"},
    {"text": "The company reported massive losses and hundreds of layoffs.", "expected": "NEGATIVE"},
    {"text": "Scientists announced a breakthrough that could cure the disease.", "expected": "POSITIVE"},
    {"text": "The stock market crashed today, wiping out billions in value.", "expected": "NEGATIVE"},
    {"text": "The rescue mission failed, and no survivors were found.", "expected": "NEGATIVE"},
    {"text": "Fans praised the incredible performance from the young athlete.", "expected": "POSITIVE"},
    # add 10+ more real rows pulled from your own data/cleaned/clean_ag_news.csv
]


def evaluate_sentiment_accuracy():
    model = get_sentiment_pipeline()
    correct = 0
    for case in manual_test_set:
        predicted = model(case["text"][:512])[0]["label"]
        is_correct = predicted == case["expected"]
        correct += int(is_correct)
        status = "OK " if is_correct else "MISS"
        print(f"[{status}] predicted={predicted} expected={case['expected']} | {case['text'][:60]}...")

    accuracy = correct / len(manual_test_set)
    print(f"\nSentiment Accuracy: {accuracy:.2%} ({correct}/{len(manual_test_set)})")
    return accuracy


if __name__ == "__main__":
    evaluate_sentiment_accuracy()