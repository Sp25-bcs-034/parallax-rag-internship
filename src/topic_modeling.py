import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# Reuse the SAME model family your embedding.py already uses.
# (Not the exact same object - BERTopic manages its own instance -
#  but keeping the model name identical keeps behavior consistent.)
topic_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def add_topics(chunk_DF, min_topic_size=10):
    """
    Runs BERTopic over chunk_DF['text'] and adds two new columns:
      - topic_id    : integer cluster ID (-1 = outlier, no clear topic)
      - topic_label : human-readable label built from the topic's top words
    """
    print("\n ---- TOPIC MODELING (BERTopic) -----")

    if chunk_DF.empty:
        print("No chunks available for topic modeling.")
        chunk_DF["topic_id"] = []
        chunk_DF["topic_label"] = []
        return chunk_DF

    documents = chunk_DF["text"].tolist()

    topic_model = BERTopic(
        embedding_model=topic_embedding_model,
        min_topic_size=min_topic_size,
        verbose=False,
    )
    topics, probs = topic_model.fit_transform(documents)

    chunk_DF["topic_id"] = topics

    # Build a readable label per topic from its top 3 words
    topic_info = topic_model.get_topic_info()
    id_to_label = {}
    for _, row in topic_info.iterrows():
        tid = row["Topic"]
        if tid == -1:
            id_to_label[tid] = "outlier"
        else:
            top_words = topic_model.get_topic(tid)
            id_to_label[tid] = ", ".join([w for w,_ in top_words[:3]])

    chunk_DF["topic_label"] = chunk_DF["topic_id"].map(id_to_label)

    n_topics = len(topic_info[topic_info["Topic"] != -1])
    n_outliers = int((chunk_DF["topic_id"] == -1).sum())
    print(f"Topics discovered: {n_topics}")
    print(f"Outlier chunks (-1): {n_outliers} / {len(chunk_DF)}")
    print(topic_info.head(10))

    topic_model.save("models/bertopic_model", serialization="pickle")

    return chunk_DF