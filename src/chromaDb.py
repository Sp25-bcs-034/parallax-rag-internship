import chromadb 
"""Utilities for storing and retrieving embeddings using ChromaDB."""
from typing import Optional
import pandas as pd   
def storing(embeded_Df: pd.DataFrame) -> Optional[chromadb.Collection]:
    """Store embedded chunks and their metadata in a persistent ChromaDB collection.

    Args:
        embeded_Df: DataFrame containing 'text', 'embedding', 'label', and
            optional NLP metadata columns (topic_id, topic_label, sentiment,
            sentiment_score).

    Returns:
        The ChromaDB collection the data was stored in, or None if the
        input DataFrame was empty.
    """    
    if embeded_Df.empty:
        print("No embeddings available.")
        return 
    client = chromadb.PersistentClient(path = "database/chroma_db") 
    collection = client.get_or_create_collection(name = "ag_news_vector_collection")
    chunk_ids = [f"chunk_{i}" for i in range (len(embeded_Df["text"]))]
    
    metadata = []
    for _, row in embeded_Df.iterrows():
        meta = {"label": row["label"]}
        if "topic_id" in embeded_Df.columns:
            meta["topic_id"] = int(row["topic_id"])
        if "topic_label" in embeded_Df.columns:
            meta["topic_label"] = str(row["topic_label"])
        if "sentiment" in embeded_Df.columns:
            meta["sentiment"] = str(row["sentiment"])
        if "sentiment_score" in embeded_Df.columns:
            meta["sentiment_score"] = float(row["sentiment_score"])
        metadata.append(meta)
        
    batch_size = 5000

    for start in range(0, len(embeded_Df), batch_size):

        end = start + batch_size

        batch = embeded_Df.iloc[start:end]

        collection.add(
            ids=chunk_ids[start:end],
            documents=batch["text"].tolist(),
            embeddings=batch["embedding"].tolist(),
            metadatas=metadata[start:end]
    )

        print(f"Stored {start} - {end}")
    return collection 
    
    
    