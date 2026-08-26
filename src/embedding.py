import pandas as pd 
import time 
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
"""Generate document and query embeddings using a Sentence Transformer model."""
def embedd (chunk_train_Df ):
    start_time = time.time()
    text_list_input =chunk_train_Df["text"].tolist()
    label_list_input=  chunk_train_Df["label"].tolist()
    text_vector_output_list = model.encode(text_list_input, show_progress_bar=True) # this is embedded list 
    embedding_list =text_vector_output_list.tolist()
    print (text_vector_output_list.shape)
    print (text_vector_output_list[:5])
    data_ ={
        "text": text_list_input,
        "embedding": embedding_list,
        "label": label_list_input
    }
    # NEW: carry NLP metadata columns through if they exist
    for optional_col in ["topic_id", "topic_label", "sentiment", "sentiment_score"]:
        if optional_col in chunk_train_Df.columns:
           data_[optional_col] = chunk_train_Df[optional_col].tolist()
    updated_ebedded_dataframe = pd.DataFrame(data_)
    end_time = time.time()
    total_time = end_time- start_time
    print ( " Total time in embedding = " , total_time)
    return updated_ebedded_dataframe
def embeded_for_query(text: str) -> list[float]:
    """Generate an embedding vector for a user query.

    Args:
        text: Query text that should be converted into an embedding.

    Returns:
        A numerical embedding vector representing the query.
    """
    query_embede_list = model.encode(text).tolist()
    return query_embede_list
       