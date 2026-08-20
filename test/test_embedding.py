from src.embedding import embedd
import pandas as pd

def test_embedding():

    df = pd.DataFrame({
        "text":[
            "Artificial Intelligence",
            "Machine Learning"
        ],
        "label":[0,1]
    })

    embedding_df = embedd(df)

    assert len(embedding_df) == 2
    assert "embedding" in embedding_df.columns
    assert embedding_df["embedding"].notnull().all()

    assert len(embedding_df["embedding"][0]) == 384