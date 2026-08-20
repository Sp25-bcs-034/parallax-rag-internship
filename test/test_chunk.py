import pandas as pd
from src.clean_data import cleaning 
from src.chunking import chunking
from  langchain_text_splitters import RecursiveCharacterTextSplitter
def test_Chunk ():
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=10, chunk_overlap = 2 )
    df = pd.DataFrame ( { "text" :["hello","",None,"hello"]
                             , "label":[1,2,3,1]})
    df = cleaning(df)
    chunk_df =chunking(splitter ,df)
    assert len(chunk_df) > 0
    assert "text" in chunk_df.columns
    assert "label" in chunk_df.columns
    assert chunk_df["text"].notnull().all()
    assert chunk_df["label"].notnull().all()
    