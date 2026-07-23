import pandas as pd
def cleaning (df ):
    df = df.drop_duplicates()
    df= df.dropna()
    df["text_length"] = df["text"].str.len()
    return df 
    
    