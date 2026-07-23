import pandas as pd
def cleaning (df ):
    df = df.drop_duplicates()
    df= df.dropna()
    df["text"] = df["text"].astype(str)
    df = df[df["text"].str.strip() != ""]
    df["text_length"] = df["text"].str.len()
    print("Duplicates removed successfully")
    print("Missing values removed successfully")
    return df 
    
    