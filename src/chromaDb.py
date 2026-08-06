import chromadb 
def storing (embeded_Df):
    if embeded_Df.empty:
        print("No embeddings available.")
        return 
    client = chromadb.PersistentClient(path = "database/chroma_db") 
    collection = client.get_or_create_collection(name = "ag_news_vector_collection")
    chunk_ids = [f"chunk_{i}" for i in range (len(embeded_Df["text"]))]
    metadata = [{"label": label }for label in embeded_Df["label"]]
    collection.add(
        ids= chunk_ids ,
        documents = embeded_Df["text"],
        embeddings= embeded_Df["embedding"],
        metadatas= metadata 
    )
    return collection 
    
    
    