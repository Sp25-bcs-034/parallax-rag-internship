import pandas as pd 
from langchain_text_splitters import  RecursiveCharacterTextSplitter
def chunking ( splitter , train_Df):
    print ("\n ---- CHUNKING  -----")
    page_content = train_Df["text"].tolist()
    metadata = train_Df[["label"]].to_dict(orient ="records")
    train_Df_doc = splitter.create_documents(page_content, metadatas= metadata )
    print("----------------")
    print("Total Chunks Created\n")
    print (len(train_Df_doc))
    print("----------------")
    print("First  Chunk  \n")
    print (train_Df_doc[0].page_content)
    print("----------------")
    print(" First  Metadata  \n")
    print(train_Df_doc[0].metadata)
    print("----------------")
    print(" length of first chunk   \n")
    print(len(train_Df_doc[0].page_content))
    print("----------------")
    print("First  3 Chunk  \n")
    for i in range(3):
        print("----------------")
        print(train_Df_doc[i].page_content)
    list_text =[]
    for doc in (train_Df_doc):
        list_text.append(doc.page_content)  
    list_metadata =[]
    for doc in  train_Df_doc:
        list_metadata.append (doc.metadata["label"])  
    data = { 
    "text" : list_text,
    "label": list_metadata }    
    chunk_DF = pd.DataFrame(data)   
    print("----------------")
    print(" chunk DF   \n")
    print (chunk_DF)
    return chunk_DF
    # Uncomment the line below to create a csv file 
    #chunk_DF.to_csv("data/chunks/chunks.csv" , index = False ) 

