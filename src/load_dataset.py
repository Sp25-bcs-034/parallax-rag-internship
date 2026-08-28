#from datasets import load_dataset
"""Utilities for loading the dataset and accessing the ChromaDB collection."""
import pandas as pd
from clean_data import cleaning  
from chunking import chunking

collection  = None 
chunk_DF = None 
def the_pipeline (): 
    train_Df = pd.read_csv("data/raw/ag_news.csv")
    #train_Df.to_csv("data/raw/ag_news.csv", index=False)
    train_Df = cleaning(train_Df)
    print (" ---- EDA -----")
    print ( " First 5  rows \n",train_Df.head())
    print ()
    print ("Info ",train_Df.info())
    print ()
    print (train_Df.describe())
    print ()
    print ("Shapes ",train_Df.shape)
    print ()
    print ("Coloumns ",train_Df.columns)
    print ()
    print ( "Value counts " ,train_Df["label"].value_counts())
    print ()
    print (train_Df.isnull().sum())
    print ("\n ---- CLEANING  -----")
    print ( "Number of duplicate " , train_Df.duplicated().sum())
    print ("Number of missin values " ,  train_Df.isnull().sum())
    print (train_Df.dtypes)
    print ("\n text length :  ")
    #train_Df["text_length"] = train_Df["text"].str.len()
    print (train_Df["text_length"].describe()) 
    print("Encoding Check")
    try:
        train_Df["text"].str.encode("utf-8")
        print("No encoding issues found.")
    except Exception as e:
        print(e)
    train_Df.to_csv("data/cleaned/clean_ag_news.csv" , index = False )
#print (pd.read_csv("data/cleaned/clean_ag_news.csv"))
# chunking 
    from langchain_text_splitters import  RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size= 350 , chunk_overlap = 50 )
    global chunk_DF
    chunk_DF=chunking(splitter , train_Df)
# Uncomment the line below to create a csv file 
#chunk_DF.to_csv("data/chunks/chunks.csv" , index = False ) 
    from topic_modeling import add_topics
    from nlp_analysis import add_sentiment, validate_edge_cases

    print ( "  ....Topic Modeling.....")
    chunk_DF = add_topics(chunk_DF)

    print ( "  ....Sentiment Analysis.....")
    chunk_DF = add_sentiment(chunk_DF)

    validate_edge_cases(chunk_DF)




# embedding 
    from embedding import embedd
    from sentence_transformers import SentenceTransformer
#model = SentenceTransformer("all-MiniLM-L6-v2")
    print ( "  ....Embedding .....")
    embedded_DF = embedd(chunk_DF )

# chromadb 
    from chromaDb import storing
    print ( "  ....storing to chromadb ......")
    global collection
    collection = storing(embedded_DF)

    

def get_chunk_Df ():
    return chunk_DF
import chromadb 
DB_PATH = "database/chroma_db"
COLLECTION_NAME = "ag_news_vector_collection"



def existing_collection():
    global collection
    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        collection = client.get_collection(COLLECTION_NAME)
        print("Collection does exist.")
        return collection
    except Exception:
        print("Collection does not exist.")
        return None

def get_collection():
    global collection
    if collection is None:
        collection = existing_collection()
    return collection    

the_pipeline()