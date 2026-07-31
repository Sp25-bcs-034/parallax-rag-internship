from datasets import load_dataset
import pandas as pd
from clean_data import cleaning  
from chunking import chunking
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
chunk_DF=chunking(splitter , train_Df)
# Uncomment the line below to create a csv file 
#chunk_DF.to_csv("data/chunks/chunks.csv" , index = False ) 
# embedding 
from embedding import embedd
from sentence_transformers import SentenceTransformer
#model = SentenceTransformer("all-MiniLM-L6-v2")
print ( "  ....Embedding .....")
embedded_DF = embedd(chunk_DF )
# chormadb 
from chromaDb import storing
print ( "  ....storing to chromadb ......")
collection = storing(embedded_DF)
# semantic search 
from semantic_Search import semnatic_Search , retrival_bechmark
print ( "  .... Semantic Search ......")
semnatic_Search(collection)
retrival_bechmark(collection)