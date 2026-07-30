from datasets import load_dataset
import pandas as pd
from clean_data import cleaning 
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
print ("\n ---- CHUNKING  -----")
from langchain_text_splitters import  RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size= 350 , chunk_overlap = 50 )
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
# Uncomment the line below to create a csv file 
#chunk_DF.to_csv("data/chunks/chunks.csv" , index = False ) 

