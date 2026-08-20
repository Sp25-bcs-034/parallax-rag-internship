from load_dataset import get_collection , get_chunk_Df  ,  the_pipeline , existing_collection
def fetch_collection ():
    if not existing_collection():
        the_pipeline()
        get_collection()
    else:
        return existing_collection()
collection = fetch_collection()
# semantic search 
from semantic_Search import semnatic_Search , retrival_bechmark , chatting
collection = get_collection()
chunk_DF = get_chunk_Df()
print ( "  .... Semantic Search ......")
semnatic_Search(collection)
retrival_bechmark(collection)
chatting (collection)
