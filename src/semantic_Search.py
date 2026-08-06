from api_calling import calling_Api
from embedding import  embeded_for_query 
import time 
def semnatic_Search (collection):
    if collection.count()==0:
        print("Database is empty.")
    text = input ("  Ask Any Question ")
    start_time = time.time()
    query_list_embeded = embeded_for_query(text)
    results = collection.query(
        query_embeddings = [query_list_embeded],
        n_results = 2
    )
    
    
   
    end_text=  time.time()
    Retrieval_time = end_text -  start_time
    print (" Semantic Search ")
    print("\nTop Results\n")

    for i in range(len(results["documents"][0])):
        print("--------------------")
        print(results["documents"][0][i])
        print(results["metadatas"][0][i])
        print ("Retrieval latency :"  ,Retrieval_time )
    
def retrival_bechmark (collection):
    if collection.count()==0:
        print("Database is empty.")
        return 
    test_cases = test_cases = [
    {"query":"oil prices",
   "correct_chunk_id":"chunk_15"},
    { "query":"football",
   "correct_chunk_id":"chunk_950"},
    {   "query":"NASA mission",
   "correct_chunk_id":"chunk_200"}
]
    hits=0
    for case in test_cases:
        query_list_embeded = embeded_for_query(case ["query"])
        results = collection.query(
            query_embeddings  = [query_list_embeded],
            n_results = 3  
        )
        retrieved_ids = results["ids"][0]
        if case["correct_chunk_id"]  in retrieved_ids :
            hits+=1
    Recall_at_3 = hits/ len(test_cases)
    print ( " .....Retrieval Bechmark .....")
    print ("Recall@3:" , Recall_at_3)
        
def chatting (collection):
    if collection.count ()==0:
        print("Database is empty.")
        return
    print ("......  Chatting Mode  .....")
    while True :
        text =input("Ask Any Question (type 'exit' to quit): ")
        if text.lower() == 'exit':
            break
        retrieval_start_time = time.time()
        # embed the query and retrieve relevant chunks from the collection
        query_list_embeded = embeded_for_query(text)
        results = collection.query(
            query_embeddings = [query_list_embeded],
            n_results = 2
        )
        retrieval_end_time = time.time()
        if  results["documents"][0] ==0:
            print("No relevant documents found.")
            return
        retrived_chunks = results["documents"][0]
        context = "\n\n".join(retrived_chunks)
        generation_start_time = time.time()
        final_ans = calling_Api(text, context)
        generation_end_time = time.time()
        print("--------------------")
        print("Final Answer:\n")
        if final_ans is None :
            print("No answer generated.")
        else:
            print(final_ans)
            
        
        Retrieval_time = retrieval_end_time -  retrieval_start_time
        generation_time = generation_end_time - generation_start_time
        print ("Generation latency :"  ,generation_time )
        print ("Retrieval latency :"  ,Retrieval_time )
        total_time = generation_time + Retrieval_time
        print ("Total latency :"  ,total_time )
    
    