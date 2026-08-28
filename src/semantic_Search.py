from embedding import  embeded_for_query 
from call_api import calling_Api 
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
    
    
    
    
def chatting(collection, text: str | None = None) -> str | None:
    """Run the RAG pipeline for a user query.

    Args:
        collection: ChromaDB collection used for retrieval.
        text: Optional query. If omitted, interactive input is used.

    Returns:
        The generated answer, or None if no answer was generated.
    """
    if collection.count ()==0:
        print("Database is empty.")
        return
    print ("......  Chatting Mode  .....")
    final_Ans = None
    while True :
        if ( text == None) :
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
        if not results["documents"][0]:
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
            final_Ans = final_ans
            
        
        Retrieval_time = retrieval_end_time -  retrieval_start_time
        generation_time = generation_end_time - generation_start_time
        print ("Generation latency :"  ,generation_time )
        print ("Retrieval latency :"  ,Retrieval_time )
        total_time = generation_time + Retrieval_time
        print ("Total latency :"  ,total_time )
    return final_Ans    
    
    
    
def filtered_search(collection):
    if collection.count() == 0:
        print("Database is empty.")
        return

    text = input("Ask a question: ")
    sentiment_filter = input("Filter by sentiment (POSITIVE/NEGATIVE/skip): ").strip().upper()

    query_list_embeded = embeded_for_query(text)

    where_clause = None
    if sentiment_filter in ("POSITIVE", "NEGATIVE"):
        where_clause = {"sentiment": sentiment_filter}

    results = collection.query(
        query_embeddings=[query_list_embeded],
        n_results=3,
        where=where_clause,
    )

    print("\nFiltered Results\n")
    for i in range(len(results["documents"][0])):
        print("--------------------")
        print(results["documents"][0][i])
        print(results["metadatas"][0][i])    
from fastapi import HTTPException        



def fastapi_question_chat(collection, question) -> dict:
    """Run the RAG pipeline for a FastAPI request and return a JSON-safe result.

    Args:
        collection: ChromaDB collection to query for relevant chunks.
        question: Pydantic request object with `.question` and `.n_results`.

    Returns:
        A dictionary containing the generated answer and latency metrics.

    Raises:
        HTTPException: If retrieval or generation fails, or no answer is produced.
    """
    ...
    
    if collection.count ()==0:
        print("Database is empty.")
        return { "message" : "Database is empty."}
    text = question.question
    n_results = question.n_results
    retrieval_start_time = time.time()
    # embed the query and retrieve relevant chunks from the collection
    try :
        query_list_embeded = embeded_for_query(text)
        results = collection.query(
            query_embeddings = [query_list_embeded],
            n_results = n_results
        )
    except Exception as e:
        print(f"Error occurred while querying the collection: {e}")
        raise HTTPException(status_code=500, detail="Error occurred while querying the collection.")

    retrieval_end_time = time.time()
    if not results["documents"][0]:
        print("No relevant documents found.")
        raise HTTPException(status_code=404, detail="No relevant documents found.")

    retrived_chunks = results["documents"][0]
    context = "\n\n".join(retrived_chunks)
    generation_start_time = time.time()
    try :
        final_ans = calling_Api(text, context)
    except Exception as e:
        print(f"Error occurred while generating the answer: {e}")
        raise HTTPException(status_code=500, detail="Error occurred while generating the answer.")
    generation_end_time = time.time()
    print("--------------------")
    print("Final Answer:\n")
    Retrieval_time = retrieval_end_time -  retrieval_start_time
    generation_time = generation_end_time - generation_start_time
    print ("Generation latency :"  ,generation_time )
    print ("Retrieval latency :"  ,Retrieval_time )
    total_time = generation_time + Retrieval_time
    print ("Total latency :"  ,total_time )
    if final_ans is None :
        print("No answer generated.")
        raise HTTPException(status_code=404, detail="No answer generated.")
    else:
        print(final_ans)
        return { "message" : final_ans}
            
        
  
        
    
def  retrival_ (collection , k=3 ):
    test_Case =[ {"query": "what is rag " , "correct_chunk_id": ["chunk_1"]},
                {"query": "what is the best model for rag" , "correct_chunk_id": ["chunk_2"]}]
    precision_scores = []
    recall_scores = []
    retrival_start_time = time.time()
    for case in test_Case:
        query_list_embedded = embeded_for_query(case["query"])
        results = collection.query(
            query_embeddings=[query_list_embedded],
            n_results=k
        )
        result_chunk_ids =  results["ids"][0]
        truth_Ans = set(case["correct_chunk_id"]) & set(result_chunk_ids)
        Recall_k = len (truth_Ans) / len (case["correct_chunk_id"])
        recall_scores.append(Recall_k)
        print(f"Query: {case['query']}, Recall@k: {Recall_k}")
        if len(result_chunk_ids) !=0  :
            precisionK = len(truth_Ans) / k
            print(f"Query: {case['query']}, Precision@k: {precisionK}")
            precision_scores.append(precisionK)
    retival_end_time = time.time()
    retrieval_time = retival_end_time - retrival_start_time
    if len(precision_scores )  !=0 :
        avg_precision = sum(precision_scores) / len(precision_scores) 
        print(f"Average Precision@k: {avg_precision}")
    if len(recall_scores) != 0:
        avg_recall = sum(recall_scores) / len(recall_scores)
        print(f"Average Recall@k: {avg_recall}")   
    print(f"Total evaluation time: {retrieval_time:.2f}s")   
     

     