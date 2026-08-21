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
        
def chatting (collection ):
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
            
        
        Retrieval_time = retrieval_end_time -  retrieval_start_time
        generation_time = generation_end_time - generation_start_time
        print ("Generation latency :"  ,generation_time )
        print ("Retrieval latency :"  ,Retrieval_time )
        total_time = generation_time + Retrieval_time
        print ("Total latency :"  ,total_time )
    
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
def fastapi_question_chat (collection , question ):
    
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
        return { "message" : "Error occurred while querying the collection."}

    retrieval_end_time = time.time()
    if not results["documents"][0]:
        print("No relevant documents found.")
        return { " message " : "No relevant documents found."}
    
    retrived_chunks = results["documents"][0]
    context = "\n\n".join(retrived_chunks)
    generation_start_time = time.time()
    try :
        final_ans = calling_Api(text, context)
    except Exception as e:
        print(f"Error occurred while generating the answer: {e}")
        return { "message" : "Error occurred while generating the answer."}
    generation_end_time = time.time()
    print("--------------------")
    print("Final Answer:\n")
    if final_ans is None :
        print("No answer generated.")
        return { "message" : "No answer generated."}
    else:
        print(final_ans)
        return { "message" : final_ans}
            
        
    Retrieval_time = retrieval_end_time -  retrieval_start_time
    generation_time = generation_end_time - generation_start_time
    print ("Generation latency :"  ,generation_time )
    print ("Retrieval latency :"  ,Retrieval_time )
    total_time = generation_time + Retrieval_time
    print ("Total latency :"  ,total_time )
        
    
def precision_recall_at_k(collection, model, test_cases, k=5):
    precision_scores = []
    recall_scores = []

    for case in test_cases:
        query_embedding = model.encode(case["query"]).tolist()
        results = collection.query(query_embeddings=[query_embedding], n_results=k)

        retrieved_ids = set(results["ids"][0])
        relevant_ids = case["relevant_ids"]

        true_positives = retrieved_ids & relevant_ids

        precision = len(true_positives) / k
        recall = len(true_positives) / len(relevant_ids)

        precision_scores.append(precision)
        recall_scores.append(recall)

        print(f"Query: '{case['query']}'")
        print(f"  Retrieved: {retrieved_ids}")
        print(f"  Relevant:  {relevant_ids}")
        print(f"  Precision@{k}: {precision:.2f} | Recall@{k}: {recall:.2f}\n")

    avg_precision = sum(precision_scores) / len(precision_scores)
    avg_recall = sum(recall_scores) / len(recall_scores)

    print(f"Average Precision@{k}: {avg_precision:.2f}")
    print(f"Average Recall@{k}: {avg_recall:.2f}")

    return avg_precision, avg_recall            