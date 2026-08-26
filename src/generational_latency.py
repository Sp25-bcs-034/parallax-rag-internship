from semantic_Search import  chatting
from load_dataset import get_collection
import time
def generational_evaluation (collection ):
    
    result_log=[]
    
    if collection.count()==0:
        print("Database is empty.")
        return
    print ("......  Generational Evaluation Mode  .....")
    test_case = [
    {
        "query": "What is RAG?",
        "keywords": ["retrieval", "generation"]
    },
    {
        "query": "What happened in the NASA mission?",
        "keywords": ["NASA", "mission"]
    },
    {
        "query": "What happened in the football match?",
        "keywords": ["football", "match"]
    },
    {
        "query": "What happened to oil prices?",
        "keywords": ["oil", "prices"]
    }
]
    for case in test_case:
        keyword_found_list =[]
        final_ans = chatting(collection, text=case["query"])
        for keyw in  case["keywords"]:
            if (  keyw in  final_ans ):
                keyword_found_list.append (keyw)
        keyward_score = (len(keyword_found_list) / len (case["keywords"]) ) * 100  
        result_log.append (
            {"query" : case["query"] , "answer" :  final_ans , "keyword_found  " : keyword_found_list, "keyword_score  " :  round(keyward_score  , 2) }
        )
    for r in result_log :
        print (f"Query : {r["query"]}")
        print (f"Answer : {r["answer"]}")
        print (f"keyword found : {r["keyword_found  "]}"  ) 
        print (f"keyword score : {r["keyword_score  "]}")           
    