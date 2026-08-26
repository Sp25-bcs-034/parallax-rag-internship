from fastapi import FastAPI  , HTTPException
from pydantic import BaseModel
from typing import List
from load_dataset import get_collection 
from semantic_Search import fastapi_question_chat
import logging
import time

logging.basicConfig(
    filename="api.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
app = FastAPI()
class question(BaseModel): 
    question : str
    n_results : int = 3
     
@app.get("/")
def read_root():
    return {"message" : "ey this is Imara's RAG application"}
        
@app.post("/ask")
def start_Chatting (  question_request : question ):
    start_time = time.time()
    logger.info(f"Recieved question: {question_request.question} with n_results: {question_request.n_results}")
    collection = get_collection()
    if not collection:
        raise HTTPException(status_code=500, detail="Failed to load the collection.")
    if not question_request.question :
        logger.warning("Rejected empty question.")
        raise HTTPException(status_code=400, detail="Question is required.")
    if question_request.n_results <= 0:
        logger.warning("Rejected invalid n_results.")
        raise HTTPException(status_code=400, detail="n_results must be a positive integer.")
    return fastapi_question_chat (collection , question_request)
    
   
    