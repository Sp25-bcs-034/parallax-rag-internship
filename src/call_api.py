import os 
import requests 
import json
from dotenv import load_dotenv
load_dotenv()
api_key = os.environ.get("OPENROUTER_API_KEY")

from typing import Optional

def calling_Api(question: str, context: str, api_key= api_key) -> Optional[str]:
    """Send a question and retrieved context to the LLM via OpenRouter.

    Args:
        question: The user's original question.
        context: Retrieved document chunks joined into a single string.
        api_key: OpenRouter API key, defaults to the value loaded from .env.

    Returns:
        The generated answer text, or None if the request failed.
    """
    ...
    if api_key is None:
        print("API key not found.")
        return None


    url ="https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization":  f"Bearer {api_key}",
    "Content-Type": "application/json",
    }
    system_sms= "You are a helpful AI assistant. Answer ONLY using the provided context.If the answer is not present in the context .reply exactly: I don't know.Do not make up information."

    user_message = f"Context:\n{context}\n\nQuestion:\n{question}"
    payload = {
        "model": "google/gemma-4-31b-it:free",
        "messages": [
    {
        "role": "system",
        "content": system_sms
    },
    {
        "role": "user",
        "content": user_message
    }
]
    }
    try :
        response = requests.post(url, headers=headers , json = payload  , timeout=30)
    except requests.exceptions.Timeout:
        print("Request timed out. Please try again later.")
        return None
    except requests.exceptions.RequestException as e:
            print (f"Error occurred while making the API request: {e}")
            return None    
   
    if (response.status_code == 200):
        data = response.json()
        return data ["choices"][0]["message"]["content"]
    elif (response.status_code == 401):
        print("Unauthorized access. Check your API key.")
        return None
    elif (response.status_code == 429):
        print("Rate limit exceeded. Please try again later.")
        return None
    elif (response.status_code == 413):
        print("Request entity too large. Please reduce the size of your request.")
        return None
    elif (response.status_code == 500):
        print("Internal server error. Please try again later.")
        return None
    else:
        print (f"Unexpected error: {response.status_code}")
        print(response.text)
        return None
print ("api calling module loaded successfully")    