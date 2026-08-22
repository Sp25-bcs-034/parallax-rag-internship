# Parallax Labs AI/ML Internship

> **Project:** Retrieval-Augmented Generation (RAG) System  
> **Internship:** Parallax Labs AI/ML Internship  
> **Dataset:** AG News

---

# 🚀 PROJECT START

## Project Overview

This project is part of the Parallax Labs AI/ML Internship.

The goal of this internship project is to progressively build a complete Retrieval-Augmented Generation (RAG) system, starting from raw dataset preparation and moving toward an API-based AI application with retrieval, NLP analysis, LLM generation, evaluation, and testing.

The project covers:

- Environment setup
- Dataset acquisition
- Exploratory Data Analysis (EDA)
- Data cleaning and validation
- Text chunking
- Embedding generation
- Vector database integration
- Semantic search
- LLM integration
- Prompt engineering
- NLP analysis
- Sentiment-based filtered retrieval
- FastAPI integration
- Retrieval evaluation
- Generation evaluation
- Latency evaluation
- API testing

---
parallax-rag-internship/

│
├── data/
│   ├── raw/
│   │   └── ag_news.csv
│   │
│   └── cleaned/
│       └── clean_ag_news.csv
│
├── src/
│   ├── load_dataset.py
│   ├── clean_data.py
│   └── verify_environment.py
│
├── tests/
│   └── test_clean_data.py
│
├── requirements.txt
├── README.md
└── .gitignore

# Week 1 – Environment Setup, Data Acquisition & Cleaning

## Objectives & Tasks

- Set up the Python development environment.
- Download and inspect a large real-world dataset.
- Perform Exploratory Data Analysis (EDA).
- Clean and validate the dataset.
- Generate a data quality report.
- Save the cleaned dataset for future RAG processing.

---

## Dataset

### Dataset: AG News

Source: Hugging Face Datasets

Number of Documents:

- Train: 120,000
- Test: 7,600

Columns:

- `text`
- `label`

The AG News dataset contains news articles belonging to different news categories and is used as the source corpus for the RAG system.

---

## Project Structure

Features Implemented
Environment Setup
Created Python virtual environment
Installed required libraries
Verified package installation
Dataset Loading

Downloaded the AG News dataset using Hugging Face Datasets and converted it into a Pandas DataFrame.

Exploratory Data Analysis (EDA)

Performed:

Display first five rows
Dataset information
Statistical summary
Shape
Column names
Label distribution
Missing value analysis
Data Cleaning

Implemented cleaning functions to:

Remove duplicate rows
Remove missing values
Remove empty text
Calculate text length
Data Quality Report

Generated:

Missing values report
Duplicate report
Data types
Label distribution
Text length statistics
Output

Generated:

data/raw/ag_news.csv

and:

data/cleaned/clean_ag_news.csv

The cleaned dataset was prepared for the next stage of the project: text chunking and embedding generation.

Technologies Used
Python
Pandas
Hugging Face Datasets
How to Run
Install dependencies
pip install -r requirements.txt
Verify environment
python src/verify_environment.py
Run data loading
python src/load_dataset.py
Week 1 Deliverables
Environment configured
Dataset acquired
Data validated
Data cleaned
Data quality report generated
Clean dataset exported
Week 2 – Chunking, Embeddings & Vector Database
Objectives & Tasks
Implement a text chunking strategy.
Generate embeddings using Sentence Transformers.
Measure embedding generation performance.
Set up ChromaDB.
Store chunks and embeddings in the vector database.
Implement semantic similarity search.
Measure retrieval latency.
Add unit tests for the implemented components.
Completed
Implemented RecursiveCharacterTextSplitter
Chunk Size = 350
Chunk Overlap = 50
Generated embeddings using SentenceTransformer (all-MiniLM-L6-v2)
Logged embedding generation time
Stored embeddings inside ChromaDB Persistent Client
Implemented semantic similarity search
Measured retrieval latency
Added unit tests for:
Data cleaning
Chunking
Embedding
ChromaDB
RAG Data Flow
Clean Dataset
      ↓
Text Chunking
      ↓
Embedding Generation
      ↓
Vector Embeddings
      ↓
ChromaDB
      ↓
Semantic Search
Week 3 – LLM Integration & Prompt Engineering
Objectives & Tasks
Taking the work from Week 2, integrate the DeepSeek/OpenRouter API to generate answers from retrieved chunks.
Implement prompt engineering best practices.
Implement system prompts and context injection.
Add robust API error handling.
Handle rate limits, timeouts, and token limitations.
Implement hallucination checks.
Handle out-of-domain and off-topic queries.
Measure end-to-end response latency.
Completed
Integrated the DeepSeek/OpenRouter API
Connected the LLM with the existing RAG pipeline
Passed retrieved chunks as context to the LLM
Implemented prompt construction
Added system prompt instructions
Implemented context injection
Added API error handling
Handled unsuccessful API responses
Added handling for hallucination and out-of-domain queries
Measured retrieval latency
Measured generation latency
Measured total response latency
Implemented RAG chatting mode
RAG Generation Flow
User Question
      ↓
Query Embedding
      ↓
ChromaDB Semantic Search
      ↓
Relevant Chunks
      ↓
Context Construction
      ↓
Question + Context
      ↓
DeepSeek/OpenRouter
      ↓
Generated Answer
Week 4 – NLP Analysis (Topic & Sentiment)
Objectives & Tasks
Taking the work from Week 3, apply NLP analysis to the corpus.
Apply topic modeling / NLP techniques to discover corpus information.
Implement sentiment analysis or NER.
Evaluate NLP results.
Validate outputs manually.
Handle edge cases such as short documents and jargon.
Integrate NLP metadata into the vector database.
Implement filtered retrieval.
Document the effectiveness of the extracted NLP metadata.
Completed
Implemented NLP analysis
Added sentiment analysis
Generated sentiment metadata
Stored sentiment information as ChromaDB metadata
Implemented metadata-based filtered retrieval
Added support for:
POSITIVE
NEGATIVE
Implemented semantic search with sentiment filtering
Validated retrieved documents and metadata
Added handling for an empty vector database
Integrated NLP metadata into the retrieval workflow
Filtered Retrieval Flow
User Question
      ↓
Query Embedding
      ↓
ChromaDB
      ↓
Sentiment Metadata Filter
      ↓
Relevant Chunks
      ↓
Retrieved Context
      ↓
LLM
      ↓
Answer
Week 5 – FastAPI, Retrieval Evaluation & Testing
Objectives & Tasks
Taking the work from Week 4, wrap the RAG system in a FastAPI application with proper endpoints.
Implement structured request logging.
Implement proper HTTP error responses:
400
422
500
Write a retrieval evaluation script calculating Precision@K and Recall@K on a test set.
Write an automated end-to-end evaluation script for generation quality and latency.
Write unit tests for the FastAPI endpoints using pytest.
Week 5 Completed
FastAPI Integration

The existing RAG pipeline was wrapped inside a FastAPI application.

FastAPI acts as the API layer between the client and the RAG pipeline.

The architecture is:

Client
   ↓
FastAPI
   ↓
Request Validation
   ↓
RAG Pipeline
   ↓
Embedding
   ↓
ChromaDB Retrieval
   ↓
Context Construction
   ↓
LLM Generation
   ↓
API Response
FastAPI Application

Created a FastAPI application using:

from fastapi import FastAPI

The application provides API endpoints for interacting with the RAG system.

Pydantic Request Model

Implemented structured request validation using Pydantic:

class QuestionRequest(BaseModel):
    question: str
    n_results: int = 3

The request contains:

question – the user's question
n_results – number of relevant chunks to retrieve

Example:

{
    "question": "What is the latest oil price?",
    "n_results": 3
}
API Endpoints
GET /

Used to verify that the API application is running.

Example response:

{
    "message": "Hey, this is Imara's RAG application"
}
POST /ask

Used to send a question to the RAG system.

The endpoint:

Receives the user's question.
Validates the request.
Gets the ChromaDB collection.
Creates the query embedding.
Performs semantic retrieval.
Retrieves relevant chunks.
Builds the context.
Sends the question and context to the LLM.
Returns the generated answer.
FastAPI RAG Flow
POST /ask
     ↓
QuestionRequest
     ↓
Get ChromaDB Collection
     ↓
Extract Question
     ↓
Create Query Embedding
     ↓
Semantic Search
     ↓
Retrieve Top-K Chunks
     ↓
Build Context
     ↓
Call LLM API
     ↓
Generate Answer
     ↓
Return JSON Response
Retrieval Evaluation

A retrieval benchmark was implemented to evaluate the performance of semantic search.

The benchmark uses predefined test cases containing:

Query
Expected chunk ID

Example:

{
    "query": "oil prices",
    "correct_chunk_id": "chunk_15"
}

Other test queries include:

football
NASA mission
Recall@K

The retrieval benchmark checks whether the expected chunk is present inside the retrieved top-K results.

Recall@K is calculated as:

Recall@K =
Number of successful retrievals
--------------------------------
Total number of test cases

The benchmark reports the retrieval performance using Recall@3.

Precision@K Evaluation

Retrieval evaluation was extended to measure Precision@K.

Precision@K evaluates how many of the retrieved results are relevant to the query.

Conceptually:

Precision@K =
Relevant Retrieved Results
--------------------------
Total Retrieved Results

This provides another way of evaluating the quality of the vector database retrieval.

Retrieval Latency

The system measures retrieval performance using timestamps.

The retrieval process measures the time taken to:

Create the query embedding
Search ChromaDB
Retrieve relevant chunks

The retrieval latency is calculated as:

Retrieval Latency =
Retrieval End Time - Retrieval Start Time
Generation Evaluation

An automated end-to-end evaluation process was implemented to evaluate the generation stage of the RAG pipeline.

The evaluation considers:

Retrieved context
Generated answer
Response quality
Generation latency
Overall response latency
End-to-End Latency

The system measures both retrieval and generation latency.

Retrieval Latency
       +
Generation Latency
       =
Total Latency

The generation latency is measured using:

Generation Latency =
Generation End Time - Generation Start Time

The total latency is measured using the retrieval and generation times.

Request Logging

Structured request logging was implemented to help monitor API activity.

The logging system records important information related to API requests and processing.

This helps with:

Debugging
Monitoring
Performance analysis
Troubleshooting
Tracking API behavior
HTTP Error Handling

The API handles common HTTP errors and invalid requests.

Implemented error categories include:

400 – Bad Request

Used when the request is invalid or cannot be processed because of incorrect client input.

422 – Validation Error

Used when the request does not match the required Pydantic schema.

For example, if the API expects:

{
    "question": "What is RAG?",
    "n_results": 3
}

but receives an invalid data type, FastAPI/Pydantic can return a validation error.

500 – Internal Server Error

Used when an unexpected server-side error occurs while processing the request.

FastAPI Testing

Unit tests were added using pytest to test the FastAPI endpoints.

Testing covers:

Root endpoint
/ask endpoint
Request validation
API responses
Error handling
RAG API integration
Complete RAG System

After completing Week 5, the project contains the complete RAG workflow:

                         AG NEWS DATASET
                                ↓
                         DATA CLEANING
                                ↓
                           CHUNKING
                                ↓
                         EMBEDDINGS
                                ↓
                           CHROMADB
                                ↓
                      SEMANTIC RETRIEVAL
                                ↓
                       NLP METADATA
                                ↓
                    FILTERED RETRIEVAL
                                ↓
                         USER QUESTION
                                ↓
                        QUERY EMBEDDING
                                ↓
                       TOP-K RETRIEVAL
                                ↓
                         CONTEXT BUILDING
                                ↓
                    DEEPSEEK / OPENROUTER
                                ↓
                       GENERATED ANSWER
                                ↓
                            FASTAPI
                                ↓
                           API RESPONSE
Complete Internship Progress
Week 1 – Environment Setup, Data Acquisition & Cleaning
Completed
Environment setup
AG News dataset acquisition
Exploratory Data Analysis
Data cleaning
Data validation
Data quality reporting
Clean dataset generation
Week 2 – Chunking, Embeddings & Vector Database
Completed
Recursive text chunking
SentenceTransformer embeddings
Embedding performance measurement
ChromaDB integration
Persistent vector database
Semantic search
Retrieval latency measurement
Unit testing
Week 3 – LLM Integration & Prompt Engineering
Completed
DeepSeek/OpenRouter API integration
Prompt engineering
System prompts
Context injection
API error handling
Hallucination handling
Out-of-domain query handling
Retrieval latency
Generation latency
End-to-end latency
RAG chatting mode
Week 4 – NLP Analysis
Completed
NLP analysis
Sentiment analysis
Sentiment metadata
ChromaDB metadata integration
Sentiment-based filtered retrieval
Metadata validation
Empty database handling
Week 5 – FastAPI, Retrieval Evaluation & Testing
Completed
FastAPI application
GET / endpoint
POST /ask endpoint
Pydantic request validation
RAG API integration
ChromaDB retrieval through API
Query embedding
Top-K retrieval
Context construction
LLM integration
Structured request logging
HTTP error handling
400 error handling
422 validation handling
500 error handling
Retrieval benchmarking
Precision@K
Recall@K
Generation-quality evaluation
Retrieval latency measurement
Generation latency measurement
End-to-end latency evaluation
pytest API tests
Technologies Used
Python
Pandas
Hugging Face Datasets
Sentence Transformers
ChromaDB
FastAPI
Pydantic
DeepSeek
OpenRouter
pytest
NLP / Sentiment Analysis
Git
GitHub
 
Project Learning Outcomes

Through the internship, the project progressed from basic data preparation to a complete AI-powered RAG application.

The project demonstrates understanding and implementation of:

Data preprocessing
Dataset analysis
Text chunking
Vector embeddings
Vector databases
Semantic search
Retrieval-Augmented Generation
Prompt engineering
LLM APIs
NLP metadata
Metadata filtering
FastAPI
REST API development
Request validation
Error handling
Retrieval evaluation
Generation evaluation
Latency measurement
Automated testing



