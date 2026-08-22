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

```text
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
```

---

## Features Implemented

### Environment Setup

- Created Python virtual environment
- Installed required libraries
- Verified package installation

---

### Dataset Loading

Downloaded the AG News dataset using Hugging Face Datasets and converted it into a Pandas DataFrame.

---

### Exploratory Data Analysis (EDA)

Performed:

- Display first five rows
- Dataset information
- Statistical summary
- Shape
- Column names
- Label distribution
- Missing value analysis

---

### Data Cleaning

Implemented cleaning functions to:

- Remove duplicate rows
- Remove missing values
- Remove empty text
- Calculate text length

---

### Data Quality Report

Generated:

- Missing values report
- Duplicate report
- Data types
- Label distribution
- Text length statistics

---

## Output

Generated:

```text
data/raw/ag_news.csv
```

and:

```text
data/cleaned/clean_ag_news.csv
```

The cleaned dataset was prepared for the next stage of the project: text chunking and embedding generation.

---

## Technologies Used

- Python
- Pandas
- Hugging Face Datasets

---

## How to Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Verify environment

```bash
python src/verify_environment.py
```

### Run data loading

```bash
python src/load_dataset.py
```

---

## Week 1 Deliverables

- Environment configured
- Dataset acquired
- Data validated
- Data cleaned
- Data quality report generated
- Clean dataset exported

---

# Week 2 – Chunking, Embeddings & Vector Database

## Objectives & Tasks

- Taking your work from Week 1, implement a text chunking strategy.
- Generate embeddings using Sentence Transformers.
- Log embedding time performance.
- Set up ChromaDB.
- Ingest chunks and embeddings into the vector database.
- Implement semantic search.
- Write a script to test retrieval performance.
- Measure and log retrieval latency.

---

## Completed

- Implemented `RecursiveCharacterTextSplitter`
- Chunk Size = 350
- Chunk Overlap = 50
- Generated embeddings using SentenceTransformer (`all-MiniLM-L6-v2`)
- Logged embedding generation time
- Stored embeddings inside ChromaDB Persistent Client
- Implemented semantic similarity search
- Measured retrieval latency
- Added unit tests for:
  - Data cleaning
  - Chunking
  - Embedding
  - ChromaDB

---

## Week 2 RAG Data Flow

```text
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
```

---

# Week 3 – LLM Integration & Prompt Engineering

## Objectives & Tasks

- Taking your work from Week 2, integrate the DeepSeek/OpenRouter API to generate answers from retrieved chunks.
- Implement prompt engineering best practices.
- Implement system prompts.
- Implement context injection.
- Add robust error handling for API calls.
- Handle rate limits, timeouts, and token limits.
- Implement hallucination checks.
- Handle out-of-domain and off-topic queries.
- Measure and log end-to-end response latency.

---

## Completed

- Integrated the DeepSeek/OpenRouter API
- Connected the LLM with the existing RAG pipeline
- Connected retrieved ChromaDB chunks with the LLM
- Implemented prompt construction
- Implemented system prompt instructions
- Implemented context injection
- Added API error handling
- Added handling for unsuccessful API responses
- Implemented hallucination/out-of-domain handling
- Measured retrieval latency
- Measured generation latency
- Measured total end-to-end latency
- Implemented terminal-based RAG chatting mode

---

## Week 3 RAG Generation Flow

```text
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
```

---

# Week 4 – NLP Analysis (Topic & Sentiment)

## Objectives & Tasks

- Taking your work from Week 3, apply topic modeling (BERTopic/LDA) to discover corpus themes.
- Implement sentiment analysis or NER, evaluating accuracy against a small manually labeled set.
- Validate topic outputs manually and handle edge cases such as short documents and jargon.
- Integrate NLP metadata (topics/sentiment) into the vector database for filtered retrieval.
- Document the effectiveness and accuracy of the extracted NLP metadata.

---

## Completed

- Implemented NLP analysis
- Added sentiment analysis
- Generated sentiment metadata for documents
- Integrated sentiment metadata into ChromaDB
- Implemented filtered semantic search using sentiment metadata
- Added support for:
  - `POSITIVE`
  - `NEGATIVE`
- Tested filtered retrieval using ChromaDB metadata filtering
- Validated retrieved documents and their metadata
- Added handling for an empty ChromaDB collection
- Integrated NLP metadata with the existing RAG retrieval pipeline

---

## Week 4 Filtered Retrieval Flow

```text
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
Generated Answer
```

---

# Week 5 – FastAPI, Retrieval Evaluation & Testing

## Objectives & Tasks

- Taking your work from Week 4, wrap the RAG system in a FastAPI application with proper endpoints.
- Implement structured request logging and proper HTTP error responses.
- Handle HTTP errors:
  - 400
  - 422
  - 500
- Write a retrieval evaluation script calculating Precision@K and Recall@K on a test set.
- Write an automated end-to-end evaluation script for generation quality and latency.
- Write unit tests for the FastAPI endpoints using pytest.

---

# Week 5 – Completed

## FastAPI Integration

The RAG system was wrapped inside a FastAPI application.

FastAPI provides the API layer that allows an external client to communicate with the RAG pipeline.

The API flow is:

```text
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
```

---

## FastAPI Application

Created the FastAPI application using:

```python
from fastapi import FastAPI

app = FastAPI()
```

The application exposes endpoints for interacting with the RAG system.

---

## Pydantic Request Model

Implemented structured request validation using Pydantic:

```python
from pydantic import BaseModel

class Question(BaseModel):
    question: str
    n_results: int = 3
```

The request contains:

- `question` – the user's question
- `n_results` – number of chunks to retrieve

Example request:

```json
{
    "question": "What is the latest oil price?",
    "n_results": 3
}
```

---

## API Endpoints

### GET `/`

Used to check whether the FastAPI application is running.

Example response:

```json
{
    "message": "Hey, this is Imara's RAG application"
}
```

---

### POST `/ask`

Used to send a question to the RAG system.

The endpoint:

1. Receives the user's question.
2. Validates the request using Pydantic.
3. Gets the ChromaDB collection.
4. Generates an embedding for the question.
5. Performs semantic retrieval.
6. Retrieves the most relevant chunks.
7. Combines the chunks into context.
8. Sends the question and context to the LLM.
9. Returns the generated answer.

---

## FastAPI RAG Flow

```text
POST /ask
     ↓
Question Request
     ↓
Pydantic Validation
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
```

---

## Semantic Search Integration

The FastAPI RAG function connects the API layer with the existing semantic search and generation pipeline.

The process is:

```text
Question
   ↓
Embedding
   ↓
ChromaDB Query
   ↓
Retrieved Documents
   ↓
Context
   ↓
LLM
   ↓
Final Answer
```

The number of retrieved documents is controlled using:

```python
n_results = question.n_results
```

This allows the API client to decide how many relevant chunks should be retrieved.

---

# Retrieval Benchmarking

A retrieval benchmark was implemented using predefined test cases.

Each test case contains:

- Query
- Expected correct chunk ID

Example:

```python
{
    "query": "oil prices",
    "correct_chunk_id": "chunk_15"
}
```

Additional test cases include:

```text
football
NASA mission
```

---

## Recall@3

The retrieval benchmark checks whether the expected chunk is present inside the top 3 retrieved results.

The calculation is:

```text
Recall@3 =
Number of successful retrievals
--------------------------------
Total number of test cases
```

The benchmark reports the retrieval performance as:

```text
Recall@3
```

---

# Precision@K

Precision@K evaluates how many of the retrieved results are relevant to the query.

The general formula is:

```text
Precision@K =
Relevant Retrieved Results
--------------------------
Total Retrieved Results
```

Precision@K provides an additional measure for evaluating the quality of semantic retrieval.

---

# Retrieval Latency

Retrieval latency is measured using timestamps before and after the retrieval process.

The system measures the time required for:

- Query embedding
- ChromaDB semantic search
- Retrieval of relevant chunks

The calculation is:

```text
Retrieval Latency =
Retrieval End Time - Retrieval Start Time
```

---

# Generation Latency

The generation stage is also measured using timestamps.

The calculation is:

```text
Generation Latency =
Generation End Time - Generation Start Time
```

This measures how long the LLM takes to generate the final answer.

---

# End-to-End Latency

The RAG system measures the overall processing time.

```text
Retrieval Latency
       +
Generation Latency
       =
Total Latency
```

This helps evaluate the performance of the complete RAG pipeline.

---

# Automated Generation Evaluation

An end-to-end evaluation process is used to evaluate the generation stage.

The evaluation considers:

- User question
- Retrieved context
- Generated answer
- Generation latency
- Total response latency

The purpose is to evaluate whether the RAG pipeline is producing useful answers from retrieved information while also measuring performance.

---

# Request Logging

Structured request logging is used to monitor API requests and application behavior.

Logging helps with:

- Debugging
- Monitoring
- Performance analysis
- Troubleshooting
- Understanding API behavior

---

# HTTP Error Handling

The FastAPI application handles common request and server errors.

## 400 – Bad Request

Used when the client sends a request that cannot be processed because of invalid input.

---

## 422 – Validation Error

FastAPI and Pydantic provide request validation.

For example, the API expects:

```json
{
    "question": "What is RAG?",
    "n_results": 3
}
```

If the request does not follow the required structure or contains an invalid data type, FastAPI can return a `422 Unprocessable Entity` response.

---

## 500 – Internal Server Error

Used when an unexpected server-side error occurs while processing the request.

---

# FastAPI Testing

Unit tests were implemented using `pytest` to test the API behavior.

Testing covers:

- Root endpoint
- `/ask` endpoint
- Request validation
- API responses
- Error handling
- RAG API integration

---

# Complete RAG System

After Week 5, the project follows the complete RAG workflow:

```text
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
```

---

# Complete Internship Progress

## Week 1 – Environment Setup, Data Acquisition & Cleaning

### Completed

- Environment setup
- AG News dataset acquisition
- Exploratory Data Analysis
- Data cleaning
- Data validation
- Data quality reporting
- Clean dataset generation

---

## Week 2 – Chunking, Embeddings & Vector Database

### Completed

- Recursive text chunking
- SentenceTransformer embeddings
- Embedding performance measurement
- ChromaDB integration
- Persistent vector database
- Semantic search
- Retrieval latency measurement
- Unit testing

---

## Week 3 – LLM Integration & Prompt Engineering

### Completed

- DeepSeek/OpenRouter API integration
- Prompt engineering
- System prompts
- Context injection
- API error handling
- Hallucination handling
- Out-of-domain query handling
- Retrieval latency
- Generation latency
- End-to-end latency
- RAG chatting mode

---

## Week 4 – NLP Analysis

### Completed

- NLP analysis
- Sentiment analysis
- Sentiment metadata
- ChromaDB metadata integration
- Sentiment-based filtered retrieval
- Metadata validation
- Empty database handling

---

## Week 5 – FastAPI, Retrieval Evaluation & Testing

### Completed

- FastAPI application
- `GET /` endpoint
- `POST /ask` endpoint
- Pydantic request validation
- RAG API integration
- ChromaDB retrieval through API
- Query embedding
- Top-K retrieval
- Context construction
- LLM integration
- Structured request logging
- HTTP error handling
- 400 error handling
- 422 validation handling
- 500 error handling
- Retrieval benchmarking
- Precision@K evaluation
- Recall@K evaluation
- Generation-quality evaluation
- Retrieval latency measurement
- Generation latency measurement
- End-to-end latency evaluation
- pytest API tests

---

# Technologies Used

- Python
- Pandas
- Hugging Face Datasets
- Sentence Transformers
- ChromaDB
- FastAPI
- Pydantic
- DeepSeek
- OpenRouter
- pytest
- NLP / Sentiment Analysis
- Git
- GitHub

---

# Overall Project Architecture

```text
┌──────────────────────────────────────────┐
│                  CLIENT                  │
│            Web / API / Application       │
└────────────────────┬─────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────┐
│                 FASTAPI                  │
│                API Layer                 │
│                                          │
│       GET /          POST /ask           │
└────────────────────┬─────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────┐
│                RAG PIPELINE              │
│                                          │
│   Question → Embedding → Retrieval       │
│                         ↓                │
│                  Context Building        │
│                         ↓                │
│                  LLM Generation          │
└────────────────┬──────────────┬──────────┘
                 │              │
                 ▼              ▼
        ┌────────────────┐  ┌────────────────┐
        │    ChromaDB    │  │      LLM       │
        │  Vector Store  │  │ DeepSeek /     │
        │                │  │  OpenRouter    │
        └────────────────┘  └────────────────┘
```

---

# Project Learning Outcomes

Through the internship, the project progressed from basic data preparation to an API-accessible AI-powered RAG application.

The project demonstrates understanding and implementation of:

- Data preprocessing
- Dataset analysis
- Text chunking
- Vector embeddings
- Vector databases
- Semantic search
- Retrieval-Augmented Generation
- Prompt engineering
- LLM APIs
- NLP metadata
- Metadata filtering
- FastAPI
- REST API development
- Request validation
- Error handling
- Retrieval evaluation
- Precision@K
- Recall@K
- Generation evaluation
- Latency measurement
- Automated testing

---

# Final Status

## Internship Progress

```text
Week 1  ✅ Completed
Week 2  ✅ Completed
Week 3  ✅ Completed
Week 4  ✅ Completed
Week 5  ✅ Completed
```

The internship project has progressed from raw dataset preparation to a complete RAG application with:

- Data preprocessing
- Chunking
- Embeddings
- Vector database
- Semantic retrieval
- NLP metadata
- Filtered retrieval
- LLM generation
- Prompt engineering
- FastAPI
- Retrieval evaluation
- Generation evaluation
- Latency measurement
- API testing

---

# 🚀 PROJECT END
