
# Parallax Labs AI/ML Internship

> **Project:** Retrieval-Augmented Generation (RAG) System  
> **Internship:** Parallax Labs AI/ML Internship  
> **Dataset:** AG News  
> **Tech Stack:** Python, LangChain, Sentence Transformers, ChromaDB, FastAPI, Pydantic, Pytest

---

# 🚀 PROJECT START

## Project Overview

This project was developed as part of the **Parallax Labs AI/ML Internship**.

The goal of the internship project was to progressively build a complete **Retrieval-Augmented Generation (RAG) system** using the AG News dataset.

The project started with data preparation and NLP preprocessing and progressively evolved into a complete AI application containing:

- Data cleaning
- Text chunking
- Text embeddings
- Vector database storage
- Semantic search
- Topic modeling
- Sentiment analysis
- Metadata filtering
- LLM integration
- Retrieval-Augmented Generation
- FastAPI deployment
- Retrieval evaluation
- Generation evaluation
- Latency benchmarking
- API testing with pytest
- Documentation and reproducibility

---

# 🎯 Project Objectives

The main objectives of this project were:

1. Understand and preprocess a real-world NLP dataset.
2. Split documents into meaningful text chunks.
3. Convert text into numerical embedding vectors.
4. Store and retrieve embeddings using a vector database.
5. Implement semantic search.
6. Add NLP metadata using topic modeling and sentiment analysis.
7. Build a Retrieval-Augmented Generation pipeline.
8. Expose the RAG system through a FastAPI API.
9. Evaluate retrieval quality using Precision@K and Recall@K.
10. Evaluate generation quality and latency.
11. Test API behavior using pytest.
12. Document the complete system and make the setup reproducible.

---

# 🏗️ System Architecture

The complete project follows this architecture:

```text
                         ┌───────────────────┐
                         │    AG News        │
                         │     Dataset       │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │   Data Cleaning   │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │     Chunking      │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │    Embeddings     │
                         │ SentenceTransform │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │     ChromaDB      │
                         │   Vector Store    │
                         └─────────┬─────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
                    ▼                             ▼
           ┌──────────────────┐         ┌──────────────────┐
           │   NLP Metadata   │         │    User Query    │
           │                  │         │                  │
           │ Topic Modeling   │         │ Query Embedding  │
           │ Sentiment        │         └────────┬─────────┘
           └────────┬─────────┘                  │
                    │                            │
                    └──────────────┬─────────────┘
                                   ▼
                         ┌───────────────────┐
                         │   Semantic Search │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │   Top-K Chunks    │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │ Context Building  │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │      LLM API      │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │   Generated       │
                         │     Answer        │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │      FastAPI      │
                         │     REST API      │
                         └─────────┬─────────┘
                                   │
                                   ▼
                              API Response
````

---

# 🔄 Complete RAG Pipeline

The RAG pipeline follows these steps:

```text
User Question
      │
      ▼
Query Embedding
      │
      ▼
Vector Similarity Search
      │
      ▼
Retrieve Top-K Chunks
      │
      ▼
Combine Retrieved Chunks
      │
      ▼
Create Context
      │
      ▼
Send Query + Context to LLM
      │
      ▼
Generate Final Answer
```

The main idea behind RAG is to provide the language model with relevant retrieved information before generating the final answer.

This helps ground the generated response in information retrieved from the project dataset.

---

# 📁 Project Structure

```text
parallax-rag-internship/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── chunks/
│
├── chroma_db/
│
├── src/
│   ├── __init__.py
│   ├── app.py
│   ├── call_api.py
│   ├── chromaDb.py
│   ├── chunking.py
│   ├── clean_data.py
│   ├── embedding.py
│   ├── fast_Api.py
│   ├── generational_latency.py
│   ├── load_dataset.py
│   ├── nlp_analysis.py
│   ├── semantic_Search.py
│   ├── topic_modeling.py
│   └── verify_environment.py
│
├── test/
│   ├── test_Data.py
│   ├── test_api.py
│   ├── test_chromadb.py
│   ├── test_chunk.py
│   ├── test_embedding.py
│   └── test_nlp_metadata.py
│
├── requirements.txt
├── packages.txt
├── nlp_metadata_report.md
└── README.md
```

---

# 🛠️ Technologies Used

| Technology               | Purpose                   |
| ------------------------ | ------------------------- |
| Python                   | Main programming language |
| Pandas                   | Dataset manipulation      |
| NumPy                    | Numerical operations      |
| LangChain Text Splitters | Text chunking             |
| Sentence Transformers    | Text embeddings           |
| ChromaDB                 | Vector database           |
| FastAPI                  | API development           |
| Pydantic                 | Request validation        |
| Pytest                   | Automated testing         |
| LLM API                  | Answer generation         |
| Logging                  | API request/error logging |

---

# 📊 Dataset

## AG News

The project uses the **AG News dataset**, a text classification dataset containing news articles from four major categories:

* World
* Sports
* Business
* Sci/Tech

The dataset was used as the source of documents for the RAG pipeline.

---

# 1️⃣ Data Preparation

The first stage of the project prepares the raw dataset.

The preprocessing pipeline includes:

* Loading the dataset
* Checking dataset structure
* Handling missing values
* Removing duplicates
* Removing empty text
* Inspecting text lengths
* Preparing data for chunking

The cleaned dataset is then passed to the chunking stage.

---

# 2️⃣ Text Chunking

Large documents are divided into smaller pieces called **chunks**.

The project uses:

```text
RecursiveCharacterTextSplitter
```

with the configured chunk size and overlap.

The purpose of chunking is to make documents suitable for embedding and semantic retrieval.

The resulting structure is approximately:

```text
Document
│
├── Chunk 1
├── Chunk 2
├── Chunk 3
└── ...
```

Each chunk retains associated metadata such as its original label.

---

# 3️⃣ Text Embeddings

Text chunks are converted into numerical vectors using a Sentence Transformer model.

The project uses:

```text
all-MiniLM-L6-v2
```

Conceptually:

```text
Text
 ↓
Embedding Model
 ↓
Vector
```

The resulting vectors represent the semantic meaning of the text and allow similarity-based retrieval.

---

# 4️⃣ Vector Database — ChromaDB

The generated embeddings are stored in **ChromaDB**.

Each stored document contains:

```text
ID
Document / Chunk
Embedding
Metadata
```

Example:

```text
ID:
chunk_15

Document:
"Oil prices increased..."

Metadata:
{
    "label": "business",
    "topic_id": 2,
    "topic_label": "Finance",
    "sentiment": "POSITIVE",
    "sentiment_score": 0.82
}
```

ChromaDB is then used to retrieve documents that are semantically similar to a user query.

---

# 5️⃣ Semantic Search

When a user asks a question:

```text
"What happened to oil prices?"
```

the query is converted into an embedding.

```text
User Query
     ↓
Query Embedding
     ↓
ChromaDB
     ↓
Similarity Search
     ↓
Top-K Relevant Chunks
```

The retrieved documents are then used as context for the LLM.

---

# 6️⃣ NLP Analysis

The project also adds NLP metadata to the chunks.

The implemented analysis includes:

## Topic Modeling

Topic modeling identifies topics associated with documents.

Metadata includes fields such as:

```text
topic_id
topic_label
```

## Sentiment Analysis

Sentiment analysis assigns sentiment information to documents.

Metadata includes:

```text
sentiment
sentiment_score
```

This metadata is stored along with the chunks.

---

# 7️⃣ Metadata Filtering

The retrieved documents can also be filtered using metadata.

For example:

```text
User Query:
"What happened to the company?"

Filter:
POSITIVE
```

The retrieval process can use the sentiment metadata to restrict results.

Conceptually:

```text
Query
 ↓
Embedding
 ↓
ChromaDB
 ↓
Metadata Filter
 ↓
Relevant Chunks
```

---

# 8️⃣ LLM Integration

After retrieving relevant chunks, the chunks are combined into a context.

```text
Chunk 1
+
Chunk 2
+
Chunk 3
      ↓
   Context
```

The context and user question are passed to the LLM.

```text
Question + Retrieved Context
             ↓
            LLM
             ↓
       Generated Answer
```

---

# 9️⃣ Retrieval-Augmented Generation

The complete RAG process combines retrieval and generation.

```text
                    USER QUESTION
                         │
                         ▼
                 Query Embedding
                         │
                         ▼
                    ChromaDB
                         │
                         ▼
                  Top-K Chunks
                         │
                         ▼
                    Build Context
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
          User Query             Context
              │                     │
              └──────────┬──────────┘
                         ▼
                        LLM
                         │
                         ▼
                  Final Answer
```

---

# 🔟 FastAPI Application

The RAG pipeline is exposed through a FastAPI application.

## Start the API

Run:

```bash
uvicorn src.fast_Api:app --reload
```

The API can then be accessed locally.

---

# 📡 API Endpoints

## GET `/`

Checks whether the API application is running.

### Example

```http
GET /
```

### Response

```json
{
    "message": "..."
}
```

---

## POST `/ask`

Sends a question to the RAG system.

### Request

```json
{
    "question": "What is RAG?",
    "n_results": 3
}
```

### Parameters

| Parameter | Type    | Description                  |
| --------- | ------- | ---------------------------- |
| question  | string  | User's question              |
| n_results | integer | Number of chunks to retrieve |

### Response

```json
{
    "message": "Generated answer..."
}
```

---

# ⚠️ HTTP Error Handling

The API handles different types of errors.

## 400 — Bad Request

Used for invalid request conditions such as:

* Empty question
* Invalid `n_results`

Example:

```json
{
    "detail": "Question is required."
}
```

---

## 422 — Validation Error

FastAPI/Pydantic automatically handles invalid request data types or missing required fields.

Example:

```json
{
    "question": "What is RAG?",
    "n_results": "three"
}
```

---

## 500 — Internal Server Error

Used when an internal operation fails, such as:

* Vector database query failure
* Embedding failure
* LLM generation failure

---

# 📝 Request Logging

The FastAPI application uses Python's logging system.

The API records information such as:

* Incoming requests
* User question
* Number of requested results
* Warnings
* Internal errors

Logs are written to:

```text
api.log
```

---

# 🧪 Testing

The project uses **pytest** for automated testing.

Tests cover:

* Root endpoint
* Valid `/ask` request
* Empty question
* Invalid data type
* Missing required field
* Invalid `n_results`

Run tests using:

```bash
pytest
```

Expected output should show all implemented tests passing.

---

# 📈 Retrieval Evaluation

The retrieval system is evaluated using:

## Precision@K

Precision@K measures how many of the retrieved K documents are relevant.

```text
Precision@K =
Relevant Retrieved Documents
----------------------------
          K
```

For example, if 2 of the top 3 retrieved chunks are relevant:

```text
Precision@3 = 2 / 3
             = 0.67
```

---

## Recall@K

Recall@K measures how many of the relevant documents were successfully retrieved.

```text
Recall@K =
Relevant Retrieved Documents
----------------------------
Total Relevant Documents
```

---

# 🤖 Generation Evaluation

The project also evaluates generated answers.

A keyword-based evaluation approach is used.

For each test question:

```text
Question
   ↓
RAG Retrieval
   ↓
LLM Generation
   ↓
Generated Answer
   ↓
Keyword Evaluation
```

The percentage of expected keywords found in the generated answer is used as the generation-quality score.

Example:

```text
Expected keywords:
capital
France

Generated answer:
"Paris is the capital of France."

Keywords found:
capital
France

Score:
100%
```

This provides a simple automated baseline for generation quality.

---

# ⚡ Performance Benchmarks

The project measures the following latency values:

### Retrieval Latency

Time required to:

```text
Query
 ↓
Embedding
 ↓
Vector Search
```

### Generation Latency

Time required for the LLM to generate the final response.

### Total Latency

```text
Total Latency =
Retrieval Latency + Generation Latency
```

---

# 📊 Evaluation Results

> **Important:** Replace the placeholders below with the actual results produced by the evaluation scripts.

## Retrieval Results

| Metric      |       Result |
| ----------- | -----------: |
| Precision@3 | `ADD_RESULT` |
| Recall@3    | `ADD_RESULT` |

## Generation Results

| Metric             |       Result |
| ------------------ | -----------: |
| Generation Quality | `ADD_RESULT` |

## Latency Results

| Metric                     |           Result |
| -------------------------- | ---------------: |
| Average Retrieval Latency  | `ADD_RESULT` sec |
| Average Generation Latency | `ADD_RESULT` sec |
| Average Total Latency      | `ADD_RESULT` sec |

---

# 📦 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/Sp25-bcs-034/parallax-rag-internship.git
```

## 2. Navigate to the Project

```bash
cd parallax-rag-internship
```

## 3. Create a Virtual Environment

Windows:

```bash
python -m venv .venv
```

Activate:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ⚙️ Environment Configuration

If the LLM API requires an API key, configure the required environment variable before running the application.

Example:

```text
API_KEY=your_api_key_here
```

Do not commit API keys or other secrets to GitHub.

---

# ▶️ Running the Project

## Run the FastAPI Application

```bash
uvicorn src.fast_Api:app --reload
```

The API will be available locally.

FastAPI also provides interactive API documentation through Swagger UI.

Open:

```text
/docs
```

---

# 🔎 Example Queries

Example questions can be sent to the `/ask` endpoint.

```json
{
    "question": "What happened to oil prices?",
    "n_results": 3
}
```

Another example:

```json
{
    "question": "What happened in the football match?",
    "n_results": 3
}
```

Another example:

```json
{
    "question": "What did NASA announce?",
    "n_results": 3
}
```

The retrieved chunks are passed to the LLM as context for answer generation.

---

# 🧪 Running Evaluation

Run the retrieval evaluation script to calculate:

```text
Precision@K
Recall@K
```

Run the generation evaluation script to evaluate:

```text
Generation Quality
Latency
```

The exact command depends on the final project entry point.

---

# 📊 Project Development by Week

## WEEK 1 — Data Preparation

Completed:

* Dataset loading
* Dataset inspection
* Data cleaning
* Missing-value handling
* Duplicate removal
* Exploratory analysis

---

## WEEK 2 — Chunking, Embeddings & Vector Database

Completed:

* Text chunking
* RecursiveCharacterTextSplitter
* Embedding generation
* Sentence Transformer
* ChromaDB integration
* Vector storage
* Semantic retrieval foundation

---

## WEEK 3 — RAG & LLM Integration

Completed:

* Semantic search
* Context construction
* LLM integration
* RAG pipeline
* Prompt-based answer generation
* Retrieval and generation latency measurement

---

## WEEK 4 — NLP Analysis

Completed:

* Topic modeling
* Topic metadata
* Sentiment analysis
* Sentiment score
* Metadata integration
* Metadata-based filtering

---

## WEEK 5 — Evaluation & API

Completed:

* FastAPI application
* `/` endpoint
* `/ask` endpoint
* Pydantic request validation
* HTTP 400 handling
* HTTP 422 validation
* HTTP 500 error handling
* Request logging
* Precision@K
* Recall@K
* Retrieval evaluation
* Generation evaluation
* Latency measurement
* pytest API tests

---

## WEEK 6 — Documentation, Benchmarks & Launch

Completed:

* Codebase documentation
* Docstrings
* Type hints
* Comprehensive README
* Architecture documentation
* Evaluation documentation
* Performance benchmark documentation
* Reproducible setup
* Requirements documentation
* Demo preparation
* Final presentation preparation

---

# 🧠 Key Concepts Learned

Through this project, the following AI Engineering concepts were implemented:

### NLP

* Text preprocessing
* Text chunking
* Topic modeling
* Sentiment analysis

### Embeddings

* Semantic representation
* Sentence Transformers
* Query embeddings
* Document embeddings

### Vector Databases

* ChromaDB
* Vector storage
* Similarity search
* Metadata filtering

### RAG

* Retrieval
* Context construction
* Prompt construction
* LLM generation

### Backend Engineering

* FastAPI
* REST endpoints
* Pydantic
* HTTP status codes
* Error handling
* Logging

### AI Evaluation

* Precision@K
* Recall@K
* Generation-quality evaluation
* Retrieval latency
* Generation latency
* Total latency

### Testing

* pytest
* API endpoint testing
* Validation testing
* Error-condition testing

---

# 🔮 Future Improvements

Possible improvements include:

* More advanced retrieval evaluation datasets
* More sophisticated generation evaluation metrics
* LLM-as-a-judge evaluation
* Reranking retrieved documents
* Hybrid keyword + vector search
* Better prompt engineering
* Streaming responses
* Authentication
* Rate limiting
* Docker deployment
* Cloud deployment
* Monitoring and observability
* Improved logging
* Production-grade database management

---

# 🎥 Demo

A demonstration of the RAG API will show:

```text
Start FastAPI
      ↓
Open Swagger UI
      ↓
POST /ask
      ↓
Enter Question
      ↓
Retrieve Relevant Chunks
      ↓
Generate Answer
      ↓
Return API Response
```

> Add the final demo GIF/video here.

---

# 📑 Final Project Summary

This project demonstrates the development of a complete Retrieval-Augmented Generation system from raw data preparation to API deployment and evaluation.

The final system combines:

```text
Data
 ↓
Cleaning
 ↓
Chunking
 ↓
Embeddings
 ↓
Vector Database
 ↓
Semantic Search
 ↓
NLP Metadata
 ↓
Filtered Retrieval
 ↓
LLM
 ↓
RAG
 ↓
FastAPI
 ↓
Evaluation
 ↓
Testing
```

The project provides practical experience in building and evaluating an end-to-end AI application rather than only developing an isolated machine-learning model.

---

# 👨‍💻 Author

**Imara Asim**

Parallax Labs AI/ML Internship

Project:

**Retrieval-Augmented Generation (RAG) System**

GitHub:

[https://github.com/Sp25-bcs-034/parallax-rag-internship](https://github.com/Sp25-bcs-034/parallax-rag-internship)

---

# 🏁 Internship Status

```text
WEEK 1  ✅
WEEK 2  ✅
WEEK 3  ✅
WEEK 4  ✅
WEEK 5  ✅
WEEK 6  🚧 In Progress
```

Final Week 6 deliverables:

```text
Code Documentation       ⬜
README                    ⬜
Evaluation Benchmarks    ⬜
Reproducible Setup       ⬜
Demo Video/GIF            ⬜
Final Presentation        ⬜
```

```
Parallax Labs AI/ML Internship
Retrieval-Augmented Generation System
```

````

