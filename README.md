# Parallax Labs AI/ML Internship

## Week 1: Environment Setup, Data Acquisition & Cleaning

## Project Overview

This project is part of the Parallax Labs AI/ML Internship.

The goal of Week 1 is to prepare a real-world dataset for building a Retrieval-Augmented Generation (RAG) system by:

- Setting up the Python environment
- Downloading a large dataset
- Performing Exploratory Data Analysis (EDA)
- Cleaning and validating the dataset
- Saving a clean dataset for future chunking and embedding

---

## Dataset

Dataset: AG News

Source:
Hugging Face Datasets

Number of Documents:

- Train: 120,000
- Test: 7,600

Columns:

- text
- label

---

## Project Structure

```
parallax-rag-internship/

│

├── data/

│   ├── raw/

│   │   └── ag_news.csv

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

### Output

Generated:

```
data/raw/ag_news.csv
```

and

```
data/cleaned/clean_ag_news.csv
```

which will be used in Week 2 for text chunking and embeddings.

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

## Next Step

Week 2 focuses on:

- Text Chunking
- Embedding Generation
- Vector Database (ChromaDB)
# Week 2 – Chunking, Embeddings & Vector Database

## Completed

- Implemented RecursiveCharacterTextSplitter
- Chunk Size = 350
- Chunk Overlap = 50
- Generated embeddings using SentenceTransformer (all-MiniLM-L6-v2)
- Logged embedding generation time
- Stored embeddings inside ChromaDB Persistent Client
- Implemented semantic similarity search
- Measured retrieval latency
- Added unit tests for data cleaning, chunking, embedding and ChromaDB
