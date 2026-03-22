# AI---RAG-Document-Chatbot
LLM and RAG (Retrieval-Augmented Generation) based document chatbot built with Streamlit, LangChain, Hugging Face embeddings, and FAISS for question answering over uploaded documents.
# Document Intelligence Assistant

## Overview

Document Intelligence Assistant is a practical AI application built to demonstrate how Retrieval Augmented Generation, Natural Language Processing, and semantic search can be used to make uploaded documents interactive and queryable.

Instead of forcing users to manually read long documents and search for important details, this system allows users to upload a supported file, detect the document type, retrieve relevant content, and generate grounded answers based on the actual uploaded document.

This project is designed as a hands on implementation of modern document understanding workflows using RAG, vector embeddings, FAISS based retrieval, and a clean Streamlit interface.

At its current stage, the application is optimized for a limited set of document types, including resumes, passports, driving licenses, bank statements, and bills.

## Why this project matters

In real life, important information is often trapped inside documents.

Recruiters need quick summaries from resumes.
Operations teams need to verify identity documents.
Finance teams need to review statements and bills.
Support teams need to search uploaded files without reading every page manually.
Business users need answers quickly, but traditional document workflows are slow and manual.

This project addresses that problem by turning static documents into searchable knowledge sources.

## What the application does

The application allows a user to:

1. Upload a PDF or DOCX document
2. Extract text from the uploaded file
3. Detect the likely document type
4. Split the document into meaningful chunks
5. Convert those chunks into semantic embeddings
6. Store and search them using FAISS
7. Suggest useful questions based on document type
8. Retrieve the most relevant context for a question
9. Generate a grounded response from the uploaded document
10. Collect viewer feedback through the interface

## Core objective

The main goal of this project is to demonstrate applied AI engineering through a realistic end to end workflow.

This is not just a simple chatbot.
It is a document aware assistant that combines retrieval, NLP, document classification, semantic matching, and user interaction in one deployable application.

## Real world problems this project can solve

### Resume screening and recruiter support

Recruiters often receive a large number of resumes and need to quickly identify key information such as candidate skills, education, university, experience, and contact details.

This project can help by allowing a recruiter to upload a resume and ask targeted questions like:

What skills are mentioned
What university is mentioned
What email address is mentioned
What companies are mentioned

This saves time and improves speed in early candidate evaluation.

### Identity document understanding

Organizations frequently work with passports, driving licenses, and similar identity documents for onboarding, verification, travel processing, and compliance checks.

This project can help by extracting and answering questions around:

Name
Nationality
Passport number
Date of birth
Expiry date
Place of issue

This reduces manual reading and supports faster verification workflows.

### Banking and financial document review

Bank statements and bills often contain critical details that users want quickly without scanning the full document.

This project can support questions such as:

What is the account number
What balances are mentioned
What is the statement period
What type of bill is this
What important dates are present

This can be useful for finance operations, audits, customer support, and internal document review.

### Internal document search

Businesses often store operational documents, onboarding files, policy files, and process records that are difficult to search effectively.

This same architecture can evolve into an internal knowledge retrieval system where employees upload or query documents and get grounded answers instead of browsing manually.

## Business value

This project is highly relevant in the business world because it demonstrates a scalable pattern for AI powered document workflows.

### Faster decision making

Instead of manually opening files and reading line by line, users can directly ask focused questions and receive immediate answers.

### Improved productivity

Teams can save time on repetitive document review and spend more time on decisions, analysis, and customer service.

### Better information access

The use of semantic retrieval means the system searches by meaning, not just exact keyword matching. This makes document search more intelligent and more useful.

### Strong foundation for enterprise AI

This architecture can be extended into:

Document copilots
Internal enterprise search systems
HR screening assistants
Financial document review tools
Compliance and verification systems
Customer support document assistants

### Portfolio and recruiter value

For recruiters, this project demonstrates:

Applied AI engineering
RAG architecture understanding
NLP based document intelligence
Vector database knowledge
Frontend and backend integration
Modular software design
Deployment readiness
Real world business use case thinking

## Architecture summary

The project follows a modular pipeline:

1. Document upload
2. Text extraction
3. Document type detection
4. Text chunking
5. Embedding generation
6. Vector indexing with FAISS
7. Similarity search
8. Context retrieval
9. Grounded answer generation
10. Feedback capture

## How RAG works in this project

Retrieval Augmented Generation is the core idea behind this application.

In a typical AI system, a model may answer from general knowledge, which can lead to vague or unsupported responses when a question depends on a specific uploaded file.

In this project, the application first retrieves relevant content from the uploaded document before generating the answer.

That means answers are based on the document itself, not on generic model memory.

This improves relevance, trust, and explainability.

## How NLP is used in this project

Natural Language Processing supports the document understanding and answering logic.

In this project, NLP is used for:

1. Normalizing extracted text
2. Detecting document type
3. Matching question intent
4. Identifying useful patterns such as names, dates, and other fields
5. Improving document specific responses

This makes the assistant more structured and more realistic than plain keyword search.

## Role of LLM style architecture

This project follows the design principles used in LLM based systems, even though it is built in a lightweight and cost conscious way.

The project separates retrieval from answer generation, which is one of the most important design patterns in modern Gen AI applications.

This makes the system easier to improve in the future with stronger hosted or local language models.

## Tech stack explained

### Python

Python is the main programming language used to build the application.

It handles:

Application logic
Document processing
Text handling
Vector retrieval workflow
Question answering logic
Session management
Feedback storage

Python is ideal for this project because of its strong AI, NLP, and data ecosystem.

### Streamlit

Streamlit powers the complete user interface.

It is used to build:

The project portfolio home page
The project details page
The live demo page
The file upload interface
The answer display section
The feedback form

Streamlit makes it possible to convert an AI pipeline into an interactive and shareable product quickly.

### LangChain style modular workflow

The project uses a LangChain style modular structure where the application logic is split into separate components for loading, splitting, vector storage, and answering.

This improves:

Maintainability
Readability
Debugging
Scalability
Future extension

### PyPDFLoader

PyPDFLoader is used for loading PDF documents and extracting text from them.

This is important because PDF is one of the most common document formats in personal and business workflows.

### Docx2txtLoader

Docx2txtLoader is used to support DOCX file uploads.

This extends the system beyond PDF and makes it more useful for resumes, reports, and business documents created in Word format.

### RecursiveCharacterTextSplitter

This module is used to break large extracted text into smaller overlapping chunks.

Chunking is necessary because semantic embeddings and retrieval work better on smaller, context aware text segments.

The overlap helps preserve meaning across chunk boundaries.

### Hugging Face Embeddings

The project uses Hugging Face embedding models to convert text into dense semantic vectors.

These vectors represent the meaning of the text, not just the exact words.

This enables more intelligent retrieval.

### sentence-transformers/all-MiniLM-L6-v2

This embedding model is lightweight, efficient, and well suited for semantic similarity tasks.

It helps keep the project practical and cost effective while still delivering good retrieval quality.

### FAISS

FAISS is the vector search engine used in the project.

Once text chunks are embedded, FAISS indexes them so the application can quickly find the most relevant chunks for a user question.

This is the retrieval backbone of the system.

### Custom answer logic

The chain logic contains custom document aware extraction and question handling.

This allows the system to behave differently for resumes, passports, driving licenses, bank statements, and general files.

This makes responses more structured and practical.

### Feedback logging

The project stores user feedback in JSON format.

This makes it possible to review comments, identify weaknesses, and improve the application iteratively.

## Current supported document types

At the current stage, the application is optimized for:

1. Resumes
2. Passports
3. Driving licenses
4. Bank statements
5. Bills

## Example supported questions

### Resume
What skills are mentioned  
What university is mentioned  
What companies are mentioned  
What email address is mentioned  

### Passport
What name is mentioned  
What passport number is mentioned  
What nationality is mentioned  
What is the expiry date  

### Driving license
What name is mentioned  
What license number is mentioned  
What date of birth is mentioned  

### Bank statement
What account number is mentioned  
What balances are mentioned  
What statement period is mentioned  

## Project structure

```text
AI---RAG-Document-Chatbot/
│
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
├── .env.example
│
├── assets/
│   └── photo.jpg
│
├── src/
│   ├── loader.py
│   ├── splitter.py
│   ├── vectorstore.py
│   └── chain.py
│
├── feedback/
└── temp/