# RAG Prototype Guide (LangChain + Node.js + Ollama)

A clean, modular Retrieval-Augmented Generation (RAG) backend prototype built with **LangChain (JS/TS)**, **Node.js/Express**, and local **Ollama** models.

##  Overview

This repository demonstrates the fundamental flow of a modern RAG system:
1. **Load a Document**: Ingests PDF documents ([`Leave_Policy.pdf`](./Leave_Policy.pdf)) using LangChain's `PDFLoader`.
2. **Text Chunking**: Splits large documents into semantically meaningful chunks with overlap using `RecursiveCharacterTextSplitter`.
3. **Vector Embeddings**: Generates mathematical vector embeddings via Ollama using `nomic-embed-text`.
4. **In-Memory Vector Store**: Stores chunks and embeddings in memory using `MemoryVectorStore`.
5. **Similarity Retrieval**: Retrieves the top-K most relevant chunks using `vectorStore.asRetriever()`.
6. **Prompt Formatting & LLM Response**: Connects context and user queries into a `ChatPromptTemplate` and generates grounded responses with `ChatOllama` (`gemma3:4b`).
7. **Hallucination Safeguard**: Strictly instructs the model to acknowledge when evidence is insufficient (*"I cannot find this information in the provided leave policy document."*).

---

##  Tech Stack & Models

- **Runtime**: Node.js (v20+ recommended) with TypeScript & `tsx`
- **Framework**: Express.js
- **Orchestration**: LangChain (`@langchain/core`, `@langchain/ollama`, `@langchain/community`, `@langchain/textsplitters`)
- **Embedding Model**: `nomic-embed-text` (running locally via Ollama)
- **LLM**: `gemma3:4b` (running locally via Ollama)

---

##  Project Structure

```
├── Leave_Policy.pdf          # Sample company leave policy document
├── src/
│   ├── config/
│   │   └── models.ts         # Ollama model initialization & configuration constants
│   ├── services/
│   │   ├── vectorStore.service.ts # In-memory vector database singleton
│   │   ├── ingestion.service.ts   # PDF loading, text chunking, and embedding
│   │   └── rag.service.ts         # Retrieval, prompt engineering, and LCEL pipeline
│   ├── controllers/
│   │   └── rag.controller.ts      # Express request handlers
│   ├── routes/
│   │   └── rag.routes.ts          # API route definitions
│   └── index.ts                   # Express server entry point
├── package.json
└── tsconfig.json
```

---

##  Getting Started

### 1. Prerequisites
Make sure Ollama is installed and running with the required models:
```bash
ollama pull nomic-embed-text
ollama pull gemma3:4b
```

### 2. Install Dependencies
```bash
npm install --legacy-peer-deps
```

### 3. Start the Server
```bash
npm run dev
```
The server will run on `http://localhost:3000`.

---

## 📬 API Endpoints (Postman / cURL)

### 1. System Status
- **Method**: `GET`
- **URL**: `http://localhost:3000/api/rag/status`

### 2. Ingest Document
Loads, splits, embeds, and indexes `Leave_Policy.pdf` into vector memory.
- **Method**: `POST`
- **URL**: `http://localhost:3000/api/rag/ingest`
- **Headers**: `Content-Type: application/json`
- **Body**: `{}`

### 3. Ask a Question
- **Method**: `POST`
- **URL**: `http://localhost:3000/api/rag/ask`
- **Headers**: `Content-Type: application/json`
- **Body**:
```json
{
  "question": "How many days of casual leave do I get per year, and can I carry it forward?"
}
```

### 4. Insufficient Evidence Test
- **Method**: `POST`
- **URL**: `http://localhost:3000/api/rag/ask`
- **Headers**: `Content-Type: application/json`
- **Body**:
```json
{
  "question": "What is the reimbursement limit for gym memberships?"
}
```
- **Response**:
```json
{
  "success": true,
  "data": {
    "question": "What is the reimbursement limit for gym memberships?",
    "answer": "I cannot find this information in the provided leave policy document.",
    "retrievedChunksCount": 2,
    "sources": [ ... ]
  }
}
```
