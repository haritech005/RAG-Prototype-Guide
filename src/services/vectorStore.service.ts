import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import type { Document } from "@langchain/core/documents";
import { embeddings } from "../config/models.js";

/**
 * ============================================================================
 * IN-MEMORY VECTOR STORE SERVICE
 * ============================================================================
 * Manages our in-memory vector database instance.
 */

// Holds the active MemoryVectorStore instance in memory (RAM)
let vectorStoreInstance: MemoryVectorStore | null = null;
let totalChunksCount = 0;

/**
 * ----------------------------------------------------------------------------
 * LangChain Component: MemoryVectorStore
 * ----------------------------------------------------------------------------
 * WHY WE USE THIS:
 * In a RAG application, we need a vector database to index and search embeddings.
 * For learning and local prototyping, setting up a separate vector database
 * (like Pinecone, Qdrant, Chroma, or Milvus) adds extra infrastructure overhead.
 * LangChain provides `MemoryVectorStore`, which keeps all vector embeddings and
 * document metadata directly in Node.js RAM!
 *
 * WHAT IT DOES:
 * 1. Stores Document objects alongside their mathematical vector embeddings.
 * 2. Computes Cosine Similarity between a query vector and all stored vectors.
 * 3. Can be turned into a LangChain "Retriever" runnable with `.asRetriever()`.
 */

/**
 * Initialize or overwrite the vector store with new Document chunks.
 */
export async function initializeVectorStoreWithDocs(
  docs: Document[]
): Promise<MemoryVectorStore> {
  console.log(`[VectorStore] Creating in-memory vector store with ${docs.length} document chunks...`);

  // MemoryVectorStore.fromDocuments takes an array of Documents and an Embeddings instance.
  // It automatically loops over each document, calls `embeddings.embedDocuments(...)`
  // using Ollama 'nomic-embed-text', and stores the vectors in RAM.
  vectorStoreInstance = await MemoryVectorStore.fromDocuments(docs, embeddings);
  totalChunksCount = docs.length;

  console.log("[VectorStore] In-memory vector store ready!");
  return vectorStoreInstance;
}

/**
 * Get the current vector store instance. Throws error if not yet ingested.
 */
export function getVectorStore(): MemoryVectorStore {
  if (!vectorStoreInstance) {
    throw new Error(
      "Vector store is empty! Please call the ingestion endpoint (POST /api/rag/ingest) first."
    );
  }
  return vectorStoreInstance;
}

/**
 * Check if the vector store has been initialized with documents.
 */
export function isVectorStoreReady(): boolean {
  return vectorStoreInstance !== null && totalChunksCount > 0;
}

/**
 * Get total chunks stored in memory.
 */
export function getStoredChunksCount(): number {
  return totalChunksCount;
}
