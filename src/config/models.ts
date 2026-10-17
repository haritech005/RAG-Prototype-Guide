import { ChatOllama, OllamaEmbeddings } from "@langchain/ollama";
import path from "node:path";

/**
 * ============================================================================
 * RAG CONFIGURATION & MODEL INITIALIZATION
 * ============================================================================
 * This file centralizes our AI models and application configurations.
 */

// Global configuration constants for our RAG pipeline
export const CONFIG = {
  // Server port
  PORT: process.env.PORT ? parseInt(process.env.PORT, 10) : 3000,

  // Default PDF document path (Acme Corporation Leave Policy)
  DEFAULT_PDF_PATH: path.resolve(process.cwd(), "Leave_Policy.pdf"),

  // Chunking parameters for RecursiveCharacterTextSplitter:
  // - CHUNK_SIZE: Maximum number of characters in each chunk
  // - CHUNK_OVERLAP: Number of shared characters between adjacent chunks
  //   (preserves context across boundaries so sentences aren't abruptly cut off)
  CHUNK_SIZE: 500,
  CHUNK_OVERLAP: 50,

  // Number of relevant chunks the retriever should return for each query
  TOP_K: 2,

  // Ollama local server URL
  OLLAMA_BASE_URL: process.env.OLLAMA_BASE_URL || "http://localhost:11434",

  // Model names in Ollama
  EMBEDDING_MODEL_NAME: "nomic-embed-text",
  LLM_MODEL_NAME: "gemma3:4b",
};

/**
 * ----------------------------------------------------------------------------
 * LangChain Component: OllamaEmbeddings
 * ----------------------------------------------------------------------------
 * WHY WE USE THIS:
 * Computers and vector databases cannot understand raw text directly; they only
 * understand mathematical vectors (arrays of numbers). An embedding model turns
 * text into a dense vector where semantically similar texts are placed close
 * together in high-dimensional space.
 *
 * WHAT IT DOES:
 * Connects to the local Ollama instance running 'nomic-embed-text' and generates
 * a 768-dimensional vector representation for any text chunk or user question.
 */
export const embeddings = new OllamaEmbeddings({
  model: CONFIG.EMBEDDING_MODEL_NAME,
  baseUrl: CONFIG.OLLAMA_BASE_URL,
});

/**
 * ----------------------------------------------------------------------------
 * LangChain Component: ChatOllama
 * ----------------------------------------------------------------------------
 * WHY WE USE THIS:
 * This is our Large Language Model (LLM) interface. LangChain's ChatOllama provides
 * a unified, standardized interface for chat models running locally via Ollama,
 * seamlessly integrating with LangChain's Runnables, Prompts, and Output Parsers.
 *
 * WHAT IT DOES:
 * Takes formatted messages (system prompt + user query + context) and sends them
 * to the 'gemma3:4b' model locally to generate a grounded, natural language answer.
 *
 * PARAMETERS:
 * - temperature: 0 -> Forces deterministic, factual responses. Essential for
 *   RAG to prevent creative hallucinations and adhere strictly to the context.
 */
export const llm = new ChatOllama({
  model: CONFIG.LLM_MODEL_NAME,
  baseUrl: CONFIG.OLLAMA_BASE_URL,
  temperature: 0,
});
