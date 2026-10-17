import type { Document } from "@langchain/core/documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { CONFIG, llm } from "../config/models.js";
import { getVectorStore, isVectorStoreReady } from "./vectorStore.service.js";

/**
 * ============================================================================
 * RAG QUERY SERVICE (RAG STEPS 4, 5, & 6)
 * ============================================================================
 * Implements the retrieval and generation pipeline using LangChain Expression
 * Language (LCEL), Runnables, and Pipes.
 *
 * Step 4: Retrieve relevant chunks for a user's question
 * Step 5: Pass question + retrieved context into a prompt template
 * Step 6: Invoke LLM with strict instructions to acknowledge insufficient evidence
 */

export interface RagResponse {
  question: string;
  answer: string;
  retrievedChunksCount: number;
  sources: Array<{
    page: number;
    content: string;
  }>;
}

/**
 * Helper function to format an array of Document objects into a clean,
 * delimited string suitable for the prompt's {context} variable.
 */
function formatDocumentsAsString(documents: Document[]): string {
  return documents
    .map((doc, index) => {
      const pageNumber = doc.metadata?.loc?.pageNumber ?? "Unknown";
      return `--- [Excerpt ${index + 1} | Page ${pageNumber}] ---\n${doc.pageContent.trim()}`;
    })
    .join("\n\n");
}

/**
 * Execute the RAG pipeline for a user question.
 *
 * @param question The user's query (e.g. "How many sick leaves do I get?")
 */
export async function answerQuestionWithRag(question: string): Promise<RagResponse> {
  // Validate that the user has ingested documents first
  if (!isVectorStoreReady()) {
    throw new Error(
      "Document has not been ingested yet. Please send a POST request to /api/rag/ingest first."
    );
  }

  const vectorStore = getVectorStore();

  console.log(`\n========================================`);
  console.log(`[RAG Query] New Question Received`);
  console.log(`[RAG Query] Question: "${question}"`);
  console.log(`========================================`);

  // ==========================================================================
  // STEP 4: RETRIEVE RELEVANT CHUNKS FOR THE QUESTION
  // ==========================================================================
  /**
   * --------------------------------------------------------------------------
   * LangChain Component: vectorStore.asRetriever(...)
   * --------------------------------------------------------------------------
   * WHY WE USE THIS:
   * A VectorStore is a database of vectors, but a "Retriever" in LangChain is a
   * higher-level Runnable abstraction.
   * By calling `asRetriever({ k: 3 })`, LangChain wraps the vector store in a
   * standardized Runnable component that:
   * 1. Takes an input string (the question).
   * 2. Automatically embeds the question with the embedding model.
   * 3. Performs similarity search in the vector space.
   * 4. Returns the top `k` closest Document chunks.
   */
  const retriever = vectorStore.asRetriever({
    k: CONFIG.TOP_K, // Retrieve top 3 most relevant chunks
  });

  console.log("[Step 4/6] Retrieving relevant chunks from vector store...");
  const relevantDocs = await retriever.invoke(question);
  console.log(`[Step 4/6] Found ${relevantDocs.length} relevant chunks.`);

  // ==========================================================================
  // STEP 5: PROMPT TEMPLATE WITH RETRIEVED CONTEXT & STRICT GROUNDING
  // ==========================================================================
  /**
   * --------------------------------------------------------------------------
   * LangChain Component: ChatPromptTemplate
   * --------------------------------------------------------------------------
   * WHY WE USE THIS:
   * Instead of manual string concatenation (which is messy, error-prone, and doesn't
   * separate system instructions from user queries), `ChatPromptTemplate` structures
   * messages cleanly with template placeholders `{context}` and `{question}`.
   *
   * STEP 6 ENFORCEMENT:
   * Notice the strict rule in the system prompt:
   * "If the answer cannot be found in the context, explicitly say:
   * 'I cannot find this information in the provided leave policy document.'"
   * This prevents hallucination when the evidence is insufficient.
   */
  const prompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      `You are an accurate and helpful HR assistant for Acme Corporation.
Your sole job is to answer employee questions strictly and exclusively based on the provided Leave Policy context.

STRICT INSTRUCTIONS:
1. Base your answer ONLY on the context provided below.
2. If the context does NOT contain sufficient evidence or information to answer the question, or if the topic is not mentioned at all, you MUST reply with this exact statement:
   "I cannot find this information in the provided leave policy document."
3. Do NOT make assumptions, do NOT guess, and do NOT use external knowledge.
4. Keep your answer factual, professional, and directly related to the question.

Context from Leave Policy Document:
{context}`,
    ],
    [
      "human",
      "{question}",
    ],
  ]);

  // ==========================================================================
  // STEP 6: LANGCHAIN RUNNABLE PIPELINE & PIPES (LCEL)
  // ==========================================================================
  /**
   * --------------------------------------------------------------------------
   * LangChain Concepts: Pipes (.pipe()), Runnables, and StringOutputParser
   * --------------------------------------------------------------------------
   * WHY WE USE THIS:
   * LangChain Expression Language (LCEL) allows composing chains declaratively.
   * Every component (retriever, prompt, llm, parser) implements the "Runnable"
   * interface with methods like `.invoke()`, `.stream()`, and `.pipe()`.
   *
   * Breakdown of this pipeline:
   *
   * 1. `prompt`: Takes `{ context, question }` and formats the chat messages.
   * 2. `.pipe(llm)`: Passes the formatted messages into ChatOllama ('gemma3:4b').
   * 3. `.pipe(new StringOutputParser())`:
   *    Chat models return an `AIMessage` object (with metadata, role, etc.).
   *    `StringOutputParser` extracts just the clean string content for our API.
   *
   * Notice how `.pipe()` acts like the Unix pipe (|), chaining the output of one
   * runnable directly into the input of the next!
   */
  console.log("[Step 5 & 6] Building LCEL Runnable chain with .pipe()...");

  const ragChain = prompt
    .pipe(llm)
    .pipe(new StringOutputParser());

  // Format the retrieved documents into the context string
  const formattedContext = formatDocumentsAsString(relevantDocs);

  console.log("[Step 6/6] Invoking LLM (gemma3:4b) via Runnable chain...");
  const generatedAnswer = await ragChain.invoke({
    context: formattedContext,
    question: question,
  });

  console.log("[Step 6/6] Response generated successfully!\n");

  // Return both the answer AND the retrieved citations so the user can verify in Postman
  return {
    question,
    answer: generatedAnswer.trim(),
    retrievedChunksCount: relevantDocs.length,
    sources: relevantDocs.map((doc: Document) => ({
      page: doc.metadata?.loc?.pageNumber ?? 1,
      content: doc.pageContent,
    })),
  };
}
