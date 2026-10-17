import type { Request, Response } from "express";
import { ingestPdfDocument } from "../services/ingestion.service.js";
import { answerQuestionWithRag } from "../services/rag.service.js";
import { getStoredChunksCount, isVectorStoreReady } from "../services/vectorStore.service.js";
import { CONFIG } from "../config/models.js";

/**
 * ============================================================================
 * RAG CONTROLLER
 * ============================================================================
 * Bridges incoming HTTP requests (from Postman) to our LangChain RAG services.
 */

/**
 * POST /api/rag/ingest
 * Trigger PDF document loading, text chunking, and vector embedding.
 * Request body (optional): { "filePath": "custom/path/to/doc.pdf" }
 */
export async function ingestDocumentHandler(req: Request, res: Response): Promise<void> {
  try {
    const { filePath } = req.body || {};
    const result = await ingestPdfDocument(filePath);
    res.status(200).json({
      success: true,
      data: result,
    });
  } catch (error: any) {
    console.error("[Controller Error - Ingestion]:", error.message);
    res.status(500).json({
      success: false,
      error: error.message || "Failed to ingest document",
    });
  }
}

/**
 * POST /api/rag/ask
 * Ask a question against the ingested document.
 * Request body: { "question": "What is the policy for sick leave?" }
 */
export async function askQuestionHandler(req: Request, res: Response): Promise<void> {
  try {
    const { question } = req.body || {};

    if (!question || typeof question !== "string" || question.trim().length === 0) {
      res.status(400).json({
        success: false,
        error: "A valid 'question' string is required in the JSON request body.",
      });
      return;
    }

    const result = await answerQuestionWithRag(question.trim());
    res.status(200).json({
      success: true,
      data: result,
    });
  } catch (error: any) {
    console.error("[Controller Error - Ask]:", error.message);
    res.status(error.message?.includes("not been ingested") ? 400 : 500).json({
      success: false,
      error: error.message || "Failed to process question",
    });
  }
}

/**
 * GET /api/rag/status
 * Check vector store readiness, loaded models, and current status.
 */
export function getStatusHandler(_req: Request, res: Response): void {
  const ready = isVectorStoreReady();
  res.status(200).json({
    success: true,
    data: {
      status: ready ? "READY" : "WAITING_FOR_INGESTION",
      vectorStoreReady: ready,
      storedChunksCount: getStoredChunksCount(),
      models: {
        embedding: CONFIG.EMBEDDING_MODEL_NAME,
        llm: CONFIG.LLM_MODEL_NAME,
        ollamaUrl: CONFIG.OLLAMA_BASE_URL,
      },
      defaultPdfPath: CONFIG.DEFAULT_PDF_PATH,
    },
  });
}
