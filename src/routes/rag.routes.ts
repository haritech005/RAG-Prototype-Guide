import { Router } from "express";
import {
  askQuestionHandler,
  getStatusHandler,
  ingestDocumentHandler,
} from "../controllers/rag.controller.js";

/**
 * ============================================================================
 * RAG API ROUTES
 * ============================================================================
 * Defines REST endpoints accessible via Postman.
 */

const router = Router();

// Ingest PDF document into vector memory
// POST http://localhost:3000/api/rag/ingest
router.post("/ingest", ingestDocumentHandler);

// Ask questions grounded on the PDF context
// POST http://localhost:3000/api/rag/ask
router.post("/ask", askQuestionHandler);

// Check system and vector store status
// GET  http://localhost:3000/api/rag/status
router.get("/status", getStatusHandler);

export default router;
