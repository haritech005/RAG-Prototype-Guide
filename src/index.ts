import express from "express";
import { CONFIG } from "./config/models.js";
import ragRouter from "./routes/rag.routes.js";

/**
 * ============================================================================
 * MAIN SERVER ENTRY POINT (Node.js + Express Backend)
 * ============================================================================
 * Runs a standalone backend API for the LangChain RAG pipeline.
 * Fully testable via Postman, curl, or any HTTP client.
 */

const app = express();

// Middleware to parse incoming JSON payloads
app.use(express.json());

// Mount our RAG API routes under /api/rag
app.use("/api/rag", ragRouter);

// Root route for quick health check
app.get("/", (_req, res) => {
  res.status(200).json({
    message: "LangChain RAG Learning Backend is running!",
    endpoints: {
      checkStatus: "GET  /api/rag/status",
      ingestPdf:   "POST /api/rag/ingest",
      askQuestion: "POST /api/rag/ask",
    },
    models: {
      embedding: CONFIG.EMBEDDING_MODEL_NAME,
      llm: CONFIG.LLM_MODEL_NAME,
    },
  });
});

// Start the Express server
app.listen(CONFIG.PORT, () => {
  console.log(`
============================================================
🚀 LangChain RAG Backend Server Started!
============================================================
📡 Port:        http://localhost:${CONFIG.PORT}
🧠 LLM Model:   ${CONFIG.LLM_MODEL_NAME} (via Ollama)
🔍 Embeddings:  ${CONFIG.EMBEDDING_MODEL_NAME} (via Ollama)
📄 Target PDF:  ${CONFIG.DEFAULT_PDF_PATH}
============================================================
Available Endpoints to test in Postman:
1. GET  http://localhost:${CONFIG.PORT}/api/rag/status
2. POST http://localhost:${CONFIG.PORT}/api/rag/ingest
3. POST http://localhost:${CONFIG.PORT}/api/rag/ask
============================================================
`);
});