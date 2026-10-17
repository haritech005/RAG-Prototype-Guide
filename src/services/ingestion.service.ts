import fs from "node:fs";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { CONFIG } from "../config/models.js";
import { initializeVectorStoreWithDocs } from "./vectorStore.service.js";

/**
 * ============================================================================
 * INGESTION SERVICE (RAG STEPS 1, 2, & 3)
 * ============================================================================
 * Handles the ingestion pipeline:
 * Step 1: Load the PDF document
 * Step 2: Split text into semantically meaningful chunks
 * Step 3: Embed each chunk and store in the in-memory vector database
 */

export interface IngestionResult {
  message: string;
  pdfPath: string;
  pagesLoaded: number;
  totalChunksCreated: number;
  sampleChunk: {
    pageContent: string;
    metadata: Record<string, any>;
  };
}

/**
 * Execute the document ingestion pipeline.
 *
 * @param customPdfPath Optional path to a PDF file. Defaults to Leave_Policy.pdf.
 */
export async function ingestPdfDocument(
  customPdfPath?: string
): Promise<IngestionResult> {
  const targetPdfPath = customPdfPath || CONFIG.DEFAULT_PDF_PATH;

  if (!fs.existsSync(targetPdfPath)) {
    throw new Error(`PDF file not found at path: ${targetPdfPath}`);
  }

  console.log(`\n========================================`);
  console.log(`[RAG Ingestion] Starting Pipeline`);
  console.log(`[RAG Ingestion] Target File: ${targetPdfPath}`);
  console.log(`========================================`);

  // ==========================================================================
  // STEP 1: LOAD THE PDF DOCUMENT
  // ==========================================================================
  /**
   * --------------------------------------------------------------------------
   * LangChain Component: PDFLoader
   * --------------------------------------------------------------------------
   * WHY WE USE THIS:
   * PDF files are binary formats containing compressed streams, layout objects,
   * fonts, and images. We cannot simply read them with `fs.readFileSync` as text.
   * LangChain's `PDFLoader` parses the raw PDF structure and converts each page
   * into a standardized LangChain `Document` object:
   *
   * {
   *   pageContent: "text extracted from this page...",
   *   metadata: { source: "path/to/file.pdf", loc: { pageNumber: 1 } }
   * }
   *
   * WHAT IT DOES:
   * - Reads the PDF file page-by-page.
   * - Extracts text while preserving line breaks.
   * - Attaches metadata (source path, page numbers) to every Document.
   */
  console.log("[Step 1/3] Loading PDF with LangChain PDFLoader...");
  const pdfLoader = new PDFLoader(targetPdfPath, {
    // splitPages: true ensures each PDF page becomes its own Document object,
    // preserving page-level metadata for citations.
    splitPages: true,
  });

  const rawDocs = await pdfLoader.load();
  console.log(`[Step 1/3] Loaded ${rawDocs.length} pages from PDF.`);

  // ==========================================================================
  // STEP 2: SPLIT INTO CHUNKS
  // ==========================================================================
  /**
   * --------------------------------------------------------------------------
   * LangChain Component: RecursiveCharacterTextSplitter
   * --------------------------------------------------------------------------
   * WHY WE USE THIS:
   * Passing an entire 100-page document into an LLM or embedding model is either:
   * 1. Impossible (exceeds the embedding model's max token limit).
   * 2. Inefficient and noisy (dilutes semantic meaning of specific answers).
   *
   * `RecursiveCharacterTextSplitter` is the gold-standard text splitter in LangChain.
   * Unlike a naive character splitter that splits blindly at 500 characters
   * (which could slice a word or sentence in half), it tries splitting by a
   * hierarchy of natural delimiters:
   *   1. Double newlines ("\n\n" -> paragraphs)
   *   2. Single newlines ("\n" -> sentences/lines)
   *   3. Spaces (" " -> words)
   *   4. Characters ("" -> fallback)
   * This keeps related sentences together in the same chunk!
   *
   * WHAT IT DOES:
   * - `chunkSize`: Target max characters per chunk (e.g., 500 characters).
   * - `chunkOverlap`: Number of characters overlapping between chunk N and N+1.
   *   Overlap is crucial so semantic context at the boundary is not lost!
   */
  console.log("[Step 2/3] Splitting documents with RecursiveCharacterTextSplitter...");
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: CONFIG.CHUNK_SIZE,
    chunkOverlap: CONFIG.CHUNK_OVERLAP,
  });

  const chunkedDocs = await textSplitter.splitDocuments(rawDocs);
  console.log(`[Step 2/3] Split ${rawDocs.length} pages into ${chunkedDocs.length} chunks.`);

  // ==========================================================================
  // STEP 3: CREATE EMBEDDINGS & STORE IN VECTOR STORE
  // ==========================================================================
  /**
   * --------------------------------------------------------------------------
   * Embeddings + Vector Store Initialization
   * --------------------------------------------------------------------------
   * WHY WE USE THIS:
   * We pass the chunked documents and our `OllamaEmbeddings` to `MemoryVectorStore`.
   * For every chunk:
   * 1. It sends chunk.pageContent to Ollama (nomic-embed-text).
   * 2. Ollama generates a 768-number vector embedding.
   * 3. It stores the vector, the text, and metadata in Node.js RAM.
   */
  console.log("[Step 3/3] Creating vector embeddings & indexing in MemoryVectorStore...");
  await initializeVectorStoreWithDocs(chunkedDocs);
  console.log("[Step 3/3] Ingestion complete!\n");

  return {
    message: "PDF successfully loaded, split into chunks, embedded, and stored in vector memory.",
    pdfPath: targetPdfPath,
    pagesLoaded: rawDocs.length,
    totalChunksCreated: chunkedDocs.length,
    sampleChunk: {
      pageContent: chunkedDocs[0]?.pageContent || "",
      metadata: chunkedDocs[0]?.metadata || {},
    },
  };
}
