import os
import time
import uuid
from typing import Optional

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from analytics.metrics import summarize
from analytics.query_log import QueryAnalytics
from backend.config import Config
from backend.inference_client import InferenceError
from backend.latency import LatencyLog, start_recording
from backend.logger import app_logger
from backend.query_cache import retrieval_cache
from backend.rag_pipeline import RAGPipeline
from backend.schemas import (
    AnalyticsSummaryResponse,
    AskResponse,
    HealthResponse,
    QuestionRequest,
    StatsResponse,
    UploadResponse,
)
from backend.utils import save_uploaded_file

app = FastAPI(title="Enterprise AI Document Assistant", version="1.2.0")

rag_pipeline = RAGPipeline()
latency_log = LatencyLog()


@app.middleware("http")
async def latency_logging_middleware(request: Request, call_next):
    """Times every request end-to-end and pairs it with the
    retrieval/generation phase breakdown `RAGPipeline.ask_question` records
    for that same request, writing one row per request to
    `logs/latency.csv`."""
    start = time.time()
    recorder = start_recording()
    response = await call_next(request)
    total_ms = (time.time() - start) * 1000

    try:
        latency_log.log({
            "timestamp": time.time(),
            "path": request.url.path,
            "retrieval_time_ms": recorder.retrieval_ms,
            "generation_time_ms": recorder.generation_ms,
            "total_time_ms": total_ms,
        })
    except Exception as e:
        app_logger.warning(f"Failed to log request latency: {e}")

    return response


async def verify_api_key(x_api_key: Optional[str] = Header(default=None)):
    """Auth stub. If API_KEY is unset the API is open; set it to require
    clients to send a matching X-API-Key header. This is intentionally
    minimal — swap in real session/JWT auth before exposing this publicly.
    """
    if Config.API_KEY and x_api_key != Config.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return True


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    app_logger.error(f"Unhandled exception on {request.url.path}: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.post("/upload", response_model=UploadResponse, summary="Upload and process a PDF document",
          dependencies=[Depends(verify_api_key)])
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF file and process it for Q&A."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    try:
        file_id = str(uuid.uuid4())
        file_path = os.path.join(Config.DATA_PATH, f"{file_id}.pdf")
        save_uploaded_file(file, file_path)

        result = rag_pipeline.process_pdf(file_path)

        if result["status"] == "success":
            return UploadResponse(
                message="PDF uploaded and processed successfully",
                file_id=file_id,
                chunks_indexed=result["chunks_indexed"],
            )
        raise HTTPException(status_code=500, detail=result["message"])
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error uploading PDF: {e}")
        raise HTTPException(status_code=500, detail="Error processing PDF")


@app.post("/ask", response_model=AskResponse, summary="Ask a question about the uploaded documents",
          dependencies=[Depends(verify_api_key)])
async def ask_question(request: QuestionRequest):
    """Ask a question and get an answer based on uploaded documents.

    `QuestionRequest.question` already rejects empty strings via
    `min_length=1`, so an empty query never reaches here — FastAPI returns
    a 422 with the validation detail first.
    """
    try:
        result = rag_pipeline.ask_question(request.question)
        return AskResponse(**result)
    except InferenceError as e:
        app_logger.error(f"Model server error while answering question: {e}")
        raise HTTPException(status_code=502, detail="The language model server is unavailable. Please try again shortly.")
    except Exception as e:
        app_logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail="Error processing question")


@app.get("/health", response_model=HealthResponse, summary="Health check")
async def health_check():
    """Report service health, including whether a vector index exists and
    whether the OpenAI credentials required for embeddings are configured."""
    if rag_pipeline.retriever.vectorstore is None:
        rag_pipeline.retriever.load_vectorstore()

    return HealthResponse(
        status="healthy",
        vectorstore_ready=rag_pipeline.retriever.vectorstore is not None,
        openai_configured=bool(Config.OPENAI_API_KEY),
    )


@app.get("/analytics/summary", response_model=AnalyticsSummaryResponse,
         summary="Query and retrieval analytics", dependencies=[Depends(verify_api_key)])
async def analytics_summary():
    """Aggregate stats over every logged query: latency distribution,
    similarity distribution and an approximate precision@k."""
    df = QueryAnalytics().load()
    return summarize(df)


@app.get("/stats", response_model=StatsResponse, summary="System/scale stats",
         dependencies=[Depends(verify_api_key)])
async def stats():
    """Real system/scale stats, computed from the live vector index and
    query log rather than hardcoded: how much has been indexed, and what a
    typical query costs in tokens/dollars."""
    vector_stats = rag_pipeline.retriever.get_stats()

    df = QueryAnalytics().load()
    avg_tokens = float(df["estimated_tokens"].mean()) if not df.empty and "estimated_tokens" in df else 0.0
    avg_cost = float(df["estimated_cost_usd"].mean()) if not df.empty and "estimated_cost_usd" in df else 0.0

    cache_stats = retrieval_cache.stats()
    return StatsResponse(
        total_documents_indexed=vector_stats["total_documents"],
        total_chunks=vector_stats["total_chunks"],
        vector_db_size_bytes=vector_stats["vector_db_size_bytes"],
        avg_tokens_per_query=avg_tokens,
        estimated_cost_per_query_usd=avg_cost,
        **cache_stats,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
