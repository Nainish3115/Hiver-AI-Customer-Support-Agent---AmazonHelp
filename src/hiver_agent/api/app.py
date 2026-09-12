"""
FastAPI REST Application for Hiver AI Support Agent.
Exposes the validated CustomerSupportPipeline via clean, high-performance, hardened REST APIs.
"""
import sys
import os
import time
import uuid
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

root_dir = str(Path(__file__).resolve().parent.parent.parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, "src"))

from hiver_agent.pipeline import CustomerSupportPipeline, EndToEndResult
from hiver_agent.api.schemas import (
    SupportProcessRequest,
    SupportProcessResponse,
    HealthResponse,
    ReadyResponse,
    StatsResponse,
    RetrievalSearchRequest,
    RetrievalSearchResponse,
    RetrievalSearchResult,
    APIErrorDetail,
    APIErrorResponse
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("hiver_agent.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager: Initializes singleton CustomerSupportPipeline on server startup.
    Ensures model, FAISS index, and classifier are loaded once and reused across all requests.
    """
    t_start = time.time()
    logger.info("Initializing Hiver AI Support Agent singleton pipeline on startup...")
    try:
        pipeline = CustomerSupportPipeline(intent_classifier_mode="calibrated_prototype_hybrid")
        app.state.pipeline = pipeline
        app.state.pipeline_ready = True
        t_load = time.time() - t_start
        logger.info(f"Pipeline successfully initialized in {t_load:.2f}s.")
    except Exception as e:
        logger.error(f"Failed to initialize CustomerSupportPipeline: {e}", exc_info=True)
        app.state.pipeline = None
        app.state.pipeline_ready = False

    yield

    logger.info("Shutting down Hiver AI Support Agent API server...")


# Create FastAPI application instance
app = FastAPI(
    title="Hiver AI Customer Support Agent API",
    description=(
        "Production-hardened REST API for AmazonHelp customer support automation. "
        "Provides end-to-end intent classification, hybrid RAG retrieval, evidence selection, "
        "Gemini 1.5 Flash response generation, safety validation, and risk-aware escalation decisioning."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# CORS Middleware Configuration
allowed_origins_env = os.environ.get("ALLOWED_ORIGINS", "")
allowed_origins = [origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()]
if not allowed_origins:
    allowed_origins = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173"
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request ID & Structured Logging Middleware
@app.middleware("http")
async def request_id_and_logging_middleware(request: Request, call_next):
    # Extract or generate X-Request-ID header
    req_id = request.headers.get("X-Request-ID")
    if not req_id or not req_id.strip():
        req_id = f"req-{uuid.uuid4().hex[:12]}"
    else:
        req_id = req_id.strip()

    request.state.request_id = req_id
    t_start = time.time()

    # Call route handler
    response = await call_next(request)

    t_elapsed_ms = round((time.time() - t_start) * 1000.0, 2)
    response.headers["X-Request-ID"] = req_id

    # Structured Operational Logging (excluding credentials/messages/secrets)
    logger.info(
        f"[{req_id}] {request.method} {request.url.path} "
        f"-> status={response.status_code} latency={t_elapsed_ms}ms"
    )

    return response


# Helper to build standardized error response JSON
def build_error_json_response(code: str, message: str, detail: Optional[str], status_code: int, request_id: Optional[str]) -> JSONResponse:
    err_detail = APIErrorDetail(
        code=code,
        message=message,
        detail=detail,
        request_id=request_id
    )
    content = {
        "error": err_detail.model_dump(),
        "code": code,
        "message": message,
        "detail": detail,
        "request_id": request_id
    }
    headers = {"X-Request-ID": request_id} if request_id else {}
    return JSONResponse(status_code=status_code, content=content, headers=headers)


# Exception Handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    req_id = getattr(request.state, "request_id", None)
    logger.warning(f"[{req_id}] Request validation error on {request.url.path}: {exc}")
    return build_error_json_response(
        code="INVALID_REQUEST",
        message="Request validation failed.",
        detail=str(exc),
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        request_id=req_id
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    req_id = getattr(request.state, "request_id", None)
    msg = exc.detail if isinstance(exc.detail, str) else "HTTP Exception"
    return build_error_json_response(
        code=f"HTTP_{exc.status_code}",
        message=msg,
        detail=None,
        status_code=exc.status_code,
        request_id=req_id
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    req_id = getattr(request.state, "request_id", None)
    logger.error(f"[{req_id}] Unhandled server error on {request.url.path}: {exc}", exc_info=True)
    return build_error_json_response(
        code="INTERNAL_SERVER_ERROR",
        message="An internal server error occurred while processing the request.",
        detail=None,
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        request_id=req_id
    )


# Routes
@app.get(
    "/api/v1/health",
    response_model=HealthResponse,
    summary="Health check endpoint (Liveness)",
    tags=["System"]
)
async def health_check(request: Request):
    """
    Liveness probe endpoint: Returns basic application process health.
    """
    req_id = getattr(request.state, "request_id", None)
    is_ready = getattr(app.state, "pipeline_ready", False)
    status_str = "ok" if is_ready else "error"
    return HealthResponse(
        status=status_str,
        service="hiver-ai-support-agent",
        pipeline_ready=is_ready,
        version="0.1.0",
        request_id=req_id
    )


@app.get(
    "/api/v1/ready",
    response_model=ReadyResponse,
    summary="Readiness check endpoint (Readiness Probe)",
    tags=["System"]
)
async def readiness_check(request: Request):
    """
    Readiness probe endpoint: Verifies runtime pipeline state, FAISS vector store index loading,
    and intent classifier availability before accepting customer support traffic. Fast non-blocking check.
    """
    req_id = getattr(request.state, "request_id", None)
    pipeline = getattr(app.state, "pipeline", None)
    is_ready = getattr(app.state, "pipeline_ready", False) and pipeline is not None

    doc_count = 0
    if is_ready and hasattr(pipeline, "base_retriever") and hasattr(pipeline.base_retriever, "vector_store"):
        doc_count = getattr(pipeline.base_retriever.vector_store, "count", lambda: 0)()

    if not is_ready or doc_count == 0:
        err_res = ReadyResponse(
            status="not_ready",
            pipeline_ready=False,
            vector_store_documents=doc_count,
            version="0.1.0",
            request_id=req_id
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=err_res.model_dump(),
            headers={"X-Request-ID": req_id} if req_id else {}
        )

    return ReadyResponse(
        status="ready",
        pipeline_ready=True,
        vector_store_documents=doc_count,
        version="0.1.0",
        request_id=req_id
    )


@app.post(
    "/api/v1/support/process",
    response_model=SupportProcessResponse,
    summary="Process customer inquiry end-to-end",
    tags=["Customer Support"]
)
async def process_support_message(payload: SupportProcessRequest, request: Request):
    """
    Executes the full end-to-end customer support RAG & decision pipeline:
    Intent Classification -> Hybrid Dense/Sparse Retrieval -> Evidence Selection ->
    Gemini 1.5 Flash Generation (with deterministic fallback) -> Grounding Safety Checker -> RiskAware Policy Decision.
    """
    req_id = getattr(request.state, "request_id", None)
    if not getattr(app.state, "pipeline_ready", False) or app.state.pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline service unavailable or not fully initialized."
        )

    t_start = time.time()
    try:
        result: EndToEndResult = app.state.pipeline.process_end_to_end(
            customer_message=payload.customer_message,
            case_id=payload.case_id,
            policy_type=payload.policy_type
        )
        t_elapsed_ms = round((time.time() - t_start) * 1000.0, 2)

        # Log process metadata without raw message secrets
        logger.info(
            f"[{req_id}] Support process complete: intent={result.predicted_intent} "
            f"decision={result.decision} risk={result.risk_score:.4f} latency={t_elapsed_ms}ms"
        )

        return SupportProcessResponse(
            case_id=result.case_id,
            customer_message=result.customer_message,
            predicted_intent=result.predicted_intent,
            intent_confidence=result.intent_confidence,
            decision=result.decision,
            reason_code=result.reason_code,
            decision_reason=result.decision_reason,
            draft_response=result.draft_response,
            risk_score=result.risk_score,
            safety_check_passed=result.safety_check_passed,
            grounding_status=result.grounding_status,
            selected_evidence_count=result.selected_evidence_count,
            retrieved_case_ids=result.retrieved_case_ids,
            retrieved_scores=result.retrieval_scores,
            signals=result.signals,
            generation_metadata=result.generation_metadata,
            latency_ms=t_elapsed_ms,
            request_id=req_id
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{req_id}] Error processing support message: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process customer support message."
        )


@app.get(
    "/api/v1/stats",
    response_model=StatsResponse,
    summary="Get verified project & benchmark metrics",
    tags=["System"]
)
async def get_project_stats(request: Request):
    """
    Returns authoritative, validated project metrics for dashboard display.
    """
    req_id = getattr(request.state, "request_id", None)
    validated_metrics = {
        "intent_accuracy_untouched_test": {
            "value": 0.5700,
            "evaluation_set": "Untouched Gold Test",
            "n_examples": 100,
            "note": "Authoritative headline metric"
        },
        "intent_macro_f1_untouched_test": {
            "value": 0.5393,
            "evaluation_set": "Untouched Gold Test",
            "n_examples": 100
        },
        "intent_accuracy_full_gold_diagnostic": {
            "value": 0.6900,
            "evaluation_set": "Full Gold Dataset",
            "n_examples": 200,
            "note": "Diagnostic score across full 200 gold dataset"
        },
        "retrieval_recall_at_1": {
            "value": 0.5250,
            "evaluation_set": "Human Gold RAG Benchmark",
            "n_examples": 200,
            "note": "Hybrid Dense + BM25 + Intent Compatible"
        },
        "retrieval_recall_at_5": {
            "value": 0.6100,
            "evaluation_set": "Human Gold RAG Benchmark",
            "n_examples": 200
        },
        "retrieval_mrr": {
            "value": 0.5570,
            "evaluation_set": "Human Gold RAG Benchmark",
            "n_examples": 200
        },
        "gemini_llm_judge_score": {
            "value": 4.72,
            "max_score": 5.0,
            "n_examples": 50,
            "note": "Evaluated on Gemini 1.5 Flash outputs"
        },
        "gemini_safety_pass_rate": {
            "value": 1.0000,
            "n_examples": 50
        },
        "escalation_agreement_rate": {
            "value": 0.7400,
            "evaluation_set": "Human Escalation Validation Set",
            "n_examples": 50
        },
        "escalation_false_auto_handle_rate": {
            "value": 0.0600,
            "evaluation_set": "Human Escalation Validation Set",
            "n_examples": 50,
            "note": "Reduced from 12.0% baseline under Day 4 integrated model"
        },
        "unit_test_count": {
            "value": 130,
            "status": "130 passed, 0 failed, 0 skipped"
        }
    }

    return StatsResponse(
        service="hiver-ai-support-agent",
        version="0.1.0",
        target_brand="AmazonHelp",
        canonical_corpus_size=15316,
        raw_conversation_turns=81186,
        intent_taxonomy_count=17,
        active_intent_classifier="CalibratedRulePrototypeHybrid",
        active_retrieval_mode="Hybrid Dense + BM25 (RRF k=60 + Intent Compatibility)",
        active_llm_model="gemini-1.5-flash",
        active_risk_threshold=0.35,
        metrics=validated_metrics,
        request_id=req_id
    )


@app.post(
    "/api/v1/retrieval/search",
    response_model=RetrievalSearchResponse,
    summary="Search RAG historical case index via POST",
    tags=["Retrieval"]
)
async def search_retrieval_corpus_post(payload: RetrievalSearchRequest, request: Request):
    """
    Search the 15,316 document RAG historical case corpus via POST.
    """
    return execute_retrieval_search(payload.query, payload.top_k, request)


@app.get(
    "/api/v1/retrieval/search",
    response_model=RetrievalSearchResponse,
    summary="Search RAG historical case index via query string",
    tags=["Retrieval"]
)
async def search_retrieval_corpus_get(
    request: Request,
    q: str = Query(..., min_length=1, max_length=1000, description="Search query string"),
    top_k: int = Query(default=10, ge=1, le=50, description="Number of results")
):
    """
    Search the 15,316 document RAG historical case corpus via GET query params.
    """
    return execute_retrieval_search(q, top_k, request)


def execute_retrieval_search(query: str, top_k: int, request: Request) -> RetrievalSearchResponse:
    req_id = getattr(request.state, "request_id", None)
    if not getattr(app.state, "pipeline_ready", False) or app.state.pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline service unavailable or not fully initialized."
        )

    t_start = time.time()
    try:
        raw_results = app.state.pipeline.base_retriever.retrieve(query, top_k=top_k)
        t_elapsed_ms = round((time.time() - t_start) * 1000.0, 2)

        results = [
            RetrievalSearchResult(
                case_id=str(r.case_id),
                customer_problem=r.customer_problem,
                score=round(float(r.score), 4),
                intent=r.intent
            )
            for r in raw_results
        ]

        return RetrievalSearchResponse(
            query=query,
            total_retrieved=len(results),
            results=results,
            latency_ms=t_elapsed_ms,
            request_id=req_id
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{req_id}] Error executing retrieval search: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to execute retrieval search."
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("hiver_agent.api.app:app", host="0.0.0.0", port=8000, reload=True)
