"""
QyverixAI — Backend API
FastAPI application with advanced middleware, rate limiting, and full analysis engine.
"""

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import settings
from .middleware import dynamic_rate_limiter
from .observability import (
    initialise_app_info,
    prometheus_metrics_middleware,
)
from .routers import (
    analyze,
    auth,
    chat,
    debugging,
    explanation,
    history,
    share,
    subscribe,
    suggestions,
    upload_file,
    user_data,
)
from .routers import health as health_router
from .routers import metrics as metrics_router
from .schemas import HealthResponse
from .services import database
from .services.scheduler import start_scheduler, stop_scheduler

try:
    import redis.asyncio as redis
    from fastapi_limiter import FastAPILimiter
except ImportError:  # pragma: no cover - exercised when optional deps are absent
    redis = None
    FastAPILimiter = None


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    await database.init_db()
    print("🚀 QyverixAI backend starting…")
    logging.getLogger(__name__).info("QyverixAI backend starting")

    if settings.redis_url and redis is not None and FastAPILimiter is not None:
        try:
            redis_client = redis.from_url(
                settings.redis_url, encoding="utf-8", decode_responses=True
            )
            await redis_client.ping()
            await FastAPILimiter.init(redis_client)
            logging.getLogger(__name__).info("Redis rate limiter initialized")
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "redis_rate_limiter_init_failed detail=%s", str(exc)
            )

    # Static info gauge so dashboards can pin version / provider labels.
    initialise_app_info(
        version="3.0.0", ai_provider=os.getenv("AI_PROVIDER", "rule-based")
    )
    start_scheduler()
    yield
    stop_scheduler()
    logging.getLogger(__name__).info("🛑 QyverixAI backend shutting down…")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="QyverixAI",
    description="AI-powered developer assistant — code explanation, debugging, and improvement.",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── Middleware ────────────────────────────────────────────────────────────────
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Prometheus instrumentation — installed early so it observes every other
# middleware's response (including rate-limited 429s and the global error
# handler's 500s). The middleware self-disables per-request when
# METRICS_ENABLED is false, so installing it unconditionally costs nothing
# operationally and lets operators flip the flag without a restart.
app.middleware("http")(prometheus_metrics_middleware)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.perf_counter()

    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed:.2f}"
    response.headers["X-QyverixAI-Version"] = "3.0.0"
    return response


@app.middleware("http")
async def add_cache_header(request: Request, call_next):
    response = await call_next(request)

    if request.url.path == "/analyze/" and request.method == "POST":
        response.headers.setdefault("X-Cache", "MISS")

    return response


# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(
    explanation.router,
    prefix="/explanation",
    tags=["Explanation"],
    dependencies=[Depends(dynamic_rate_limiter)],
)
app.include_router(
    debugging.router,
    prefix="/debugging",
    tags=["Debugging"],
    dependencies=[Depends(dynamic_rate_limiter)],
)
app.include_router(
    suggestions.router,
    prefix="/suggestions",
    tags=["Suggestions"],
    dependencies=[Depends(dynamic_rate_limiter)],
)
app.include_router(
    analyze.router,
    prefix="/analyze",
    tags=["Full Analysis"],
    dependencies=[Depends(dynamic_rate_limiter)],
)
app.include_router(subscribe.router, prefix="/subscribe", tags=["Subscription"])
app.include_router(history.router, prefix="/history", tags=["History"])
app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(share.router)
app.include_router(user_data.router)
app.include_router(upload_file.router, prefix="/upload", tags=["Upload File"])


# Operational endpoints: /healthz/live, /healthz/ready, /metrics
app.include_router(health_router.router)
app.include_router(metrics_router.router)


# ── Core Endpoints ────────────────────────────────────────────────────────────
@app.get("/", response_model=HealthResponse, tags=["System"])
async def root():
    return {
        "status": "ok",
        "version": "3.0.0",
        "message": "QyverixAI API is running.",
        "endpoints": [
            "/auth/signup",
            "/auth/login",
            "/auth/me",
            "/explanation/",
            "/debugging/",
            "/suggestions/",
            "/analyze/",
            "/subscribe/",
            "/share/",
            "/auth/",
            "/chat/",
            "/user/",
            "/analyze/zip/",
            "/analyze/zip/",
            "/subscribe/",
            "/share/",
            "/history/",
        ],
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    return {
        "status": "ok",
        "version": "3.0.0",
        "message": "QyverixAI is healthy",
        "endpoints": [
            "/auth/signup",
            "/auth/login",
            "/auth/me",
            "/explanation/",
            "/debugging/",
            "/suggestions/",
            "/analyze/",
            "/subscribe/",
            "/share/",
            "/auth/",
            "/chat/",
            "/user/",
            "/analyze/zip/",
            "/history/",
        ],
    }


@app.get("/ping", tags=["System"])
async def ping():
    return {"message": "pong"}


# ── Static / Frontend ─────────────────────────────────────────────────────────
_frontend = os.path.join(os.path.dirname(__file__), "..", "..", "frontend")
if os.path.isdir(_frontend):
    app.mount("/app", StaticFiles(directory=_frontend, html=True), name="frontend")


# ── Global error handler ──────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logging.exception("Unhandled error")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again."},
    )
