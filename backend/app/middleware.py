import logging
import time
import uuid
from collections import defaultdict, deque
from threading import Lock

from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse

from .config import settings

try:
    from fastapi_limiter import FastAPILimiter
    from fastapi_limiter.depends import RateLimiter
except ImportError:  # pragma: no cover - exercised when optional deps are absent
    FastAPILimiter = None
    RateLimiter = None

logger = logging.getLogger("ai_assistant.api")

_rate_limit_buckets: dict[str, deque[float]] = defaultdict(deque)
_rate_limit_lock = Lock()


def get_client_key(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    if xff:
        return xff
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


async def request_id_and_logging_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    started_at = time.perf_counter()
    logger.info(
        "request_started request_id=%s method=%s path=%s",
        request_id,
        request.method,
        request.url.path,
    )

    response = await call_next(request)

    elapsed_ms = (time.perf_counter() - started_at) * 1000
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request_finished request_id=%s method=%s path=%s status=%s elapsed_ms=%.2f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


async def request_size_limit_middleware(request: Request, call_next):
    if request.method not in {"POST", "PUT", "PATCH"}:
        return await call_next(request)

    content_length = request.headers.get("content-length")
    if content_length:
        try:
            declared_size = int(content_length)
            if declared_size > settings.max_request_bytes:
                return JSONResponse(
                    status_code=413,
                    content={
                        "error": "payload_too_large",
                        "detail": f"Request body exceeds {settings.max_request_bytes} bytes limit.",
                    },
                )
        except ValueError:
            pass

    return await call_next(request)


async def rate_limit_middleware(request: Request, call_next):
    limiter_response = Response()
    await dynamic_rate_limiter(request, limiter_response)
    response = await call_next(request)
    response.headers.update(limiter_response.headers)
    return response


async def dynamic_rate_limiter(request: Request, response: Response):
    """Use Redis-backed limits when initialized, otherwise fall back in memory."""
    if (
        FastAPILimiter is not None
        and RateLimiter is not None
        and getattr(FastAPILimiter, "redis", None) is not None
    ):
        try:
            limiter = RateLimiter(
                times=settings.rate_limit_requests,
                seconds=settings.rate_limit_window_seconds,
            )
            await limiter(request=request, response=response)
            return
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning("redis_rate_limit_failed detail=%s", str(exc))

    client_key = get_client_key(request)
    now = time.time()
    cutoff = now - settings.rate_limit_window_seconds

    with _rate_limit_lock:
        bucket = _rate_limit_buckets[client_key]
        while bucket and bucket[0] < cutoff:
            bucket.popleft()

        if len(bucket) >= settings.rate_limit_requests:
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Too many requests. Limit is {settings.rate_limit_requests} requests "
                    f"per {settings.rate_limit_window_seconds} seconds."
                ),
                headers={
                    "Retry-After": str(settings.rate_limit_window_seconds),
                    "X-RateLimit-Limit": str(settings.rate_limit_requests),
                    "X-RateLimit-Remaining": "0",
                },
            )

        bucket.append(now)
        remaining = settings.rate_limit_requests - len(bucket)

    response.headers["X-RateLimit-Limit"] = str(settings.rate_limit_requests)
    response.headers["X-RateLimit-Remaining"] = str(remaining)
