"""Middleware components for FastAPI application."""

import time
from typing import Callable, Dict
from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from collections import defaultdict
from datetime import datetime, timedelta
import asyncio

from ..utils.logger import get_logger
from ..utils.config import settings

logger = get_logger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware using sliding window algorithm.

    Limits requests per IP address within a time window.
    Default: 100 requests per 15 minutes.
    """

    def __init__(self, app, requests_limit: int = None, window_seconds: int = None):
        super().__init__(app)
        self.requests_limit = requests_limit or settings.rate_limit_requests
        self.window_seconds = window_seconds or settings.rate_limit_window
        self.request_counts: Dict[str, list] = defaultdict(list)
        self.cleanup_interval = 60  # Cleanup every 60 seconds
        self.last_cleanup = time.time()

        logger.info(
            "Rate limiter initialized",
            requests_limit=self.requests_limit,
            window_seconds=self.window_seconds
        )

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with rate limiting."""
        # Get client IP
        client_ip = self._get_client_ip(request)

        # Check if rate limit exceeded
        if self._is_rate_limited(client_ip):
            logger.warning(
                "Rate limit exceeded",
                client_ip=client_ip,
                path=request.url.path
            )

            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "RateLimitExceeded",
                    "message": f"Too many requests. Limit: {self.requests_limit} requests per {self.window_seconds // 60} minutes",
                    "retry_after": self._get_retry_after(client_ip)
                }
            )

        # Record request
        self._record_request(client_ip)

        # Periodic cleanup
        if time.time() - self.last_cleanup > self.cleanup_interval:
            self._cleanup_old_requests()

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        remaining = self._get_remaining_requests(client_ip)
        response.headers["X-RateLimit-Limit"] = str(self.requests_limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(time.time() + self.window_seconds))

        return response

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request."""
        # Check X-Forwarded-For header first (for proxies)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()

        # Check X-Real-IP header
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fall back to client host
        return request.client.host if request.client else "unknown"

    def _is_rate_limited(self, client_ip: str) -> bool:
        """Check if client has exceeded rate limit."""
        current_time = time.time()
        cutoff_time = current_time - self.window_seconds

        # Filter requests within window
        recent_requests = [
            req_time for req_time in self.request_counts[client_ip]
            if req_time > cutoff_time
        ]

        return len(recent_requests) >= self.requests_limit

    def _record_request(self, client_ip: str) -> None:
        """Record a request timestamp."""
        self.request_counts[client_ip].append(time.time())

    def _get_remaining_requests(self, client_ip: str) -> int:
        """Get remaining requests for client."""
        current_time = time.time()
        cutoff_time = current_time - self.window_seconds

        recent_requests = [
            req_time for req_time in self.request_counts[client_ip]
            if req_time > cutoff_time
        ]

        return max(0, self.requests_limit - len(recent_requests))

    def _get_retry_after(self, client_ip: str) -> int:
        """Get seconds until rate limit resets."""
        if not self.request_counts[client_ip]:
            return 0

        oldest_request = min(self.request_counts[client_ip])
        reset_time = oldest_request + self.window_seconds
        retry_after = max(0, int(reset_time - time.time()))

        return retry_after

    def _cleanup_old_requests(self) -> None:
        """Clean up old request timestamps to prevent memory growth."""
        current_time = time.time()
        cutoff_time = current_time - self.window_seconds

        for client_ip in list(self.request_counts.keys()):
            # Filter out old requests
            self.request_counts[client_ip] = [
                req_time for req_time in self.request_counts[client_ip]
                if req_time > cutoff_time
            ]

            # Remove empty entries
            if not self.request_counts[client_ip]:
                del self.request_counts[client_ip]

        self.last_cleanup = current_time


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for structured request/response logging."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Log request and response details."""
        # Generate request ID
        request_id = f"req_{int(time.time() * 1000)}"
        request.state.request_id = request_id

        # Start timer
        start_time = time.time()

        # Log request
        logger.info(
            "Request started",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            client_ip=request.client.host if request.client else "unknown"
        )

        # Process request
        try:
            response = await call_next(request)

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000

            # Log response
            logger.info(
                "Request completed",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                processing_time_ms=round(processing_time, 2)
            )

            # Add headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Processing-Time-MS"] = str(round(processing_time, 2))

            return response

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000

            logger.error(
                "Request failed",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                error=str(e),
                processing_time_ms=round(processing_time, 2)
            )
            raise


class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware for performance monitoring and metrics collection."""

    def __init__(self, app):
        super().__init__(app)
        self.metrics = {
            'total_requests': 0,
            'total_errors': 0,
            'latencies': [],
            'endpoint_counts': defaultdict(int),
            'status_counts': defaultdict(int),
            'start_time': time.time()
        }
        self.max_latencies = 10000  # Keep last 10K latencies for percentile calculation

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Monitor request performance."""
        start_time = time.time()

        try:
            response = await call_next(request)

            # Record metrics
            latency = (time.time() - start_time) * 1000
            self._record_metrics(request, response, latency)

            return response

        except Exception as e:
            latency = (time.time() - start_time) * 1000
            self.metrics['total_errors'] += 1
            self._record_metrics(request, None, latency, error=True)
            raise

    def _record_metrics(
        self,
        request: Request,
        response: Response = None,
        latency: float = 0,
        error: bool = False
    ) -> None:
        """Record request metrics."""
        self.metrics['total_requests'] += 1

        # Record latency
        self.metrics['latencies'].append(latency)

        # Keep only last N latencies
        if len(self.metrics['latencies']) > self.max_latencies:
            self.metrics['latencies'] = self.metrics['latencies'][-self.max_latencies:]

        # Record endpoint
        endpoint = f"{request.method} {request.url.path}"
        self.metrics['endpoint_counts'][endpoint] += 1

        # Record status
        if response:
            self.metrics['status_counts'][response.status_code] += 1
        elif error:
            self.metrics['status_counts'][500] += 1

    def get_metrics(self) -> Dict:
        """Get current metrics summary."""
        latencies = self.metrics['latencies']

        if latencies:
            latencies_sorted = sorted(latencies)
            p50 = latencies_sorted[len(latencies_sorted) // 2]
            p95 = latencies_sorted[int(len(latencies_sorted) * 0.95)]
            p99 = latencies_sorted[int(len(latencies_sorted) * 0.99)]
            avg = sum(latencies) / len(latencies)
        else:
            p50 = p95 = p99 = avg = 0

        uptime = time.time() - self.metrics['start_time']

        return {
            'total_requests': self.metrics['total_requests'],
            'total_errors': self.metrics['total_errors'],
            'error_rate': self.metrics['total_errors'] / max(1, self.metrics['total_requests']),
            'uptime_seconds': uptime,
            'avg_latency_ms': round(avg, 2),
            'p50_latency_ms': round(p50, 2),
            'p95_latency_ms': round(p95, 2),
            'p99_latency_ms': round(p99, 2),
            'endpoint_counts': dict(self.metrics['endpoint_counts']),
            'status_counts': dict(self.metrics['status_counts'])
        }


class CORSMiddleware(BaseHTTPMiddleware):
    """Custom CORS middleware for API."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add CORS headers to response."""
        response = await call_next(request)

        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Request-ID"
        response.headers["Access-Control-Max-Age"] = "3600"

        return response
