"""Prometheus metrics for monitoring."""

from prometheus_client import Counter, Histogram, Gauge


# API Metrics
http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"]
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    ["method", "endpoint"]
)

http_requests_in_progress = Gauge(
    "http_requests_in_progress",
    "HTTP requests in progress",
    ["method", "endpoint"]
)

# Model Metrics
model_predictions_total = Counter(
    "model_predictions_total",
    "Total model predictions",
    ["model_version"]
)

model_inference_duration_seconds = Histogram(
    "model_inference_duration_seconds",
    "Model inference latency",
    ["model_version"]
)

model_profitability_score = Histogram(
    "model_profitability_score",
    "Profitability score distribution",
    buckets=[0, 20, 40, 60, 80, 100]
)

# Database Metrics
database_connections_active = Gauge(
    "database_connections_active",
    "Active database connections"
)

database_query_duration_seconds = Histogram(
    "database_query_duration_seconds",
    "Database query latency",
    ["query_type"]
)

# Cache Metrics
cache_hits_total = Counter(
    "cache_hits_total",
    "Total cache hits"
)

cache_misses_total = Counter(
    "cache_misses_total",
    "Total cache misses"
)

cache_hit_rate = Gauge(
    "cache_hit_rate",
    "Cache hit rate percentage"
)
