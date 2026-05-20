# KisanCredit Load Testing Scenarios

This document describes load testing scenarios for the KisanCredit API using Locust.

## Quick Start

```bash
# Install Locust
pip install locust

# Start API server
uvicorn src.api.main:app --reload

# Run load test
locust -f tests/locustfile.py --host=http://localhost:8000

# Open web UI
# Navigate to: http://localhost:8089
```

## Test Scenarios

### 1. Normal Load Test
**Purpose**: Validate performance under expected daily load

**Configuration**:
- Users: 50-100
- Spawn rate: 5 users/sec
- Duration: 10 minutes
- User class: `KisanCreditUser`

**Expected Results**:
- P95 latency: <100ms
- Error rate: <1%
- Throughput: >500 req/sec

**Command**:
```bash
locust -f tests/locustfile.py --host=http://localhost:8000 \
  --users 100 --spawn-rate 5 --run-time 10m
```

### 2. Stress Test
**Purpose**: Find breaking point and validate graceful degradation

**Configuration**:
- Users: 500-1000
- Spawn rate: 20 users/sec
- Duration: 5 minutes
- User class: `StressTestUser`

**Expected Results**:
- System remains responsive
- Rate limiting engages properly
- No crashes or data loss

**Command**:
```bash
locust -f tests/locustfile.py --host=http://localhost:8000 \
  --user-classes StressTestUser \
  --users 1000 --spawn-rate 20 --run-time 5m
```

### 3. Batch Processing Test
**Purpose**: Validate batch endpoint performance

**Configuration**:
- Users: 20-50
- Spawn rate: 2 users/sec
- Duration: 10 minutes
- User class: `HeavyBatchUser`

**Expected Results**:
- Batch P95 latency: <500ms
- Throughput: >1000 predictions/sec
- No timeout errors

**Command**:
```bash
locust -f tests/locustfile.py --host=http://localhost:8000 \
  --user-classes HeavyBatchUser \
  --users 50 --spawn-rate 2 --run-time 10m
```

### 4. Spike Test
**Purpose**: Validate recovery from sudden traffic spike

**Configuration**:
- Phase 1: 50 users for 5 minutes (baseline)
- Phase 2: Spike to 500 users instantly
- Phase 3: Hold 500 users for 2 minutes
- Phase 4: Drop to 50 users
- Phase 5: Observe recovery for 5 minutes

**Manual execution**:
1. Start with 50 users
2. After 5 min, increase to 500
3. Monitor for 2 min
4. Reduce to 50
5. Observe for 5 min

**Expected Results**:
- API handles spike without crashing
- Rate limiter protects system
- Quick recovery to normal performance

### 5. Endurance Test
**Purpose**: Validate stability over extended period

**Configuration**:
- Users: 200
- Spawn rate: 10 users/sec
- Duration: 4 hours
- User class: `KisanCreditUser`

**Expected Results**:
- No memory leaks
- Consistent performance throughout
- No degradation over time

**Command**:
```bash
locust -f tests/locustfile.py --host=http://localhost:8000 \
  --users 200 --spawn-rate 10 --run-time 4h
```

## Monitoring During Tests

### 1. Grafana Dashboards
Access: http://localhost:3000

Monitor:
- API Performance dashboard
- System Health dashboard
- Model Metrics dashboard

### 2. Locust Web UI
Access: http://localhost:8089

Monitor:
- Request statistics
- Response time percentiles
- Failures
- Users over time

### 3. Prometheus Metrics
Access: http://localhost:9090

Key queries:
```promql
# Request rate
rate(http_requests_total[1m])

# P95 latency
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# Error rate
100 * sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m]))
```

## Performance Targets

| Metric | Target | Critical Threshold |
|--------|--------|-------------------|
| P95 Latency | <100ms | <200ms |
| P99 Latency | <200ms | <500ms |
| Throughput | >500 req/sec | >300 req/sec |
| Error Rate | <1% | <5% |
| CPU Usage | <70% | <90% |
| Memory Usage | <80% | <95% |
| Cache Hit Rate | >70% | >50% |

## Common Issues & Solutions

### Issue: High Error Rate (429 Too Many Requests)
**Cause**: Rate limiter protecting system
**Solution**: Reduce user count or increase rate limits in .env
```
RATE_LIMIT_REQUESTS=1000
RATE_LIMIT_WINDOW=900
```

### Issue: Timeouts
**Cause**: Database or model inference bottleneck
**Solution**:
- Check database connection pool size
- Verify model is loaded in memory
- Check Redis connectivity

### Issue: Memory Growth
**Cause**: Potential memory leak
**Solution**:
- Monitor with `docker stats`
- Check for unreleased resources
- Review SHAP explainer caching

### Issue: Inconsistent Latency
**Cause**: Cold start or GC pauses
**Solution**:
- Warm up API before test: `locust --headless --users 10 --run-time 2m`
- Increase uvicorn workers: `--workers 8`
- Tune Python GC settings

## Results Interpretation

### Good Results
- P95 < 100ms consistently
- Error rate < 1%
- Throughput scales linearly with users
- No degradation over time

### Warning Signs
- P95 > 200ms
- Error rate 1-5%
- Throughput plateaus early
- Gradual performance degradation

### Critical Issues
- P95 > 500ms
- Error rate > 5%
- Crashes or restarts
- Memory leaks

## Headless Mode (CI/CD)

For automated testing in CI/CD:

```bash
# Run test without web UI
locust -f tests/locustfile.py --host=http://localhost:8000 \
  --headless --users 100 --spawn-rate 10 --run-time 5m \
  --html results/load_test_report.html \
  --csv results/load_test_stats

# Exit with error code if failure rate > 1%
locust ... --exit-code-on-error 1
```

## Next Steps

1. **Baseline Metrics**: Run normal load test, document results
2. **Optimize**: Identify bottlenecks, optimize code
3. **Retest**: Verify improvements
4. **Automate**: Integrate into CI/CD pipeline
5. **Alert**: Set up monitoring alerts based on thresholds

## References

- [Locust Documentation](https://docs.locust.io/)
- [Performance Testing Guide](https://martinfowler.com/articles/practical-test-pyramid.html#IntegrationTests)
- [SRE Book: Load Testing](https://sre.google/workbook/load-balancing/)
