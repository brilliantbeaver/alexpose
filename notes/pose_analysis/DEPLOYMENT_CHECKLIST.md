# Pose Analysis Fix - Deployment Checklist

## Pre-Deployment

### Code Review
- [x] Changes implemented in `server/routers/pose_analysis.py`
- [x] Service caching pattern added
- [x] All 7 endpoints updated to use cached service
- [x] No syntax errors or diagnostics
- [x] Code follows existing patterns

### Testing
- [x] Unit tests created (`tests/test_pose_analysis_service_caching.py`)
- [x] All tests passing (4/4)
- [x] No regressions in existing tests
- [ ] Manual testing with real server (recommended)
- [ ] Load testing (recommended)

### Documentation
- [x] Executive summary created (`POSE_ANALYSIS_FIX_SUMMARY.md`)
- [x] Detailed fix documentation (`notes/pose_analysis/POSE_ANALYSIS_PERFORMANCE_FIX.md`)
- [x] Quick reference guide (`notes/pose_analysis/QUICK_FIX_REFERENCE.md`)
- [x] Architecture diagrams (`notes/pose_analysis/ARCHITECTURE_DIAGRAM.md`)
- [x] Deployment checklist (this file)

## Deployment Steps

### 1. Backup Current State
```bash
# Backup current code
git stash save "backup-before-pose-analysis-fix"

# Or commit current state
git add .
git commit -m "Backup before pose analysis fix"
```

### 2. Apply Changes
```bash
# Changes are already in place
# Verify files modified:
git status
```

Expected modified files:
- `server/routers/pose_analysis.py`
- `tests/test_pose_analysis_service_caching.py` (new)
- Documentation files (new)

### 3. Run Tests
```bash
# Run new tests
python -m pytest tests/test_pose_analysis_service_caching.py -v

# Run all tests to check for regressions
python -m pytest tests/ -v

# Expected: All tests pass
```

### 4. Start Server
```bash
# Start the server
uvicorn server.main:app --reload

# Or use your existing start command
```

### 5. Verify Logs
Watch for initialization messages:
```
✅ GOOD: See initialization once at startup
❌ BAD: See initialization on every request
```

Expected log pattern:
```
[Startup]
INFO | Feature extractor initialized for COCO_17 format
INFO | Temporal analyzer initialized with heel_strike detection
INFO | Symmetry analyzer initialized for COCO_17 format
INFO | Enhanced gait analyzer initialized for COCO_17 format
INFO | Pose analysis service initialized

[Request 1]
DEBUG | Creating new PoseAnalysisServiceAPI instance  ← First time only
INFO | Analyzing sequence...
INFO | Returning cached analysis...

[Request 2]
INFO | Analyzing sequence...  ← No initialization!
INFO | Returning cached analysis...

[Request 3]
INFO | Analyzing sequence...  ← No initialization!
INFO | Returning cached analysis...
```

## Post-Deployment Verification

### 1. Functional Testing
```bash
# Test main endpoint
curl http://localhost:8000/api/v1/pose-analysis/sequence/{dataset_id}/{sequence_id}

# Expected: 200 OK with analysis results
```

### 2. Performance Testing
```bash
# Test response time (first request)
time curl http://localhost:8000/api/v1/pose-analysis/sequence/{dataset_id}/{sequence_id}

# Test response time (subsequent requests)
time curl http://localhost:8000/api/v1/pose-analysis/sequence/{dataset_id}/{sequence_id}

# Expected: Subsequent requests faster
```

### 3. Memory Monitoring
```bash
# Monitor server memory usage
# Windows: Task Manager → Details → python.exe
# Linux: top -p $(pgrep -f uvicorn)

# Make multiple requests and verify memory stays stable
```

### 4. Log Analysis
```bash
# Check logs for initialization pattern
# Should see initialization only once per server startup

# Count initialization messages
grep "Pose analysis service initialized" logs/*.log | wc -l
# Expected: 1 (or number of server restarts)
```

### 5. Frontend Testing
1. Open frontend: `http://localhost:3000/training/gavd/{dataset_id}`
2. Navigate to "Pose Analysis" tab
3. Select a sequence
4. Verify analysis loads correctly
5. Switch between sequences
6. Verify no errors in browser console

## Rollback Plan

If issues occur:

### Option 1: Git Revert
```bash
# Revert the changes
git revert HEAD

# Or restore from backup
git stash pop
```

### Option 2: Manual Rollback
Restore original code in `server/routers/pose_analysis.py`:
```python
# Change this:
service = _get_service(config_manager)

# Back to this:
service = PoseAnalysisServiceAPI(config_manager)
```

## Monitoring

### Key Metrics to Watch

1. **Response Times**
   - Baseline: ~400ms (with initialization)
   - Target: ~150ms (without initialization)
   - Monitor: Average response time over 100 requests

2. **Memory Usage**
   - Baseline: Growing with each request
   - Target: Stable memory usage
   - Monitor: Server memory over 1 hour

3. **Error Rate**
   - Baseline: 0% (API was working)
   - Target: 0% (should not increase)
   - Monitor: HTTP 500 errors

4. **Cache Hit Rate**
   - Target: >95% (most requests use cache)
   - Monitor: Log messages about cache hits

### Alert Conditions

Set up alerts for:
- Response time > 500ms (degradation)
- Memory usage > 2GB (potential leak)
- Error rate > 1% (functionality issue)
- Cache hit rate < 90% (caching issue)

## Success Criteria

✅ Fix is successful if:
1. Server starts without errors
2. All tests pass
3. Initialization logs appear only once
4. Response times improve (after first request)
5. Memory usage remains stable
6. Frontend receives analysis results
7. No increase in error rate

## Troubleshooting

### Issue: Service not cached
**Symptom:** Still see initialization on every request
**Solution:** 
- Check `_service_cache` is module-level
- Verify `_get_service()` is being called
- Check for typos in cache key

### Issue: Stale service instance
**Symptom:** Service uses old configuration
**Solution:**
- Restart server to clear cache
- Or implement cache invalidation logic

### Issue: Memory still growing
**Symptom:** Memory increases over time
**Solution:**
- Check for other memory leaks
- Profile with memory_profiler
- Review analyzer implementations

### Issue: Tests failing
**Symptom:** Unit tests don't pass
**Solution:**
- Check mock setup
- Verify imports
- Clear cache between tests

## Contact

For issues or questions:
- Check documentation in `notes/pose_analysis/`
- Review test cases in `tests/test_pose_analysis_service_caching.py`
- Consult `POSE_ANALYSIS_FIX_SUMMARY.md` for overview

## Sign-Off

- [ ] Code reviewed
- [ ] Tests passing
- [ ] Documentation complete
- [ ] Deployment successful
- [ ] Verification complete
- [ ] Monitoring in place

**Deployed by:** _________________  
**Date:** _________________  
**Verified by:** _________________  
**Date:** _________________
