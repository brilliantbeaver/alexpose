# Pose Analysis Testing Guide

## Quick Start Testing

### Prerequisites
- Python environment with all dependencies installed
- Node.js and npm installed
- GAVD dataset uploaded and processed with pose data

### Step 1: Start Backend Server

```bash
cd server
python -m uvicorn main:app --reload
```

**Expected Output**:
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

**Verify Health Check**:
```bash
curl http://localhost:8000/api/v1/pose-analysis/health
```

Expected response:
```json
{
  "status": "healthy",
  "service": "pose_analysis",
  "version": "1.0.0"
}
```

### Step 2: Start Frontend Dev Server

```bash
cd frontend
npm run dev
```

**Expected Output**:
```
> frontend@0.1.0 dev
> next dev

   ▲ Next.js 14.x.x
   - Local:        http://localhost:3000
   - Ready in X.Xs
```

### Step 3: Navigate to GAVD Dataset

1. Open browser: http://localhost:3000
2. Navigate to Training → GAVD
3. Select a processed dataset
4. You should see the dataset analysis page

### Step 4: Test Pose Analysis Tab

#### Test Case 1: Basic Loading
1. Click on "Pose Analysis" tab
2. **Expected**: Loading spinner appears
3. **Expected**: After 1-2 seconds, analysis results display
4. **Verify**: All cards render correctly
5. **Verify**: No console errors

#### Test Case 2: Sequence Selection
1. Use the sequence dropdown to select different sequences
2. **Expected**: Loading spinner appears
3. **Expected**: Analysis updates for new sequence
4. **Verify**: Data changes appropriately

#### Test Case 3: Refresh Button
1. Click "Refresh Analysis" button
2. **Expected**: Loading spinner appears
3. **Expected**: Analysis reloads (may be from cache)
4. **Verify**: Data remains consistent

#### Test Case 4: Error Handling - No Pose Data
1. Select a sequence without pose data (if available)
2. **Expected**: Error message displays
3. **Expected**: Error message says "No pose data available for this sequence"
4. **Verify**: UI remains stable

#### Test Case 5: Error Handling - Network Error
1. Stop the backend server
2. Try to load pose analysis
3. **Expected**: Error message displays
4. **Expected**: Error message indicates connection failure
5. **Verify**: UI remains stable
6. Restart backend server

#### Test Case 6: Tab Switching
1. Switch between different tabs (Overview, Sequences, Visualization, Pose)
2. Return to Pose Analysis tab
3. **Expected**: Analysis data persists (doesn't reload unnecessarily)
4. **Verify**: No flickering or layout shifts

## Detailed Verification Checklist

### Overall Assessment Card
- [ ] Overall Level badge displays (Good/Moderate/Poor)
- [ ] Confidence level displays
- [ ] Symmetry classification displays
- [ ] Symmetry score displays (0.000 format)
- [ ] Icons render correctly
- [ ] Colors match assessment level (green=good, yellow=moderate, red=poor)

### Key Metrics Cards
- [ ] Cadence card shows value in steps/minute
- [ ] Cadence level badge displays
- [ ] Stability card shows level badge
- [ ] Gait Cycles card shows count
- [ ] Gait Cycles shows average duration
- [ ] Movement card shows consistency and smoothness

### Recommendations Section
- [ ] Recommendations list displays (if available)
- [ ] Each recommendation has checkmark icon
- [ ] Recommendations are readable and formatted correctly
- [ ] Section only shows if recommendations exist

### Sequence Information Card
- [ ] Frames count displays
- [ ] Duration displays in seconds (X.XX format)
- [ ] FPS displays
- [ ] Format displays (e.g., COCO_17)
- [ ] Performance metrics display (analysis time, processing speed)

### Asymmetry Details Card
- [ ] Most asymmetric joints list displays (if available)
- [ ] Each joint shows name and asymmetry value
- [ ] Badges show correct severity (High/Moderate/Low)
- [ ] Section only shows if asymmetric joints exist

### Loading State
- [ ] Spinner animation is smooth
- [ ] Loading message displays
- [ ] Card structure is maintained during loading
- [ ] No layout shift when loading completes

### Error State
- [ ] Error alert displays with red styling
- [ ] Error icon shows
- [ ] Error message is clear and helpful
- [ ] No console errors

## Performance Testing

### Test 1: Initial Load Time
1. Clear browser cache
2. Navigate to Pose Analysis tab
3. **Measure**: Time from click to data display
4. **Expected**: 1-3 seconds for first load

### Test 2: Cached Load Time
1. Load analysis for a sequence
2. Switch to another tab
3. Return to Pose Analysis tab
4. **Measure**: Time to display
5. **Expected**: <100ms (instant from cache)

### Test 3: Sequence Switching
1. Load analysis for sequence A
2. Switch to sequence B
3. **Measure**: Time to load new analysis
4. **Expected**: 1-3 seconds (or <100ms if cached)

### Test 4: Multiple Sequences
1. Load analysis for 5 different sequences
2. **Verify**: No memory leaks
3. **Verify**: Performance remains consistent
4. **Check**: Browser memory usage (should be stable)

## Browser Compatibility Testing

Test on the following browsers:
- [ ] Chrome (latest)
- [ ] Firefox (latest)
- [ ] Safari (latest)
- [ ] Edge (latest)

For each browser, verify:
- [ ] Layout renders correctly
- [ ] Colors display correctly
- [ ] Icons render correctly
- [ ] Animations are smooth
- [ ] No console errors

## Responsive Design Testing

Test on the following viewport sizes:
- [ ] Desktop (1920x1080)
- [ ] Laptop (1366x768)
- [ ] Tablet (768x1024)
- [ ] Mobile (375x667)

For each size, verify:
- [ ] Cards stack appropriately
- [ ] Text is readable
- [ ] No horizontal scrolling
- [ ] Buttons are accessible

## API Testing

### Test API Endpoints Directly

#### 1. Complete Analysis
```bash
curl http://localhost:8000/api/v1/pose-analysis/sequence/{dataset_id}/{sequence_id}
```

**Expected**: JSON response with complete analysis

#### 2. Features Only
```bash
curl http://localhost:8000/api/v1/pose-analysis/sequence/{dataset_id}/{sequence_id}/features
```

**Expected**: JSON response with features only

#### 3. Gait Cycles Only
```bash
curl http://localhost:8000/api/v1/pose-analysis/sequence/{dataset_id}/{sequence_id}/cycles
```

**Expected**: JSON response with gait cycles

#### 4. Symmetry Only
```bash
curl http://localhost:8000/api/v1/pose-analysis/sequence/{dataset_id}/{sequence_id}/symmetry
```

**Expected**: JSON response with symmetry analysis

#### 5. Cache Stats
```bash
curl http://localhost:8000/api/v1/pose-analysis/cache/stats
```

**Expected**: JSON response with cache statistics

#### 6. Clear Cache
```bash
curl -X DELETE http://localhost:8000/api/v1/pose-analysis/cache/{dataset_id}/{sequence_id}
```

**Expected**: Success message

## Common Issues & Solutions

### Issue 1: Backend Not Starting
**Symptoms**: `uvicorn` command fails
**Solution**: 
- Check Python environment is activated
- Verify all dependencies installed: `pip install -r requirements.txt`
- Check port 8000 is not in use

### Issue 2: Frontend Not Starting
**Symptoms**: `npm run dev` fails
**Solution**:
- Run `npm install` to install dependencies
- Check Node.js version (should be 18+)
- Check port 3000 is not in use

### Issue 3: CORS Errors
**Symptoms**: Browser console shows CORS errors
**Solution**:
- Verify backend CORS settings in `server/main.py`
- Check frontend is accessing correct backend URL

### Issue 4: No Pose Data Error
**Symptoms**: Error message "No pose data available"
**Solution**:
- Verify dataset has been processed with pose estimation
- Check sequence actually has pose data in database
- Try a different sequence

### Issue 5: Slow Analysis
**Symptoms**: Analysis takes >5 seconds
**Solution**:
- Check backend logs for errors
- Verify pose data is properly formatted
- Check system resources (CPU, memory)

## Success Criteria

All tests pass when:
- [x] Backend server starts without errors
- [x] Frontend server starts without errors
- [x] Pose Analysis tab loads successfully
- [x] Analysis displays correctly for sequences with pose data
- [x] Error handling works for sequences without pose data
- [x] Loading states are smooth and informative
- [x] All metrics display correctly
- [x] Sequence switching works properly
- [x] Refresh button works
- [x] No console errors
- [x] No TypeScript errors
- [x] Performance is acceptable (<3s initial load, <100ms cached)
- [x] UI is responsive on different screen sizes
- [x] Works on all major browsers

## Next Steps After Testing

1. **Document Issues**: Create list of bugs found
2. **Prioritize Fixes**: Categorize as critical/high/medium/low
3. **Fix Critical Bugs**: Address any blocking issues
4. **Polish UI**: Improve styling and animations
5. **Update Documentation**: Add any missing details
6. **Prepare for Deployment**: Create deployment checklist

---

**Testing Date**: _____________  
**Tester**: _____________  
**Environment**: _____________  
**Results**: _____________
