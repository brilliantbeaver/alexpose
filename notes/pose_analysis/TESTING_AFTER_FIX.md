# Testing Guide After Infinite Loop Fix

**Date:** January 4, 2026  
**Fixes Applied:** Frontend infinite loop + Backend service caching

## Quick Test (5 minutes)

### 1. Start Servers

**Terminal 1 - Backend:**
```bash
cd server
python -m uvicorn main:app --reload
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

### 2. Test Pose Analysis Tab

1. Open browser: http://localhost:3000
2. Navigate to: Training → GAVD → [Select Dataset]
3. Click: **"Pose Analysis"** tab
4. **Watch server logs** in Terminal 1

### 3. Expected Behavior ✅

**Server logs should show:**
```
INFO | Received analysis request for [dataset]/[sequence]
DEBUG | Cache hit for [sequence] (age: XXXs)
INFO | Returning cached analysis for [sequence]
INFO | Request completed
[... NO MORE REQUESTS ...]
```

**Frontend should show:**
- Loading spinner briefly
- Analysis results display
- No freezing or hanging
- No console errors

### 4. What to Look For

✅ **GOOD SIGNS:**
- Only **ONE** "Received analysis request" log entry
- Analysis loads and displays within 1-3 seconds
- UI remains responsive
- Can switch tabs without issues
- Switching sequences triggers new request (expected)

❌ **BAD SIGNS (Bug Still Present):**
- Multiple "Received analysis request" entries (repeating)
- Logs flooding continuously
- UI appears frozen
- Browser becomes unresponsive

---

## Detailed Test (15 minutes)

### Test Case 1: Initial Load
1. Navigate to GAVD dataset page
2. Click "Pose Analysis" tab
3. **Verify:** Single API request in logs
4. **Verify:** Analysis displays correctly
5. **Verify:** No console errors

### Test Case 2: Tab Switching
1. With analysis loaded, switch to "Overview" tab
2. Switch back to "Pose Analysis" tab
3. **Verify:** NO new API request (uses cached data)
4. **Verify:** Analysis displays instantly
5. **Verify:** No console errors

### Test Case 3: Sequence Switching
1. With analysis loaded, select different sequence
2. **Verify:** ONE new API request for new sequence
3. **Verify:** Analysis updates for new sequence
4. **Verify:** No infinite loop
5. Switch back to first sequence
6. **Verify:** ONE new API request (or cached if recent)

### Test Case 4: Refresh Button
1. With analysis loaded, click "Refresh Analysis" button
2. **Verify:** ONE new API request
3. **Verify:** Analysis reloads
4. **Verify:** No infinite loop

### Test Case 5: Multiple Sequences
1. Load analysis for sequence A
2. Switch to sequence B
3. Switch to sequence C
4. Switch back to sequence A
5. **Verify:** Each switch triggers ONE request
6. **Verify:** No infinite loops at any point

### Test Case 6: Error Handling
1. Select sequence without pose data (if available)
2. **Verify:** Error message displays
3. **Verify:** No infinite loop of failed requests
4. **Verify:** Can recover by selecting different sequence

---

## Performance Verification

### Backend Performance

**Check server logs for:**
```
INFO | Pose analysis service initialized  ← Should appear ONCE at startup
```

**NOT:**
```
INFO | Pose analysis service initialized  ← Should NOT repeat on every request
INFO | Feature extractor initialized
INFO | Temporal analyzer initialized
[... repeating ...]
```

### Frontend Performance

**Open browser DevTools (F12) → Network tab:**

1. Click "Pose Analysis" tab
2. **Verify:** ONE request to `/api/v1/pose-analysis/sequence/...`
3. **Verify:** Request completes in 1-3 seconds (first time) or <100ms (cached)
4. **Verify:** No repeated requests

**Console tab:**
- **Verify:** No errors
- **Verify:** Log shows: `[loadPoseAnalysis] Starting to load analysis for sequence: ...`
- **Verify:** Log shows: `[loadPoseAnalysis] Analysis loaded successfully for sequence ...`
- **Verify:** These logs appear ONCE per sequence

---

## Stress Test (Optional)

### Test Rapid Tab Switching
1. Quickly switch between tabs: Overview → Pose → Sequences → Pose → Visualization → Pose
2. **Verify:** No infinite loops
3. **Verify:** Analysis loads correctly each time
4. **Verify:** Server logs remain clean

### Test Rapid Sequence Switching
1. Quickly switch between 5 different sequences
2. **Verify:** Each triggers ONE request
3. **Verify:** No infinite loops
4. **Verify:** UI remains responsive

---

## Troubleshooting

### If Infinite Loop Still Occurs

**Check:**
1. Did you save the file changes?
2. Did you restart the frontend dev server?
3. Did you clear browser cache? (Ctrl+Shift+R)
4. Are you on the correct branch?

**Verify the fix is applied:**
```bash
# Check the file
grep -A 5 "Load pose analysis when switching to pose tab" frontend/app/training/gavd/[datasetId]/page.tsx
```

Should show:
```typescript
useEffect(() => {
  if (activeTab === 'pose' && selectedSequence && !loadingPoseAnalysis && !poseAnalysis) {
    loadPoseAnalysis(selectedSequence);
  }
  // eslint-disable-next-line react-hooks/exhaustive-deps
}, [activeTab, selectedSequence]);
```

### If Backend Service Re-initializes

**Check:**
```bash
# Verify the fix is applied
grep -A 10 "_get_service" server/routers/pose_analysis.py
```

Should show the caching function.

---

## Success Criteria

All tests pass when:

- [x] Single API request per sequence
- [x] No infinite loops
- [x] Analysis loads correctly
- [x] UI remains responsive
- [x] Tab switching works smoothly
- [x] Sequence switching works correctly
- [x] Refresh button works
- [x] Error handling works
- [x] No console errors
- [x] Server logs are clean
- [x] Backend service initializes once

---

## Reporting Issues

If you find any issues:

1. **Capture server logs** (last 50 lines)
2. **Capture browser console** (F12 → Console tab)
3. **Capture network requests** (F12 → Network tab)
4. **Note exact steps** to reproduce
5. **Document expected vs actual behavior**

---

**Testing Date:** _____________  
**Tester:** _____________  
**Result:** ☐ PASS  ☐ FAIL  
**Notes:** _____________________________________________

