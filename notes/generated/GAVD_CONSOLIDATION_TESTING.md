# GAVD Consolidation - Testing Guide

**Date:** January 4, 2026  
**Changes:** Unified sequence viewer, eliminated duplication

## Quick Test (5 minutes)

### 1. Start Servers

```bash
# Terminal 1 - Backend
cd server
python -m uvicorn main:app --reload

# Terminal 2 - Frontend
cd frontend
npm run dev
```

### 2. Test Dashboard Link

1. Open http://localhost:3000/dashboard
2. Find a GAVD dataset in "Recent Analyses"
3. Click "View →" button
4. **Expected:**
   - ✅ Navigates to `/training/gavd/[dataset_id]`
   - ✅ Shows all 4 tabs (Overview, Sequences, Visualization, Pose Analysis)
   - ✅ Can switch between tabs
   - ✅ All functionality works

### 3. Test Direct Sequence URL

1. Get a sequence URL (e.g., from old bookmark)
2. Navigate to: `http://localhost:3000/gavd/[dataset_id]/sequence/[sequence_id]`
3. **Expected:**
   - ✅ Shows loading spinner briefly
   - ✅ Redirects to `/training/gavd/[dataset_id]?sequence=[sequence_id]&tab=visualization`
   - ✅ Auto-selects the sequence
   - ✅ Auto-switches to Visualization tab
   - ✅ Video player loads correctly

### 4. Test Query Parameters

1. Navigate to: `http://localhost:3000/training/gavd/[dataset_id]?sequence=[seq_id]&tab=pose`
2. **Expected:**
   - ✅ Auto-selects specified sequence
   - ✅ Auto-switches to Pose Analysis tab
   - ✅ Loads pose analysis for that sequence

---

## Detailed Test Cases

### Test Case 1: Dashboard to Dataset

**Steps:**
1. Go to Dashboard
2. Click on GAVD dataset

**Expected Results:**
- ✅ URL: `/training/gavd/[dataset_id]`
- ✅ All 4 tabs visible
- ✅ Overview tab active by default
- ✅ First sequence selected by default
- ✅ No console errors

### Test Case 2: Old Sequence URL Redirect

**Steps:**
1. Navigate to `/gavd/[dataset_id]/sequence/[sequence_id]`

**Expected Results:**
- ✅ Loading spinner appears
- ✅ Redirects to `/training/gavd/[dataset_id]?sequence=[sequence_id]&tab=visualization`
- ✅ Specified sequence auto-selected
- ✅ Visualization tab active
- ✅ Video player loads
- ✅ No console errors

### Test Case 3: Query Parameter - Sequence Only

**Steps:**
1. Navigate to `/training/gavd/[dataset_id]?sequence=[seq_id]`

**Expected Results:**
- ✅ Specified sequence auto-selected
- ✅ Current tab remains (or default)
- ✅ Sequence loads correctly
- ✅ No console errors

### Test Case 4: Query Parameter - Tab Only

**Steps:**
1. Navigate to `/training/gavd/[dataset_id]?tab=pose`

**Expected Results:**
- ✅ Pose Analysis tab active
- ✅ First sequence selected (default)
- ✅ Pose analysis loads
- ✅ No console errors

### Test Case 5: Query Parameter - Both

**Steps:**
1. Navigate to `/training/gavd/[dataset_id]?sequence=[seq_id]&tab=visualization`

**Expected Results:**
- ✅ Specified sequence auto-selected
- ✅ Visualization tab active
- ✅ Video player loads for that sequence
- ✅ No console errors

### Test Case 6: Invalid Sequence Parameter

**Steps:**
1. Navigate to `/training/gavd/[dataset_id]?sequence=invalid_id`

**Expected Results:**
- ✅ Ignores invalid sequence
- ✅ Defaults to first sequence
- ✅ No crash
- ✅ Console may show warning (acceptable)

### Test Case 7: Invalid Tab Parameter

**Steps:**
1. Navigate to `/training/gavd/[dataset_id]?tab=invalid_tab`

**Expected Results:**
- ✅ Ignores invalid tab
- ✅ Defaults to current/default tab
- ✅ No crash
- ✅ Console may show warning (acceptable)

### Test Case 8: All Tabs Work

**Steps:**
1. Navigate to dataset page
2. Click each tab: Overview, Sequences, Visualization, Pose Analysis

**Expected Results:**
- ✅ Overview tab: Shows dataset info and sequence list
- ✅ Sequences tab: Shows sequence selector and details
- ✅ Visualization tab: Shows video player with controls
- ✅ Pose Analysis tab: Shows pose analysis results
- ✅ All tabs load correctly
- ✅ No console errors

### Test Case 9: Sequence Switching

**Steps:**
1. Navigate to dataset page
2. Select different sequences from dropdown

**Expected Results:**
- ✅ Sequence changes correctly
- ✅ Data updates for new sequence
- ✅ Video player updates (if on Visualization tab)
- ✅ Pose analysis updates (if on Pose tab)
- ✅ No console errors

### Test Case 10: Bookmarking

**Steps:**
1. Navigate to `/training/gavd/[dataset_id]?sequence=[seq_id]&tab=pose`
2. Bookmark the page
3. Close browser
4. Open bookmark

**Expected Results:**
- ✅ Returns to exact same view
- ✅ Same sequence selected
- ✅ Same tab active
- ✅ Data loads correctly

---

## Browser Compatibility

Test on:
- [ ] Chrome (latest)
- [ ] Firefox (latest)
- [ ] Safari (latest)
- [ ] Edge (latest)

For each browser, verify:
- [ ] Redirect works
- [ ] Query parameters work
- [ ] All tabs work
- [ ] No console errors

---

## Performance Checks

### Page Load Time
- [ ] Dashboard → Dataset: < 2 seconds
- [ ] Sequence redirect: < 500ms
- [ ] Tab switching: < 100ms

### Memory Usage
- [ ] No memory leaks
- [ ] Stable memory usage
- [ ] No excessive re-renders

### Network Requests
- [ ] No duplicate API calls
- [ ] Efficient data loading
- [ ] Proper caching

---

## Console Checks

### Expected Logs
```
[Query Param] Auto-selecting sequence: xxx
[Query Param] Auto-switching to tab: yyy
[Sequence Redirect] Redirecting to: /training/gavd/...
```

### No Errors
- [ ] No React errors
- [ ] No TypeScript errors
- [ ] No network errors
- [ ] No 404s

---

## Edge Cases

### Test Case 11: No Sequences
**Steps:**
1. Navigate to dataset with no sequences

**Expected:**
- ✅ Handles gracefully
- ✅ Shows appropriate message
- ✅ No crash

### Test Case 12: Slow Network
**Steps:**
1. Throttle network to "Slow 3G"
2. Test redirect and query parameters

**Expected:**
- ✅ Loading states show
- ✅ Eventually loads correctly
- ✅ No timeout errors

### Test Case 13: Rapid Navigation
**Steps:**
1. Quickly switch between tabs
2. Quickly switch between sequences

**Expected:**
- ✅ Handles rapid changes
- ✅ No race conditions
- ✅ Final state is correct

---

## Regression Testing

### Existing Functionality
- [ ] Dataset overview still works
- [ ] Sequence list still works
- [ ] Video player still works
- [ ] Pose analysis still works
- [ ] Bounding box overlay still works
- [ ] Pose overlay still works
- [ ] Frame navigation still works

### No Breaking Changes
- [ ] All existing URLs still work
- [ ] All existing features still work
- [ ] No performance degradation
- [ ] No new bugs introduced

---

## Success Criteria

All tests pass when:
- [x] Code changes complete
- [x] TypeScript errors resolved
- [ ] Dashboard links work correctly
- [ ] Sequence redirect works correctly
- [ ] Query parameters work correctly
- [ ] All 4 tabs accessible from all entry points
- [ ] No code duplication
- [ ] No console errors
- [ ] No regressions
- [ ] Performance acceptable
- [ ] Browser compatibility confirmed

---

## Troubleshooting

### Issue: Redirect Not Working
**Check:**
- Frontend server restarted?
- Browser cache cleared?
- Correct URL format?

### Issue: Query Parameters Ignored
**Check:**
- Sequence exists in dataset?
- Tab name is valid?
- Console for warnings?

### Issue: Console Errors
**Check:**
- TypeScript compilation clean?
- All imports correct?
- No syntax errors?

---

**Testing Date:** _____________  
**Tester:** _____________  
**Result:** ☐ PASS  ☐ FAIL  
**Notes:** _____________________________________________
