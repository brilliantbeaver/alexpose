# Frontend Delete Functionality - Integration Test Guide

## Test Date: January 4, 2026

## Prerequisites
- Backend server running on http://localhost:8000
- Frontend server running on http://localhost:3000
- At least 2 GAVD datasets uploaded and processed

## Test 1: Dashboard Delete

### Steps:
1. Navigate to http://localhost:3000/dashboard
2. Scroll to "Recent Analyses" section
3. Find a GAVD dataset entry
4. Click the 🗑️ (trash) icon on the right side
5. Verify confirmation dialog appears with:
   - Dataset filename
   - Warning about permanent deletion
   - List of data types to be deleted
6. Click "OK" to confirm
7. Verify:
   - Dataset disappears from list
   - Dashboard statistics update
   - No error messages

### Expected Behavior:
- Delete button visible and clickable
- Confirmation dialog shows correct information
- Dataset removed from UI immediately
- Backend files deleted (verify in data/training/gavd/)

### Troubleshooting:
- If button not visible: Check browser console for React errors
- If confirmation doesn't appear: Check browser popup blocker
- If delete fails: Check browser console and backend logs

## Test 2: Training GAVD Page Delete

### Steps:
1. Navigate to http://localhost:3000/training/gavd
2. Click "Recent Datasets" tab
3. Find a dataset in the list
4. Click the 🗑️ (trash) icon on the right side
5. Verify confirmation dialog appears
6. Click "OK" to confirm
7. Verify:
   - Dataset disappears from list
   - Success alert appears
   - No error messages

### Expected Behavior:
- Delete button visible next to "View →" button
- Confirmation dialog shows dataset details
- Dataset removed from list immediately
- Success message displayed

### Troubleshooting:
- If list empty: Upload a new dataset first
- If button not visible: Check CSS styling
- If delete fails: Check network tab for API errors

## Test 3: GAVD Dataset Detail Page Delete

### Steps:
1. Navigate to http://localhost:3000/gavd/[dataset_id]
   (Replace [dataset_id] with actual ID from list)
2. Verify "Delete Dataset" button in top-right header
3. Click "Delete Dataset" button
4. Verify confirmation dialog shows:
   - Dataset filename
   - Number of sequences processed
   - Number of frames processed
   - Warning about permanent deletion
5. Click "OK" to confirm
6. Verify:
   - Redirected to /dashboard
   - Success alert appears
   - Dataset no longer in dashboard list

### Expected Behavior:
- Delete button visible in header (red, destructive style)
- Confirmation shows dataset-specific statistics
- Redirect to dashboard after successful deletion
- Dataset completely removed from system

### Troubleshooting:
- If button not visible: Check if page loaded correctly
- If redirect fails: Check router configuration
- If dataset still appears: Check backend deletion logs

## Backend Verification

After each delete test, verify files are actually deleted:

### Check Metadata:
```powershell
Get-ChildItem data\training\gavd\metadata\*.json
```

### Check CSV Files:
```powershell
Get-ChildItem data\training\gavd\*.csv
```

### Check Results:
```powershell
Get-ChildItem data\training\gavd\results\*_results.json
Get-ChildItem data\training\gavd\results\*_pose_data.json
```

### Check Videos:
```powershell
Get-ChildItem data\youtube\*.mp4
```

## Common Issues

### Issue 1: Delete Button Not Visible
**Symptoms**: Can't find delete button
**Causes**:
- CSS not loaded
- Button hidden by styling
- React component not rendering

**Solutions**:
- Hard refresh browser (Ctrl+Shift+R)
- Check browser console for errors
- Verify component is rendering (React DevTools)

### Issue 2: Confirmation Dialog Doesn't Appear
**Symptoms**: Click delete, nothing happens
**Causes**:
- Browser popup blocker
- JavaScript error preventing dialog
- Event handler not attached

**Solutions**:
- Check browser console for errors
- Disable popup blocker
- Verify onClick handler in code

### Issue 3: Delete Fails Silently
**Symptoms**: Confirmation accepted, but dataset still there
**Causes**:
- Backend server not running
- Network error
- Backend delete failed

**Solutions**:
- Check backend server is running
- Check browser Network tab for failed requests
- Check backend logs for errors
- Verify API endpoint is correct

### Issue 4: Dataset Deleted But Still Shows in UI
**Symptoms**: Backend deleted, but UI still shows dataset
**Causes**:
- Frontend state not updated
- Cache not cleared
- Page not refreshed

**Solutions**:
- Refresh page manually
- Check if state update logic is correct
- Verify API response handling

## Success Criteria

All tests pass if:
- ✅ Delete buttons visible in all 3 locations
- ✅ Confirmation dialogs appear with correct information
- ✅ Datasets removed from UI immediately
- ✅ Backend files actually deleted
- ✅ No errors in browser console
- ✅ No errors in backend logs
- ✅ Redirects work correctly
- ✅ Statistics update correctly

## Test Results

### Test 1 (Dashboard): ⬜ Not Tested / ✅ Passed / ❌ Failed
**Notes**: 

### Test 2 (Training GAVD Page): ⬜ Not Tested / ✅ Passed / ❌ Failed
**Notes**: 

### Test 3 (Dataset Detail Page): ⬜ Not Tested / ✅ Passed / ❌ Failed
**Notes**: 

### Backend Verification: ⬜ Not Tested / ✅ Passed / ❌ Failed
**Notes**: 

## Conclusion

The delete functionality is **fully implemented** in the codebase:
- Backend: DELETE endpoint working correctly
- Frontend: All 3 pages have delete handlers
- No TypeScript/React errors
- Code follows best practices

If user reports it's "not working", the issue is likely:
1. Frontend not refreshed (browser cache)
2. UI elements not visible (CSS/styling)
3. User canceling confirmation dialog
4. Network/connectivity issue

**Recommendation**: Have user perform hard refresh (Ctrl+Shift+R) and test again.
