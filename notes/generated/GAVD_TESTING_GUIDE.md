# GAVD Dataset Analysis - Testing Guide

## Quick Start
1. Start the backend server: `python -m uvicorn server.main:app --reload`
2. Start the frontend: `cd frontend && npm run dev`
3. Navigate to: `http://localhost:3000/gavd`

## Test Scenarios

### 1. Upload and Process Dataset
**Steps:**
1. Go to `/gavd`
2. Click "Upload Dataset" tab
3. Drag and drop or select a GAVD CSV file
4. Ensure "Process immediately after upload" is checked
5. Click "Upload and Process Dataset"

**Expected Results:**
- ✅ Upload progress indicator appears
- ✅ Processing status card shows real-time updates
- ✅ Upon completion, "View Dataset Analysis" button appears
- ✅ Dataset appears in "Recent Datasets" tab

### 2. View Dataset List
**Steps:**
1. Go to `/gavd`
2. Click "Recent Datasets" tab

**Expected Results:**
- ✅ List of uploaded datasets with status badges
- ✅ Each dataset shows: filename, sequence count, row count, upload time
- ✅ "View →" button for each dataset
- ✅ Delete button (🗑️) for each dataset

### 3. Dataset Detail Page - Overview Tab
**Steps:**
1. From dataset list, click "View →" on any completed dataset
2. Should land on Overview tab by default

**Expected Results:**
- ✅ 4 tabs visible: Overview, Sequences, Visualization, Pose Analysis
- ✅ Dataset statistics cards show: Total Sequences, Total Frames, Avg Frames/Seq, Processing Status
- ✅ Dataset information section shows: Dataset ID, Uploaded date, Filename, Status
- ✅ Sequence list preview shows first 5 sequences
- ✅ Clicking on a sequence switches to Sequences tab

### 4. Sequences Tab
**Steps:**
1. Click "Sequences" tab
2. Select a sequence from dropdown

**Expected Results:**
- ✅ Sequence selector dropdown populated with all sequences
- ✅ Sequence metadata displayed: Sequence ID, Total Frames, Frame Range
- ✅ Frame timeline slider appears
- ✅ Frame details shown: Frame Number, Camera View, Gait Event, Video Resolution
- ✅ Slider updates frame details in real-time

### 5. Visualization Tab
**Steps:**
1. Click "Visualization" tab
2. Ensure a sequence is selected

**Expected Results:**
- ✅ GAVDVideoPlayer component loads
- ✅ Frame image displays correctly
- ✅ "Show Bounding Box" checkbox toggles bbox overlay
- ✅ "Show Pose Overlay" checkbox toggles pose keypoints
- ✅ Frame navigation controls work (play, pause, next, previous)
- ✅ Frame slider updates video player
- ✅ Bounding box and video info displayed below player
- ✅ Frame count shows "X frames with pose data"

### 6. Pose Analysis Tab
**Steps:**
1. Click "Pose Analysis" tab
2. Ensure a sequence with pose data is selected

**Expected Results:**
- ✅ PoseAnalysisOverview component loads
- ✅ Loading spinner appears while fetching analysis
- ✅ Analysis results display when loaded:
  - Gait cycle detection
  - Feature extraction metrics
  - Symmetry assessment
  - Temporal analysis
- ✅ "Refresh Analysis" button works
- ✅ Error message if no pose data available

### 7. Deep Linking
**Steps:**
1. Navigate to: `/gavd/[dataset_id]?sequence=[sequence_id]&tab=visualization`
   (Replace with actual IDs from your data)

**Expected Results:**
- ✅ Page loads with specified sequence pre-selected
- ✅ Visualization tab is automatically active
- ✅ Frames load for the specified sequence
- ✅ All other tabs remain accessible

### 8. Sequence Redirect
**Steps:**
1. Navigate to: `/gavd/[dataset_id]/sequence/[sequence_id]`
   (Replace with actual IDs from your data)

**Expected Results:**
- ✅ Automatic redirect to: `/gavd/[dataset_id]?sequence=[sequence_id]&tab=visualization`
- ✅ Loading spinner shows during redirect
- ✅ Sequence pre-selected and visualization tab active

### 9. Navigation Menu
**Steps:**
1. Click on "Analyze" in top navigation

**Expected Results:**
- ✅ Dropdown menu appears
- ✅ "GAVD Dataset Analysis" option visible
- ✅ Clicking it navigates to `/gavd`

### 10. Legacy Route Redirects
**Steps:**
1. Navigate to: `/training/gavd`
2. Navigate to: `/training/gavd/[datasetId]` (with actual ID)

**Expected Results:**
- ✅ `/training/gavd` redirects to `/gavd`
- ✅ `/training/gavd/[datasetId]` redirects to `/gavd/[dataset_id]`
- ✅ Redirect is seamless with loading indicator

## Error Scenarios

### No Backend Connection
**Steps:**
1. Stop backend server
2. Try to load `/gavd`

**Expected Results:**
- ✅ Error message: "Cannot connect to server"
- ✅ Troubleshooting instructions displayed
- ✅ "Retry" button available

### No Pose Data
**Steps:**
1. Select a sequence without pose data
2. Go to Pose Analysis tab

**Expected Results:**
- ✅ Error message: "No pose data available for this sequence"
- ✅ Graceful degradation (no crash)

### Empty Dataset
**Steps:**
1. Upload a CSV with no valid sequences

**Expected Results:**
- ✅ Processing completes
- ✅ "No sequences found" message
- ✅ No crash or undefined errors

## Performance Checks

### Caching
**Steps:**
1. Load pose analysis for a sequence
2. Switch to another tab
3. Return to pose analysis tab

**Expected Results:**
- ✅ Analysis loads instantly from cache
- ✅ No redundant API calls (check browser network tab)

### Frame Loading
**Steps:**
1. Select a sequence with many frames (>100)
2. Switch to Visualization tab

**Expected Results:**
- ✅ Loading spinner appears
- ✅ Frames load progressively
- ✅ No browser freeze or crash

## Browser Compatibility
Test in:
- ✅ Chrome/Edge (Chromium)
- ✅ Firefox
- ✅ Safari (if available)

## Mobile Responsiveness
Test on mobile viewport:
- ✅ Navigation menu collapses correctly
- ✅ Tabs stack or scroll horizontally
- ✅ Video player scales appropriately
- ✅ Touch interactions work

## Known Limitations
1. Large datasets (>1000 sequences) may have slow initial load
2. Pose analysis requires backend processing time
3. Video player requires frames to be pre-extracted
4. Deep linking requires exact sequence IDs (no fuzzy matching)

## Troubleshooting

### Build Errors
```bash
cd frontend
npm run build
```
Should complete with no errors.

### TypeScript Errors
```bash
cd frontend
npm run type-check
```
Should show no type errors.

### Backend Not Running
```bash
python -m uvicorn server.main:app --reload
```
Should start on `http://localhost:8000`

### Frontend Not Running
```bash
cd frontend
npm run dev
```
Should start on `http://localhost:3000`

## Success Criteria
All test scenarios pass with:
- ✅ No console errors
- ✅ No TypeScript errors
- ✅ No build errors
- ✅ Smooth user experience
- ✅ Correct data display
- ✅ Proper error handling
