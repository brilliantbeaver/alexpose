# Navigation Flow - AlexPose Frontend

## Complete User Journey

### Starting Points

```
┌─────────────────────────────────────────────────────────────┐
│                        Homepage (/)                          │
│  • Hero section with "Get Started" and "View Demo"          │
│  • Feature cards                                             │
│  • Quick stats                                               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ↓
                    ┌─────────┴─────────┐
                    │                   │
                    ↓                   ↓
        ┌───────────────────┐  ┌──────────────────┐
        │   Dashboard       │  │  Top Navigation  │
        │   (/dashboard)    │  │  (Always visible)│
        └───────────────────┘  └──────────────────┘
```

---

## Main Navigation Paths

### Path 1: New Analysis

```
Dashboard
    │
    ↓ Click "New Analysis" or "📤 Upload Video"
    │
    ├─→ Upload Video (/analyze/upload)
    │       │
    │       ↓ Upload file
    │       │
    │       ↓ Processing...
    │       │
    │       └─→ Results Detail (/results/{id})
    │
    └─→ YouTube URL (/analyze/youtube)
            │
            ↓ Enter URL
            │
            ↓ Processing...
            │
            └─→ Results Detail (/results/{id})
```

### Path 2: View Existing Results

```
Dashboard
    │
    ↓ Click "View →" on Recent Analysis
    │
    └─→ Results Detail (/results/{id})
            │
            ├─→ View Gait Metrics tab
            ├─→ View Temporal Analysis tab
            ├─→ View Spatial Analysis tab
            ├─→ View AI Analysis tab
            └─→ View Video tab
            │
            ↓ Click "Back" or "← Back to Results"
            │
            └─→ Results List (/results)
```

### Path 3: Browse All Results

```
Dashboard or Top Nav
    │
    ↓ Click "Results" in navigation
    │
    └─→ Results List (/results)
            │
            ├─→ Tab: All Results
            ├─→ Tab: Normal
            └─→ Tab: Abnormal
            │
            ↓ Click "📊 View Details"
            │
            └─→ Results Detail (/results/{id})
                    │
                    ├─→ Export Report
                    ├─→ Compare
                    └─→ Re-analyze
```

---

## Detailed Page Flows

### Dashboard Page Flow

```
┌──────────────────────────────────────────────────────────┐
│                      Dashboard                            │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  [📤 New Analysis]  ← Click to upload                    │
│                                                           │
│  ┌─────────┬─────────┬─────────┬─────────┐             │
│  │ Total   │ Normal  │Abnormal │  Avg.   │             │
│  │   24    │   18    │    6    │  92%    │             │
│  └─────────┴─────────┴─────────┴─────────┘             │
│                                                           │
│  Recent Analyses:                                        │
│  ┌────────────────────────────────────────┐             │
│  │ 1  Walking Test 1    [Normal]  [View →]│ ← Click     │
│  ├────────────────────────────────────────┤             │
│  │ 2  Gait Analysis 2 [Abnormal] [View →]│             │
│  ├────────────────────────────────────────┤             │
│  │ 3  Patient Video 3   [Normal]  [View →]│             │
│  └────────────────────────────────────────┘             │
│                                                           │
│  ✅ System Status: Operational                           │
└──────────────────────────────────────────────────────────┘
```

### Results Detail Page Flow

```
┌──────────────────────────────────────────────────────────┐
│  [← Back]              Walking Test 1                     │
│  Analysis #1 • 2024-01-03 at 14:30:00                    │
│                                                           │
│  [💾 Export] [📊 Compare] [🔄 Re-analyze]                │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  Analysis Summary                        [Normal Badge]  │
│  ┌────────────────────────────────────────────────────┐ │
│  │ Confidence: 95%  │ Duration: 00:00:45 │ Model: GPT │ │
│  │ ████████████████ │ 1350 frames        │ 4.1        │ │
│  └────────────────────────────────────────────────────┘ │
│                                                           │
│  ┌─────────────────────────────────────────────────────┐│
│  │ [Gait Metrics] [Temporal] [Spatial] [AI] [Video]   ││
│  ├─────────────────────────────────────────────────────┤│
│  │                                                      ││
│  │  Cadence: 112 steps/min        ✓ Normal            ││
│  │  ████████████████████ 100%                          ││
│  │  Normal range: 100-120                              ││
│  │                                                      ││
│  │  Stride Length: 1.42 m         ✓ Normal            ││
│  │  ████████████████████ 100%                          ││
│  │  Normal range: 1.2-1.6                              ││
│  │                                                      ││
│  │  ... (4 more metrics)                               ││
│  │                                                      ││
│  └─────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────┘
```

---

## Navigation Menu Structure

### Top Navigation Bar

```
┌────────────────────────────────────────────────────────────┐
│ [AP] AlexPose  │ Dashboard │ Analyze ▼ │ Results ▼ │ ... │
└────────────────────────────────────────────────────────────┘
                                  │              │
                                  │              └─→ History
                                  │                  Compare
                                  │                  Export
                                  │
                                  └─→ Upload Video
                                      YouTube URL
                                      Live Camera (Soon)
                                      Batch Process
```

### Breadcrumb Navigation

```
Home / Dashboard
Home / Dashboard / Results
Home / Dashboard / Results / Analysis #1
Home / Analyze / Upload
Home / Analyze / YouTube
```

---

## Mobile Navigation Flow

### Mobile Menu

```
┌──────────────────────┐
│ [☰]  AlexPose   [🌙] │
└──────────────────────┘
        │
        ↓ Click hamburger
        │
┌──────────────────────┐
│  Navigation          │
├──────────────────────┤
│ 🏠 Dashboard         │
│                      │
│ 📹 Analyze           │
│   📤 Upload Video    │
│   🔗 YouTube URL     │
│   📷 Live Camera     │
│   📊 Batch Process   │
│                      │
│ 📈 Results           │
│   📋 History         │
│   🔍 Compare         │
│   💾 Export          │
│                      │
│ 🤖 Models            │
│ ❓ Help              │
└──────────────────────┘
```

---

## Action Flows

### Upload and Analyze Flow

```
1. Click "New Analysis" or "Upload Video"
   ↓
2. Drag & drop video or click to browse
   ↓
3. File validation (format, size)
   ↓
4. Upload progress bar
   ↓
5. Processing notification
   ↓
6. Redirect to Results Detail (/results/{id})
   ↓
7. View comprehensive analysis
```

### View Results Flow

```
1. Navigate to Dashboard or Results List
   ↓
2. See list of analyses with status badges
   ↓
3. Click "View" or "View Details"
   ↓
4. Results Detail page loads
   ↓
5. View Summary Card (status, confidence)
   ↓
6. Switch between tabs:
   - Gait Metrics (measurements)
   - Temporal (time-based)
   - Spatial (distance-based)
   - AI Analysis (insights)
   - Video (playback)
   ↓
7. Take action:
   - Export report
   - Compare with others
   - Re-analyze
   - Go back to list
```

### Compare Analyses Flow (Future)

```
1. From Results Detail, click "Compare"
   ↓
2. Select additional analyses to compare
   ↓
3. View side-by-side comparison
   ↓
4. See differences highlighted
   ↓
5. Export comparison report
```

---

## Error Handling Flows

### 404 - Analysis Not Found

```
URL: /results/999 (doesn't exist)
   ↓
┌──────────────────────────┐
│         ❌                │
│  Analysis Not Found      │
│                          │
│  The analysis with ID    │
│  999 could not be found. │
│                          │
│  [← Back to Results]     │
└──────────────────────────┘
```

### Upload Error

```
Upload file
   ↓
File too large / Invalid format
   ↓
┌──────────────────────────┐
│  ⚠️ Upload Error          │
│                          │
│  File must be:           │
│  • MP4, AVI, MOV, WebM   │
│  • Under 500MB           │
│                          │
│  [Try Again]             │
└──────────────────────────┘
```

---

## Quick Access Patterns

### From Anywhere

```
Top Navigation (always visible)
   ↓
Click any menu item
   ↓
Instant navigation to:
   • Dashboard
   • Upload
   • YouTube
   • Results
   • Models
   • Help
```

### Breadcrumbs

```
Any page
   ↓
Click breadcrumb link
   ↓
Navigate up hierarchy:
   Results Detail → Results List → Dashboard → Home
```

### Back Button

```
Results Detail
   ↓
Click "← Back"
   ↓
Return to previous page (Results List or Dashboard)
```

---

## User Journey Examples

### First-Time User

```
1. Land on Homepage
2. Click "Get Started"
3. Arrive at Dashboard
4. Click "New Analysis"
5. Upload first video
6. View results
7. Explore tabs
8. Read AI recommendations
```

### Returning User

```
1. Land on Homepage
2. Navigate to Dashboard
3. See recent analyses
4. Click "View" on latest
5. Review results
6. Compare with previous
7. Export report
```

### Clinical User

```
1. Navigate to Dashboard
2. Click "New Analysis"
3. Upload patient video
4. Wait for processing
5. Review detailed metrics
6. Check AI analysis
7. Read recommendations
8. Export PDF report
9. Share with team
```

---

## Navigation Best Practices

### Always Available
✅ Top navigation bar (sticky)  
✅ Breadcrumbs (context)  
✅ Back buttons (escape route)  
✅ Logo (home link)  

### Clear Indicators
✅ Active page highlighting  
✅ Hover effects  
✅ Status badges  
✅ Progress indicators  

### Keyboard Navigation
✅ Tab through elements  
✅ Arrow keys in menus  
✅ Escape to close  
✅ Enter to activate  

### Mobile Friendly
✅ Hamburger menu  
✅ Touch targets  
✅ Swipe gestures  
✅ Responsive layout  

---

## Summary

The AlexPose frontend provides intuitive navigation with:

✅ **Multiple entry points** - Homepage, dashboard, direct links  
✅ **Clear hierarchy** - Breadcrumbs and back buttons  
✅ **Consistent patterns** - Same navigation everywhere  
✅ **Error handling** - Graceful 404 and error pages  
✅ **Mobile support** - Responsive hamburger menu  
✅ **Accessibility** - Keyboard and screen reader support  

Every page is reachable within 3 clicks from any other page.

---

**Last Updated**: January 3, 2026  
**Status**: ✅ Complete navigation system implemented
