# Pose Analysis Architecture Diagram

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         FRONTEND (Next.js + React)                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  GAVD Dataset Page (/training/gavd/[datasetId])                  │   │
│  │                                                                   │   │
│  │  ┌──────────┬──────────┬──────────────┬─────────────────┐      │   │
│  │  │ Overview │ Sequences│ Visualization│ Pose Analysis ⭐│      │   │
│  │  └──────────┴──────────┴──────────────┴─────────────────┘      │   │
│  │                                              │                   │   │
│  │                                              ▼                   │   │
│  │                                   ┌──────────────────┐          │   │
│  │                                   │ PoseAnalysisView │          │   │
│  │                                   └────────┬─────────┘          │   │
│  │                                            │                     │   │
│  │                                            ▼                     │   │
│  │                              ┌─────────────────────────┐        │   │
│  │                              │ PoseAnalysisOverview    │        │   │
│  │                              │ - Summary Cards         │        │   │
│  │                              │ - Key Metrics           │        │   │
│  │                              │ - Recommendations       │        │   │
│  │                              └─────────────────────────┘        │   │
│  │                                                                   │   │
│  └───────────────────────────────────────────────────────────────┘   │
│                                      │                                 │
│                                      │ HTTP GET                        │
│                                      │ /api/v1/pose-analysis/          │
│                                      │ sequence/{dataset_id}/          │
│                                      │ {sequence_id}                   │
│                                      ▼                                 │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                         SERVER (FastAPI)                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  API Router (server/routers/pose_analysis.py)                    │   │
│  │                                                                   │   │
│  │  GET /api/v1/pose-analysis/sequence/{dataset_id}/{sequence_id}  │   │
│  │  ├─ Request validation                                           │   │
│  │  ├─ Error handling                                               │   │
│  │  └─ Response formatting                                          │   │
│  └───────────────────────────────┬─────────────────────────────────┘   │
│                                   │                                      │
│                                   ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Service Layer (server/services/pose_analysis_service.py)        │   │
│  │                                                                   │   │
│  │  PoseAnalysisServiceAPI                                          │   │
│  │  ├─ get_sequence_analysis()                                      │   │
│  │  ├─ _load_pose_sequence()                                        │   │
│  │  └─ Result caching                                               │   │
│  └───────────────────────────────┬─────────────────────────────────┘   │
│                                   │                                      │
│                                   ├─────────────┐                        │
│                                   │             │                        │
│                                   ▼             ▼                        │
│                    ┌──────────────────┐  ┌──────────────┐              │
│                    │  GAVDService     │  │  Analysis    │              │
│                    │  - Load frames   │  │  Components  │              │
│                    │  - Load pose data│  │              │              │
│                    └──────────────────┘  └──────┬───────┘              │
│                                                  │                       │
└──────────────────────────────────────────────────┼───────────────────────┘
                                                   │
┌──────────────────────────────────────────────────┼───────────────────────┐
│                         AMBIENT PACKAGE          │                       │
├──────────────────────────────────────────────────┼───────────────────────┤
│                                                   │                       │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  EnhancedGaitAnalyzer (ambient/analysis/gait_analyzer.py)      │    │
│  │                                                                  │    │
│  │  analyze_gait_sequence(pose_sequence, metadata)                │    │
│  │  ├─ Orchestrates all analysis components                       │    │
│  │  ├─ Generates summary assessment                               │    │
│  │  └─ Returns comprehensive results                              │    │
│  └────────────────────────────────┬───────────────────────────────┘    │
│                                    │                                     │
│                    ┌───────────────┼───────────────┐                    │
│                    │               │               │                    │
│                    ▼               ▼               ▼                    │
│  ┌──────────────────────┐ ┌──────────────┐ ┌──────────────────┐      │
│  │  FeatureExtractor    │ │ Temporal     │ │ Symmetry         │      │
│  │                      │ │ Analyzer     │ │ Analyzer         │      │
│  │  - Kinematic         │ │              │ │                  │      │
│  │  - Joint angles      │ │ - Gait cycles│ │ - Left/Right     │      │
│  │  - Temporal          │ │ - Heel strike│ │ - Movement       │      │
│  │  - Stride            │ │ - Toe-off    │ │ - Angular        │      │
│  │  - Symmetry          │ │ - Phases     │ │ - Temporal       │      │
│  │  - Stability         │ │ - Timing     │ │ - Overall score  │      │
│  └──────────────────────┘ └──────────────┘ └──────────────────┘      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                         DATA STORAGE                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  data/training/gavd/                                                     │
│  ├─ {dataset_id}.csv                    (Original GAVD CSV)             │
│  ├─ metadata/{dataset_id}.json          (Dataset metadata)              │
│  └─ results/                                                             │
│     ├─ {dataset_id}_results.json        (Processing summary)            │
│     └─ {dataset_id}_pose_data.json      (Pose keypoints) ⭐            │
│                                                                           │
│  data/youtube/                                                           │
│  └─ {video_id}.mp4                      (Cached videos)                 │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

## Data Flow Diagram

```
┌─────────────┐
│   User      │
│   Clicks    │
│ "Pose       │
│  Analysis"  │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. Frontend: PoseAnalysisView Component                     │
│    - Detects tab change                                      │
│    - Checks if sequence is selected                          │
│    - Initiates data fetch                                    │
└──────┬──────────────────────────────────────────────────────┘
       │
       │ HTTP GET Request
       │ /api/v1/pose-analysis/sequence/{dataset_id}/{sequence_id}
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Server: API Router (pose_analysis.py)                    │
│    - Validates request parameters                            │
│    - Extracts dataset_id and sequence_id                     │
│    - Calls service layer                                     │
└──────┬──────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Server: PoseAnalysisServiceAPI                           │
│    - Calls get_sequence_analysis()                          │
│    - Checks cache (if implemented)                          │
│    - Loads pose data from GAVD service                      │
└──────┬──────────────────────────────────────────────────────┘
       │
       ├─────────────────┐
       │                 │
       ▼                 ▼
┌──────────────┐  ┌──────────────────────────────────────────┐
│ GAVDService  │  │ 4. Load Pose Sequence                    │
│              │  │    - Get sequence frames                  │
│ - Get frames │  │    - For each frame:                      │
│ - Get pose   │  │      • Load pose keypoints                │
│   data       │  │      • Build pose_sequence list           │
└──────────────┘  └──────┬───────────────────────────────────┘
                         │
                         ▼
                  ┌──────────────────────────────────────────┐
                  │ 5. Ambient: EnhancedGaitAnalyzer         │
                  │    analyze_gait_sequence()               │
                  │                                           │
                  │    ┌─────────────────────────────────┐  │
                  │    │ FeatureExtractor                │  │
                  │    │ - Extract 50+ features          │  │
                  │    │ - Kinematic, angles, temporal   │  │
                  │    └─────────────────────────────────┘  │
                  │                                           │
                  │    ┌─────────────────────────────────┐  │
                  │    │ TemporalAnalyzer                │  │
                  │    │ - Detect gait cycles            │  │
                  │    │ - Heel strikes, toe-offs        │  │
                  │    │ - Phase analysis                │  │
                  │    └─────────────────────────────────┘  │
                  │                                           │
                  │    ┌─────────────────────────────────┐  │
                  │    │ SymmetryAnalyzer                │  │
                  │    │ - Left/Right comparison         │  │
                  │    │ - Movement correlation          │  │
                  │    │ - Overall symmetry score        │  │
                  │    └─────────────────────────────────┘  │
                  │                                           │
                  │    ┌─────────────────────────────────┐  │
                  │    │ Generate Summary                │  │
                  │    │ - Overall assessment            │  │
                  │    │ - Recommendations               │  │
                  │    │ - Quality scores                │  │
                  │    └─────────────────────────────────┘  │
                  └──────┬───────────────────────────────────┘
                         │
                         ▼
                  ┌──────────────────────────────────────────┐
                  │ 6. Return Analysis Results               │
                  │    {                                      │
                  │      "metadata": {...},                   │
                  │      "features": {...},                   │
                  │      "gait_cycles": [...],                │
                  │      "timing_analysis": {...},            │
                  │      "symmetry_analysis": {...},          │
                  │      "summary": {...}                     │
                  │    }                                      │
                  └──────┬───────────────────────────────────┘
                         │
                         │ JSON Response
                         │
                         ▼
                  ┌──────────────────────────────────────────┐
                  │ 7. Frontend: Display Results             │
                  │    PoseAnalysisOverview Component        │
                  │                                           │
                  │    ┌─────────────────────────────────┐  │
                  │    │ Overall Assessment Card         │  │
                  │    │ - Level: Good/Moderate/Poor     │  │
                  │    │ - Confidence: High/Medium/Low   │  │
                  │    └─────────────────────────────────┘  │
                  │                                           │
                  │    ┌─────────────────────────────────┐  │
                  │    │ Key Metrics Cards               │  │
                  │    │ - Symmetry Score                │  │
                  │    │ - Cadence (steps/min)           │  │
                  │    │ - Stability Level               │  │
                  │    │ - Gait Cycles Count             │  │
                  │    └─────────────────────────────────┘  │
                  │                                           │
                  │    ┌─────────────────────────────────┐  │
                  │    │ Recommendations List            │  │
                  │    │ - Clinical suggestions          │  │
                  │    │ - Follow-up actions             │  │
                  │    └─────────────────────────────────┘  │
                  │                                           │
                  │    ┌─────────────────────────────────┐  │
                  │    │ Sequence Information            │  │
                  │    │ - Frames, Duration, FPS         │  │
                  │    │ - Keypoint Format               │  │
                  │    └─────────────────────────────────┘  │
                  └───────────────────────────────────────────┘
```

## Component Interaction Diagram

```
┌────────────────────────────────────────────────────────────────────┐
│                    COMPONENT INTERACTIONS                           │
└────────────────────────────────────────────────────────────────────┘

Frontend Components:
┌──────────────────────┐
│ GAVDAnalysisPage     │
│ (Main Page)          │
└──────┬───────────────┘
       │
       │ Contains
       │
       ▼
┌──────────────────────┐
│ Tabs Component       │
│ - Overview           │
│ - Sequences          │
│ - Visualization      │
│ - Pose Analysis ⭐   │
└──────┬───────────────┘
       │
       │ Renders
       │
       ▼
┌──────────────────────┐
│ PoseAnalysisView     │
│ - Manages state      │
│ - Fetches data       │
│ - Handles errors     │
└──────┬───────────────┘
       │
       │ Renders
       │
       ▼
┌──────────────────────┐
│ PoseAnalysisOverview │
│ - Displays results   │
│ - Shows metrics      │
│ - Lists recommendations
└──────────────────────┘

Backend Components:
┌──────────────────────┐
│ FastAPI App          │
│ (main.py)            │
└──────┬───────────────┘
       │
       │ Includes
       │
       ▼
┌──────────────────────┐
│ pose_analysis router │
│ (routers/)           │
└──────┬───────────────┘
       │
       │ Uses
       │
       ▼
┌──────────────────────┐
│ PoseAnalysisServiceAPI
│ (services/)          │
└──────┬───────────────┘
       │
       ├─────────────────┐
       │                 │
       ▼                 ▼
┌──────────────┐  ┌──────────────────┐
│ GAVDService  │  │ EnhancedGait     │
│              │  │ Analyzer         │
└──────────────┘  └──────┬───────────┘
                         │
                         ├─────────────────┬─────────────────┐
                         │                 │                 │
                         ▼                 ▼                 ▼
                  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
                  │ Feature      │ │ Temporal     │ │ Symmetry     │
                  │ Extractor    │ │ Analyzer     │ │ Analyzer     │
                  └──────────────┘ └──────────────┘ └──────────────┘
```

## File Structure

```
alexpose/
├── ambient/
│   └── analysis/
│       ├── gait_analyzer.py          ✅ EXISTS (EnhancedGaitAnalyzer)
│       ├── feature_extractor.py      ✅ EXISTS (FeatureExtractor)
│       ├── temporal_analyzer.py      ✅ EXISTS (TemporalAnalyzer)
│       ├── symmetry_analyzer.py      ✅ EXISTS (SymmetryAnalyzer)
│       └── pose_analysis_service.py  ❌ NEW (Optional - for caching)
│
├── server/
│   ├── services/
│   │   ├── gavd_service.py           ✅ EXISTS
│   │   └── pose_analysis_service.py  ❌ NEW (REQUIRED)
│   │
│   ├── routers/
│   │   ├── gavd.py                   ✅ EXISTS
│   │   └── pose_analysis.py          ❌ NEW (REQUIRED)
│   │
│   └── main.py                       ✅ EXISTS (needs router registration)
│
├── frontend/
│   ├── components/
│   │   └── pose-analysis/
│   │       ├── PoseAnalysisOverview.tsx      ❌ NEW (REQUIRED for MVP)
│   │       ├── GaitCycleVisualization.tsx    ❌ NEW (Enhancement)
│   │       ├── SymmetryAnalysis.tsx          ❌ NEW (Enhancement)
│   │       ├── FeatureMetrics.tsx            ❌ NEW (Enhancement)
│   │       ├── JointAngleCharts.tsx          ❌ NEW (Enhancement)
│   │       ├── StabilityMetrics.tsx          ❌ NEW (Enhancement)
│   │       ├── FrameByFrameAnalysis.tsx      ❌ NEW (Enhancement)
│   │       ├── ComparisonView.tsx            ❌ NEW (Enhancement)
│   │       └── ExportOptions.tsx             ❌ NEW (Enhancement)
│   │
│   ├── hooks/
│   │   ├── usePoseAnalysis.ts        ❌ NEW (Optional)
│   │   ├── useGaitCycles.ts          ❌ NEW (Optional)
│   │   └── useSymmetryAnalysis.ts    ❌ NEW (Optional)
│   │
│   ├── types/
│   │   └── pose-analysis.ts          ❌ NEW (Optional)
│   │
│   └── app/
│       └── training/
│           └── gavd/
│               └── [datasetId]/
│                   └── page.tsx       ✅ EXISTS (needs update)
│
├── data/
│   └── training/
│       └── gavd/
│           ├── {dataset_id}.csv              ✅ EXISTS
│           ├── metadata/
│           │   └── {dataset_id}.json         ✅ EXISTS
│           └── results/
│               ├── {dataset_id}_results.json ✅ EXISTS
│               └── {dataset_id}_pose_data.json ✅ EXISTS
│
└── notes/
    ├── POSE_ANALYSIS_IMPLEMENTATION_PLAN.md     ✅ CREATED
    ├── POSE_ANALYSIS_EXECUTIVE_SUMMARY.md       ✅ CREATED
    ├── POSE_ANALYSIS_CHECKLIST.md               ✅ CREATED
    ├── POSE_ANALYSIS_QUICK_START.md             ✅ CREATED
    ├── POSE_ANALYSIS_ARCHITECTURE_DIAGRAM.md    ✅ CREATED
    └── 05. Pose Analysis Implementation Summary.md ✅ CREATED
```

## Legend

- ✅ EXISTS - Component already exists and works
- ❌ NEW - Component needs to be created
- ⭐ - Focus area for implementation

## Summary

**What Exists**: All analysis components in `ambient/analysis/`
**What's Missing**: Service layer and frontend UI
**Main Work**: Connect existing components through API and build UI
**Estimated Time**: 12-16 hours (MVP) or 119 hours (Complete)

---

**Created**: January 4, 2026  
**Purpose**: Visual guide for Pose Analysis implementation  
**Status**: Ready for implementation
