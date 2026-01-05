# Pose Analysis API Integration

## Overview

The frontend has been successfully refactored to integrate with the backend pose analysis API, moving away from mock data to real backend processing.

## Architecture

### Backend (Business Logic)
- **Ambient Package** (`ambient/analysis/`): Core pose analysis algorithms
  - `GaitAnalyzer`: Main orchestrator for comprehensive gait analysis
  - `FeatureExtractor`: Extracts kinematic features, joint angles, velocities
  - `TemporalAnalyzer`: Detects gait cycles, heel strikes, toe-offs
  - `SymmetryAnalyzer`: Analyzes left-right symmetry

- **Server Services** (`server/services/pose_analysis_service.py`): High-level API
  - Orchestrates analysis pipeline
  - Manages caching (file + database)
  - Handles error recovery

- **API Endpoints** (`server/routers/pose_analysis.py`): REST endpoints
  - `/api/v1/pose-analysis/sequence/{dataset_id}/{sequence_id}` - Full analysis
  - `/api/v1/pose-analysis/sequence/{dataset_id}/{sequence_id}/features` - Features only
  - `/api/v1/pose-analysis/sequence/{dataset_id}/{sequence_id}/cycles` - Gait cycles only
  - `/api/v1/pose-analysis/sequence/{dataset_id}/{sequence_id}/symmetry` - Symmetry only

### Frontend (Display Layer)
- **API Client** (`frontend/applib/api-client.ts`): Handles all backend communication
- **React Hooks** (`frontend/hooks/usePoseAnalysis.ts`): Manages API state
- **Components** (`frontend/components/pose-analysis/`): Display analysis results
- **Pages** (`frontend/app/results/[id]/page.tsx`): Integrated with real API

## Key Features

### API Integration
- ✅ Real-time API calls to backend
- ✅ Comprehensive error handling
- ✅ Loading states and progress indicators
- ✅ Automatic retry mechanisms
- ✅ Cache management

### Data Flow
```
Frontend Request
    ↓
API Client (frontend/applib/api-client.ts)
    ↓
Backend API (server/routers/pose_analysis.py)
    ↓
Pose Analysis Service (server/services/pose_analysis_service.py)
    ↓
Ambient Analysis Package (ambient/analysis/)
    ↓
Analysis Results
    ↓
Frontend Display (React Components)
```

### Environment Configuration
- `NEXT_PUBLIC_API_URL=http://localhost:8000` in `frontend/.env.local`
- Configurable API base URL for different environments

## Usage

### Starting the System
1. **Backend**: `cd server && python main.py` (runs on port 8000)
2. **Frontend**: `cd frontend && npm run dev` (runs on port 3000)

### Viewing Results
- Navigate to `/results/{dataset_id}_{sequence_id}` or `/results/{dataset_id}-{sequence_id}`
- Example: `/results/gavd_seq001` or `/results/default-test123`

### API Endpoints Available
- **Full Analysis**: GET `/api/v1/pose-analysis/sequence/{dataset_id}/{sequence_id}`
- **Features Only**: GET `/api/v1/pose-analysis/sequence/{dataset_id}/{sequence_id}/features`
- **Gait Cycles**: GET `/api/v1/pose-analysis/sequence/{dataset_id}/{sequence_id}/cycles`
- **Symmetry**: GET `/api/v1/pose-analysis/sequence/{dataset_id}/{sequence_id}/symmetry`
- **Status Check**: GET `/api/v1/pose-analysis/sequence/{dataset_id}/{sequence_id}/status`

## Error Handling

### Frontend Error States
- **404**: Analysis not found - sequence may not have been analyzed yet
- **400**: Invalid request - check dataset and sequence IDs
- **500**: Server error during analysis processing
- **Network**: Connection issues with backend

### User Experience
- Loading spinners during API calls
- Detailed error messages with suggested actions
- Retry buttons for failed requests
- Graceful fallbacks for missing data

## Caching Strategy

### Dual-Layer Caching
1. **File Cache**: JSON files in `data/cache/pose_analysis/`
2. **Database Cache**: SQLite storage for persistence

### Cache Control
- `use_cache=true` (default): Use cached results if available
- `force_refresh=true`: Re-analyze even if cached
- Cache statistics and management endpoints available

## Type Safety

### TypeScript Interfaces
- `PoseAnalysisResult`: Complete analysis response
- `PoseAnalysisSummary`: Summary with assessments
- `GaitCycle`: Individual gait cycle data
- `SymmetryAssessment`: Symmetry analysis results
- `ClinicalRecommendation`: Clinical recommendations with evidence

### API Response Types
All API responses are properly typed with comprehensive interfaces that match the backend data structures.

## Benefits of This Architecture

### ✅ Proper Separation of Concerns
- Backend handles all computation and business logic
- Frontend focuses on user interface and experience
- Clear API boundaries between layers

### ✅ Scalability
- Backend can handle multiple concurrent analyses
- Caching reduces redundant computations
- API can be consumed by multiple frontends

### ✅ Maintainability
- Business logic centralized in backend
- Frontend components are reusable
- Type safety prevents runtime errors

### ✅ Performance
- Efficient caching strategy
- Optimized API endpoints for specific data needs
- Progressive loading of analysis components

## Next Steps

1. **Real Data Integration**: Connect to actual GAVD dataset sequences
2. **WebSocket Support**: Real-time updates for long-running analyses
3. **Batch Processing**: Support for analyzing multiple sequences
4. **Export Features**: Enhanced report generation and export options
5. **Visualization**: Advanced charts and graphs for analysis results