# Pose Analysis - Quick Start Guide

## 🚀 Get Started in 30 Minutes

This guide will help you implement a **minimal viable Pose Analysis tab** that displays basic analysis results.

## Prerequisites

- ✅ Backend server running (`python -m server.main`)
- ✅ Frontend running (`npm run dev` in `frontend/`)
- ✅ GAVD dataset uploaded and processed
- ✅ Python packages: `numpy`, `scipy`, `loguru`, `fastapi`
- ✅ Node packages: `recharts` (install with `npm install recharts`)

## Step 1: Create Backend Service (10 minutes)

### 1.1 Create `server/services/pose_analysis_service.py`

```python
"""
Pose analysis service for analyzing gait sequences.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
from loguru import logger
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ambient.analysis.gait_analyzer import EnhancedGaitAnalyzer
from server.services.gavd_service import GAVDService


class PoseAnalysisServiceAPI:
    """
    Service for analyzing pose data from GAVD sequences.
    """
    
    def __init__(self, config_manager):
        """Initialize pose analysis service."""
        self.config = config_manager
        self.gavd_service = GAVDService(config_manager)
        self.analyzer = EnhancedGaitAnalyzer(
            keypoint_format="COCO_17",
            fps=30.0
        )
        logger.info("Pose analysis service initialized")
    
    def get_sequence_analysis(
        self, 
        dataset_id: str, 
        sequence_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get complete analysis for a sequence.
        
        Args:
            dataset_id: Dataset ID
            sequence_id: Sequence ID
            
        Returns:
            Analysis results dictionary or None if error
        """
        try:
            logger.info(f"Analyzing sequence {sequence_id} in dataset {dataset_id}")
            
            # Load pose data from GAVD service
            pose_sequence = self._load_pose_sequence(dataset_id, sequence_id)
            
            if not pose_sequence:
                logger.warning(f"No pose data found for sequence {sequence_id}")
                return None
            
            # Run analysis
            metadata = {
                "dataset_id": dataset_id,
                "sequence_id": sequence_id
            }
            
            results = self.analyzer.analyze_gait_sequence(pose_sequence, metadata)
            
            logger.info(f"Analysis complete for sequence {sequence_id}")
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing sequence {sequence_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _load_pose_sequence(
        self, 
        dataset_id: str, 
        sequence_id: str
    ) -> List[Dict[str, Any]]:
        """Load pose data for a sequence."""
        try:
            # Get frames for sequence
            frames = self.gavd_service.get_sequence_frames(dataset_id, sequence_id)
            
            if not frames:
                return []
            
            # Load pose data for each frame
            pose_sequence = []
            
            for frame in frames:
                frame_num = frame["frame_num"]
                pose_data = self.gavd_service.get_frame_pose_data(
                    dataset_id, 
                    sequence_id, 
                    frame_num
                )
                
                if pose_data:
                    # Handle both old and new format
                    if isinstance(pose_data, dict):
                        keypoints = pose_data.get('keypoints', [])
                    else:
                        keypoints = pose_data
                    
                    if keypoints:
                        pose_sequence.append({
                            "frame_num": frame_num,
                            "keypoints": keypoints
                        })
            
            logger.info(f"Loaded {len(pose_sequence)} frames with pose data")
            return pose_sequence
            
        except Exception as e:
            logger.error(f"Error loading pose sequence: {str(e)}")
            return []
```

### 1.2 Create `server/routers/pose_analysis.py`

```python
"""
Pose analysis endpoints for AlexPose FastAPI application.
"""

from fastapi import APIRouter, HTTPException, Request
from typing import Dict, Any
from loguru import logger

from server.services.pose_analysis_service import PoseAnalysisServiceAPI

router = APIRouter(prefix="/api/v1/pose-analysis", tags=["pose-analysis"])


@router.get("/sequence/{dataset_id}/{sequence_id}")
async def get_sequence_analysis(
    request: Request,
    dataset_id: str,
    sequence_id: str
) -> Dict[str, Any]:
    """
    Get complete pose analysis for a sequence.
    
    Args:
        dataset_id: Dataset ID
        sequence_id: Sequence ID
        
    Returns:
        Complete analysis results including features, cycles, symmetry, and summary
    """
    config_manager = request.app.state.config
    service = PoseAnalysisServiceAPI(config_manager)
    
    try:
        results = service.get_sequence_analysis(dataset_id, sequence_id)
        
        if not results:
            raise HTTPException(
                status_code=404, 
                detail="Analysis results not found. Sequence may not have pose data."
            )
        
        return {
            "success": True,
            "dataset_id": dataset_id,
            "sequence_id": sequence_id,
            "analysis": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in pose analysis endpoint: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to analyze sequence: {str(e)}"
        )
```

### 1.3 Register Router in `server/main.py`

Add this import at the top:
```python
from server.routers import pose_analysis
```

Add this line after other router registrations:
```python
app.include_router(pose_analysis.router)
```

## Step 2: Create Frontend Component (15 minutes)

### 2.1 Create `frontend/components/pose-analysis/PoseAnalysisOverview.tsx`

```typescript
/**
 * Pose Analysis Overview Component
 * Displays summary of gait analysis results
 */

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';

interface PoseAnalysisOverviewProps {
  analysis: any;
  loading?: boolean;
  error?: string | null;
}

export default function PoseAnalysisOverview({ 
  analysis, 
  loading = false, 
  error = null 
}: PoseAnalysisOverviewProps) {
  if (loading) {
    return (
      <div className="space-y-4">
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-center py-8">
              <div className="text-center space-y-2">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto"></div>
                <p className="text-sm text-muted-foreground">Analyzing gait sequence...</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (error) {
    return (
      <Alert variant="destructive">
        <AlertTitle>Analysis Error</AlertTitle>
        <AlertDescription>{error}</AlertDescription>
      </Alert>
    );
  }

  if (!analysis) {
    return (
      <Alert>
        <AlertTitle>No Analysis Available</AlertTitle>
        <AlertDescription>
          Select a sequence to view pose analysis results.
        </AlertDescription>
      </Alert>
    );
  }

  const summary = analysis.summary || {};
  const overallAssessment = summary.overall_assessment || {};
  const symmetryAssessment = summary.symmetry_assessment || {};
  const cadenceAssessment = summary.cadence_assessment || {};
  const stabilityAssessment = summary.stability_assessment || {};

  const getAssessmentColor = (level: string) => {
    switch (level) {
      case 'good':
      case 'high':
      case 'normal':
      case 'symmetric':
        return 'bg-green-100 text-green-800 border-green-200';
      case 'moderate':
      case 'mildly_asymmetric':
        return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      case 'poor':
      case 'low':
      case 'severely_asymmetric':
        return 'bg-red-100 text-red-800 border-red-200';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  return (
    <div className="space-y-6">
      {/* Overall Assessment */}
      <Card>
        <CardHeader>
          <CardTitle>Overall Assessment</CardTitle>
          <CardDescription>Summary of gait analysis results</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="border rounded-lg p-4">
              <p className="text-sm text-muted-foreground mb-2">Overall Level</p>
              <Badge className={getAssessmentColor(overallAssessment.overall_level || 'unknown')}>
                {(overallAssessment.overall_level || 'Unknown').toUpperCase()}
              </Badge>
              <p className="text-xs text-muted-foreground mt-2">
                Confidence: {overallAssessment.confidence || 'N/A'}
              </p>
            </div>
            
            <div className="border rounded-lg p-4">
              <p className="text-sm text-muted-foreground mb-2">Symmetry</p>
              <Badge className={getAssessmentColor(symmetryAssessment.symmetry_classification || 'unknown')}>
                {(symmetryAssessment.symmetry_classification || 'Unknown').replace('_', ' ').toUpperCase()}
              </Badge>
              <p className="text-xs text-muted-foreground mt-2">
                Score: {symmetryAssessment.symmetry_score?.toFixed(3) || 'N/A'}
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium">Cadence</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {cadenceAssessment.cadence_value?.toFixed(1) || 'N/A'}
            </div>
            <p className="text-xs text-muted-foreground">steps/minute</p>
            <Badge className={`mt-2 ${getAssessmentColor(cadenceAssessment.cadence_level || 'unknown')}`}>
              {(cadenceAssessment.cadence_level || 'Unknown').toUpperCase()}
            </Badge>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium">Stability</CardTitle>
          </CardHeader>
          <CardContent>
            <Badge className={getAssessmentColor(stabilityAssessment.stability_level || 'unknown')}>
              {(stabilityAssessment.stability_level || 'Unknown').toUpperCase()}
            </Badge>
            <p className="text-xs text-muted-foreground mt-2">
              Center of mass stability
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium">Gait Cycles</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {analysis.gait_cycles?.length || 0}
            </div>
            <p className="text-xs text-muted-foreground">detected cycles</p>
          </CardContent>
        </Card>
      </div>

      {/* Recommendations */}
      {overallAssessment.recommendations && overallAssessment.recommendations.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Recommendations</CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="space-y-2">
              {overallAssessment.recommendations.map((rec: string, idx: number) => (
                <li key={idx} className="flex items-start space-x-2">
                  <span className="text-primary mt-1">•</span>
                  <span className="text-sm">{rec}</span>
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      )}

      {/* Sequence Info */}
      <Card>
        <CardHeader>
          <CardTitle>Sequence Information</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <p className="text-muted-foreground">Frames</p>
              <p className="font-medium">{analysis.sequence_info?.num_frames || 'N/A'}</p>
            </div>
            <div>
              <p className="text-muted-foreground">Duration</p>
              <p className="font-medium">{analysis.sequence_info?.duration_seconds?.toFixed(2) || 'N/A'}s</p>
            </div>
            <div>
              <p className="text-muted-foreground">FPS</p>
              <p className="font-medium">{analysis.sequence_info?.fps || 'N/A'}</p>
            </div>
            <div>
              <p className="text-muted-foreground">Format</p>
              <p className="font-medium">{analysis.sequence_info?.keypoint_format || 'N/A'}</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
```

### 2.2 Update `frontend/app/training/gavd/[datasetId]/page.tsx`

Replace the Pose Analysis tab content (around line 450) with:

```typescript
{/* Pose Analysis Tab */}
<TabsContent value="pose" className="space-y-4">
  <Card>
    <CardHeader>
      <CardTitle>Pose Estimation Analysis</CardTitle>
      <CardDescription>
        Comprehensive gait analysis with feature extraction and symmetry assessment
        {selectedSequence && ` • Sequence: ${selectedSequence}`}
      </CardDescription>
    </CardHeader>
    <CardContent>
      <PoseAnalysisView 
        datasetId={datasetId} 
        sequenceId={selectedSequence} 
      />
    </CardContent>
  </Card>
</TabsContent>
```

Add this component at the bottom of the file:

```typescript
function PoseAnalysisView({ 
  datasetId, 
  sequenceId 
}: { 
  datasetId: string; 
  sequenceId: string | null; 
}) {
  const [analysis, setAnalysis] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!sequenceId) {
      setAnalysis(null);
      setError(null);
      return;
    }

    const loadAnalysis = async () => {
      setLoading(true);
      setError(null);
      
      try {
        const response = await fetch(
          `http://localhost:8000/api/v1/pose-analysis/sequence/${datasetId}/${sequenceId}`
        );
        
        if (!response.ok) {
          throw new Error(`Failed to load analysis: ${response.statusText}`);
        }
        
        const result = await response.json();
        setAnalysis(result.analysis);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load analysis');
      } finally {
        setLoading(false);
      }
    };

    loadAnalysis();
  }, [datasetId, sequenceId]);

  return (
    <PoseAnalysisOverview 
      analysis={analysis} 
      loading={loading} 
      error={error} 
    />
  );
}
```

Add the import at the top:
```typescript
import PoseAnalysisOverview from '@/components/pose-analysis/PoseAnalysisOverview';
```

## Step 3: Test It! (5 minutes)

1. **Restart the backend server**:
   ```bash
   # Stop the server (Ctrl+C)
   # Start it again
   python -m server.main
   ```

2. **Restart the frontend** (if needed):
   ```bash
   # In frontend/ directory
   npm run dev
   ```

3. **Navigate to a GAVD dataset**:
   - Go to http://localhost:3000/training/gavd
   - Click on a completed dataset
   - Click the "Pose Analysis" tab
   - Select a sequence

4. **You should see**:
   - Overall assessment badge (Good/Moderate/Poor)
   - Symmetry classification
   - Cadence value and level
   - Stability assessment
   - Number of detected gait cycles
   - Recommendations (if any)
   - Sequence information

## Troubleshooting

### Backend Issues

**Error: "Module not found: ambient"**
- Solution: Make sure you're running from the project root
- Check: `sys.path.insert(0, str(project_root))` is in the service file

**Error: "No pose data found"**
- Solution: Make sure the dataset has been processed with pose estimation
- Check: Look for `{dataset_id}_pose_data.json` in `data/training/gavd/results/`

**Error: "Analysis failed"**
- Solution: Check server logs for detailed error
- Check: Make sure numpy and scipy are installed

### Frontend Issues

**Error: "Failed to load analysis"**
- Solution: Check that backend server is running on port 8000
- Check: Open browser console for detailed error

**Component not rendering**
- Solution: Check that import path is correct
- Check: Make sure component is exported properly

**Blank screen**
- Solution: Check browser console for errors
- Check: Make sure all dependencies are installed (`npm install`)

## Next Steps

Now that you have a basic Pose Analysis tab working, you can:

1. **Add more visualizations**:
   - Gait cycle timeline
   - Joint angle charts
   - Symmetry comparison charts

2. **Add more features**:
   - Export functionality
   - Comparison between sequences
   - Frame-by-frame analysis

3. **Improve UX**:
   - Add loading skeletons
   - Add error recovery
   - Add tooltips and help text

4. **Add tests**:
   - Unit tests for service
   - API endpoint tests
   - Component tests

## Resources

- **Full Implementation Plan**: `notes/POSE_ANALYSIS_IMPLEMENTATION_PLAN.md`
- **Executive Summary**: `notes/POSE_ANALYSIS_EXECUTIVE_SUMMARY.md`
- **Detailed Checklist**: `notes/POSE_ANALYSIS_CHECKLIST.md`
- **Design Document**: `docs/specs/design.md`
- **Requirements**: `docs/specs/requirements.md`

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review the full implementation plan
3. Check server logs: `logs/alexpose_*.log`
4. Check browser console for frontend errors

---

**Congratulations!** 🎉 You now have a working Pose Analysis tab that displays comprehensive gait analysis results!
