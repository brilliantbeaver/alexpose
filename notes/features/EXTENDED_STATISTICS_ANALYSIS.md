# Extended Joint Angle Statistics Analysis

## Current Situation

You're seeing **0.00 values** for extended joint angle statistics (std, max, min) despite having non-zero mean and range values:

```
left_hip_mean                 :  173.19  ✅ NON-ZERO
left_hip_range                :   18.19  ✅ NON-ZERO
left_hip_std                  :    0.00  ❌ ZERO
left_hip_max                  :    0.00  ❌ ZERO
left_hip_min                  :    0.00  ❌ ZERO
```

## Root Cause Analysis

### Why Are They Zero?

Based on code analysis, there are three possible reasons:

1. **`include_joint_statistics=False`** when FeatureExtractor runs
   - Check: `analyzer.feature_extractor.include_joint_statistics`
   - Should be: `True` (default when `comprehensive_features=True`)

2. **Features not in `features_dict`** after extraction
   - Check: Run `diagnose_extended_stats.py` to see what's in the dict
   - If missing: FeatureExtractor is not creating them

3. **Features in dict but not extracted** by GaitFeatureVector
   - Check: Compare `features_dict` vs `feature_vector` attributes
   - If mismatch: `from_analysis_results()` has extraction bug

### Most Likely Cause

Looking at your output, you have:
- ✅ Non-zero mean (173.19) - proves angle arrays are NOT empty
- ✅ Non-zero range (18.19) - proves min/max calculation works
- ❌ Zero std/max/min - suggests they're not being created

**Hypothesis**: The extended statistics are NOT being created in `features_dict` because `include_joint_statistics` is somehow `False` when extraction runs, OR there's a conditional logic issue in the code.

## Are Extended Statistics Redundant?

### Mathematical Relationships

```python
# What we have:
mean = average(angles)
range = max(angles) - min(angles)

# What extended stats add:
std = standard_deviation(angles)  # Variability around mean
max = maximum(angles)             # Upper limit of ROM
min = minimum(angles)             # Lower limit of ROM
```

### Redundancy Analysis

| Feature | Information | Redundant? | Clinical Value |
|---------|-------------|------------|----------------|
| `mean` | Central tendency | ❌ No | Essential - average joint angle |
| `range` | Spread (max-min) | ⚠️ Partial | Simple ROM metric |
| `std` | Variability | ❌ No | Gait consistency/stability |
| `max` | Upper limit | ⚠️ Partial | ROM upper bound |
| `min` | Lower limit | ⚠️ Partial | ROM lower bound |

**Key Insight**: `range`, `max`, and `min` are mathematically related:
```python
range = max - min
```

So if you have `max` and `min`, you can calculate `range`. Having all three IS redundant.

### Clinical Perspective

**Standard Deviation (`std`)**:
- ✅ **Keep** - Provides unique information about gait variability
- Used in research: High std indicates inconsistent gait (fall risk, neurological issues)
- NOT redundant with range (different statistical properties)

**Max and Min**:
- ⚠️ **Questionable** - Provide ROM limits but are partially redundant with range
- Clinical use: Identifying ROM restrictions (e.g., limited knee flexion)
- Could be derived: `max = mean + (range/2)`, `min = mean - (range/2)` (approximate)

**Range**:
- ✅ **Keep** - Simple, interpretable ROM metric
- Widely used in clinical practice
- Easier to understand than max/min separately

## Recommendations

### Option 1: Keep All Features (Current Approach)
**Pros**:
- Maximum information preservation
- Flexibility for different classifiers
- Some ML algorithms can handle redundancy

**Cons**:
- 18 extra features (6 joints × 3 stats)
- Potential multicollinearity in models
- More complex feature space

### Option 2: Remove Max/Min, Keep Std (Recommended)
**Pros**:
- Eliminates redundancy (range = max - min)
- Keeps unique information (std)
- Reduces feature count by 12 (6 joints × 2 stats)
- Simpler feature space

**Cons**:
- Loses explicit ROM limits
- Can't directly see max/min values

### Option 3: Remove All Extended Stats
**Pros**:
- Simplest approach
- Reduces feature count by 18
- Mean + range may be sufficient for many use cases

**Cons**:
- Loses variability information (std)
- Less comprehensive analysis

## Implementation

### If You Want to Fix the Zero Values

1. **Verify Configuration**:
```python
analyzer = EnhancedGaitAnalyzer(
    comprehensive_features=True,  # Ensures include_joint_statistics=True
    feature_extraction_config={
        "include_joint_statistics": True  # Explicit override
    }
)
```

2. **Run Diagnostic**:
```bash
python diagnose_extended_stats.py
```

3. **Check Logs**:
Look for:
```
DEBUG | Extracted extended stats for left_hip: std=X.XX, max=X.XX, min=X.XX
```

If you see these logs, the features ARE being created. If not, `include_joint_statistics=False`.

### If You Want to Remove Redundant Features

**Option A: Remove Max/Min (Keep Std)**

Edit `ambient/classification/features.py`:

```python
@dataclass
class GaitFeatureVector:
    # ... existing fields ...
    
    # Extended joint angle statistics (std only, not max/min)
    left_hip_std: float = 0.0
    left_knee_std: float = 0.0
    left_ankle_std: float = 0.0
    right_hip_std: float = 0.0
    right_knee_std: float = 0.0
    right_ankle_std: float = 0.0
    
    # Remove these fields:
    # left_hip_max, left_hip_min
    # left_knee_max, left_knee_min
    # left_ankle_max, left_ankle_min
    # right_hip_max, right_hip_min
    # right_knee_max, right_knee_min
    # right_ankle_max, right_ankle_min
```

Edit `ambient/analysis/feature_extractor.py`:

```python
# Extended statistics (optional)
if self.include_joint_statistics:
    features[f"{angle_name}_std"] = np.std(angle_values)
    # Remove max/min extraction
    logger.debug(f"Extracted std for {angle_name}: {features[f'{angle_name}_std']:.2f}")
```

**Option B: Remove All Extended Stats**

Set `include_joint_statistics=False`:

```python
analyzer = EnhancedGaitAnalyzer(
    comprehensive_features=True,
    feature_extraction_config={
        "include_joint_statistics": False  # Disable extended stats
    }
)
```

## Feature Count Impact

| Configuration | Feature Count | Notes |
|---------------|---------------|-------|
| Current (all) | 94 features | Includes std, max, min for 6 joints |
| Remove max/min | 82 features | Keeps std, removes max/min (12 features) |
| Remove all extended | 76 features | Removes std, max, min (18 features) |

## My Recommendation

**Remove max/min, keep std**:

1. **Eliminates redundancy**: range = max - min, so having all three is redundant
2. **Preserves unique information**: std provides variability info that range doesn't
3. **Reduces complexity**: 12 fewer features
4. **Maintains clinical value**: std is used in gait research for stability assessment

**Implementation**:
```python
# In FeatureExtractor._extract_joint_angle_features()
if self.include_joint_statistics:
    features[f"{angle_name}_std"] = np.std(angle_values)
    # Don't create max/min - they're redundant with range
```

This gives you:
- ✅ Mean (central tendency)
- ✅ Range (ROM spread)
- ✅ Std (variability/consistency)
- ❌ Max/Min (redundant with range)

**Result**: 82 comprehensive features instead of 94, with no loss of unique information.

## Next Steps

1. **Decide**: Do you want to fix the zero values OR remove redundant features?

2. **If fixing zero values**:
   - Run `diagnose_extended_stats.py`
   - Check if `include_joint_statistics=True`
   - Verify features are in `features_dict`

3. **If removing redundant features**:
   - Remove max/min fields from `GaitFeatureVector`
   - Update `FeatureExtractor` to not create them
   - Update `from_analysis_results()` to not extract them
   - Update documentation to reflect 82 features (not 94)

4. **Update documentation**:
   - Change "94+ features" to "82 features" (if removing max/min)
   - Update feature count in all docs
   - Update steering files

Let me know which direction you want to go!
