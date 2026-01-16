# Import Error Fix - January 15, 2026

## Problem

Jupyter notebooks were failing with import errors:
```python
ImportError: cannot import name 'get_pose_estimator' from 'ambient.pose.factory'
```

## Root Cause Analysis

### 1. Stale Python Bytecode Cache
- Python caches compiled bytecode in `.pyc` files and `__pycache__` directories
- After file migrations/renames, the cache can become stale
- Jupyter notebook kernels can hold onto old cached imports

### 2. Why Command Line Worked But Notebooks Failed
- Command line Python starts fresh each time
- Jupyter kernels persist across cell executions
- Kernel may have loaded old module versions before migration

## Solution

### Immediate Fix
1. **Clear Python bytecode cache:**
   ```bash
   python scripts/clear_python_cache.py
   ```

2. **Restart Jupyter kernel:**
   - In Jupyter: Kernel → Restart Kernel
   - Or restart the entire Jupyter server

### Verification
```bash
# Test imports work
python -c "from ambient.pose import OpenPoseEstimator; print('OK')"
python -c "from ambient.pose.pose_estimators import get_pose_estimator; print('OK')"
```

## Prevention

### When to Clear Cache
Clear Python cache after:
- Moving/renaming Python files
- Refactoring import statements
- Seeing "cannot import name" errors for functions that exist
- Major code reorganization

### Quick Commands

**Windows PowerShell:**
```powershell
# Remove .pyc files
Get-ChildItem -Path . -Recurse -Filter "*.pyc" | Remove-Item -Force

# Remove __pycache__ directories
Get-ChildItem -Path . -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
```

**Linux/macOS:**
```bash
# Remove .pyc files and __pycache__ directories
find . -type f -name "*.pyc" -delete
find . -type d -name "__pycache__" -exec rm -rf {} +
```

**Python script (cross-platform):**
```bash
python scripts/clear_python_cache.py
```

## Technical Details

### Import Chain
```
notebooks/explore3.ipynb
  → from ambient.pose import OpenPoseEstimator
    → ambient/pose/__init__.py
      → from ambient.pose.openpose_estimator import OpenPoseEstimator
        → ambient/pose/openpose_estimator.py (OK)
      → from ambient.pose.factory import get_pose_estimator
        → ambient/pose/factory.py (OK - function exists at line 373)
```

### Why This Happened
1. Previous migration renamed `ambient/utils/logging.py` → `log_config.py`
2. Python cached the old module structure
3. Jupyter kernel loaded stale cached imports
4. New code couldn't find functions in cached modules

## Files Modified
- Created: `scripts/clear_python_cache.py` - Cache clearing utility
- Created: `notes/IMPORT_ERROR_FIX.md` - This documentation

## Related Issues
- Circular import fix (logging.py → log_config.py)
- Unicode encoding fix (emoji removal)
- notebooks/utils migration to ambient/utils

## Testing
```bash
# All these should work after fix:
python -c "from ambient.pose import OpenPoseEstimator"
python -c "from ambient.pose.pose_estimators import get_pose_estimator"
python -c "from ambient.pose.factory import get_pose_estimator"
python -c "from ambient.gavd import GaitDataProcessor, GAVDDataLoader"
```

## Jupyter Notebook Users

**After any code migration, always:**
1. Clear Python cache: `python scripts/clear_python_cache.py`
2. Restart Jupyter kernel: Kernel → Restart Kernel
3. Re-run imports from the top of the notebook

**Pro tip:** Add this to the first cell of notebooks:
```python
# Force reload modules (useful during development)
%load_ext autoreload
%autoreload 2
```

This automatically reloads modules when they change, reducing cache issues.
