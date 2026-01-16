# Jupyter Notebook Setup Guide

## Quick Start

If you're seeing import errors in Jupyter notebooks, follow these steps:

### 1. Clear Python Cache
```bash
python scripts/clear_python_cache.py
```

### 2. Restart Jupyter Kernel
- In Jupyter: **Kernel → Restart Kernel**
- Or restart the entire Jupyter server

### 3. Enable Auto-Reload (Recommended)
Add this to the first cell of your notebook:
```python
%load_ext autoreload
%autoreload 2
```

This automatically reloads modules when they change, preventing most cache issues.

## Common Import Errors

### Error: "cannot import name 'X' from 'Y'"
**Cause:** Stale Python bytecode cache  
**Solution:** Clear cache + restart kernel (steps 1-2 above)

### Error: "UnsupportedOperation: fileno"
**Cause:** This was a Jupyter compatibility issue (now fixed!)  
**Solution:** Update to latest code - the fix is already in place

### Error: "ModuleNotFoundError"
**Cause:** Missing dependencies or wrong Python environment  
**Solution:**
```bash
# Ensure you're in the virtual environment
uv sync
```

## Standard Notebook Template

Use this template at the start of your notebooks:

```python
# Cell 1: Setup and Auto-reload
%load_ext autoreload
%autoreload 2

import sys
from pathlib import Path

# Add project root to path
project_root = Path.cwd().parent
sys.path.insert(0, str(project_root))
print(f"Project root: {project_root}")

# Cell 2: Imports
from ambient.gavd import GaitDataProcessor, GAVDDataLoader, PoseDataConverter
from ambient.pose import MediaPipeEstimator, get_pose_estimator
from ambient.utils.csv_parser import parse_csv_with_dicts
import pandas as pd
import numpy as np

print("[OK] All imports successful")
```

## Troubleshooting

### Imports work in terminal but not in Jupyter
1. Check you're using the correct kernel (should be `alexpose` virtual environment)
2. Clear cache: `python scripts/clear_python_cache.py`
3. Restart kernel

### Changes to code not reflected in notebook
- Enable autoreload (see step 3 above)
- Or manually restart kernel after code changes

### "MediaPipe detects 33 landmarks" message
This is normal - it's MediaPipe initializing. Not an error.

## Best Practices

1. **Always use autoreload** during development
2. **Restart kernel** after major code refactoring
3. **Clear cache** if you see import errors
4. **Use relative imports** from project root
5. **Keep notebooks in `notebooks/` directory** for consistent paths

## Related Files

- `scripts/clear_python_cache.py` - Cache clearing utility
- `notes/IMPORT_ERROR_FIX.md` - Detailed technical documentation
- `.venv/` - Virtual environment (don't commit this)
