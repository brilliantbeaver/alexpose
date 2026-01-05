# GAVD Upload - Quick Fix Guide

## Problem
When clicking "Upload GAVD Dataset", you get an error because the backend server is not running.

## Solution

### Option 1: Use the Startup Script (Recommended)

**Windows:**
```powershell
# From project root
.\scripts\start-dev.ps1
```

**Mac/Linux:**
```bash
# From project root
chmod +x scripts/start-dev.sh
./scripts/start-dev.sh
```

This will:
- ✅ Start the backend server on http://localhost:8000
- ✅ Start the frontend server on http://localhost:3000
- ✅ Check dependencies
- ✅ Show you the URLs to access

### Option 2: Start Manually

**Terminal 1 - Backend:**
```bash
cd C:\Users\theod\dev\alex\alexpose
uv run uvicorn server.main:app --reload --host 127.0.0.1 --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd C:\Users\theod\dev\alex\alexpose\frontend
npm run dev
```

## Verify It's Working

1. **Check Backend Health:**
   ```bash
   curl http://localhost:8000/health
   ```
   Should return: `{"status":"healthy",...}`

2. **Check Frontend:**
   Open http://localhost:3000 in your browser

3. **Test GAVD Upload:**
   - Go to http://localhost:3000/training/gavd
   - You should see the upload page
   - Try uploading a CSV file

## Expected GAVD CSV Format

Your CSV file should have these columns:
- `seq` - Sequence ID (e.g., "cljan9b4p00043n6ligceanyp")
- `frame_num` - Frame number (e.g., 1757)
- `bbox` - Bounding box JSON: `{"left":156,"top":125,"width":238,"height":500}`
- `url` - YouTube URL: `https://www.youtube.com/watch?v=B5hrxKe2nP8`
- `vid_info` - Video info JSON: `{"width":1280,"height":720}`
- Optional: `cam_view`, `gait_event`, `dataset`, `gait_pat`, `id`

## What Happens After Upload

1. **CSV Validation** - Checks file format and required columns
2. **YouTube Download** - Downloads videos from URLs (cached in `data/youtube/`)
3. **Frame Extraction** - Extracts frames at specified frame numbers
4. **Pose Estimation** - Runs MediaPipe/OpenPose on each frame
5. **Results Saved** - Stores processed data in `data/training/gavd/`

## Troubleshooting

### Backend won't start
- Check if port 8000 is already in use: `netstat -ano | findstr ":8000"`
- Check Python environment: `uv run python --version`
- Check logs in `logs/` directory

### Frontend won't start
- Check if port 3000 is already in use: `netstat -ano | findstr ":3000"`
- Install dependencies: `cd frontend && npm install`
- Check Node version: `node --version` (should be 18+)

### Upload fails
- Check backend is running: `curl http://localhost:8000/health`
- Check browser console for errors (F12)
- Check backend logs for detailed error messages
- Verify CSV format matches requirements

### YouTube download fails
- Videos may be age-restricted or region-locked
- Add cookies file: `config/yt_cookies.txt` (see YouTube-DL docs)
- Check internet connection
- Try a different video URL

### Pose estimation fails
- Check if MediaPipe is installed: `uv run python -c "import mediapipe"`
- Falls back to placeholder keypoints if estimator fails
- Check logs for specific error messages

## Sample Test CSV

Create a file `test_gavd.csv`:

```csv
seq,frame_num,cam_view,gait_event,dataset,gait_pat,bbox,vid_info,id,url
test_seq_001,100,front,heel_strike,GAVD,normal,"{""left"":100,""top"":50,""width"":200,""height"":400}","{""width"":1280,""height"":720}",test001,https://www.youtube.com/watch?v=dQw4w9WgXcQ
test_seq_001,101,front,toe_off,GAVD,normal,"{""left"":102,""top"":51,""width"":200,""height"":400}","{""width"":1280,""height"":720}",test001,https://www.youtube.com/watch?v=dQw4w9WgXcQ
```

## Next Steps

Once both servers are running:

1. Navigate to http://localhost:3000/training/gavd
2. Upload your GAVD CSV file
3. Wait for processing to complete
4. View results in the analysis page

## Need More Help?

- Check `GAVD_UPLOAD_ERROR_ANALYSIS.md` for detailed technical analysis
- Review `notebooks/explore2.ipynb` for processing examples
- Check server logs in `logs/` directory
- Open browser DevTools (F12) to see network errors
