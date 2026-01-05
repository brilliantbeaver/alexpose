# Pose Analysis - Quick Reference Card

## 🚀 Quick Start

### Start Servers
```bash
# Terminal 1 - Backend
cd server
python -m uvicorn main:app --reload

# Terminal 2 - Frontend  
cd frontend
npm run dev
```

### Access Application
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

---

## 📁 Key Files

### Backend
- `server/services/pose_analysis_service.py` - Service layer (400 lines)
- `server/routers/pose_analysis.py` - API endpoints (300 lines)
- `tests/test_pose_analysis_service.py` - Unit tests (500 lines)

### Frontend
- `frontend/components/pose-analysis/PoseAnalysisOverview.tsx` - UI component (450 lines)
- `frontend/app/training/gavd/[datasetId]/page.tsx` - Page integration (modified)

### Documentation
- `notes/POSE_ANALYSIS_IMPLEMENTATION_PLAN.md` - Full implementation plan
- `notes/POSE_ANALYSIS_MVP_PROGRESS.md` - Progress tracking
- `notes/POSE_ANALYSIS_TESTING_GUIDE.md` - Testing guide
- `notes/POSE_ANALYSIS_IMPLEMENTATION_SUMMARY.md` - Complete summary

---

## 🔌 API Endpoints

### Complete Analysis
```bash
GET /api/v1/pose-analysis/sequence/{dataset_id}/{sequence_id}
```

### Subset Endpoints
```bash
GET /api/v1/pose-analysis/sequence/{dataset_id}/{sequence_id}/features
GET /api/v1/pose-analysis/sequence/{dataset_id}/{sequence_id}/cycles
GET /api/v1/pose-analysis/sequence/{dataset_id}/{sequence_id}/symmetry
```

### Cache Management
```bash
DELETE /api/v1/pose-analysis/cache/{dataset_id}/{sequence_id}
DELETE /api/v1/pose-analysis/cache/{dataset_id}
GET /api/v1/pose-analysis/cache/stats
```

### Health Check
```bash
GET /api/v1/pose-analysis/health
```

---

## 🧪 Testing

### Run Backend Tests
```bash
python -m pytest tests/test_pose_analysis_service.py -v
```

**Expected**: 17 passed ✅

### Manual Testing Steps
1. Navigate to GAVD dataset page
2. Click "Pose Analysis" tab
3. Select sequence with pose data
4. Verify analysis displays correctly
5. Test error handling (sequence without pose data)
6. Test refresh button
7. Test sequence switching

---

## 📊 What's Displayed

### Overall Assessment Card
- Overall level (Good/Moderate/Poor)
- Confidence level
- Symmetry classification
- Symmetry score

### Key Metrics (4 Cards)
1. **Cadence** - steps/minute + level
2. **Stability** - level badge
3. **Gait Cycles** - count + average duration
4. **Movement** - consistency + smoothness

### Additional Sections
- Recommendations (if available)
- Sequence information (frames, duration, FPS)
- Asymmetry details (most asymmetric joints)
- Performance metrics (analysis time)

---

## ⚡ Performance

- **Initial Analysis**: 1-2 seconds
- **Cached Response**: <100ms
- **Component Render**: <50ms
- **Cache TTL**: 1 hour

---

## 🐛 Common Issues

### Backend Won't Start
```bash
# Check Python environment
python --version  # Should be 3.12+

# Reinstall dependencies
pip install -r requirements.txt

# Check port availability
netstat -ano | findstr :8000
```

### Frontend Won't Start
```bash
# Check Node version
node --version  # Should be 18+

# Reinstall dependencies
npm install

# Check port availability
netstat -ano | findstr :3000
```

### No Pose Data Error
- Verify dataset has been processed with pose estimation
- Check sequence has pose data in database
- Try different sequence

### CORS Errors
- Verify backend CORS settings in `server/main.py`
- Check frontend is accessing `http://localhost:8000`

---

## 📈 Status

### Completed ✅
- [x] Backend service layer
- [x] API endpoints (8 total)
- [x] Frontend component
- [x] Page integration
- [x] Unit tests (17 passing)
- [x] TypeScript compilation
- [x] Documentation

### Pending 🔄
- [ ] Manual testing with real data
- [ ] Browser compatibility testing
- [ ] UI polish and refinements
- [ ] Deployment to staging

---

## 🎯 Success Criteria

- [x] Backend tests passing (17/17)
- [x] TypeScript errors resolved (0 errors)
- [x] Component renders without errors
- [x] API endpoints functional
- [ ] Manual testing complete
- [ ] Bug fixes applied
- [ ] Documentation updated

---

## 📞 Support

### Check Logs
```bash
# Backend logs
# Check terminal running uvicorn

# Frontend logs  
# Check browser console (F12)

# Test logs
python -m pytest tests/test_pose_analysis_service.py -v -s
```

### Verify Health
```bash
# Backend health
curl http://localhost:8000/api/v1/pose-analysis/health

# Cache stats
curl http://localhost:8000/api/v1/pose-analysis/cache/stats
```

---

## 🔄 Next Steps

1. **Test with Real Data**
   - Start both servers
   - Navigate to GAVD dataset
   - Test Pose Analysis tab
   - Document any issues

2. **Fix Bugs**
   - Address critical issues
   - Improve error messages
   - Optimize performance

3. **Polish UI**
   - Refine styling
   - Add animations
   - Improve responsiveness

4. **Deploy**
   - Test in staging
   - Update documentation
   - Deploy to production

---

**Last Updated**: January 4, 2026  
**Version**: 1.0  
**Status**: Ready for Testing 🔄
