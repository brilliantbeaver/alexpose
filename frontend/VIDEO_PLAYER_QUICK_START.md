# Video Player - Quick Start Guide

## Overview

The AlexPose video player is a custom-built component designed specifically for gait analysis, featuring frame-by-frame navigation and precise playback controls.

## Accessing the Video Player

### From Dashboard
1. Navigate to Dashboard (`/dashboard`)
2. Find an analysis in "Recent Analyses"
3. Click "View →"
4. Click the "Video" tab

### From Results List
1. Navigate to Results (`/results`)
2. Click "📊 View Details" on any analysis
3. Click the "Video" tab

## Quick Controls Reference

### Essential Controls

```
┌─────────────────────────────────────────────┐
│ [▶️] Play/Pause                              │
│ [⏮] [⏭] Previous/Next Frame                 │
│ [⏪] [⏩] Skip Backward/Forward (5s)          │
│ [━━━━━] Seek Bar (drag to any point)        │
│ [🔊] Volume Control                          │
│ [1x ▼] Playback Speed (0.25x - 2x)          │
│ [⛶] Fullscreen                               │
└─────────────────────────────────────────────┘
```

## Common Use Cases

### 1. Frame-by-Frame Analysis

**Goal**: Examine a specific movement in detail

**Steps**:
1. Play video to approximate location
2. Click **Pause** (or press Space)
3. Use **⏮ Previous Frame** and **⏭ Next Frame** buttons
4. Observe frame counter in top-right corner
5. Note specific frame numbers for documentation

**Tip**: Use keyboard shortcuts `,` (comma) and `.` (period) for faster frame navigation.

### 2. Slow Motion Review

**Goal**: Watch movement at reduced speed

**Steps**:
1. Click the **Speed Selector** (shows "1x")
2. Select **0.5x** or **0.25x**
3. Click **Play**
4. Movement plays at half or quarter speed

**Tip**: Use 0.25x for very detailed analysis of complex movements.

### 3. Comparing Two Points in Time

**Goal**: Compare gait at different times in video

**Steps**:
1. Play to first point of interest
2. Note the **frame number** (top-right)
3. Use **seek bar** to jump to second point
4. Note the second frame number
5. Use frame navigation to switch between points

**Tip**: Write down frame numbers for later reference.

### 4. Analyzing Specific Gait Cycles

**Goal**: Focus on particular gait cycles

**Steps**:
1. Identify start of gait cycle
2. Note **start frame** number
3. Use **⏭ Next Frame** to advance through cycle
4. Note **end frame** number
5. Calculate cycle duration: (end - start) / 30 fps

**Example**: Frames 150-180 = 30 frames = 1 second at 30 fps

## Keyboard Shortcuts

### Playback
- **Space**: Play/Pause
- **Left Arrow**: Skip backward 5 seconds
- **Right Arrow**: Skip forward 5 seconds

### Frame Navigation
- **,** (comma): Previous frame
- **.** (period): Next frame

### Volume
- **Up Arrow**: Increase volume
- **Down Arrow**: Decrease volume
- **M**: Mute/Unmute

### View
- **F**: Toggle fullscreen

## Understanding the Display

### Frame Counter (Top-Right)
```
Frame: 450 / 1350
       ↑       ↑
   Current  Total
```

**Meaning**: Currently at frame 450 out of 1350 total frames

**Calculation**: 
- At 30 fps: 450 frames = 15 seconds
- Total: 1350 frames = 45 seconds

### Time Display (Below Seek Bar)
```
00:15          00:45
  ↑              ↑
Current      Duration
```

**Format**: MM:SS (minutes:seconds)

### Info Bar (Bottom)
```
Frame Rate: 30 fps • Duration: 00:45
Speed: 1x • Frame: 450/1350
```

**Details**:
- **Frame Rate**: Frames per second (usually 30)
- **Duration**: Total video length
- **Speed**: Current playback rate
- **Frame**: Current/Total frames

## Tips & Tricks

### 1. Finding Specific Events

**Problem**: Need to find when patient starts limping

**Solution**:
1. Play video at **1x** speed
2. When you see the event, click **Pause**
3. Use **⏮ Previous Frame** to go back
4. Find exact frame where event starts
5. Note frame number

### 2. Measuring Time Between Events

**Problem**: How long between two steps?

**Solution**:
1. Find first step (note frame number, e.g., 150)
2. Find second step (note frame number, e.g., 180)
3. Calculate: (180 - 150) / 30 fps = 1 second

### 3. Reviewing Difficult Sections

**Problem**: Movement too fast to analyze

**Solution**:
1. Set speed to **0.25x**
2. Play through difficult section
3. Use **⏮** and **⏭** for frame-by-frame if needed

### 4. Comparing Left vs Right

**Problem**: Need to compare left and right leg movements

**Solution**:
1. Find left leg event (note frame)
2. Find right leg event (note frame)
3. Use seek bar to jump between frames
4. Compare movements

## Troubleshooting

### Video Won't Load

**Symptoms**: Loading spinner doesn't disappear

**Solutions**:
1. Check internet connection
2. Refresh the page
3. Try a different browser
4. Check if video URL is valid

### Video Stutters

**Symptoms**: Playback is choppy

**Solutions**:
1. Pause and wait for buffering
2. Reduce playback speed
3. Check internet speed
4. Close other browser tabs

### Controls Not Responding

**Symptoms**: Buttons don't work

**Solutions**:
1. Click on video area first
2. Refresh the page
3. Check browser console for errors
4. Try different browser

### Frame Counter Not Updating

**Symptoms**: Frame number stuck

**Solutions**:
1. Pause and play again
2. Seek to different position
3. Refresh the page

## Advanced Features

### Callbacks (For Developers)

The video player provides callbacks for integration:

```typescript
<VideoPlayer
  videoUrl={url}
  videoName={name}
  onTimeUpdate={(time) => {
    // Called every time video time updates
    console.log('Current time:', time);
  }}
  onFrameChange={(frame) => {
    // Called every time frame changes
    console.log('Current frame:', frame);
  }}
/>
```

**Use Cases**:
- Sync metrics display with video
- Update pose overlay
- Trigger analysis at specific frames
- Log viewing behavior

### Custom Frame Rate

Default is 30 fps, but can be customized:

```typescript
<VideoPlayer
  videoUrl={url}
  videoName={name}
  frameRate={60}  // For 60 fps videos
/>
```

## Best Practices

### For Clinical Analysis

1. **Always note frame numbers** for documentation
2. **Use slow motion** (0.5x or 0.25x) for detailed review
3. **Review multiple times** at different speeds
4. **Compare symmetry** by noting left/right frame numbers
5. **Document findings** with specific frame references

### For Research

1. **Record frame numbers** for reproducibility
2. **Use consistent playback speed** for comparisons
3. **Note video metadata** (frame rate, duration)
4. **Export frame numbers** with findings
5. **Verify frame accuracy** with frame counter

### For Teaching

1. **Use slow motion** to demonstrate movements
2. **Pause at key frames** to explain concepts
3. **Use frame-by-frame** to show details
4. **Note frame numbers** for student reference
5. **Demonstrate comparisons** using seek bar

## Common Workflows

### Workflow 1: Initial Review
```
1. Play video at 1x speed (full review)
2. Note areas of interest
3. Replay at 0.5x speed (detailed review)
4. Pause at key frames
5. Document findings
```

### Workflow 2: Detailed Analysis
```
1. Navigate to area of interest
2. Set speed to 0.25x
3. Play through section
4. Use frame-by-frame for critical points
5. Note frame numbers
6. Document observations
```

### Workflow 3: Comparison
```
1. Find first event (note frame)
2. Find second event (note frame)
3. Jump between frames using seek bar
4. Compare movements
5. Calculate time difference
6. Document comparison
```

## Summary

The AlexPose video player provides professional-grade tools for gait analysis:

✅ **Frame-accurate navigation** for precise analysis  
✅ **Variable playback speed** for detailed examination  
✅ **Comprehensive controls** for efficient workflow  
✅ **Real-time frame tracking** for documentation  
✅ **Keyboard shortcuts** for power users  

Master these controls to perform thorough, accurate gait analysis with confidence.

---

**Need Help?**
- See [VIDEO_PLAYER_DOCUMENTATION.md](./components/video/VIDEO_PLAYER_DOCUMENTATION.md) for technical details
- See [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) for common issues
- Contact support for additional assistance

**Quick Reference Card**:
```
▶️/⏸  Play/Pause        Space
⏮     Previous Frame    , (comma)
⏭     Next Frame        . (period)
⏪     Skip Back         Left Arrow
⏩     Skip Forward      Right Arrow
🔊/🔇  Mute             M
⛶     Fullscreen        F
```

---

**Last Updated**: January 3, 2026  
**Version**: 1.0.0  
**Status**: ✅ Ready for use
