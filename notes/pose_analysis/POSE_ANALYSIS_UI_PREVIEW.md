# Pose Analysis UI Preview

## Tooltip Examples

### Overall Assessment Card
```
┌─────────────────────────────────────────────────────────┐
│ 🎯 Overall Assessment (?)  [Learn more →]              │
│ Summary of gait analysis results                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────────┐  ┌──────────────────┐           │
│  │ Overall Level (?)│  │ Symmetry      (?)│           │
│  │                  │  │                  │           │
│  │   [Moderate]     │  │   [Symmetric]    │           │
│  │ Confidence: Med  │  │ Score: 0.078     │           │
│  └──────────────────┘  └──────────────────┘           │
│                                                          │
└─────────────────────────────────────────────────────────┘

Hover on (?):
┌────────────────────────────────────────┐
│ Overall Gait Quality                   │
│                                        │
│ Comprehensive assessment combining     │
│ all gait metrics into a single        │
│ quality rating                         │
│                                        │
│ Good: Gait pattern is within normal   │
│       limits with symmetric, stable   │
│       movement                         │
│                                        │
│ Moderate: Some deviations detected    │
│           but generally functional    │
│                                        │
│ Poor: Significant abnormalities       │
│       requiring clinical attention    │
└────────────────────────────────────────┘
```

### Key Metrics Cards
```
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ 📊 Cadence    (?)│  │ ⚡ Stability  (?)│  │ 📈 Gait Cycles(?)│  │ 📊 Movement   (?)│
│                  │  │                  │  │                  │  │                  │
│      93.4        │  │     [Low]        │  │       11         │  │ Consistency:     │
│  steps/minute    │  │                  │  │  detected cycles │  │    [Poor]        │
│                  │  │ Center of mass   │  │                  │  │                  │
│     [Slow]       │  │    stability     │  │  Avg: 1.05s      │  │ Smoothness:      │
│                  │  │                  │  │                  │  │   [Smooth]       │
└──────────────────┘  └──────────────────┘  └──────────────────┘  └──────────────────┘

Hover on Cadence (?):
┌────────────────────────────────────────┐
│ Cadence (Steps/Minute)                 │
│                                        │
│ Number of steps taken per minute -    │
│ a fundamental measure of walking speed │
│                                        │
│ Normal: 100-120 steps/min             │
│         Typical adult walking pace     │
│                                        │
│ Slow: < 100 steps/min                 │
│       May indicate caution, pain, or  │
│       reduced mobility                 │
│                                        │
│ Fast: > 120 steps/min                 │
│       Rapid walking or running gait    │
│                                        │
│ Clinical: Slow cadence may indicate   │
│          fall risk or mobility issues  │
└────────────────────────────────────────┘
```

### Recommendations Section
```
┌─────────────────────────────────────────────────────────┐
│ ℹ️  Recommendations                                      │
│ Clinical suggestions based on analysis                   │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ✓ Consider balance training or stability exercises     │
│                                                          │
│  ✓ Monitor for fall risk due to low stability          │
│                                                          │
│  ✓ Evaluate for underlying causes of slow cadence      │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Sequence Information
```
┌─────────────────────────────────────────────────────────┐
│ Sequence Information                                     │
│ Technical details about the analyzed sequence            │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Frames: 215    Duration: 7.17s    FPS: 30    COCO_17  │
│                                                          │
│  Performance                                             │
│  Analysis time: 0.02s    Processing Speed: 666.5 fps    │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Asymmetry Details
```
┌─────────────────────────────────────────────────────────┐
│ Asymmetry Details                                        │
│ Joints showing the most asymmetry                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Hip Velocity          Asymmetry: 0.337      [High]     │
│  Ankle Velocity        Asymmetry: 0.279      [High]     │
│  Ankle Variance        Asymmetry: 0.201      [High]     │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## Help Page Structure

### Navigation
```
┌─────────────────────────────────────────────────────────────────┐
│ 📚 Pose Analysis Guide                                          │
│ Understanding gait analysis metrics, scores, and clinical       │
│ interpretations                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Quick Navigation                                               │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌────────┐│
│  │ Overall      │ │ Cadence      │ │ Symmetry     │ │Stability││
│  │ Assessment   │ │              │ │              │ │         ││
│  └──────────────┘ └──────────────┘ └──────────────┘ └────────┘│
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Tabs
```
┌─────────────────────────────────────────────────────────────────┐
│ [Overview] [Metrics] [Interpretation] [Clinical Use] [FAQ]      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Content for selected tab...                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Cadence Section (Metrics Tab)
```
┌─────────────────────────────────────────────────────────────────┐
│ 📊 Cadence (Steps per Minute)                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ Cadence measures the number of steps taken per minute.         │
│ It's a fundamental indicator of walking speed and efficiency.   │
│                                                                 │
│ ┃ Normal Range                                                  │
│ ┃ 100-120 steps/minute for typical adult walking               │
│                                                                 │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐          │
│  │ [Slow]       │ │ [Normal]     │ │ [Fast]       │          │
│  │ < 100        │ │ 100-120      │ │ > 120        │          │
│  │              │ │              │ │              │          │
│  │ May indicate │ │ Typical      │ │ Rapid        │          │
│  │ caution,     │ │ adult        │ │ walking or   │          │
│  │ pain, or     │ │ walking      │ │ jogging      │          │
│  │ reduced      │ │ pace         │ │ gait         │          │
│  │ mobility     │ │              │ │              │          │
│  └──────────────┘ └──────────────┘ └──────────────┘          │
│                                                                 │
│  ⚠️ Clinical Significance                                       │
│  Slow cadence (< 100 steps/min) may indicate increased fall   │
│  risk or mobility limitations. Cadence is affected by age,     │
│  height, fitness level, and walking surface.                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Symmetry Section with Diagram
```
┌─────────────────────────────────────────────────────────────────┐
│ 🎯 Gait Symmetry                                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ Symmetry measures the balance between left and right side      │
│ movements during walking.                                       │
│                                                                 │
│  ✓ Symmetric (< 0.10)                                          │
│    Excellent left-right balance. Normal gait pattern.          │
│                                                                 │
│  ⚠️ Mildly Asymmetric (0.10-0.20)                              │
│    Minor imbalances. May be normal variation.                  │
│                                                                 │
│  ⚠️ Moderately Asymmetric (0.20-0.30)                          │
│    Noticeable imbalance. Consider evaluation.                  │
│                                                                 │
│  ❌ Severely Asymmetric (> 0.30)                               │
│    Significant imbalance requiring attention.                  │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ Gait Symmetry Visualization                             │  │
│  │                                                          │  │
│  │  [Symmetric]    [Mildly Asymmetric]  [Severely Asymm.]  │  │
│  │   < 0.10           0.10-0.20            > 0.30          │  │
│  │                                                          │  │
│  │   ██  ██          ██  ▓▓              ██  ░░            │  │
│  │   ██  ██          ██  ▓▓              ██  ░░            │  │
│  │   ██  ██          ██  ▓▓              ██                │  │
│  │   ██  ██          ██                  ██                │  │
│  │   L   R           L   R               L   R             │  │
│  │                                                          │  │
│  │  Equal movement   Minor difference   Significant diff   │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Gait Cycle Diagram
```
┌─────────────────────────────────────────────────────────────────┐
│ Gait Cycle Phases                                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌────────────────────────────────┬──────────────────────┐     │
│  │   Stance Phase (60%)           │  Swing Phase (40%)   │     │
│  │   Foot in contact with ground  │  Foot moving in air  │     │
│  └────────────────────────────────┴──────────────────────┘     │
│  Heel Strike              Toe Off              Heel Strike     │
│                                                                 │
│  Stance Sub-phases:              Swing Sub-phases:             │
│  • Initial Contact (0-2%)        • Initial Swing (60-73%)      │
│  • Loading Response (2-12%)      • Mid Swing (73-87%)          │
│  • Mid Stance (12-31%)           • Terminal Swing (87-100%)    │
│  • Terminal Stance (31-50%)                                    │
│  • Pre-swing (50-60%)                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Color Coding

### Assessment Levels
- 🟢 **Green**: Good, Normal, Symmetric, High (positive)
- 🟡 **Yellow**: Moderate, Mildly Asymmetric, Medium (caution)
- 🟠 **Orange**: Moderately Asymmetric (warning)
- 🔴 **Red**: Poor, Severely Asymmetric, Low (alert)

### UI Elements
- **Primary**: Blue (#3B82F6) - Interactive elements
- **Success**: Green (#10B981) - Positive indicators
- **Warning**: Yellow (#F59E0B) - Caution indicators
- **Danger**: Red (#EF4444) - Alert indicators
- **Muted**: Gray (#6B7280) - Secondary text

---

## Interaction Flow

### 1. Quick Understanding (Tooltip)
```
User hovers on (?) icon
  ↓
Tooltip appears with:
  - Metric title
  - Brief description
  - Key interpretation points
  - Clinical significance
  ↓
User gets immediate understanding
```

### 2. Detailed Learning (Help Page)
```
User clicks "Learn more" link
  ↓
Help page opens in new tab
  ↓
User navigates to relevant section
  ↓
User reads detailed explanation
  ↓
User views visual diagrams
  ↓
User understands concept deeply
```

### 3. Progressive Disclosure
```
Level 1: Metric value displayed
  ↓
Level 2: Hover tooltip (2-3 seconds)
  ↓
Level 3: Expanded tooltip with examples (10-20 seconds)
  ↓
Level 4: Full help page with diagrams (2-5 minutes)
```

---

## Mobile Experience

### Tooltips on Mobile
- Tap (?) icon to show tooltip
- Tap outside to dismiss
- Scrollable content if needed
- Touch-friendly targets (44x44px minimum)

### Help Page on Mobile
- Responsive grid layouts
- Stacked cards on small screens
- Collapsible sections
- Easy navigation
- Readable text sizes

---

## Accessibility

### Keyboard Navigation
- Tab to focus on (?) icons
- Enter/Space to activate tooltip
- Escape to close tooltip
- Tab through help page sections

### Screen Readers
- ARIA labels on all interactive elements
- Semantic HTML structure
- Alt text for diagrams
- Descriptive link text

### Visual
- High contrast colors
- Clear focus indicators
- Readable font sizes (14px minimum)
- Color + text indicators (not color alone)

---

## Summary

The UI now provides:
- ✅ Immediate help via tooltips
- ✅ Detailed explanations via help page
- ✅ Visual learning via diagrams
- ✅ Progressive disclosure
- ✅ Mobile-friendly design
- ✅ Accessible to all users
- ✅ Professional appearance
- ✅ Clinical context throughout

Users can understand metrics at a glance, get detailed explanations when needed, and access comprehensive documentation for deep learning.
