# AlexPose Analysis Flow Diagrams

## Complete System Architecture

```mermaid
graph TB
    subgraph "Input Layer"
        A[Video Upload] --> B[Dataset Processing]
        B --> C[Pose Estimation<br/>MediaPipe/OpenPose]
        C --> D[Pose Sequence<br/>Keypoint Data]
    end
    
    subgraph "Analysis Engine"
        D --> E[EnhancedGaitAnalyzer]
        E --> F[FeatureExtractor<br/>60+ Features]
        E --> G[TemporalAnalyzer<br/>Gait Cycles]
        E --> H[SymmetryAnalyzer<br/>L/R Comparison]
        
        F --> I[Kinematic Features<br/>Velocity, Acceleration]
        F --> J[Joint Angle Features<br/>ROM, Patterns]
        F --> K[Stability Features<br/>COM, Balance]
        
        G --> L[Cycle Detection<br/>Heel Strike/Toe Off]
        G --> M[Timing Analysis<br/>Cadence, Phases]
        G --> N[Rhythm Assessment<br/>Regularity, Frequency]
        
        H --> O[Bilateral Comparison<br/>Joint Symmetry]
        H --> P[Asymmetry Detection<br/>Pattern Analysis]
        H --> Q[Coordination Analysis<br/>Inter-limb Timing]
    end
    
    subgraph "Clinical Assessment"
        I --> R[Clinical Summary<br/>Assessment Generator]
        J --> R
        K --> R
        L --> R
        M --> R
        N --> R
        O --> R
        P --> R
        Q --> R
        
        R --> S[Overall Assessment<br/>Good/Moderate/Poor]
        R --> T[Key Metrics<br/>Cadence, Symmetry, Stability]
        R --> U[Recommendations<br/>Clinical Suggestions]
    end
    
    subgraph "Storage & Caching"
        S --> V[Database Storage<br/>SQLite/PostgreSQL]
        T --> V
        U --> V
        V --> W[File Cache<br/>7-day Expiration]
        W --> X[Hash-based<br/>Deduplication]
    end
    
    subgraph "Web Interface"
        V --> Y[REST API<br/>FastAPI Endpoints]
        Y --> Z[React Frontend<br/>NextJS + Shadcn UI]
        
        Z --> AA[Dataset Browser<br/>Upload & Management]
        Z --> BB[Video Player<br/>Frame Navigation]
        Z --> CC[Analysis Dashboard<br/>Real-time Results]
        
        CC --> DD[Overall Assessment Card<br/>Color-coded Status]
        CC --> EE[Key Metrics Grid<br/>Interactive Cards]
        CC --> FF[Detailed Panels<br/>Tooltips & Help]
        
        DD --> GG[Symmetry Assessment<br/>Score & Classification]
        EE --> HH[Cadence Display<br/>Steps/min with Ranges]
        EE --> II[Stability Indicator<br/>Balance Assessment]
        EE --> JJ[Gait Cycles Count<br/>Detection Quality]
        FF --> KK[Recommendations<br/>Clinical Suggestions]
        FF --> LL[Sequence Info<br/>Technical Details]
    end
    
    style E fill:#e1f5ff
    style R fill:#fff4e1
    style Z fill:#e8f5e9
    style CC fill:#f3e5f5
```

## Frontend User Interface Flow

```mermaid
graph LR
    subgraph "Dataset Management"
        A1[Upload Dataset<br/>Drag & Drop] --> A2[Processing Status<br/>Real-time Progress]
        A2 --> A3[Dataset Browser<br/>Sequence List]
        A3 --> A4[Search & Filter<br/>Gait Patterns]
    end
    
    subgraph "Sequence Selection"
        A4 --> B1[Select Sequence<br/>Click to Analyze]
        B1 --> B2[Loading State<br/>Analysis Progress]
        B2 --> B3[Cache Check<br/>Database/File]
        
        B3 --> B4{Cache Hit?}
        B4 -->|Yes| B5[Load Cached<br/>Results ⚡]
        B4 -->|No| B6[Run Analysis<br/>Real-time Processing]
        B6 --> B7[Save to Cache<br/>Future Access]
    end
    
    subgraph "Analysis Display"
        B5 --> C1[Analysis Dashboard<br/>Main Interface]
        B7 --> C1
        
        C1 --> C2[Overall Assessment<br/>Good/Moderate/Poor]
        C1 --> C3[Key Metrics Grid<br/>4 Primary Cards]
        C1 --> C4[Detailed Sections<br/>Expandable Panels]
        
        C2 --> C5[Symmetry Score<br/>Color-coded Badge]
        C2 --> C6[Confidence Level<br/>Analysis Reliability]
        
        C3 --> C7[Cadence Card<br/>Steps/min Display]
        C3 --> C8[Stability Card<br/>Balance Assessment]
        C3 --> C9[Gait Cycles<br/>Count & Quality]
        C3 --> C10[Movement Quality<br/>Consistency & Smoothness]
        
        C4 --> C11[Recommendations<br/>Clinical Suggestions]
        C4 --> C12[Asymmetry Details<br/>Joint-specific Analysis]
        C4 --> C13[Sequence Info<br/>Technical Metadata]
    end
    
    subgraph "Interactive Features"
        C1 --> D1[Video Player<br/>Frame Navigation]
        D1 --> D2[Pose Overlay<br/>Keypoint Visualization]
        D2 --> D3[Timeline Scrubbing<br/>Frame-by-frame]
        
        C1 --> D4[Tooltip System<br/>Contextual Help]
        D4 --> D5[Clinical Explanations<br/>Educational Content]
        D5 --> D6[Normal Ranges<br/>Reference Values]
        
        C1 --> D7[Error Handling<br/>Graceful Degradation]
        D7 --> D8[Retry Options<br/>User Recovery]
        D8 --> D9[Status Messages<br/>Clear Feedback]
    end
    
    style B2 fill:#e1f5ff
    style C1 fill:#fff4e1
    style D4 fill:#e8f5e9
```

## Analysis Processing Pipeline

```mermaid
sequenceDiagram
    participant User
    participant Frontend as React Frontend
    participant API as FastAPI Server
    participant Service as PoseAnalysisService
    participant Analyzer as EnhancedGaitAnalyzer
    participant DB as Database
    participant Cache as File Cache
    
    User->>Frontend: Select sequence
    Frontend->>Frontend: Show loading state
    Frontend->>API: GET /api/pose-analysis/{datasetId}/{sequenceId}
    
    API->>Service: get_sequence_analysis()
    Service->>DB: Check database cache
    
    alt Cache Hit (Database)
        DB-->>Service: Return cached results
        Service-->>API: Cached analysis
    else Cache Miss
        Service->>Service: Load pose sequence from GAVD
        Service->>Analyzer: analyze_gait_sequence()
        
        par Feature Extraction
            Analyzer->>Analyzer: Extract 60+ features
        and Temporal Analysis
            Analyzer->>Analyzer: Detect gait cycles
            Analyzer->>Analyzer: Calculate timing parameters
        and Symmetry Analysis
            Analyzer->>Analyzer: Bilateral comparison
            Analyzer->>Analyzer: Asymmetry detection
        end
        
        Analyzer->>Analyzer: Generate clinical summary
        Analyzer-->>Service: Complete analysis results
        
        Service->>DB: Store in database
        Service->>Cache: Save to file cache
        Service-->>API: Fresh analysis results
    end
    
    API-->>Frontend: JSON analysis data
    Frontend->>Frontend: Render analysis dashboard
    Frontend->>Frontend: Update UI components
    Frontend-->>User: Display results with tooltips
    
    Note over Frontend,User: Real-time updates with<br/>loading states and progress
```

## Feature Extraction Flow

```mermaid
graph TD
    subgraph "Input Processing"
        A[Pose Sequence<br/>Keypoint Data] --> B[Data Validation<br/>Quality Check]
        B --> C[Keypoint Filtering<br/>Confidence Threshold]
        C --> D[Smoothing<br/>Noise Reduction]
    end
    
    subgraph "Kinematic Analysis"
        D --> E[Velocity Calculation<br/>Frame-to-frame]
        E --> F[Acceleration Analysis<br/>Second Derivative]
        F --> G[Jerk Computation<br/>Third Derivative]
        
        E --> H[Movement Smoothness<br/>Consistency Metrics]
        F --> I[Coordination Quality<br/>Pattern Analysis]
        G --> J[Motor Control<br/>Stability Assessment]
    end
    
    subgraph "Joint Analysis"
        D --> K[Joint Angle Calculation<br/>3-point Angles]
        K --> L[Range of Motion<br/>Min/Max/Mean]
        L --> M[Joint Patterns<br/>Gait Phase Analysis]
        
        K --> N[Bilateral Comparison<br/>Left vs Right]
        N --> O[Symmetry Indices<br/>Joint-specific]
        O --> P[Asymmetry Detection<br/>Threshold Analysis]
    end
    
    subgraph "Spatial Analysis"
        D --> Q[Step Length Calculation<br/>Ankle Distances]
        Q --> R[Stride Characteristics<br/>Spatial Patterns]
        R --> S[Step Width Analysis<br/>Balance Indicators]
        
        D --> T[Center of Mass<br/>Hip Midpoint]
        T --> U[Stability Metrics<br/>COM Movement]
        U --> V[Postural Control<br/>Sway Analysis]
    end
    
    subgraph "Feature Integration"
        H --> W[Clinical Assessment<br/>Feature Synthesis]
        I --> W
        J --> W
        M --> W
        P --> W
        S --> W
        V --> W
        
        W --> X[Overall Quality Score<br/>Good/Moderate/Poor]
        W --> Y[Key Metrics<br/>Primary Indicators]
        W --> Z[Detailed Features<br/>60+ Parameters]
    end
    
    subgraph "UI Display"
        X --> AA[Assessment Cards<br/>Color-coded Status]
        Y --> BB[Metric Displays<br/>Interactive Elements]
        Z --> CC[Detailed Panels<br/>Expandable Views]
        
        AA --> DD[Overall Assessment<br/>Primary Display]
        BB --> EE[Key Metrics Grid<br/>4 Main Cards]
        CC --> FF[Feature Tables<br/>Technical Details]
    end
    
    style W fill:#e1f5ff
    style AA fill:#fff4e1
    style EE fill:#e8f5e9
```

## Temporal Analysis Workflow

```mermaid
graph TB
    subgraph "Gait Event Detection"
        A[Pose Sequence] --> B[Ankle Trajectory<br/>Extraction]
        B --> C[Vertical Position<br/>Analysis]
        C --> D[Velocity Calculation<br/>Movement Speed]
        
        D --> E{Detection Method}
        E -->|Heel Strike| F[Local Minima<br/>Detection]
        E -->|Toe Off| G[Acceleration Peaks<br/>Push-off Events]
        E -->|Combined| H[Multi-method<br/>Validation]
        
        F --> I[Heel Strike Events<br/>Frame Numbers]
        G --> J[Toe Off Events<br/>Frame Numbers]
        H --> K[Validated Events<br/>High Confidence]
    end
    
    subgraph "Cycle Segmentation"
        I --> L[Cycle Boundaries<br/>Strike to Strike]
        J --> L
        K --> L
        
        L --> M[Cycle Validation<br/>Duration Checks]
        M --> N[Quality Assessment<br/>Confidence Scoring]
        N --> O[Gait Cycles<br/>Segmented Data]
    end
    
    subgraph "Temporal Parameter Extraction"
        O --> P[Cadence Calculation<br/>Steps per Minute]
        O --> Q[Cycle Duration<br/>Mean & Variability]
        O --> R[Phase Analysis<br/>Stance/Swing/Double Support]
        
        P --> S[Rhythm Assessment<br/>Regularity Analysis]
        Q --> T[Timing Consistency<br/>Coefficient of Variation]
        R --> U[Phase Distribution<br/>Percentage Breakdown]
    end
    
    subgraph "Clinical Assessment"
        S --> V[Cadence Classification<br/>Normal/Slow/Fast]
        T --> W[Regularity Scoring<br/>Good/Moderate/Poor]
        U --> X[Phase Balance<br/>Normal Ranges]
        
        V --> Y[Clinical Interpretation<br/>Temporal Quality]
        W --> Y
        X --> Y
    end
    
    subgraph "UI Integration"
        Y --> Z[Cadence Card<br/>Primary Display]
        Y --> AA[Gait Cycles Card<br/>Count & Quality]
        Y --> BB[Temporal Metrics<br/>Detailed View]
        
        Z --> CC[Large Numeric<br/>Steps/min]
        Z --> DD[Classification Badge<br/>Color-coded]
        Z --> EE[Normal Range<br/>Reference Display]
        
        AA --> FF[Cycle Count<br/>Detection Results]
        AA --> GG[Average Duration<br/>Timing Display]
        AA --> HH[Quality Indicator<br/>Reliability Score]
    end
    
    style Y fill:#e1f5ff
    style Z fill:#fff4e1
    style AA fill:#e8f5e9
```

## Error Handling and Quality Control

```mermaid
graph LR
    subgraph "Input Validation"
        A[Pose Sequence] --> B{Sufficient Data?}
        B -->|No| C[Error: Insufficient Data<br/>< 10 frames]
        B -->|Yes| D{Keypoint Quality?}
        D -->|Low| E[Warning: Low Confidence<br/>< 0.3 average]
        D -->|Good| F[Proceed with Analysis]
    end
    
    subgraph "Analysis Quality Control"
        F --> G[Feature Extraction]
        G --> H{Features Valid?}
        H -->|Invalid| I[Error: Feature Extraction Failed<br/>NaN or Inf values]
        H -->|Valid| J[Temporal Analysis]
        
        J --> K{Cycles Detected?}
        K -->|< 2 cycles| L[Warning: Insufficient Cycles<br/>Limited reliability]
        K -->|≥ 2 cycles| M[Symmetry Analysis]
        
        M --> N{Bilateral Data?}
        N -->|Missing| O[Warning: Limited Symmetry<br/>Incomplete bilateral data]
        N -->|Complete| P[Generate Assessment]
    end
    
    subgraph "UI Error Display"
        C --> Q[Error Alert<br/>Red notification]
        E --> R[Warning Banner<br/>Yellow notification]
        I --> Q
        L --> R
        O --> R
        
        Q --> S[Error Message<br/>User-friendly explanation]
        Q --> T[Retry Button<br/>User action]
        Q --> U[Help Link<br/>Troubleshooting guide]
        
        R --> V[Warning Details<br/>Impact explanation]
        R --> W[Continue Option<br/>Proceed with limitations]
        R --> X[Quality Score<br/>Reliability indicator]
    end
    
    subgraph "Graceful Degradation"
        P --> Y[Full Analysis<br/>All components]
        L --> Z[Partial Analysis<br/>Limited temporal data]
        O --> AA[Reduced Analysis<br/>No symmetry assessment]
        
        Y --> BB[Complete Dashboard<br/>All cards active]
        Z --> CC[Modified Dashboard<br/>Temporal warnings]
        AA --> DD[Basic Dashboard<br/>Symmetry disabled]
    end
    
    style Q fill:#ffebee
    style R fill:#fff3e0
    style BB fill:#e8f5e9
```