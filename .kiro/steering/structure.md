# AlexPose - Project Structure

```
alexpose/
├── ambient/                    # Core Python package
│   ├── analysis/              # Gait analysis components
│   │   ├── feature_extractor.py   # 60+ gait features
│   │   ├── gait_analyzer.py       # Main analyzer + EnhancedGaitAnalyzer
│   │   ├── temporal_analyzer.py   # Gait cycle detection
│   │   └── symmetry_analyzer.py   # Left-right symmetry
│   ├── classification/        # LLM-based classification
│   │   ├── llm_classifier.py      # Multi-model LLM support
│   │   └── prompt_manager.py      # YAML prompt templates
│   ├── cli/                   # Command-line interface
│   ├── core/                  # Interfaces, data models, config
│   ├── data/                  # Data management, augmentation
│   ├── gavd/                  # GAVD dataset processing
│   ├── pose/                  # Pose estimation backends
│   │   ├── factory.py             # Estimator factory
│   │   ├── ultralytics_estimator.py
│   │   └── alphapose_estimator.py
│   ├── storage/               # SQLite storage, backups
│   ├── utils/                 # Logging, video utils, CSV parser
│   └── video/                 # Video processing, YouTube handler
│
├── server/                    # FastAPI backend
│   ├── main.py               # App entry point
│   ├── routers/              # API route handlers
│   │   ├── analysis.py           # /api/analysis/*
│   │   ├── gavd.py               # /api/gavd/*
│   │   ├── models.py             # /api/models/*
│   │   ├── pose_analysis.py      # /api/pose-analysis/*
│   │   ├── upload.py             # /api/upload/*
│   │   └── video.py              # /api/video/*
│   ├── services/             # Business logic layer
│   └── middleware/           # Auth, CORS, logging
│
├── frontend/                  # Next.js frontend
│   ├── app/                  # App router pages
│   │   ├── analyze/              # Video analysis page
│   │   ├── dashboard/            # Main dashboard
│   │   ├── gavd/                 # GAVD dataset browser
│   │   ├── models/               # Model management
│   │   └── results/              # Analysis results
│   ├── applib/               # Shared utilities (api-client, types)
│   ├── components/           # React components
│   │   ├── ui/                   # Shadcn UI components
│   │   ├── navigation/           # Nav components
│   │   └── pose-analysis/        # Pose visualization
│   └── hooks/                # Custom React hooks
│
├── config/                    # Configuration files
│   ├── alexpose.yaml             # Main config
│   ├── llm_models.yaml           # LLM model specs
│   ├── classification_prompts.yaml
│   ├── development.yaml          # Dev overrides
│   └── production.yaml           # Prod overrides
│
├── data/                      # Data storage
│   ├── gavd/                     # GAVD dataset by condition
│   ├── models/                   # Pose model files
│   ├── storage/                  # SQLite database
│   └── videos/                   # Uploaded videos
│
├── tests/                     # Test suite
│   ├── ambient/                  # Core library tests
│   ├── api/                      # API endpoint tests
│   ├── integration/              # Integration tests
│   └── property/                 # Hypothesis property tests
│
├── docs/                      # Documentation
│   ├── analysis/                 # Analysis component docs
│   ├── architecture/             # System design
│   ├── guides/                   # User guides
│   └── specs/                    # Requirements, design specs
│
├── scripts/                   # Dev/CI scripts
├── examples/                  # Usage examples
└── logs/                      # Application logs
```

## Key Conventions

- Backend services in `server/services/` wrap `ambient/` library calls
- API routes follow REST patterns: `/api/{resource}/{action}`
- Frontend uses `applib/` for shared code (migrated from `lib/`)
- Config uses YAML with environment-specific overrides
- Tests mirror source structure in `tests/`
