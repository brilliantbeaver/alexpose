# AlexPose - Tech Stack

## Backend (Python 3.12+)

- **Package Manager**: uv (preferred over pip)
- **Web Framework**: FastAPI with uvicorn
- **Database**: SQLite (dev) / PostgreSQL (prod) via SQLAlchemy
- **AI/ML**: OpenAI GPT, Google Gemini, MediaPipe
- **Video Processing**: OpenCV, FFmpeg, yt-dlp
- **Data**: NumPy, Pandas, Matplotlib
- **Logging**: Loguru with structured logging
- **Config**: YAML-based with environment overrides

## Frontend (Node.js 18+)

- **Framework**: Next.js 16 with React 19
- **Language**: TypeScript
- **UI Components**: Radix UI primitives, Shadcn UI
- **Styling**: Tailwind CSS 4
- **Charts**: Recharts
- **Testing**: Jest with Testing Library

## Code Quality

- **Formatting**: Black (line-length 88), isort
- **Linting**: Flake8, ESLint
- **Type Checking**: mypy (strict mode)
- **Pre-commit**: Hooks configured

## Common Commands

### Python/Backend
```bash
# Install dependencies
uv sync

# Run dev server
uvicorn server.main:app --reload

# Run tests
pytest                    # All tests
pytest -m "not slow"      # Fast tests only
pytest --cov=ambient      # With coverage

# Code quality
make format               # Format code
make lint                 # Run linters
```

### Frontend
```bash
cd frontend
npm install               # Install deps
npm run dev               # Dev server (port 3000)
npm run build             # Production build
npm test                  # Run tests
npm run lint              # ESLint
```

### Full Stack Development
```bash
# Start both servers (from project root)
./scripts/start-dev.ps1  # Windows
./scripts/start-dev.sh   # Linux/macOS
```

## Environment Variables

Required in `.env`:
```
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AI...
```

## Test Markers

- `@pytest.mark.slow` - Resource-intensive tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.property` - Hypothesis property tests
- `@pytest.mark.unit` - Unit tests
