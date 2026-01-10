# AlexPose Frontend

Modern NextJS web frontend for the AlexPose Gait Analysis System using Shadcn UI components with sophisticated navigation and design language.

## Features

- ✨ Modern UI with Shadcn components
- 🎨 Glass morphism design with subtle shadows and hover effects
- 🧭 Intuitive top navigation with clear context switching
- 📱 Fully responsive (desktop, tablet, mobile)
- 🌙 Dark/light theme support
- 📤 Drag-and-drop video upload
- 🔗 YouTube URL analysis
- 📊 Interactive results dashboard
- 💡 Comprehensive tooltips and help system

## Known Issues & Solutions

### Hydration Warning (Resolved)

**Issue**: Console warning about hydration mismatch with Radix UI components.

**Root Cause**: Radix UI generates random IDs that differ between server and client renders.

**Solution**: Added `suppressHydrationWarning` to affected components. This is safe and doesn't affect functionality.

**Details**: See [HYDRATION_FIX.md](./HYDRATION_FIX.md) for complete analysis and solution.

## Tech Stack

- **Framework**: Next.js 15 with App Router
- **Language**: TypeScript
- **Styling**: Tailwind CSS v4
- **Components**: Shadcn UI
- **Icons**: Emoji-based for simplicity

## Getting Started

### Prerequisites

- Node.js 18+ 
- npm or yarn

### Installation

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build

# Start production server
npm start
```

### Development

The development server will start at [http://localhost:3000](http://localhost:3000)

## Project Structure

```
frontend/
├── app/                    # Next.js app directory
│   ├── analyze/           # Analysis pages
│   │   ├── upload/        # Video upload
│   │   └── youtube/       # YouTube URL input
│   ├── dashboard/         # Dashboard page
│   ├── results/           # Results pages
│   ├── layout.tsx         # Root layout with navigation
│   └── page.tsx           # Homepage
├── components/            # React components
│   ├── navigation/        # Navigation components
│   │   ├── TopNavBar.tsx
│   │   ├── NavigationMenu.tsx
│   │   └── Breadcrumbs.tsx
│   └── ui/                # Shadcn UI components
├── hooks/                 # Custom React hooks
│   └── useNavigation.ts
├── applib/                # Utilities and config
│   ├── api-client.ts
│   ├── index.ts
│   ├── navigation-config.ts
│   ├── navigation-types.ts
│   ├── pose-analysis-tooltips.ts
│   ├── pose-analysis-types.ts
│   └── utils.ts
└── public/                # Static assets
```

## Navigation Structure

```
AlexPose Logo | Dashboard | Analyze | Results | Models | Help | Profile
                            ↓         ↓        ↓       ↓     ↓
                     [Upload Video] [History] [Browse] [Docs] [Settings]
                     [YouTube URL]  [Compare] [Train]  [FAQ]  [Account]
                     [Live Camera]  [Export]  [Deploy] [API]  [Logout]
                     [Batch Process]
```

## Key Pages

### Homepage (`/`)
- Hero section with call-to-action buttons
- Feature cards highlighting key capabilities
- Quick stats and metrics

### Dashboard (`/dashboard`)
- Overview of recent analyses
- Quick action cards
- System status

### Upload Video (`/analyze/upload`)
- Drag-and-drop interface
- File validation
- Upload progress tracking
- Supported formats: MP4, AVI, MOV, WebM

### YouTube Analysis (`/analyze/youtube`)
- URL input with validation
- Video preview
- Real-time validation feedback

### Results List (`/results`)
- Tabbed interface (All, Normal, Abnormal)
- Result cards with status badges
- Export functionality
- Detailed view links

### Result Detail (`/results/[id]`) ✨ NEW
- Comprehensive analysis view
- 5 detailed tabs:
  - **Gait Metrics**: 6 key measurements with normal ranges
  - **Temporal Analysis**: Gait cycles, stance/swing phases
  - **Spatial Analysis**: Step lengths, variability
  - **AI Analysis**: Classification, reasoning, recommendations
  - **Video**: Full-featured video player with frame-by-frame controls ✨ NEW
- Status indicators and confidence scores
- Actionable recommendations
- Export and comparison options

#### Video Player Features ✨ NEW
- Play/Pause, Seek, Volume controls
- Frame-by-frame navigation (Previous/Next frame)
- Playback speed control (0.25x to 2x)
- Skip forward/backward (5 seconds)
- Frame counter overlay
- Fullscreen mode
- Loading states and error handling
- Keyboard shortcuts support

See [RESULT_DETAIL_PAGE.md](./RESULT_DETAIL_PAGE.md) for detailed documentation.  
See [VIDEO_PLAYER_DOCUMENTATION.md](./components/video/VIDEO_PLAYER_DOCUMENTATION.md) for video player details.

## API Integration

The frontend connects to the FastAPI backend at:
- Development: `http://localhost:8000`
- Production: Configure via environment variables

### Environment Variables

Create a `.env.local` file:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_API_VERSION=v1
```

## Styling

### Theme

The app uses a neutral color scheme with:
- Primary: Blue (600)
- Secondary: Purple (600)
- Accent: Gradient from blue to purple
- Glass morphism effects with backdrop blur

### Responsive Breakpoints

- Mobile: < 768px
- Tablet: 768px - 1024px
- Desktop: > 1024px

## Testing

```bash
# Run tests (when implemented)
npm test

# Run E2E tests
npm run test:e2e

# Run linting
npm run lint
```

## Deployment

### Vercel (Recommended)

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel
```

### Docker

```bash
# Build image
docker build -t alexpose-frontend .

# Run container
docker run -p 3000:3000 alexpose-frontend
```

## Contributing

1. Follow the existing code style
2. Use TypeScript for type safety
3. Add comments for complex logic
4. Test on multiple screen sizes
5. Ensure accessibility compliance

## License

MIT License - see LICENSE file for details

## Support

For issues and questions:
- GitHub Issues: [github.com/alexpose/alexpose](https://github.com/alexpose/alexpose)
- Documentation: [docs.alexpose.com](https://docs.alexpose.com)
- Email: support@alexpose.com
