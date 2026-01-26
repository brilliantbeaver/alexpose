# Frontend Troubleshooting Guide - Realtime Feature

## Issue: Missing Radix UI Dependencies

### Problem
The frontend build was failing with the following error:
```
Module not found: Can't resolve '@radix-ui/react-label'
Module not found: Can't resolve '@radix-ui/react-switch'
```

### Root Cause
The Shadcn UI components (`label.tsx` and `switch.tsx`) were added to the project but their required Radix UI peer dependencies were not installed in `package.json`.

### Solution
Install the missing Radix UI packages:

```bash
npm install --prefix frontend @radix-ui/react-label @radix-ui/react-switch
```

### Verification
After installation, verify the packages are in `package.json`:

```json
{
  "dependencies": {
    "@radix-ui/react-label": "^2.1.8",
    "@radix-ui/react-switch": "^1.2.6",
    // ... other dependencies
  }
}
```

## Common Frontend Issues

### 1. Missing Radix UI Packages

**Symptoms:**
- Build errors: `Module not found: Can't resolve '@radix-ui/react-*'`
- Components fail to render

**Solution:**
Check which Radix UI components are used and install missing packages:

```bash
# Common Radix UI packages for Shadcn UI
npm install --prefix frontend \
  @radix-ui/react-label \
  @radix-ui/react-switch \
  @radix-ui/react-dialog \
  @radix-ui/react-dropdown-menu \
  @radix-ui/react-select \
  @radix-ui/react-slider \
  @radix-ui/react-separator \
  @radix-ui/react-progress \
  @radix-ui/react-tabs \
  @radix-ui/react-tooltip \
  @radix-ui/react-slot
```

### 2. TypeScript Errors

**Symptoms:**
- Type errors in IDE
- Build fails with type checking errors

**Solution:**
1. Ensure all type definitions are installed:
```bash
npm install --prefix frontend --save-dev \
  @types/node \
  @types/react \
  @types/react-dom
```

2. Check `tsconfig.json` is properly configured:
```json
{
  "compilerOptions": {
    "target": "ES2017",
    "lib": ["dom", "dom.iterable", "esnext"],
    "allowJs": true,
    "skipLibCheck": true,
    "strict": true,
    "forceConsistentCasingInFileNames": true,
    "noEmit": true,
    "esModuleInterop": true,
    "module": "esnext",
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "jsx": "preserve",
    "incremental": true,
    "plugins": [
      {
        "name": "next"
      }
    ],
    "paths": {
      "@/*": ["./*"]
    }
  }
}
```

### 3. WebSocket Connection Issues

**Symptoms:**
- "Failed to connect to WebSocket" errors
- Connection drops frequently
- No pose data received

**Solution:**

1. **Check backend is running:**
```bash
# Verify FastAPI server is running
curl http://localhost:8000/api/realtime/health
```

2. **Check WebSocket URL:**
```typescript
// In useRealtimeAnalysis.ts
const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const host = window.location.host;
const wsUrl = `${protocol}//${host}/api/realtime/stream`;
```

3. **Check CORS settings:**
```python
# In server/main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 4. Camera Permission Issues

**Symptoms:**
- "Camera access denied" message
- Black screen instead of camera feed
- Permission prompt doesn't appear

**Solution:**

1. **Check browser permissions:**
   - Chrome: Settings → Privacy and security → Site Settings → Camera
   - Firefox: Preferences → Privacy & Security → Permissions → Camera
   - Safari: Preferences → Websites → Camera

2. **Ensure HTTPS or localhost:**
   - WebRTC requires HTTPS or localhost
   - Use `http://localhost:3000` for development

3. **Check camera availability:**
```javascript
navigator.mediaDevices.enumerateDevices()
  .then(devices => {
    const cameras = devices.filter(d => d.kind === 'videoinput');
    console.log('Available cameras:', cameras);
  });
```

### 5. Performance Issues

**Symptoms:**
- Laggy camera feed
- High CPU usage
- Frame drops

**Solution:**

1. **Reduce processing quality:**
   - Switch to "Fast" mode
   - Lower target FPS
   - Reduce buffer size

2. **Optimize frame capture:**
```typescript
// In RealtimeCamera.tsx
const targetFPS = processing_mode === 'fast' ? 30 : 
                  processing_mode === 'accurate' ? 15 : 20;
const frameInterval = 1000 / targetFPS;
```

3. **Check browser performance:**
   - Close other tabs
   - Disable browser extensions
   - Use hardware acceleration

### 6. Build Errors

**Symptoms:**
- `npm run build` fails
- Deployment errors

**Solution:**

1. **Clear build cache:**
```bash
rm -rf frontend/.next
rm -rf frontend/node_modules
npm install --prefix frontend
npm run build --prefix frontend
```

2. **Check for ESLint errors:**
```bash
npm run lint --prefix frontend
```

3. **Verify all imports:**
```bash
# Check for unused imports
npx eslint frontend --ext .ts,.tsx
```

### 7. Runtime Errors

**Symptoms:**
- White screen
- "Hydration failed" errors
- Component crashes

**Solution:**

1. **Check browser console:**
   - Open DevTools (F12)
   - Look for error messages
   - Check Network tab for failed requests

2. **Verify component props:**
```typescript
// Ensure all required props are provided
<RealtimeCamera
  isActive={isConnected && isProcessing}
  currentPose={currentPose}
  onFrame={sendFrame}
  config={config}
  onPermissionChange={setCameraPermission}
/>
```

3. **Check for null/undefined:**
```typescript
// Add null checks
if (!currentPose) {
  return <div>Loading...</div>;
}
```

## Debugging Tips

### 1. Enable Verbose Logging

```typescript
// In useRealtimeAnalysis.ts
ws.onmessage = (event) => {
  console.log('WebSocket message:', event.data);
  const message = JSON.parse(event.data);
  handleWebSocketMessage(message);
};
```

### 2. Monitor Performance

```typescript
// In RealtimeCamera.tsx
const captureAndSendFrame = () => {
  const startTime = performance.now();
  
  // ... capture logic ...
  
  const endTime = performance.now();
  console.log(`Frame capture took ${endTime - startTime}ms`);
};
```

### 3. Check Network Traffic

1. Open DevTools → Network tab
2. Filter by "WS" (WebSocket)
3. Click on WebSocket connection
4. View Messages tab to see data flow

### 4. Test Components Individually

```typescript
// Create a test page
export default function TestPage() {
  return (
    <div>
      <RealtimeCamera
        isActive={true}
        currentPose={null}
        onFrame={(data) => console.log('Frame:', data)}
        config={{}}
      />
    </div>
  );
}
```

## Prevention

### 1. Use Dependency Checklist

When adding new Shadcn UI components:
- [ ] Check component file for Radix UI imports
- [ ] Install required Radix UI packages
- [ ] Test build locally
- [ ] Verify in browser

### 2. Automated Checks

Add to `package.json`:
```json
{
  "scripts": {
    "check-deps": "npm ls @radix-ui/react-*",
    "prebuild": "npm run check-deps"
  }
}
```

### 3. Documentation

Keep track of required dependencies:
```markdown
# Required Radix UI Packages

- @radix-ui/react-label (for Label component)
- @radix-ui/react-switch (for Switch component)
- @radix-ui/react-slider (for Slider component)
- @radix-ui/react-select (for Select component)
```

## Getting Help

If issues persist:

1. **Check logs:**
   - Browser console (F12)
   - Backend logs (`logs/alexpose_*.log`)
   - Network tab in DevTools

2. **Verify versions:**
```bash
node --version  # Should be 18+
npm --version   # Should be 9+
```

3. **Clean install:**
```bash
rm -rf frontend/node_modules frontend/.next
npm install --prefix frontend
npm run dev --prefix frontend
```

4. **Check documentation:**
   - [Realtime README](README.md)
   - [API Reference](api-reference.md)
   - [Architecture](architecture.md)

## Related Issues

- [Next.js Module Resolution](https://nextjs.org/docs/messages/module-not-found)
- [Radix UI Documentation](https://www.radix-ui.com/docs/primitives/overview/introduction)
- [Shadcn UI Components](https://ui.shadcn.com/docs/components)
- [WebRTC Troubleshooting](https://developer.mozilla.org/en-US/docs/Web/API/WebRTC_API)
