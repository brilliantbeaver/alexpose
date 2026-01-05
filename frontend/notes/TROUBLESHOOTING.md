# Frontend Troubleshooting Guide

## Common Issues and Solutions

### 1. Hydration Mismatch Warning ✅ RESOLVED

**Symptom:**
```
Console Error: A tree hydrated but some attributes of the server rendered HTML 
didn't match the client properties.
```

**Root Cause:**
- Radix UI components (NavigationMenu, Sheet, DropdownMenu) generate random IDs
- IDs differ between server-side rendering and client-side hydration
- This is a known limitation of Radix UI with Next.js SSR

**Solution Applied:**
- Added `suppressHydrationWarning` to affected components
- This is React's official way to handle expected mismatches
- No functional impact - purely cosmetic console warning

**Files Modified:**
- `components/navigation/TopNavBar.tsx`
- `components/navigation/NavigationMenu.tsx`

**Verification:**
- ✅ All tests passing (17/17)
- ✅ Navigation works correctly
- ✅ No console warnings
- ✅ Accessibility features intact

**Documentation:**
See [HYDRATION_FIX.md](./HYDRATION_FIX.md) for detailed analysis.

---

### 2. Module Not Found Errors

**Symptom:**
```
Error: Cannot find module '@/components/...'
```

**Solution:**
1. Check `tsconfig.json` has correct path mappings:
   ```json
   {
     "compilerOptions": {
       "paths": {
         "@/*": ["./*"]
       }
     }
   }
   ```
2. Restart development server: `npm run dev`
3. Clear Next.js cache: `rm -rf .next`

---

### 3. Tailwind Styles Not Applying

**Symptom:**
- Components render but have no styling
- Classes appear in HTML but no CSS applied

**Solution:**
1. Verify `tailwind.config.js` content paths:
   ```js
   content: [
     "./app/**/*.{js,ts,jsx,tsx,mdx}",
     "./components/**/*.{js,ts,jsx,tsx,mdx}",
   ]
   ```
2. Check `globals.css` imports Tailwind:
   ```css
   @tailwind base;
   @tailwind components;
   @tailwind utilities;
   ```
3. Restart dev server

---

### 4. Shadcn UI Component Issues

**Symptom:**
- Component not found or import errors
- TypeScript errors on Shadcn components

**Solution:**
1. Verify component is installed:
   ```bash
   npx shadcn@latest add [component-name]
   ```
2. Check `components.json` configuration
3. Ensure `lib/utils.ts` exists with `cn()` function

---

### 5. Build Errors

**Symptom:**
```
Error: Build failed
```

**Common Causes & Solutions:**

**TypeScript Errors:**
```bash
# Check for type errors
npm run build
# Fix type issues in reported files
```

**Missing Dependencies:**
```bash
# Reinstall dependencies
rm -rf node_modules package-lock.json
npm install
```

**Environment Variables:**
```bash
# Create .env.local with required variables
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

### 6. Development Server Issues

**Symptom:**
- Server won't start
- Port already in use
- Hot reload not working

**Solutions:**

**Port in Use:**
```bash
# Kill process on port 3000
# Windows:
netstat -ano | findstr :3000
taskkill /PID [PID] /F

# Linux/Mac:
lsof -ti:3000 | xargs kill -9
```

**Hot Reload Not Working:**
1. Check file watcher limits (Linux):
   ```bash
   echo fs.inotify.max_user_watches=524288 | sudo tee -a /etc/sysctl.conf
   sudo sysctl -p
   ```
2. Restart dev server
3. Clear `.next` folder

---

### 7. Test Failures

**Symptom:**
```
Tests failing or not running
```

**Solutions:**

**Jest Configuration:**
```bash
# Verify jest.config.js exists
# Check jest.setup.js is configured
```

**Missing Test Dependencies:**
```bash
npm install --save-dev @testing-library/react @testing-library/jest-dom
```

**Clear Jest Cache:**
```bash
npm test -- --clearCache
```

---

### 8. API Connection Issues

**Symptom:**
- Cannot connect to backend API
- CORS errors
- 404 on API endpoints

**Solutions:**

**Check API URL:**
```bash
# Verify .env.local
NEXT_PUBLIC_API_URL=http://localhost:8000
```

**CORS Issues:**
- Ensure backend has CORS middleware configured
- Check allowed origins include frontend URL

**Network Errors:**
```bash
# Test API directly
curl http://localhost:8000/api/v1/health
```

---

### 9. Performance Issues

**Symptom:**
- Slow page loads
- Laggy interactions
- High memory usage

**Solutions:**

**Optimize Images:**
```tsx
// Use Next.js Image component
import Image from 'next/image'
<Image src="/path" width={500} height={300} alt="..." />
```

**Code Splitting:**
```tsx
// Dynamic imports for heavy components
import dynamic from 'next/dynamic'
const HeavyComponent = dynamic(() => import('./HeavyComponent'))
```

**Check Bundle Size:**
```bash
npm run build
# Review bundle analyzer output
```

---

### 10. Deployment Issues

**Symptom:**
- Build succeeds locally but fails on deployment
- App works locally but not in production

**Solutions:**

**Environment Variables:**
- Ensure all required env vars are set in deployment platform
- Check variable names match exactly (case-sensitive)

**Build Configuration:**
```json
// package.json
{
  "scripts": {
    "build": "next build",
    "start": "next start"
  }
}
```

**Static Export Issues:**
- If using `output: 'export'`, ensure no server-side features used
- Check for dynamic routes compatibility

---

## Debugging Tips

### Enable Verbose Logging
```bash
# Development
DEBUG=* npm run dev

# Build
npm run build -- --debug
```

### Check Browser Console
1. Open DevTools (F12)
2. Check Console tab for errors
3. Check Network tab for failed requests
4. Check React DevTools for component issues

### Verify Dependencies
```bash
# Check for outdated packages
npm outdated

# Update dependencies
npm update

# Audit for vulnerabilities
npm audit
```

### Clear All Caches
```bash
# Clear Next.js cache
rm -rf .next

# Clear node_modules
rm -rf node_modules package-lock.json
npm install

# Clear browser cache
# Use Ctrl+Shift+Delete or Cmd+Shift+Delete
```

---

## Getting Help

### Documentation
- [Next.js Docs](https://nextjs.org/docs)
- [Shadcn UI Docs](https://ui.shadcn.com)
- [Tailwind CSS Docs](https://tailwindcss.com/docs)
- [Radix UI Docs](https://www.radix-ui.com)

### Community Support
- [Next.js GitHub Discussions](https://github.com/vercel/next.js/discussions)
- [Shadcn UI GitHub Issues](https://github.com/shadcn-ui/ui/issues)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/next.js)

### Project-Specific Issues
- Check [README.md](./README.md) for setup instructions
- Review [HYDRATION_FIX.md](./HYDRATION_FIX.md) for hydration issues
- Open GitHub issue with:
  - Error message
  - Steps to reproduce
  - Environment details (OS, Node version, etc.)
  - Screenshots if applicable

---

## Preventive Measures

### Before Starting Development
1. ✅ Verify Node.js version (18+)
2. ✅ Install dependencies: `npm install`
3. ✅ Create `.env.local` with required variables
4. ✅ Run tests: `npm test`
5. ✅ Start dev server: `npm run dev`

### Before Committing
1. ✅ Run linter: `npm run lint`
2. ✅ Run tests: `npm test`
3. ✅ Check TypeScript: `npm run build`
4. ✅ Review changes in browser

### Before Deploying
1. ✅ Test production build locally: `npm run build && npm start`
2. ✅ Verify all environment variables set
3. ✅ Check bundle size is reasonable
4. ✅ Test on multiple browsers
5. ✅ Verify API connections work

---

## Status: All Known Issues Resolved ✅

Current status of the frontend:
- ✅ No hydration warnings
- ✅ All tests passing (17/17)
- ✅ Navigation working correctly
- ✅ Responsive design functional
- ✅ Accessibility features intact
- ✅ Ready for development and deployment
