# GAVD Frontend Consolidation - Refactoring Plan

**Date:** January 4, 2026  
**Goal:** Consolidate duplicate GAVD sequence viewing functionality  
**Principles:** DRY, SOLID, YAGNI, Modularity, Robustness, Extensibility

## Problem Analysis

### Current Architecture Issues

**Duplicate Pages:**
1. `/gavd/[dataset_id]/page.tsx` - Full dataset analysis with 4 tabs (Overview, Sequences, Visualization, Pose Analysis)
2. `/gavd/[dataset_id]/sequence/[sequence_id]/page.tsx` - Single sequence viewer (only Visualization)

**Problems:**
- ❌ Code duplication (frame loading, pose data loading, video player setup)
- ❌ Inconsistent UX (dashboard links to limited view, dataset page has full view)
- ❌ Missing functionality (sequence page lacks Pose Analysis, Overview, etc.)
- ❌ Maintenance burden (changes must be made in two places)
- ❌ Violates DRY principle

### User Flow Issues

**Current Flow:**
```
Dashboard → Click sequence → Limited viewer (only video)
Dataset page → Select sequence → Full viewer (4 tabs)
```

**Desired Flow:**
```
Dashboard → Click sequence → Full viewer (4 tabs)
Dataset page → Select sequence → Full viewer (4 tabs)
```

## Proposed Solution

### Architecture: Unified Sequence Viewer Component

**Key Principle:** Single source of truth for sequence viewing logic


### Refactoring Strategy

**Option 1: Redirect to Dataset Page (RECOMMENDED)**
- Simplest solution
- Leverage existing full-featured implementation
- Minimal code changes
- Consistent UX

**Option 2: Create Shared Component**
- Extract common logic into reusable component
- Both pages use same component
- More complex but more flexible

**Option 3: Standalone Sequence Page**
- Duplicate all 4 tabs to sequence page
- Most code duplication
- Not recommended

**DECISION: Option 1 - Redirect to Dataset Page**

## Implementation Plan

### Phase 1: Redirect Sequence Page to Dataset Page

**File:** `frontend/app/gavd/[dataset_id]/sequence/[sequence_id]/page.tsx`

**Changes:**
1. Redirect to dataset page with sequence pre-selected
2. Use URL query parameter to specify sequence
3. Minimal code, maximum reuse

**New URL Pattern:**
```
/gavd/[dataset_id]?sequence=[sequence_id]&tab=visualization
```

### Phase 2: Update Dataset Page to Handle Query Parameters

**File:** `frontend/app/training/gavd/[datasetId]/page.tsx`

**Changes:**
1. Read `sequence` query parameter on mount
2. Read `tab` query parameter on mount
3. Auto-select sequence if provided
4. Auto-switch to tab if provided


### Phase 3: Update Dashboard Links

**File:** `frontend/app/dashboard/page.tsx`

**Changes:**
1. Update `getAnalysisLink()` to link to dataset page (not sequence page)
2. Include sequence ID in query parameter if available

## Detailed Implementation

### Step 1: Modify Sequence Page to Redirect

**File:** `frontend/app/gavd/[dataset_id]/sequence/[sequence_id]/page.tsx`

**Replace entire file with:**
```typescript
'use client';

import { useEffect } from 'react';
import { useParams, useRouter } from 'next/navigation';

export default function GAVDSequencePage() {
  const params = useParams();
  const router = useRouter();
  const dataset_id = params.dataset_id as string;
  const sequence_id = params.sequence_id as string;

  useEffect(() => {
    // Redirect to dataset page with sequence pre-selected
    router.replace(`/training/gavd/${dataset_id}?sequence=${sequence_id}&tab=visualization`);
  }, [dataset_id, sequence_id, router]);

  return (
    <div className="flex items-center justify-center min-h-screen">
      <div className="text-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-gray-900 mx-auto"></div>
        <p className="mt-4 text-muted-foreground">Redirecting to sequence viewer...</p>
      </div>
    </div>
  );
}
```

**Benefits:**
- ✅ Minimal code (20 lines vs 400+)
- ✅ No duplication
- ✅ Automatic redirect
- ✅ Preserves URL structure


### Step 2: Update Dataset Page to Handle Query Parameters

**File:** `frontend/app/training/gavd/[datasetId]/page.tsx`

**Add at top of component:**
```typescript
import { useSearchParams } from 'next/navigation';

export default function GAVDAnalysisPage() {
  const params = useParams();
  const searchParams = useSearchParams();
  const datasetId = params.datasetId as string;
  
  // Get query parameters
  const sequenceParam = searchParams.get('sequence');
  const tabParam = searchParams.get('tab');
  
  // ... existing state ...
```

**Add useEffect to handle query parameters:**
```typescript
// Handle URL query parameters (sequence and tab)
useEffect(() => {
  if (sequenceParam && sequences.length > 0) {
    // Check if sequence exists in loaded sequences
    const sequenceExists = sequences.some(seq => seq.sequence_id === sequenceParam);
    if (sequenceExists && selectedSequence !== sequenceParam) {
      console.log(`[Query Param] Auto-selecting sequence: ${sequenceParam}`);
      setSelectedSequence(sequenceParam);
    }
  }
  
  if (tabParam && tabParam !== activeTab) {
    console.log(`[Query Param] Auto-switching to tab: ${tabParam}`);
    setActiveTab(tabParam);
  }
}, [sequenceParam, tabParam, sequences, selectedSequence, activeTab]);
```

**Benefits:**
- ✅ Seamless integration
- ✅ Preserves existing functionality
- ✅ Enables deep linking
- ✅ No breaking changes


### Step 3: Update Dashboard to Link Correctly

**File:** `frontend/app/dashboard/page.tsx`

**Current implementation:**
```typescript
const getAnalysisLink = (analysis: RecentAnalysis) => {
  if (analysis.type === 'gavd_dataset') {
    return `/gavd/${analysis.dataset_id}`;  // ❌ Links to old path
  } else {
    return `/results/${analysis.analysis_id}`;
  }
};
```

**Updated implementation:**
```typescript
const getAnalysisLink = (analysis: RecentAnalysis) => {
  if (analysis.type === 'gavd_dataset') {
    // Link to dataset page (training/gavd path)
    return `/training/gavd/${analysis.dataset_id}`;  // ✅ Correct path
  } else {
    return `/results/${analysis.analysis_id}`;
  }
};
```

**Note:** If we want to link to a specific sequence from dashboard:
```typescript
const getAnalysisLink = (analysis: RecentAnalysis) => {
  if (analysis.type === 'gavd_dataset') {
    // If we have a specific sequence to show, add it as query param
    const baseUrl = `/training/gavd/${analysis.dataset_id}`;
    // Could add: ?sequence=xxx&tab=visualization if we track last viewed sequence
    return baseUrl;
  } else {
    return `/results/${analysis.analysis_id}`;
  }
};
```

## Benefits of This Approach

### DRY (Don't Repeat Yourself)
- ✅ Single implementation of sequence viewing logic
- ✅ No code duplication
- ✅ Changes made once, apply everywhere

### SOLID Principles

**Single Responsibility:**
- ✅ Dataset page: Manages dataset-level view and sequence selection
- ✅ Sequence page: Simple redirect (single responsibility)

**Open/Closed:**
- ✅ Dataset page open for extension (can add more tabs)
- ✅ Closed for modification (existing functionality preserved)

**Liskov Substitution:**
- ✅ Both entry points lead to same viewer
- ✅ Consistent behavior regardless of entry point

**Interface Segregation:**
- ✅ Clean URL interface with query parameters
- ✅ No unnecessary dependencies

**Dependency Inversion:**
- ✅ Both pages depend on URL routing abstraction
- ✅ Not tightly coupled to specific implementations


### YAGNI (You Aren't Gonna Need It)
- ✅ No over-engineering
- ✅ Simple redirect solution
- ✅ Avoids premature abstraction
- ✅ Can refactor later if needs change

### Modularity
- ✅ Clear separation of concerns
- ✅ Dataset page handles all sequence viewing
- ✅ Sequence page handles routing only
- ✅ Easy to test and maintain

### Robustness
- ✅ Handles missing query parameters gracefully
- ✅ Validates sequence exists before selecting
- ✅ Preserves existing error handling
- ✅ No breaking changes to existing functionality

### Extensibility
- ✅ Easy to add more query parameters (e.g., frame number)
- ✅ Can add more tabs without changing redirect logic
- ✅ Future features automatically available from both entry points
- ✅ URL-based state enables bookmarking and sharing

## Testing Plan

### Test Case 1: Direct Dataset Access
1. Navigate to `/training/gavd/[dataset_id]`
2. ✅ Should show dataset page with all 4 tabs
3. ✅ Should default to first sequence
4. ✅ Should default to Overview tab

### Test Case 2: Sequence Page Redirect
1. Navigate to `/gavd/[dataset_id]/sequence/[sequence_id]`
2. ✅ Should redirect to `/training/gavd/[dataset_id]?sequence=[sequence_id]&tab=visualization`
3. ✅ Should auto-select specified sequence
4. ✅ Should auto-switch to Visualization tab

### Test Case 3: Dashboard Link
1. Click GAVD dataset from dashboard
2. ✅ Should navigate to dataset page
3. ✅ Should show all 4 tabs
4. ✅ Should work correctly

### Test Case 4: Query Parameter Handling
1. Navigate to `/training/gavd/[dataset_id]?sequence=xxx&tab=pose`
2. ✅ Should auto-select sequence xxx
3. ✅ Should auto-switch to Pose Analysis tab
4. ✅ Should load pose analysis for that sequence

### Test Case 5: Invalid Query Parameters
1. Navigate with invalid sequence ID
2. ✅ Should ignore invalid sequence
3. ✅ Should default to first sequence
4. ✅ Should not crash


## Migration Path

### Phase 1: Implement Redirect (Immediate)
- Update sequence page to redirect
- Update dataset page to handle query params
- Update dashboard links
- **Time:** 30 minutes
- **Risk:** Low

### Phase 2: Test and Validate (Same Day)
- Manual testing of all flows
- Verify no regressions
- Check console for errors
- **Time:** 15 minutes
- **Risk:** Low

### Phase 3: Monitor and Iterate (Ongoing)
- Monitor user feedback
- Check for edge cases
- Refine as needed
- **Time:** Ongoing
- **Risk:** Minimal

## Alternative Approaches Considered

### Option A: Shared Component (Not Chosen)
**Pros:**
- More "proper" architecture
- Reusable component

**Cons:**
- More complex
- More code to maintain
- Harder to test
- Over-engineering for current needs

### Option B: Duplicate Everything (Not Chosen)
**Pros:**
- Independent pages
- No coupling

**Cons:**
- Massive code duplication
- Maintenance nightmare
- Violates DRY
- Inconsistent UX

### Option C: Redirect (CHOSEN)
**Pros:**
- Simple and effective
- No duplication
- Consistent UX
- Easy to maintain
- Follows YAGNI

**Cons:**
- Extra redirect (minimal performance impact)
- URL changes (but preserves functionality)

## Success Criteria

- [x] Plan documented
- [ ] Sequence page redirects correctly
- [ ] Dataset page handles query parameters
- [ ] Dashboard links work
- [ ] All 4 tabs accessible from both entry points
- [ ] No code duplication
- [ ] No regressions
- [ ] Clean console (no errors)
- [ ] User testing passes

## Files to Modify

1. ✅ `frontend/app/gavd/[dataset_id]/sequence/[sequence_id]/page.tsx` - Redirect
2. ✅ `frontend/app/training/gavd/[datasetId]/page.tsx` - Query param handling
3. ✅ `frontend/app/dashboard/page.tsx` - Update links

**Total:** 3 files, ~50 lines of code changes

---

**Status:** Ready for implementation  
**Estimated Time:** 45 minutes  
**Risk Level:** Low  
**Impact:** High (better UX, less code)

