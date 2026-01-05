/**
 * Tests for useNavigation hook
 */

import { renderHook } from '@testing-library/react';
import { useNavigation } from '@/hooks/useNavigation';

// Mock next/navigation
jest.mock('next/navigation', () => ({
  usePathname: jest.fn(),
}));

import { usePathname } from 'next/navigation';

describe('useNavigation', () => {
  it('returns navigation items', () => {
    (usePathname as jest.Mock).mockReturnValue('/');
    const { result } = renderHook(() => useNavigation());
    
    expect(result.current.navigationItems).toBeDefined();
    expect(result.current.navigationItems.length).toBeGreaterThan(0);
  });

  it('identifies active item from pathname', () => {
    (usePathname as jest.Mock).mockReturnValue('/dashboard');
    const { result } = renderHook(() => useNavigation());
    
    expect(result.current.activeItem).toBe('dashboard');
  });

  it('generates breadcrumbs from pathname', () => {
    (usePathname as jest.Mock).mockReturnValue('/analyze/upload');
    const { result } = renderHook(() => useNavigation());
    
    expect(result.current.breadcrumbs).toEqual(['Analyze', 'Upload']);
  });

  it('returns null active item for unknown path', () => {
    (usePathname as jest.Mock).mockReturnValue('/unknown-path');
    const { result } = renderHook(() => useNavigation());
    
    expect(result.current.activeItem).toBeNull();
  });

  it('handles root path correctly', () => {
    (usePathname as jest.Mock).mockReturnValue('/');
    const { result } = renderHook(() => useNavigation());
    
    expect(result.current.breadcrumbs).toEqual([]);
  });
});
