/**
 * Custom hook for navigation state management
 */

'use client';

import { useState, useEffect } from 'react';
import { usePathname } from 'next/navigation';
import { NavigationItem } from '@/lib/navigation-types';
import { navigationConfig } from '@/lib/navigation-config';

export function useNavigation() {
  const pathname = usePathname();
  const [activeItem, setActiveItem] = useState<string | null>(null);
  const [breadcrumbs, setBreadcrumbs] = useState<string[]>([]);

  useEffect(() => {
    // Determine active navigation item based on current path
    const findActiveItem = () => {
      for (const item of navigationConfig) {
        if (pathname === item.href || pathname.startsWith(item.href + '/')) {
          return item.id;
        }
        if (item.submenu) {
          for (const subitem of item.submenu) {
            if (pathname === subitem.href) {
              return item.id;
            }
          }
        }
      }
      return null;
    };

    // Generate breadcrumbs from current path
    const generateBreadcrumbs = () => {
      const segments = pathname.split('/').filter(Boolean);
      return segments.map((segment, index) => {
        const path = '/' + segments.slice(0, index + 1).join('/');
        return segment.charAt(0).toUpperCase() + segment.slice(1);
      });
    };

    setActiveItem(findActiveItem());
    setBreadcrumbs(generateBreadcrumbs());
  }, [pathname]);

  return {
    activeItem,
    breadcrumbs,
    pathname,
    navigationItems: navigationConfig,
  };
}
