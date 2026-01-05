/**
 * Breadcrumb navigation component
 * Shows current location in the app hierarchy
 */

'use client';

import React from 'react';
import Link from 'next/link';
import { useNavigation } from '@/hooks/useNavigation';

export function Breadcrumbs() {
  const { breadcrumbs, pathname } = useNavigation();

  if (breadcrumbs.length === 0) {
    return null;
  }

  // Define route segments that should not be clickable (they're just path structure, not pages)
  const nonClickableSegments = ['sequence', 'frame', 'analysis'];

  return (
    <nav aria-label="Breadcrumb" className="flex items-center space-x-2 text-sm text-muted-foreground">
      <Link href="/" className="hover:text-foreground transition-colors">
        🏠 Home
      </Link>
      {breadcrumbs.map((crumb, index) => {
        const isLast = index === breadcrumbs.length - 1;
        const href = '/' + pathname.split('/').slice(1, index + 2).join('/');
        const crumbLower = crumb.toLowerCase();
        
        // Check if this segment should be non-clickable
        const isNonClickable = nonClickableSegments.includes(crumbLower);
        
        return (
          <React.Fragment key={index}>
            <span>/</span>
            {isLast || isNonClickable ? (
              <span className={isLast ? "font-medium text-foreground" : "text-muted-foreground"}>
                {crumb}
              </span>
            ) : (
              <Link href={href} className="hover:text-foreground transition-colors">
                {crumb}
              </Link>
            )}
          </React.Fragment>
        );
      })}
    </nav>
  );
}
