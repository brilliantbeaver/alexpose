/**
 * GAVD Training Layout
 * Provides consistent navigation and context for GAVD dataset workflows
 */

import { ReactNode } from 'react';

export default function GAVDLayout({ children }: { children: ReactNode }) {
  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-blue-50 to-white">
      <div className="container mx-auto px-4 py-8">
        {/* Breadcrumb Navigation */}
        <div className="mb-6 flex items-center space-x-2 text-sm text-muted-foreground">
          <a href="/" className="hover:text-foreground transition-colors">Home</a>
          <span>→</span>
          <a href="/training/gavd" className="hover:text-foreground transition-colors">GAVD Training</a>
        </div>
        
        {children}
      </div>
    </div>
  );
}
