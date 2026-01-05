/**
 * Redirect page for legacy /training/gavd route
 * Redirects to /gavd
 */

'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';

export default function TrainingGAVDRedirect() {
  const router = useRouter();

  useEffect(() => {
    router.replace('/gavd');
  }, [router]);

  return (
    <div className="flex items-center justify-center min-h-screen">
      <div className="text-center space-y-4">
        <div className="text-4xl animate-spin">⏳</div>
        <p className="text-lg">Redirecting to GAVD Dataset Analysis...</p>
      </div>
    </div>
  );
}