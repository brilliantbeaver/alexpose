/**
 * Redirect page for legacy /training/gavd/[datasetId] route
 * Redirects to /gavd/[datasetId]
 */

'use client';

import { use, useEffect } from 'react';
import { useRouter } from 'next/navigation';

interface TrainingGAVDDatasetRedirectProps {
  params: Promise<{ datasetId: string }>;
}

export default function TrainingGAVDDatasetRedirect({ params }: TrainingGAVDDatasetRedirectProps) {
  const { datasetId } = use(params);
  const router = useRouter();

  useEffect(() => {
    router.replace(`/gavd/${datasetId}`);
  }, [router, datasetId]);

  return (
    <div className="flex items-center justify-center min-h-screen">
      <div className="text-center space-y-4">
        <div className="text-4xl animate-spin">⏳</div>
        <p className="text-lg">Redirecting to dataset analysis...</p>
      </div>
    </div>
  );
}