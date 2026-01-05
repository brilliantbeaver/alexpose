/**
 * GAVD Sequence Viewer Page - Redirect to Dataset Page
 * 
 * This page redirects to the full dataset analysis page with the sequence pre-selected.
 * This consolidates functionality and provides a consistent UX with all 4 tabs
 * (Overview, Sequences, Visualization, Pose Analysis) accessible from any entry point.
 * 
 * Redirect Pattern: /gavd/[dataset_id]/sequence/[sequence_id]
 *                → /gavd/[dataset_id]?sequence=[sequence_id]&tab=visualization
 */

'use client';

import { useEffect } from 'react';
import { useParams, useRouter } from 'next/navigation';
import { LoadingState } from '@/components/ui/loading-spinner';
import { ArrowRight } from 'lucide-react';


export default function GAVDSequencePage() {
  const params = useParams();
  const router = useRouter();
  const dataset_id = params.dataset_id as string;
  const sequence_id = params.sequence_id as string;

  useEffect(() => {
    // Redirect to dataset page with sequence pre-selected and visualization tab active
    const targetUrl = `/gavd/${dataset_id}?sequence=${sequence_id}&tab=visualization`;
    console.log(`[Sequence Redirect] Redirecting to: ${targetUrl}`);
    router.replace(targetUrl);
  }, [dataset_id, sequence_id, router]);

  return (
    <LoadingState
      title="Redirecting to Sequence Viewer"
      description={`Loading full analysis interface for sequence ${sequence_id}...`}
      icon={<ArrowRight className="w-6 h-6" />}
      variant="fullscreen"
    />
  );
}
