/**
 * @jest-environment jsdom
 */

import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { AnalysesTable } from '@/components/ui/analyses-table';

// Mock Next.js Link component
jest.mock('next/link', () => {
  return ({ children, href }: { children: React.ReactNode; href: string }) => (
    <a href={href}>{children}</a>
  );
});

const mockGaitAnalysis = {
  type: 'gait_analysis' as const,
  analysis_id: 'test-analysis-123',
  is_normal: true,
  confidence: 0.85,
  frame_count: 120,
  status: 'completed',
  created_at: '2024-01-01T10:00:00Z',
  identified_conditions: ['Normal gait pattern'],
  seq: 'cljan9b4p00043n6ligceanyp',
  gait_pat: 'normal',
};

const mockGavdDataset = {
  type: 'gavd_dataset' as const,
  dataset_id: 'test-dataset-456',
  filename: 'GAVD_Clinical_Annotations_1.csv',
  status: 'completed',
  uploaded_at: '2024-01-01T09:00:00Z',
  total_sequences_processed: 50,
  total_frames_processed: 1500,
  seq: 'cljan9b4p00043n6ligceanyp',
  gait_pat: 'parkinsons',
};

describe('AnalysesTable', () => {
  it('renders empty state when no analyses provided', () => {
    render(<AnalysesTable analyses={[]} />);
    
    expect(screen.getByText('No analyses yet')).toBeInTheDocument();
    expect(screen.getByText('Upload a video or GAVD dataset to get started')).toBeInTheDocument();
  });

  it('renders gait analysis correctly', () => {
    render(<AnalysesTable analyses={[mockGaitAnalysis]} showFilters={false} />);
    
    expect(screen.getByText('Sequence')).toBeInTheDocument();
    expect(screen.getByText('Gait Pattern')).toBeInTheDocument();
    expect(screen.getByText('cljan9b4p00043n6ligceanyp')).toBeInTheDocument();
    expect(screen.getByText('normal')).toBeInTheDocument();
    expect(screen.getByText('Normal')).toBeInTheDocument();
    expect(screen.getByText(/120 frames/)).toBeInTheDocument();
  });

  it('renders GAVD dataset correctly', () => {
    render(<AnalysesTable analyses={[mockGavdDataset]} showFilters={false} />);
    
    expect(screen.getByText('Sequence')).toBeInTheDocument();
    expect(screen.getByText('Gait Pattern')).toBeInTheDocument();
    expect(screen.getByText('cljan9b4p00043n6ligceanyp')).toBeInTheDocument();
    expect(screen.getByText('parkinsons')).toBeInTheDocument();
    expect(screen.getByText('Completed')).toBeInTheDocument();
    expect(screen.getByText(/50 sequences/)).toBeInTheDocument();
    expect(screen.getByText(/1500 frames/)).toBeInTheDocument();
  });

  it('renders mixed analyses correctly', () => {
    render(<AnalysesTable analyses={[mockGaitAnalysis, mockGavdDataset]} showFilters={false} />);
    
    // Should show both sequences (full, not truncated)
    expect(screen.getAllByText('cljan9b4p00043n6ligceanyp')).toHaveLength(2);
    
    // Should show different gait patterns
    expect(screen.getByText('normal')).toBeInTheDocument();
    expect(screen.getByText('parkinsons')).toBeInTheDocument();
  });

  it('limits rows when maxRows is specified', () => {
    const manyAnalyses = Array(10).fill(null).map((_, i) => ({
      ...mockGaitAnalysis,
      analysis_id: `test-analysis-${i}`,
      seq: `seq-${i}`,
    }));
    
    render(<AnalysesTable analyses={manyAnalyses} maxRows={3} showFilters={false} />);
    
    // Should only show 3 rows
    const rows = screen.getAllByRole('row');
    // 1 header row + 3 data rows = 4 total
    expect(rows).toHaveLength(4);
  });

  it('shows actions column when showActions is true', () => {
    render(<AnalysesTable analyses={[mockGaitAnalysis]} showActions={true} showFilters={false} />);
    
    expect(screen.getByText('Actions')).toBeInTheDocument();
  });

  it('hides actions column when showActions is false', () => {
    render(<AnalysesTable analyses={[mockGaitAnalysis]} showActions={false} showFilters={false} />);
    
    expect(screen.queryByText('Actions')).not.toBeInTheDocument();
  });

  it('shows filters when showFilters is true', () => {
    render(<AnalysesTable analyses={[mockGaitAnalysis, mockGavdDataset]} showFilters={true} />);
    
    expect(screen.getByText('Filter by:')).toBeInTheDocument();
    expect(screen.getByPlaceholderText('Search sequences...')).toBeInTheDocument();
  });

  it('hides filters when showFilters is false', () => {
    render(<AnalysesTable analyses={[mockGaitAnalysis, mockGavdDataset]} showFilters={false} />);
    
    expect(screen.queryByText('Filter by:')).not.toBeInTheDocument();
    expect(screen.queryByPlaceholderText('Search sequences...')).not.toBeInTheDocument();
  });

  it('filters analyses by sequence', () => {
    const analyses = [
      { ...mockGaitAnalysis, seq: 'seq-abc123' },
      { ...mockGavdDataset, seq: 'seq-def456' }
    ];
    
    render(<AnalysesTable analyses={analyses} showFilters={true} />);
    
    // Initially both should be visible
    expect(screen.getByText('seq-abc123')).toBeInTheDocument();
    expect(screen.getByText('seq-def456')).toBeInTheDocument();
    
    // Filter by sequence
    const seqInput = screen.getByPlaceholderText('Search sequences...');
    fireEvent.change(seqInput, { target: { value: 'abc' } });
    
    // Only matching sequence should be visible
    expect(screen.getByText('seq-abc123')).toBeInTheDocument();
    expect(screen.queryByText('seq-def456')).not.toBeInTheDocument();
  });
});