'use client';

import { useState } from 'react';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';

interface RecentAnalysis {
  type: 'gait_analysis' | 'gavd_dataset';
  // Gait analysis fields
  analysis_id?: string;
  file_id?: string;
  is_normal?: boolean | null;
  confidence?: number;
  explanation?: string;
  identified_conditions?: string[];
  frame_count?: number;
  duration?: number;
  // GAVD dataset fields
  dataset_id?: string;
  filename?: string;
  sequence_count?: number;
  row_count?: number;
  total_sequences_processed?: number;
  total_frames_processed?: number;
  progress?: string;
  // GAVD sequence-specific fields
  seq?: string;
  gait_pat?: string;
  // Common fields
  status: string;
  created_at?: string;
  uploaded_at?: string;
  completed_at?: string;
}

interface AnalysesTableProps {
  analyses: RecentAnalysis[];
  onDelete?: (analysis: RecentAnalysis) => void;
  deletingAnalysis?: string | null;
  showActions?: boolean;
  maxRows?: number;
  showFilters?: boolean;
}

export function AnalysesTable({ 
  analyses, 
  onDelete, 
  deletingAnalysis, 
  showActions = true,
  maxRows,
  showFilters = true
}: AnalysesTableProps) {
  const [seqFilter, setSeqFilter] = useState('');
  const [gaitPatFilter, setGaitPatFilter] = useState('all_patterns');
  const [filterType, setFilterType] = useState<'seq' | 'gait_pat' | 'all'>('all');

  // Filter analyses based on current filters
  const filteredAnalyses = analyses.filter(analysis => {
    if (filterType === 'seq' && seqFilter) {
      return analysis.seq?.toLowerCase().includes(seqFilter.toLowerCase());
    }
    if (filterType === 'gait_pat' && gaitPatFilter && gaitPatFilter !== 'all_patterns') {
      return analysis.gait_pat?.toLowerCase().includes(gaitPatFilter.toLowerCase());
    }
    if (filterType === 'all') {
      const seqMatch = !seqFilter || analysis.seq?.toLowerCase().includes(seqFilter.toLowerCase());
      const gaitPatMatch = !gaitPatFilter || gaitPatFilter === 'all_patterns' || analysis.gait_pat?.toLowerCase().includes(gaitPatFilter.toLowerCase());
      return seqMatch && gaitPatMatch;
    }
    return true;
  });

  const displayedAnalyses = maxRows ? filteredAnalyses.slice(0, maxRows) : filteredAnalyses;

  // Get unique values for filter dropdowns
  const uniqueSeqs = [...new Set(analyses.map(a => a.seq).filter(Boolean))];
  const uniqueGaitPats = [...new Set(analyses.map(a => a.gait_pat).filter(Boolean))];

  const formatTimeAgo = (dateString: string): string => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    return `${diffDays}d ago`;
  };

  const getStatusBadge = (analysis: RecentAnalysis) => {
    if (analysis.type === 'gavd_dataset') {
      const status = analysis.status;
      if (status === 'completed') return <Badge variant="default" className="bg-green-600">Completed</Badge>;
      if (status === 'processing') return <Badge variant="default" className="bg-blue-600">Processing</Badge>;
      if (status === 'uploaded') return <Badge variant="secondary">Uploaded</Badge>;
      if (status === 'error') return <Badge variant="destructive">Error</Badge>;
      return <Badge variant="secondary">{status}</Badge>;
    } else {
      // Gait analysis
      const isNormal = analysis.is_normal;
      if (isNormal === null) return <Badge variant="secondary">Analyzing</Badge>;
      return isNormal ? (
        <Badge variant="default" className="bg-green-600">Normal</Badge>
      ) : (
        <Badge variant="destructive">Abnormal</Badge>
      );
    }
  };

  const getAnalysisTitle = (analysis: RecentAnalysis) => {
    if (analysis.type === 'gavd_dataset') {
      // Rename GAVD files based on their content/diagnosis
      const filename = analysis.filename || 'GAVD Dataset';
      
      // Map common GAVD filenames to meaningful diagnosis names
      if (filename.includes('GAVD_Clinical_Annotations_1.csv') || filename.includes('GAVD_Clinical_Annotations_1.1.csv')) {
        return 'Parkinson\'s Disease Dataset';
      } else if (filename.includes('GAVD_Clinical_Annotations_1.2.csv')) {
        return 'General Abnormal Gait Dataset';
      } else if (filename.includes('GAVD_Clinical_Annotations_1.3.csv')) {
        return 'Abnormal Gait Patterns Dataset';
      } else if (filename.toLowerCase().includes('parkinson')) {
        return 'Parkinson\'s Disease Dataset';
      } else if (filename.toLowerCase().includes('abnormal')) {
        return 'Abnormal Gait Dataset';
      } else if (filename.toLowerCase().includes('normal')) {
        return 'Normal Gait Dataset';
      } else if (filename.toLowerCase().includes('stroke')) {
        return 'Stroke Gait Dataset';
      } else if (filename.toLowerCase().includes('cerebral')) {
        return 'Cerebral Palsy Dataset';
      } else if (filename.toLowerCase().includes('spinal')) {
        return 'Spinal Cord Injury Dataset';
      } else if (filename.toLowerCase().includes('muscular')) {
        return 'Muscular Dystrophy Dataset';
      } else if (filename.toLowerCase().includes('arthritis')) {
        return 'Arthritis Gait Dataset';
      } else if (filename.toLowerCase().includes('gavd')) {
        return 'GAVD Clinical Dataset';
      } else {
        return filename;
      }
    } else {
      return `Gait Analysis ${(analysis.analysis_id || '').substring(0, 8)}`;
    }
  };

  const getAnalysisLink = (analysis: RecentAnalysis) => {
    if (analysis.type === 'gavd_dataset') {
      return `/gavd/${analysis.dataset_id}`;
    } else {
      return `/results/${analysis.analysis_id}`;
    }
  };

  const getAnalysisDetails = (analysis: RecentAnalysis) => {
    if (analysis.type === 'gavd_dataset') {
      const details = [];
      if (analysis.total_sequences_processed) {
        details.push(`${analysis.total_sequences_processed} sequences`);
      }
      if (analysis.total_frames_processed) {
        details.push(`${analysis.total_frames_processed} frames`);
      }
      
      // Add diagnosis-specific context
      const title = getAnalysisTitle(analysis);
      if (title.includes('Parkinson\'s')) {
        details.push('Movement disorder analysis');
      } else if (title.includes('Abnormal')) {
        details.push('Pathological gait patterns');
      } else if (title.includes('Normal')) {
        details.push('Healthy gait patterns');
      } else if (title.includes('Stroke')) {
        details.push('Post-stroke gait analysis');
      } else if (title.includes('Cerebral')) {
        details.push('Neurological condition');
      } else if (title.includes('Spinal')) {
        details.push('Spinal injury patterns');
      } else if (title.includes('Muscular')) {
        details.push('Muscle disorder analysis');
      } else if (title.includes('Arthritis')) {
        details.push('Joint condition analysis');
      }
      
      return details.join(' • ');
    } else {
      const details = [];
      if (analysis.frame_count) {
        details.push(`${analysis.frame_count} frames`);
      }
      if (analysis.confidence !== undefined) {
        details.push(`${(analysis.confidence * 100).toFixed(0)}% confidence`);
      }
      if (analysis.identified_conditions && analysis.identified_conditions.length > 0) {
        details.push(`Conditions: ${analysis.identified_conditions.join(', ')}`);
      }
      return details.join(' • ');
    }
  };

  const handleDelete = (analysis: RecentAnalysis, event: React.MouseEvent) => {
    event.preventDefault();
    event.stopPropagation();
    if (onDelete) {
      onDelete(analysis);
    }
  };

  if (analyses.length === 0) {
    return (
      <div className="text-center py-8 text-muted-foreground">
        <div className="text-4xl mb-4">📊</div>
        <p className="text-lg font-medium mb-2">No analyses yet</p>
        <p className="text-sm">Upload a video or GAVD dataset to get started</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Filter Controls */}
      {showFilters && (
        <div className="flex flex-wrap gap-4 p-4 bg-muted/50 rounded-lg">
          <div className="flex items-center gap-2">
            <label className="text-sm font-medium">Filter by:</label>
            <Select value={filterType} onValueChange={(value: 'seq' | 'gait_pat' | 'all') => setFilterType(value)}>
              <SelectTrigger className="w-32">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All</SelectItem>
                <SelectItem value="seq">Sequence</SelectItem>
                <SelectItem value="gait_pat">Gait Pattern</SelectItem>
              </SelectContent>
            </Select>
          </div>
          
          {(filterType === 'seq' || filterType === 'all') && (
            <div className="flex items-center gap-2">
              <label className="text-sm font-medium">Sequence:</label>
              <Input
                placeholder="Search sequences..."
                value={seqFilter}
                onChange={(e) => setSeqFilter(e.target.value)}
                className="w-48"
              />
            </div>
          )}
          
          {(filterType === 'gait_pat' || filterType === 'all') && (
            <div className="flex items-center gap-2">
              <label className="text-sm font-medium">Gait Pattern:</label>
              <Select value={gaitPatFilter} onValueChange={setGaitPatFilter}>
                <SelectTrigger className="w-48">
                  <SelectValue placeholder="Select pattern..." />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all_patterns">All patterns</SelectItem>
                  {uniqueGaitPats.map(pattern => (
                    <SelectItem key={pattern} value={pattern || ''}>{pattern}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}
          
          {(seqFilter || (gaitPatFilter && gaitPatFilter !== 'all_patterns')) && (
            <Button 
              variant="outline" 
              size="sm"
              onClick={() => {
                setSeqFilter('');
                setGaitPatFilter('all_patterns');
                setFilterType('all');
              }}
            >
              Clear Filters
            </Button>
          )}
        </div>
      )}

      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Sequence</TableHead>
            <TableHead>Gait Pattern</TableHead>
            <TableHead>Status</TableHead>
            <TableHead>Details</TableHead>
            <TableHead>Date</TableHead>
            {showActions && <TableHead className="w-[100px]">Actions</TableHead>}
          </TableRow>
        </TableHeader>
        <TableBody>
          {displayedAnalyses.map((analysis, index) => {
            const analysisId = analysis.dataset_id || analysis.analysis_id || `${index}`;
            const date = analysis.completed_at || analysis.uploaded_at || analysis.created_at || '';
            
            return (
              <TableRow key={analysisId} className="cursor-pointer hover:bg-muted/50">
                <TableCell>
                  <Link 
                    href={getAnalysisLink(analysis)}
                    className="font-medium hover:underline"
                  >
                    {analysis.seq ? (
                      <div className="font-mono text-sm">{analysis.seq}</div>
                    ) : (
                      <div className="text-muted-foreground text-sm">No sequence data</div>
                    )}
                  </Link>
                </TableCell>
                <TableCell>
                  {analysis.gait_pat ? (
                    <Badge variant={analysis.gait_pat === 'parkinsons' ? 'destructive' : 'secondary'}>
                      {analysis.gait_pat}
                    </Badge>
                  ) : (
                    <span className="text-muted-foreground text-sm">N/A</span>
                  )}
                </TableCell>
                <TableCell>
                  {getStatusBadge(analysis)}
                </TableCell>
                <TableCell>
                  <div className="text-sm text-muted-foreground">
                    {getAnalysisDetails(analysis)}
                  </div>
                </TableCell>
                <TableCell>
                  <div className="text-sm text-muted-foreground">
                    {date ? formatTimeAgo(date) : 'Unknown'}
                  </div>
                </TableCell>
                {showActions && (
                  <TableCell>
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button variant="ghost" size="sm">
                          ⋮
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end">
                        <DropdownMenuItem asChild>
                          <Link href={getAnalysisLink(analysis)}>
                            View Details
                          </Link>
                        </DropdownMenuItem>
                        {onDelete && (
                          <DropdownMenuItem
                            onClick={(e) => handleDelete(analysis, e)}
                            disabled={deletingAnalysis === analysisId}
                            className="text-red-600 focus:text-red-600"
                          >
                            {deletingAnalysis === analysisId ? (
                              <span className="flex items-center gap-2">
                                <span className="animate-spin">⏳</span>
                                Deleting...
                              </span>
                            ) : (
                              'Delete'
                            )}
                          </DropdownMenuItem>
                        )}
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </TableCell>
                )}
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
      
      {filteredAnalyses.length === 0 && analyses.length > 0 && (
        <div className="text-center py-8 text-muted-foreground">
          <div className="text-4xl mb-4">🔍</div>
          <p className="text-lg font-medium mb-2">No matching analyses</p>
          <p className="text-sm">Try adjusting your filters</p>
        </div>
      )}
    </div>
  );
}