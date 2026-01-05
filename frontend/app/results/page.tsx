/**
 * Results Page
 * Display analysis results with visualizations
 */

import Link from 'next/link';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

export default function ResultsPage() {
  return (
    <div className="space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Analysis Results</h1>
          <p className="text-muted-foreground">View and manage your gait analysis results</p>
        </div>
        <Button asChild>
          <Link href="/analyze/upload">
            📤 New Analysis
          </Link>
        </Button>
      </div>

      <Tabs defaultValue="all" className="w-full">
        <TabsList>
          <TabsTrigger value="all">All Results</TabsTrigger>
          <TabsTrigger value="normal">Normal</TabsTrigger>
          <TabsTrigger value="abnormal">Abnormal</TabsTrigger>
        </TabsList>

        <TabsContent value="all" className="space-y-4 mt-6">
          {[
            { id: 1, name: 'Walking Test 1', date: '2024-01-03', status: 'Normal', confidence: 95, conditions: [] },
            { id: 2, name: 'Gait Analysis 2', date: '2024-01-03', status: 'Abnormal', confidence: 88, conditions: ['Limping', 'Asymmetry'] },
            { id: 3, name: 'Patient Video 3', date: '2024-01-02', status: 'Normal', confidence: 92, conditions: [] },
            { id: 4, name: 'YouTube Analysis 4', date: '2024-01-02', status: 'Abnormal', confidence: 85, conditions: ['Reduced Cadence'] },
            { id: 5, name: 'Batch Test 5', date: '2024-01-01', status: 'Normal', confidence: 94, conditions: [] },
          ].map((result) => (
            <Card key={result.id} className="hover:shadow-md transition-shadow">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-4">
                    <div className="w-16 h-16 rounded-lg bg-gradient-to-br from-blue-400 to-purple-500 flex items-center justify-center text-white font-bold text-xl">
                      #{result.id}
                    </div>
                    <div>
                      <CardTitle>{result.name}</CardTitle>
                      <CardDescription>{result.date}</CardDescription>
                    </div>
                  </div>
                  <div className="flex items-center space-x-4">
                    <Badge variant={result.status === 'Normal' ? 'default' : 'destructive'}>
                      {result.status}
                    </Badge>
                    <div className="text-sm text-muted-foreground">
                      {result.confidence}% confidence
                    </div>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div className="flex items-center justify-between">
                  <div className="flex flex-wrap gap-2">
                    {result.conditions.length > 0 ? (
                      result.conditions.map((condition, idx) => (
                        <Badge key={idx} variant="outline">
                          {condition}
                        </Badge>
                      ))
                    ) : (
                      <span className="text-sm text-muted-foreground">No conditions detected</span>
                    )}
                  </div>
                  <div className="flex space-x-2">
                    <Button variant="outline" size="sm" asChild>
                      <Link href={`/results/${result.id}`}>
                        📊 View Details
                      </Link>
                    </Button>
                    <Button variant="ghost" size="sm">
                      💾 Export
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </TabsContent>

        <TabsContent value="normal" className="space-y-4 mt-6">
          <Card>
            <CardContent className="pt-6">
              <p className="text-center text-muted-foreground">
                Showing only normal gait patterns
              </p>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="abnormal" className="space-y-4 mt-6">
          <Card>
            <CardContent className="pt-6">
              <p className="text-center text-muted-foreground">
                Showing only abnormal gait patterns
              </p>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
