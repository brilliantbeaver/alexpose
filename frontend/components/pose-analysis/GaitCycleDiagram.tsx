/**
 * Gait Cycle Diagram Component
 * Visual representation of gait cycle phases
 */

'use client';

export default function GaitCycleDiagram() {
  return (
    <div className="w-full max-w-2xl mx-auto p-6 border rounded-lg bg-muted/30">
      <h3 className="text-lg font-semibold mb-4 text-center">Gait Cycle Phases</h3>
      
      {/* Timeline */}
      <div className="relative h-32 mb-8">
        {/* Stance Phase */}
        <div className="absolute left-0 top-0 w-[60%] h-full">
          <div className="h-16 bg-blue-500/20 border-2 border-blue-500 rounded-l-lg flex items-center justify-center">
            <span className="font-semibold text-blue-700 dark:text-blue-300">Stance Phase (60%)</span>
          </div>
          <div className="mt-2 text-xs text-center text-muted-foreground">
            Foot in contact with ground
          </div>
        </div>
        
        {/* Swing Phase */}
        <div className="absolute right-0 top-0 w-[40%] h-full">
          <div className="h-16 bg-green-500/20 border-2 border-green-500 rounded-r-lg flex items-center justify-center">
            <span className="font-semibold text-green-700 dark:text-green-300">Swing Phase (40%)</span>
          </div>
          <div className="mt-2 text-xs text-center text-muted-foreground">
            Foot moving through air
          </div>
        </div>
        
        {/* Markers */}
        <div className="absolute left-0 bottom-0 text-xs font-semibold">
          Heel Strike
        </div>
        <div className="absolute left-[60%] bottom-0 text-xs font-semibold transform -translate-x-1/2">
          Toe Off
        </div>
        <div className="absolute right-0 bottom-0 text-xs font-semibold text-right">
          Heel Strike
        </div>
      </div>

      {/* Sub-phases */}
      <div className="grid grid-cols-2 gap-4 mt-6">
        <div className="space-y-2">
          <h4 className="font-semibold text-sm text-blue-700 dark:text-blue-300">Stance Sub-phases</h4>
          <ul className="text-xs space-y-1 text-muted-foreground">
            <li>• Initial Contact (0-2%)</li>
            <li>• Loading Response (2-12%)</li>
            <li>• Mid Stance (12-31%)</li>
            <li>• Terminal Stance (31-50%)</li>
            <li>• Pre-swing (50-60%)</li>
          </ul>
        </div>
        <div className="space-y-2">
          <h4 className="font-semibold text-sm text-green-700 dark:text-green-300">Swing Sub-phases</h4>
          <ul className="text-xs space-y-1 text-muted-foreground">
            <li>• Initial Swing (60-73%)</li>
            <li>• Mid Swing (73-87%)</li>
            <li>• Terminal Swing (87-100%)</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
