/**
 * Symmetry Diagram Component
 * Visual representation of gait symmetry concept
 */

'use client';

export default function SymmetryDiagram() {
  return (
    <div className="w-full max-w-3xl mx-auto p-6 border rounded-lg bg-muted/30">
      <h3 className="text-lg font-semibold mb-6 text-center">Gait Symmetry Visualization</h3>
      
      <div className="grid grid-cols-3 gap-6">
        {/* Perfect Symmetry */}
        <div className="space-y-3">
          <div className="text-center">
            <div className="inline-block px-3 py-1 bg-green-100 dark:bg-green-900 text-green-800 dark:text-green-100 rounded-full text-xs font-semibold mb-2">
              Symmetric (&lt; 0.10)
            </div>
          </div>
          <div className="flex justify-center gap-4">
            {/* Left leg */}
            <div className="space-y-1">
              <div className="w-12 h-32 bg-blue-500 rounded-lg"></div>
              <p className="text-xs text-center text-muted-foreground">Left</p>
            </div>
            {/* Right leg */}
            <div className="space-y-1">
              <div className="w-12 h-32 bg-blue-500 rounded-lg"></div>
              <p className="text-xs text-center text-muted-foreground">Right</p>
            </div>
          </div>
          <p className="text-xs text-center text-muted-foreground">
            Equal movement on both sides
          </p>
        </div>

        {/* Mild Asymmetry */}
        <div className="space-y-3">
          <div className="text-center">
            <div className="inline-block px-3 py-1 bg-yellow-100 dark:bg-yellow-900 text-yellow-800 dark:text-yellow-100 rounded-full text-xs font-semibold mb-2">
              Mildly Asymmetric (0.10-0.20)
            </div>
          </div>
          <div className="flex justify-center gap-4">
            {/* Left leg */}
            <div className="space-y-1">
              <div className="w-12 h-32 bg-blue-500 rounded-lg"></div>
              <p className="text-xs text-center text-muted-foreground">Left</p>
            </div>
            {/* Right leg - slightly shorter */}
            <div className="space-y-1">
              <div className="w-12 h-28 bg-blue-400 rounded-lg mt-4"></div>
              <p className="text-xs text-center text-muted-foreground">Right</p>
            </div>
          </div>
          <p className="text-xs text-center text-muted-foreground">
            Minor difference between sides
          </p>
        </div>

        {/* Severe Asymmetry */}
        <div className="space-y-3">
          <div className="text-center">
            <div className="inline-block px-3 py-1 bg-red-100 dark:bg-red-900 text-red-800 dark:text-red-100 rounded-full text-xs font-semibold mb-2">
              Severely Asymmetric (&gt; 0.30)
            </div>
          </div>
          <div className="flex justify-center gap-4">
            {/* Left leg */}
            <div className="space-y-1">
              <div className="w-12 h-32 bg-blue-500 rounded-lg"></div>
              <p className="text-xs text-center text-muted-foreground">Left</p>
            </div>
            {/* Right leg - much shorter */}
            <div className="space-y-1">
              <div className="w-12 h-20 bg-blue-300 rounded-lg mt-12"></div>
              <p className="text-xs text-center text-muted-foreground">Right</p>
            </div>
          </div>
          <p className="text-xs text-center text-muted-foreground">
            Significant difference requiring attention
          </p>
        </div>
      </div>

      {/* Legend */}
      <div className="mt-6 pt-6 border-t">
        <p className="text-xs text-muted-foreground text-center">
          Bar height represents movement amplitude, stride length, or joint angle range.
          Symmetry index measures the average difference between left and right sides.
        </p>
      </div>
    </div>
  );
}
