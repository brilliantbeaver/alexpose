#!/usr/bin/env python3
"""
Example demonstrating lazy loading and memory management capabilities in AlexPose Frame system.

This example shows how to:
1. Use lazy loading to defer frame loading until needed
2. Monitor memory usage across frames
3. Use automatic memory management
4. Process frame sequences efficiently with batch operations
5. Optimize memory usage for large sequences

Run this example with: python examples/memory_management_example.py
"""

import numpy as np
import time
from pathlib import Path

from ambient.core.frame import (
    Frame, 
    FrameSequence, 
    MemoryOptimizedFrameSequence,
    TemporaryMemorySettings,
    get_global_memory_stats,
    set_global_memory_threshold,
    force_global_cleanup
)


def demonstrate_lazy_loading():
    """Demonstrate lazy loading functionality."""
    print("=== Lazy Loading Demonstration ===")
    
    # Create a large frame with lazy loading
    large_array = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
    
    # Create frame with lazy loading enabled
    frame = Frame.from_array(large_array)
    frame.lazy_load = True
    frame.unload()  # Start unloaded
    
    print(f"Frame created - Loaded: {frame.is_loaded}, Memory: {frame.memory_usage_mb:.2f}MB")
    
    # Access frame data - this triggers loading
    print("Accessing frame data...")
    data = frame.load()
    print(f"After load() - Loaded: {frame.is_loaded}, Memory: {frame.memory_usage_mb:.2f}MB")
    
    # Unload to free memory
    frame.unload()
    print(f"After unload() - Loaded: {frame.is_loaded}, Memory: {frame.memory_usage_mb:.2f}MB")
    
    print()


def demonstrate_memory_monitoring():
    """Demonstrate memory usage monitoring."""
    print("=== Memory Monitoring Demonstration ===")
    
    # Clear existing frames
    force_global_cleanup()
    
    # Create multiple frames
    frames = []
    for i in range(5):
        array = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
        frame = Frame.from_array(array)
        frames.append(frame)
        
        stats = get_global_memory_stats()
        print(f"Frame {i+1}: {stats['loaded_frames']} frames, {stats['total_memory_mb']:.2f}MB total")
    
    print(f"Final stats: {get_global_memory_stats()}")
    print()


def demonstrate_automatic_cleanup():
    """Demonstrate automatic memory cleanup."""
    print("=== Automatic Memory Cleanup Demonstration ===")
    
    # Set a low memory threshold for demonstration
    original_threshold = get_global_memory_stats()['memory_threshold_mb']
    
    try:
        # Set threshold to 5MB
        set_global_memory_threshold(5)
        print(f"Set memory threshold to 5MB")
        
        # Create frames that will exceed the threshold
        frames = []
        for i in range(10):
            # Create 1MB frames
            array = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
            frame = Frame.from_array(array)
            frame.lazy_load = True  # Enable automatic unloading
            frames.append(frame)
            
            stats = get_global_memory_stats()
            print(f"Frame {i+1}: {stats['loaded_frames']} loaded, {stats['total_memory_mb']:.2f}MB")
            
            # Small delay to allow memory manager to work
            time.sleep(0.1)
        
        print("Automatic cleanup should have occurred to stay under 5MB threshold")
        
    finally:
        # Restore original threshold
        set_global_memory_threshold(original_threshold)
    
    print()


def demonstrate_batch_processing():
    """Demonstrate efficient batch processing."""
    print("=== Batch Processing Demonstration ===")
    
    # Create a large sequence of frames
    frames = []
    for i in range(20):
        array = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        frame = Frame.from_array(array)
        frame.lazy_load = True
        frame.unload()  # Start unloaded
        frames.append(frame)
    
    sequence = FrameSequence(frames=frames, lazy_load=True)
    sequence.set_batch_size(5)
    
    print(f"Created sequence with {len(frames)} frames")
    
    # Define a simple processing function
    def process_batch(batch_data, start_idx):
        """Simple processing: calculate mean brightness of each frame."""
        results = []
        for i, frame_data in enumerate(batch_data):
            mean_brightness = np.mean(frame_data)
            results.append(mean_brightness)
        print(f"  Processed batch starting at frame {start_idx}: {len(batch_data)} frames")
        return results
    
    # Process in batches with automatic memory management
    print("Processing frames in batches...")
    results = sequence.process_in_batches(
        process_batch, 
        batch_size=5, 
        unload_after_processing=True
    )
    
    print(f"Processed {len(results)} batches")
    
    # Check memory usage after processing
    stats = sequence.get_memory_stats()
    print(f"After processing: {stats['loaded_frames']}/{stats['total_frames']} frames loaded")
    print()


def demonstrate_smart_preloading():
    """Demonstrate smart preloading in sequences."""
    print("=== Smart Preloading Demonstration ===")
    
    # Create sequence with frames
    frames = []
    for i in range(15):
        array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        frame = Frame.from_array(array)
        frame.lazy_load = True
        frame.unload()
        frames.append(frame)
    
    sequence = FrameSequence(frames=frames, lazy_load=True)
    sequence.set_preload_window(3)  # Preload 3 frames around current position
    
    print(f"Created sequence with {len(frames)} frames, preload window = 3")
    
    # Access frame in the middle
    print("Accessing frame 7...")
    frame_7 = sequence[7]
    
    # Check which frames are loaded
    loaded_frames = [i for i, frame in enumerate(frames) if frame.is_loaded]
    print(f"Loaded frames after accessing frame 7: {loaded_frames}")
    print("(Should include frames around position 7)")
    
    print()


def demonstrate_memory_optimization():
    """Demonstrate memory optimization features."""
    print("=== Memory Optimization Demonstration ===")
    
    # Use temporary memory settings
    with TemporaryMemorySettings(threshold_mb=10, max_frames=5):
        print("Using temporary memory settings: 10MB threshold, max 5 frames")
        
        # Create memory-optimized sequence
        frames = []
        for i in range(12):
            array = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
            frame = Frame.from_array(array)
            frames.append(frame)
        
        sequence = MemoryOptimizedFrameSequence(frames=frames, lazy_load=True)
        sequence.enable_auto_optimization(enabled=True, interval=3)
        
        print(f"Created optimized sequence with {len(frames)} frames")
        
        # Access frames to trigger optimization
        for i in range(8):
            sequence[i]
            stats = sequence.get_memory_stats()
            print(f"  Access {i}: {stats['loaded_frames']} frames loaded")
        
        print("Auto-optimization should have kept memory usage under control")
    
    print("Temporary settings restored")
    print()


def main():
    """Run all demonstrations."""
    print("AlexPose Frame Memory Management Examples")
    print("=" * 50)
    print()
    
    demonstrate_lazy_loading()
    demonstrate_memory_monitoring()
    demonstrate_automatic_cleanup()
    demonstrate_batch_processing()
    demonstrate_smart_preloading()
    demonstrate_memory_optimization()
    
    # Final cleanup
    force_global_cleanup()
    final_stats = get_global_memory_stats()
    print(f"Final memory stats: {final_stats}")
    print("\nAll demonstrations completed successfully!")


if __name__ == "__main__":
    main()