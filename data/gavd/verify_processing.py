#!/usr/bin/env python3
"""
Script to verify the GAVD CSV processing was successful.

This script:
1. Checks that all gait pattern folders were created
2. Counts the number of CSV files and rows in each folder
3. Verifies header consistency across all generated files
4. Provides a summary of the processing results

Usage: python verify_processing.py
"""

import os
import csv
from collections import defaultdict

def verify_processing():
    """Verify that the CSV processing was successful."""
    gavd_path = 'data/gavd'
    
    if not os.path.exists(gavd_path):
        print("Error: gavd folder not found!")
        return False
    
    # Get all gait pattern folders
    gait_folders = [f for f in os.listdir(gavd_path) 
                   if os.path.isdir(os.path.join(gavd_path, f)) and not f.startswith('.')]
    print(f"Found {len(gait_folders)} gait pattern folders:")
    
    total_files = 0
    total_rows = 0
    
    for folder in sorted(gait_folders):
        folder_path = os.path.join(gavd_path, folder)
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        
        folder_rows = 0
        for csv_file in csv_files:
            csv_path = os.path.join(folder_path, csv_file)
            with open(csv_path, 'r', newline='', encoding='utf-8') as file:
                reader = csv.reader(file)
                rows = sum(1 for row in reader) - 1  # Subtract header row
                folder_rows += rows
        
        print(f"  {folder}: {len(csv_files)} files, {folder_rows} total rows")
        total_files += len(csv_files)
        total_rows += folder_rows
    
    print(f"\nSummary:")
    print(f"Total gait pattern folders: {len(gait_folders)}")
    print(f"Total CSV files created: {total_files}")
    print(f"Total data rows processed: {total_rows}")
    
    # Verify header consistency
    print(f"\nVerifying header consistency...")
    expected_header = ['seq', 'frame_num', 'cam_view', 'gait_event', 'dataset', 'gait_pat', 'bbox', 'vid_info', 'id', 'url']
    
    header_issues = 0
    for folder in gait_folders:
        folder_path = os.path.join(gavd_path, folder)
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        
        if csv_files:  # Check first file in each folder
            csv_path = os.path.join(folder_path, csv_files[0])
            with open(csv_path, 'r', newline='', encoding='utf-8') as file:
                reader = csv.reader(file)
                header = next(reader)
                if header != expected_header:
                    print(f"  WARNING: Header mismatch in {folder}/{csv_files[0]}")
                    print(f"    Expected: {expected_header}")
                    print(f"    Found: {header}")
                    header_issues += 1
                else:
                    print(f"  ✓ {folder}: Header correct")
    
    if header_issues == 0:
        print(f"\n✓ All headers are consistent and correct!")
    else:
        print(f"\n⚠ Found {header_issues} header issues")
    
    return header_issues == 0

def get_detailed_stats():
    """Get detailed statistics about the processed data."""
    gavd_path = 'data/gavd'
    
    if not os.path.exists(gavd_path):
        print("Error: gavd folder not found!")
        return
    
    print("\nDetailed Statistics:")
    print("=" * 50)
    
    gait_folders = [f for f in os.listdir(gavd_path) 
                   if os.path.isdir(os.path.join(gavd_path, f)) and not f.startswith('.')]
    
    for folder in sorted(gait_folders):
        folder_path = os.path.join(gavd_path, folder)
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        
        if not csv_files:
            continue
            
        print(f"\n{folder.upper()} ({len(csv_files)} sequences):")
        
        # Get row counts for each file
        file_stats = []
        for csv_file in csv_files:
            csv_path = os.path.join(folder_path, csv_file)
            with open(csv_path, 'r', newline='', encoding='utf-8') as file:
                reader = csv.reader(file)
                rows = sum(1 for row in reader) - 1  # Subtract header row
                file_stats.append((csv_file, rows))
        
        # Sort by row count
        file_stats.sort(key=lambda x: x[1], reverse=True)
        
        # Show top 5 largest sequences
        print(f"  Largest sequences:")
        for i, (filename, rows) in enumerate(file_stats[:5]):
            seq_id = filename.replace('.csv', '')
            print(f"    {i+1}. {seq_id}: {rows} frames")
        
        if len(file_stats) > 5:
            print(f"    ... and {len(file_stats) - 5} more sequences")

if __name__ == "__main__":
    success = verify_processing()
    
    if success:
        get_detailed_stats()
    else:
        print("Verification failed. Please check the processing.")