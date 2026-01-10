#!/usr/bin/env python3
"""
Script to process GAVD_Clinical_Annotations_1.csv and organize data by gait patterns.

This script:
1. Analyzes the CSV file to find unique gait patterns and sequences
2. Creates a folder structure with subfolders for each gait pattern
3. Splits the data by sequence ID into separate CSV files within appropriate gait pattern folders
4. Preserves the original column structure in all output files

Usage: python process_gavd.py
"""

import csv
import os
from collections import defaultdict
import sys

def analyze_csv_structure(csv_file):
    """Analyze the CSV file to understand unique gait patterns and sequences."""
    gait_patterns = set()
    sequences = set()
    seq_to_gait_pat = {}
    
    print(f"Analyzing {csv_file}...")
    
    with open(csv_file, 'r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        
        for i, row in enumerate(reader):
            if i % 10000 == 0:
                print(f"Processed {i} rows...")
            
            seq = row['seq']
            gait_pat = row['gait_pat']
            
            gait_patterns.add(gait_pat)
            sequences.add(seq)
            seq_to_gait_pat[seq] = gait_pat
    
    print(f"\nAnalysis complete:")
    print(f"Total unique gait patterns: {len(gait_patterns)}")
    print(f"Total unique sequences: {len(sequences)}")
    print(f"Gait patterns found: {sorted(gait_patterns)}")
    
    return gait_patterns, sequences, seq_to_gait_pat

def create_folder_structure(base_path, gait_patterns):
    """Create the folder structure for organizing CSV files."""
    gavd_path = os.path.join(base_path, 'gavd')
    
    # Create main gavd folder
    os.makedirs(gavd_path, exist_ok=True)
    print(f"Created directory: {gavd_path}")
    
    # Create subfolders for each gait pattern
    for pattern in gait_patterns:
        pattern_path = os.path.join(gavd_path, pattern)
        os.makedirs(pattern_path, exist_ok=True)
        print(f"Created directory: {pattern_path}")
    
    return gavd_path

def process_and_split_csv(csv_file, gavd_path, seq_to_gait_pat):
    """Process the CSV file and split it by sequences into appropriate gait pattern folders."""
    
    # Dictionary to store data for each sequence
    sequence_data = defaultdict(list)
    
    print(f"Reading and organizing data from {csv_file}...")
    
    # Read the CSV file and organize data by sequence
    with open(csv_file, 'r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        header = reader.fieldnames
        
        for i, row in enumerate(reader):
            if i % 10000 == 0:
                print(f"Processed {i} rows...")
            
            seq = row['seq']
            sequence_data[seq].append(row)
    
    print(f"Writing individual CSV files for {len(sequence_data)} sequences...")
    
    # Write each sequence to its appropriate CSV file
    for seq, rows in sequence_data.items():
        gait_pat = seq_to_gait_pat[seq]
        
        # Create the file path
        csv_filename = f"{seq}.csv"
        csv_filepath = os.path.join(gavd_path, gait_pat, csv_filename)
        
        # Write the CSV file
        with open(csv_filepath, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"Created: {csv_filepath} with {len(rows)} rows")

def main():
    """Main function to process the GAVD CSV file."""
    csv_file = 'data/GAVD_Clinical_Annotations_1.csv'
    base_path = 'data'
    
    # Check if input file exists
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found!")
        sys.exit(1)
    
    try:
        # Step 1: Analyze the CSV structure
        gait_patterns, sequences, seq_to_gait_pat = analyze_csv_structure(csv_file)
        
        # Step 2: Create folder structure
        gavd_path = create_folder_structure(base_path, gait_patterns)
        
        # Step 3: Process and split the CSV file
        process_and_split_csv(csv_file, gavd_path, seq_to_gait_pat)
        
        print(f"\nProcessing complete!")
        print(f"Created {len(gait_patterns)} gait pattern folders")
        print(f"Generated {len(sequences)} individual CSV files")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()