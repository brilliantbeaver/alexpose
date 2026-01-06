# GAVD CSV Processing Scripts Documentation

This directory contains two Python scripts for processing and verifying the GAVD (Gait Analysis Video Dataset) clinical annotations CSV file. These scripts help organize the large dataset into manageable, structured folders based on gait patterns.

## Overview

The GAVD dataset contains clinical annotations for gait analysis videos with multiple frames per sequence across different gait patterns. These scripts automate the process of:

1. **Splitting** the large CSV file into smaller, organized files
2. **Grouping** sequences by gait pattern type
3. **Verifying** the processing results for data integrity

## Files in this Directory

- `process_gavd.py` - Main processing script
- `verify_processing.py` - Verification and statistics script
- `instructions.md` - This documentation file
- Various gait pattern folders (created after processing)

---

## Script 1: process_gavd.py

### Purpose
Processes the main `GAVD_Clinical_Annotations_1.csv` file and organizes it into a structured folder hierarchy based on gait patterns and sequence IDs.

### What it does
1. **Analyzes** the source CSV to identify unique gait patterns and sequences
2. **Creates** a folder structure with subfolders for each gait pattern
3. **Splits** data by sequence ID into separate CSV files
4. **Preserves** the original column structure in all output files

### Input Requirements
- Source file: `data/GAVD_Clinical_Annotations_1.csv`
- The CSV must contain columns: `seq`, `gait_pat`, and other standard GAVD columns

### Output Structure
```
data/gavd/
├── abnormal/
│   ├── sequence1.csv
│   ├── sequence2.csv
│   └── ...
├── normal/
│   ├── sequence3.csv
│   └── ...
├── parkinsons/
├── stroke/
└── [other gait patterns]/
```

### Usage

#### Basic Usage
```bash
# Run from project root directory
python data/gavd/process_gavd.py
```

#### Step-by-Step Process
1. **Navigate to project root**:
   ```bash
   cd /path/to/your/project
   ```

2. **Ensure the source file exists**:
   ```bash
   # Check that this file exists
   ls data/GAVD_Clinical_Annotations_1.csv
   ```

3. **Run the processing script**:
   ```bash
   python data/gavd/process_gavd.py
   ```

4. **Monitor progress**:
   The script will display progress updates every 10,000 rows processed.

#### Expected Output
```
Analyzing data/GAVD_Clinical_Annotations_1.csv...
Processed 0 rows...
Processed 10000 rows...
...
Analysis complete:
Total unique gait patterns: 11
Total unique sequences: 374
Gait patterns found: ['abnormal', 'antalgic', 'cerebral palsy', ...]

Created directory: data\gavd
Created directory: data\gavd\abnormal
...
Processing complete!
Created 11 gait pattern folders
Generated 374 individual CSV files
```

### Error Handling
- **File not found**: Script will exit with error message if source CSV is missing
- **Permission errors**: Ensure write permissions to the `data/` directory
- **Memory issues**: For very large files, the script processes data in chunks

### Customization Options

#### Modify Source File Path
Edit line 95 in `process_gavd.py`:
```python
csv_file = 'path/to/your/source/file.csv'
```

#### Change Output Directory
Edit line 96 in `process_gavd.py`:
```python
base_path = 'your/output/directory'
```

---

## Script 2: verify_processing.py

### Purpose
Verifies that the CSV processing completed successfully and provides detailed statistics about the organized data.

### What it does
1. **Counts** files and rows in each gait pattern folder
2. **Verifies** header consistency across all generated files
3. **Provides** detailed statistics and summaries
4. **Identifies** any processing issues or inconsistencies

### Usage

#### Basic Verification
```bash
# Run from project root directory
python data/gavd/verify_processing.py
```

#### What to Expect
The script will output:
1. **Folder Summary**: Number of files and rows per gait pattern
2. **Total Statistics**: Overall counts and totals
3. **Header Verification**: Confirms all files have correct column structure
4. **Detailed Statistics**: Top sequences by frame count in each category

#### Sample Output
```
Found 11 gait pattern folders:
  abnormal: 190 files, 51031 total rows
  antalgic: 4 files, 975 total rows
  cerebral palsy: 16 files, 5546 total rows
  ...

Summary:
Total gait pattern folders: 11
Total CSV files created: 374
Total data rows processed: 91624

Verifying header consistency...
  ✓ abnormal: Header correct
  ✓ antalgic: Header correct
  ...

✓ All headers are consistent and correct!

Detailed Statistics:
==================================================

ABNORMAL (190 sequences):
  Largest sequences:
    1. cljaqrnsq00ae3n6lo53132n5: 1226 frames
    2. cljaqfh2m009g3n6l1e9tgem1: 1169 frames
    ...
```

### Interpreting Results

#### Success Indicators
- ✅ All gait pattern folders found
- ✅ File counts match expected numbers
- ✅ All headers marked as "correct"
- ✅ Total row count matches original dataset

#### Warning Signs
- ⚠️ Missing folders or files
- ⚠️ Header mismatch warnings
- ⚠️ Unexpected row counts
- ⚠️ Permission or access errors

---

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. "File not found" Error
**Problem**: `GAVD_Clinical_Annotations_1.csv` not found
**Solution**: 
```bash
# Check file location
ls data/GAVD_Clinical_Annotations_1.csv

# If in different location, update script path or move file
mv /path/to/file/GAVD_Clinical_Annotations_1.csv data/
```

#### 2. Permission Denied
**Problem**: Cannot create directories or files
**Solution**:
```bash
# Check permissions
ls -la data/

# Fix permissions if needed
chmod 755 data/
```

#### 3. Memory Issues
**Problem**: Script runs out of memory with very large files
**Solution**: The script is designed to handle large files efficiently, but if issues persist:
- Close other applications
- Run on a machine with more RAM
- Process in smaller batches (modify script)

#### 4. Incomplete Processing
**Problem**: Script stops before completion
**Solution**:
```bash
# Check for partial results
python data/gavd/verify_processing.py

# Re-run processing if needed
python data/gavd/process_gavd.py
```

#### 5. Header Mismatch Warnings
**Problem**: Verification shows header inconsistencies
**Solution**:
- Check source file format
- Ensure CSV uses standard delimiters
- Re-run processing script

### Data Validation

#### Verify Row Counts
```bash
# Count rows in original file (subtract 1 for header)
wc -l data/GAVD_Clinical_Annotations_1.csv

# Compare with verification script output
python data/gavd/verify_processing.py
```

#### Check File Integrity
```bash
# Verify a sample file has correct structure
head -5 data/gavd/normal/[any_sequence_file].csv
```

---

## Advanced Usage

### Batch Processing Multiple Files
If you have multiple GAVD annotation files:

1. **Modify the script** to accept command-line arguments:
```python
import sys
csv_file = sys.argv[1] if len(sys.argv) > 1 else 'data/GAVD_Clinical_Annotations_1.csv'
```

2. **Run for each file**:
```bash
python data/gavd/process_gavd.py data/GAVD_Clinical_Annotations_1.csv
python data/gavd/process_gavd.py data/GAVD_Clinical_Annotations_2.csv
```

### Custom Analysis
Use the verification script as a base for custom analysis:

```python
# Add to verify_processing.py for custom metrics
def analyze_frame_distribution():
    # Your custom analysis code here
    pass
```

### Integration with Other Tools
The processed CSV files can be easily integrated with:
- **Pandas**: `pd.read_csv('data/gavd/normal/sequence.csv')`
- **Machine Learning pipelines**: Organized by gait pattern for classification
- **Video processing tools**: Using the sequence IDs to match with video files

---

## File Format Specifications

### Input CSV Format
The source CSV must contain these columns:
- `seq`: Sequence identifier (string)
- `frame_num`: Frame number within sequence (integer)
- `cam_view`: Camera view angle (string)
- `gait_event`: Gait event marker (string, may be empty)
- `dataset`: Dataset category (string)
- `gait_pat`: Gait pattern classification (string)
- `bbox`: Bounding box coordinates (JSON string)
- `vid_info`: Video metadata (JSON string)
- `id`: Video ID (string)
- `url`: Source URL (string)

### Output CSV Format
Each output CSV file:
- Contains the same column structure as input
- Includes only rows for a single sequence ID
- Is named using the sequence ID: `{sequence_id}.csv`
- Is placed in the folder corresponding to its gait pattern

---

## Performance Notes

### Processing Time
- **Small datasets** (< 10K rows): < 1 minute
- **Medium datasets** (10K-100K rows): 1-5 minutes
- **Large datasets** (> 100K rows): 5-15 minutes

### Memory Usage
- Script uses approximately 2-3x the source file size in RAM
- For a 50MB CSV file, expect ~100-150MB memory usage

### Disk Space
- Output files will be approximately the same total size as input
- Additional space needed for folder structure overhead

---

## Support and Maintenance

### Regular Maintenance
1. **Verify processing** after each run
2. **Backup** original data before processing
3. **Monitor** disk space usage
4. **Update** scripts if CSV format changes

### Getting Help
If you encounter issues:
1. Check this documentation first
2. Run the verification script to identify problems
3. Check file permissions and paths
4. Ensure Python environment has required modules (csv, os)

### Contributing
To improve these scripts:
1. Test changes on a small subset of data first
2. Update documentation for any new features
3. Maintain backward compatibility when possible
4. Add appropriate error handling for new functionality

---

## Version History

- **v1.0**: Initial implementation with basic processing and verification
- Current version handles all standard GAVD CSV formats and provides comprehensive error checking

---

*Last updated: January 2026*