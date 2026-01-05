#!/bin/bash

# Coverage Reporter Script
# Generates comprehensive coverage reports and uploads to external services

set -e  # Exit on any error

# Configuration
COVERAGE_THRESHOLD=${COVERAGE_THRESHOLD:-80}
OUTPUT_DIR=${OUTPUT_DIR:-"coverage_reports"}
CODECOV_TOKEN=${CODECOV_TOKEN:-""}
UPLOAD_TO_CODECOV=${UPLOAD_TO_CODECOV:-false}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to setup output directory
setup_output_dir() {
    log_info "Setting up output directory: $OUTPUT_DIR"
    
    mkdir -p "$OUTPUT_DIR"
    
    # Create subdirectories
    mkdir -p "$OUTPUT_DIR/html"
    mkdir -p "$OUTPUT_DIR/xml"
    mkdir -p "$OUTPUT_DIR/json"
    mkdir -p "$OUTPUT_DIR/reports"
    
    log_success "Output directory setup completed"
}

# Function to run coverage analysis
run_coverage_analysis() {
    log_info "Running coverage analysis..."
    
    # Run tests with coverage
    uv run pytest \
        --cov=ambient \
        --cov=server \
        --cov-report=html:"$OUTPUT_DIR/html" \
        --cov-report=xml:"$OUTPUT_DIR/xml/coverage.xml" \
        --cov-report=json:"$OUTPUT_DIR/json/coverage.json" \
        --cov-report=term-missing \
        --cov-fail-under=$COVERAGE_THRESHOLD \
        -q
    
    log_success "Coverage analysis completed"
}

# Function to generate detailed coverage report
generate_detailed_report() {
    log_info "Generating detailed coverage report..."
    
    # Use our custom coverage analyzer
    uv run python tests/coverage/coverage_analyzer.py > "$OUTPUT_DIR/reports/coverage_analysis.md"
    
    # Generate JSON report for programmatic access
    uv run python -c "
import json
from tests.coverage.coverage_analyzer import CoverageAnalyzer

analyzer = CoverageAnalyzer()
analysis = analyzer.run_coverage_analysis()

with open('$OUTPUT_DIR/reports/coverage_analysis.json', 'w') as f:
    json.dump(analysis, f, indent=2)

print('Detailed coverage report generated')
"
    
    log_success "Detailed coverage report generated"
}

# Function to generate component-specific reports
generate_component_reports() {
    log_info "Generating component-specific coverage reports..."
    
    # Core components
    uv run pytest \
        --cov=ambient/core \
        --cov-report=html:"$OUTPUT_DIR/html/core" \
        --cov-report=xml:"$OUTPUT_DIR/xml/core_coverage.xml" \
        -q tests/ambient/core/ || true
    
    # Domain components
    uv run pytest \
        --cov=ambient/analysis \
        --cov=ambient/classification \
        --cov=ambient/gavd \
        --cov-report=html:"$OUTPUT_DIR/html/domain" \
        --cov-report=xml:"$OUTPUT_DIR/xml/domain_coverage.xml" \
        -q tests/ambient/ || true
    
    # Integration components
    uv run pytest \
        --cov=server \
        --cov-report=html:"$OUTPUT_DIR/html/integration" \
        --cov-report=xml:"$OUTPUT_DIR/xml/integration_coverage.xml" \
        -q tests/integration/ || true
    
    log_success "Component-specific reports generated"
}

# Function to generate coverage badges
generate_coverage_badges() {
    log_info "Generating coverage badges..."
    
    # Install coverage-badge if not available
    if ! command_exists coverage-badge; then
        uv add coverage-badge
    fi
    
    # Generate badge
    if [ -f "$OUTPUT_DIR/xml/coverage.xml" ]; then
        uv run coverage-badge -f -o "$OUTPUT_DIR/coverage-badge.svg"
        log_success "Coverage badge generated"
    else
        log_warning "No coverage XML file found for badge generation"
    fi
}

# Function to create coverage summary
create_coverage_summary() {
    log_info "Creating coverage summary..."
    
    cat > "$OUTPUT_DIR/coverage_summary.md" << EOF
# Coverage Report Summary

Generated on: $(date)

## Overall Coverage

EOF
    
    # Extract overall coverage from XML if available
    if [ -f "$OUTPUT_DIR/xml/coverage.xml" ]; then
        OVERALL_COVERAGE=$(python3 -c "
import xml.etree.ElementTree as ET
try:
    tree = ET.parse('$OUTPUT_DIR/xml/coverage.xml')
    root = tree.getroot()
    line_rate = float(root.attrib.get('line-rate', 0))
    print(f'{line_rate * 100:.1f}%')
except:
    print('N/A')
")
        
        echo "- **Overall Coverage**: $OVERALL_COVERAGE" >> "$OUTPUT_DIR/coverage_summary.md"
    fi
    
    cat >> "$OUTPUT_DIR/coverage_summary.md" << EOF

## Component Coverage

- **Core Components**: See core_coverage.xml
- **Domain Components**: See domain_coverage.xml  
- **Integration Components**: See integration_coverage.xml

## Files Generated

- \`html/\`: HTML coverage reports
- \`xml/\`: XML coverage data
- \`json/\`: JSON coverage data
- \`reports/\`: Detailed analysis reports
- \`coverage-badge.svg\`: Coverage badge

## Coverage Targets

- Core (Frame, Config, Interfaces): 95%
- Domain (Pose, Gait, Classification): 90%
- Integration (API, Video, Storage): 85%
- Overall System: 85%

EOF
    
    log_success "Coverage summary created"
}

# Function to upload to Codecov
upload_to_codecov() {
    if [ "$UPLOAD_TO_CODECOV" = true ] && [ -n "$CODECOV_TOKEN" ]; then
        log_info "Uploading coverage to Codecov..."
        
        # Install codecov if not available
        if ! command_exists codecov; then
            pip install codecov
        fi
        
        # Upload coverage
        codecov -f "$OUTPUT_DIR/xml/coverage.xml" -t "$CODECOV_TOKEN"
        
        log_success "Coverage uploaded to Codecov"
    else
        log_info "Skipping Codecov upload (not configured)"
    fi
}

# Function to validate coverage thresholds
validate_coverage_thresholds() {
    log_info "Validating coverage thresholds..."
    
    if [ -f "$OUTPUT_DIR/reports/coverage_analysis.json" ]; then
        VALIDATION_RESULT=$(python3 -c "
import json
import sys

with open('$OUTPUT_DIR/reports/coverage_analysis.json', 'r') as f:
    analysis = json.load(f)

if not analysis.get('success', False):
    print('Coverage analysis failed')
    sys.exit(1)

targets_met = analysis.get('targets_met', {})
failed_targets = [comp for comp, met in targets_met.items() if not met]

if failed_targets:
    print(f'Coverage targets not met for: {', '.join(failed_targets)}')
    sys.exit(1)
else:
    print('All coverage targets met')
    sys.exit(0)
")
        
        if [ $? -eq 0 ]; then
            log_success "Coverage thresholds validation passed"
        else
            log_error "Coverage thresholds validation failed: $VALIDATION_RESULT"
            return 1
        fi
    else
        log_warning "No coverage analysis data available for validation"
    fi
}

# Function to create archive
create_coverage_archive() {
    log_info "Creating coverage archive..."
    
    ARCHIVE_NAME="coverage_report_$(date +%Y%m%d_%H%M%S).tar.gz"
    
    tar -czf "$ARCHIVE_NAME" "$OUTPUT_DIR"
    
    log_success "Coverage archive created: $ARCHIVE_NAME"
}

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -t, --threshold THRESHOLD    Coverage threshold (default: 80)"
    echo "  -o, --output-dir DIR         Output directory (default: coverage_reports)"
    echo "  -u, --upload-codecov         Upload to Codecov (requires CODECOV_TOKEN)"
    echo "  -a, --archive                Create coverage archive"
    echo "  -v, --validate               Validate coverage thresholds"
    echo "  -h, --help                   Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  CODECOV_TOKEN               Token for Codecov upload"
    echo "  COVERAGE_THRESHOLD          Coverage threshold percentage"
    echo ""
    echo "Examples:"
    echo "  $0 --threshold 85 --upload-codecov"
    echo "  $0 --output-dir reports --validate --archive"
}

# Parse command line arguments
VALIDATE_THRESHOLDS=false
CREATE_ARCHIVE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--threshold)
            COVERAGE_THRESHOLD="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -u|--upload-codecov)
            UPLOAD_TO_CODECOV=true
            shift
            ;;
        -a|--archive)
            CREATE_ARCHIVE=true
            shift
            ;;
        -v|--validate)
            VALIDATE_THRESHOLDS=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Main execution
main() {
    log_info "Starting coverage reporter..."
    log_info "Coverage threshold: $COVERAGE_THRESHOLD%"
    log_info "Output directory: $OUTPUT_DIR"
    
    # Setup
    setup_output_dir
    
    # Run coverage analysis
    run_coverage_analysis
    
    # Generate reports
    generate_detailed_report
    generate_component_reports
    generate_coverage_badges
    create_coverage_summary
    
    # Upload to external services
    upload_to_codecov
    
    # Validation
    if [ "$VALIDATE_THRESHOLDS" = true ]; then
        validate_coverage_thresholds
    fi
    
    # Create archive
    if [ "$CREATE_ARCHIVE" = true ]; then
        create_coverage_archive
    fi
    
    log_success "Coverage reporting completed!"
    log_info "Reports available in: $OUTPUT_DIR"
}

# Run main function
main "$@"