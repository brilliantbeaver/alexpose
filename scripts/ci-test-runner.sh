#!/bin/bash

# CI Test Runner Script
# Provides standardized test execution for CI/CD environments

set -e  # Exit on any error

# Configuration
PYTHON_VERSION=${PYTHON_VERSION:-"3.12"}
TEST_CATEGORY=${TEST_CATEGORY:-"all"}
COVERAGE_THRESHOLD=${COVERAGE_THRESHOLD:-80}
HYPOTHESIS_PROFILE=${HYPOTHESIS_PROFILE:-"ci"}
MAX_TEST_DURATION=${MAX_TEST_DURATION:-300}  # 5 minutes

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

# Function to setup environment
setup_environment() {
    log_info "Setting up test environment..."
    
    # Check for required tools
    if ! command_exists uv; then
        log_error "uv is not installed. Please install uv first."
        exit 1
    fi
    
    if ! command_exists python; then
        log_error "Python is not installed."
        exit 1
    fi
    
    # Install dependencies
    log_info "Installing dependencies..."
    uv sync --all-extras --dev
    
    # Set up environment variables
    export HYPOTHESIS_PROFILE=$HYPOTHESIS_PROFILE
    export PYTHONPATH="${PYTHONPATH}:$(pwd)"
    
    log_success "Environment setup completed"
}

# Function to run fast tests
run_fast_tests() {
    log_info "Running fast tests..."
    
    uv run pytest \
        -m "fast" \
        --tb=short \
        --cov=ambient \
        --cov=server \
        --cov-report=xml \
        --cov-report=html:htmlcov \
        --cov-report=term-missing \
        --cov-fail-under=$COVERAGE_THRESHOLD \
        --maxfail=5 \
        --timeout=$MAX_TEST_DURATION \
        -v
    
    log_success "Fast tests completed"
}

# Function to run slow tests
run_slow_tests() {
    log_info "Running slow tests..."
    
    uv run pytest \
        -m "slow and not performance" \
        --tb=short \
        --cov=ambient \
        --cov=server \
        --cov-report=xml \
        --cov-report=html:htmlcov \
        --cov-report=term-missing \
        --maxfail=3 \
        --timeout=$((MAX_TEST_DURATION * 2)) \
        -v
    
    log_success "Slow tests completed"
}

# Function to run integration tests
run_integration_tests() {
    log_info "Running integration tests..."
    
    uv run pytest \
        -m "integration" \
        --tb=short \
        --cov=ambient \
        --cov=server \
        --cov-report=xml \
        --cov-report=html:htmlcov \
        --cov-report=term-missing \
        --maxfail=3 \
        --timeout=$((MAX_TEST_DURATION * 3)) \
        -v
    
    log_success "Integration tests completed"
}

# Function to run property-based tests
run_property_tests() {
    log_info "Running property-based tests with profile: $HYPOTHESIS_PROFILE..."
    
    uv run pytest \
        -m "property" \
        --hypothesis-profile=$HYPOTHESIS_PROFILE \
        --tb=short \
        --maxfail=5 \
        --timeout=$((MAX_TEST_DURATION * 2)) \
        -v
    
    log_success "Property-based tests completed"
}

# Function to run performance tests
run_performance_tests() {
    log_info "Running performance tests..."
    
    uv run pytest \
        -m "performance" \
        --tb=short \
        --maxfail=3 \
        --timeout=$((MAX_TEST_DURATION * 4)) \
        -v
    
    log_success "Performance tests completed"
}

# Function to run all tests
run_all_tests() {
    log_info "Running all tests..."
    
    # Run tests in order of increasing duration
    run_fast_tests
    run_property_tests
    run_slow_tests
    run_integration_tests
    
    log_success "All tests completed"
}

# Function to run quality gates
run_quality_gates() {
    log_info "Running quality gates..."
    
    uv run python tests/quality/quality_gates.py --report quality_report.json
    
    if [ $? -eq 0 ]; then
        log_success "Quality gates passed"
    else
        log_error "Quality gates failed"
        return 1
    fi
}

# Function to generate coverage report
generate_coverage_report() {
    log_info "Generating coverage report..."
    
    if [ -f "coverage.xml" ]; then
        uv run python tests/coverage/coverage_analyzer.py
        log_success "Coverage report generated"
    else
        log_warning "No coverage data found"
    fi
}

# Function to cleanup
cleanup() {
    log_info "Cleaning up..."
    
    # Remove temporary files
    rm -f .coverage
    rm -f test-results.xml
    rm -f pytest.ini.bak
    
    # Clean up Python cache
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    
    log_success "Cleanup completed"
}

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -c, --category CATEGORY    Test category to run (fast|slow|integration|property|performance|all)"
    echo "  -t, --threshold THRESHOLD  Coverage threshold (default: 80)"
    echo "  -p, --profile PROFILE      Hypothesis profile (dev|ci|thorough)"
    echo "  -d, --duration DURATION    Max test duration in seconds (default: 300)"
    echo "  -q, --quality-gates        Run quality gates after tests"
    echo "  -r, --coverage-report      Generate coverage report"
    echo "  -h, --help                 Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --category fast --quality-gates"
    echo "  $0 --category all --threshold 85 --coverage-report"
    echo "  $0 --category property --profile thorough"
}

# Parse command line arguments
QUALITY_GATES=false
COVERAGE_REPORT=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--category)
            TEST_CATEGORY="$2"
            shift 2
            ;;
        -t|--threshold)
            COVERAGE_THRESHOLD="$2"
            shift 2
            ;;
        -p|--profile)
            HYPOTHESIS_PROFILE="$2"
            shift 2
            ;;
        -d|--duration)
            MAX_TEST_DURATION="$2"
            shift 2
            ;;
        -q|--quality-gates)
            QUALITY_GATES=true
            shift
            ;;
        -r|--coverage-report)
            COVERAGE_REPORT=true
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
    log_info "Starting CI test runner..."
    log_info "Test category: $TEST_CATEGORY"
    log_info "Coverage threshold: $COVERAGE_THRESHOLD%"
    log_info "Hypothesis profile: $HYPOTHESIS_PROFILE"
    log_info "Max test duration: ${MAX_TEST_DURATION}s"
    
    # Setup environment
    setup_environment
    
    # Run tests based on category
    case $TEST_CATEGORY in
        fast)
            run_fast_tests
            ;;
        slow)
            run_slow_tests
            ;;
        integration)
            run_integration_tests
            ;;
        property)
            run_property_tests
            ;;
        performance)
            run_performance_tests
            ;;
        all)
            run_all_tests
            ;;
        *)
            log_error "Invalid test category: $TEST_CATEGORY"
            usage
            exit 1
            ;;
    esac
    
    # Run quality gates if requested
    if [ "$QUALITY_GATES" = true ]; then
        run_quality_gates
    fi
    
    # Generate coverage report if requested
    if [ "$COVERAGE_REPORT" = true ]; then
        generate_coverage_report
    fi
    
    # Cleanup
    cleanup
    
    log_success "CI test runner completed successfully!"
}

# Trap to ensure cleanup on exit
trap cleanup EXIT

# Run main function
main "$@"