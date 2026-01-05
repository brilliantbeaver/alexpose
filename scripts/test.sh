#!/bin/bash

# AlexPose Testing Framework - Comprehensive Test Execution Script
# Based on the Testing Strategy and Enhancement Specification

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if uv is available
check_uv() {
    if command -v uv &> /dev/null; then
        PYTHON_CMD="uv run python"
        PYTEST_CMD="uv run pytest"
    else
        print_warning "uv not found, falling back to system python"
        PYTHON_CMD="python"
        PYTEST_CMD="pytest"
    fi
}

# Function to display help
show_help() {
    echo "AlexPose Testing Framework"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  fast          Run fast tests only (< 1 second each) - Development workflow"
    echo "  slow          Run slow tests (1-30 seconds each)"
    echo "  performance   Run performance tests (30+ seconds) - Benchmarking"
    echo "  integration   Run integration tests with external resources"
    echo "  property      Run property-based tests with Hypothesis"
    echo "  unit          Run unit tests only"
    echo "  e2e           Run end-to-end tests"
    echo "  all           Run all tests except performance"
    echo "  ci            Run CI/CD test suite with coverage"
    echo "  coverage      Generate coverage report"
    echo "  clean         Clean test artifacts and cache"
    echo ""
    echo "Options:"
    echo "  --profile PROFILE    Set Hypothesis profile (dev, ci, thorough)"
    echo "  --parallel           Run tests in parallel using pytest-xdist"
    echo "  --verbose            Verbose output"
    echo "  --no-cov             Skip coverage reporting"
    echo "  --fail-fast          Stop on first failure"
    echo ""
    echo "Examples:"
    echo "  $0 fast                    # Quick development tests"
    echo "  $0 property --profile ci   # Property tests with CI profile"
    echo "  $0 all --parallel          # All tests in parallel"
    echo "  $0 ci                      # Full CI test suite"
}

# Function to run fast tests (development workflow)
run_fast_tests() {
    print_status "Running fast tests (< 1 second each) - Development workflow"
    
    local cmd="$PYTEST_CMD -v -m \"fast or (not slow and not performance and not integration)\""
    
    if [[ "$PARALLEL" == "true" ]]; then
        cmd="$cmd -n auto"
    fi
    
    if [[ "$NO_COV" == "true" ]]; then
        cmd="$cmd --no-cov"
    fi
    
    if [[ "$FAIL_FAST" == "true" ]]; then
        cmd="$cmd -x"
    fi
    
    if [[ "$VERBOSE" == "true" ]]; then
        cmd="$cmd -vv"
    fi
    
    eval $cmd
    print_success "Fast tests completed"
}

# Function to run slow tests
run_slow_tests() {
    print_status "Running slow tests (1-30 seconds each)"
    
    local cmd="$PYTEST_CMD -v -m slow"
    
    if [[ "$PARALLEL" == "true" ]]; then
        cmd="$cmd -n auto"
    fi
    
    if [[ "$NO_COV" == "true" ]]; then
        cmd="$cmd --no-cov"
    fi
    
    eval $cmd
    print_success "Slow tests completed"
}

# Function to run performance tests
run_performance_tests() {
    print_status "Running performance tests (30+ seconds) - Benchmarking"
    print_warning "Performance tests may take several minutes to complete"
    
    local cmd="$PYTEST_CMD -v -m performance --benchmark-only"
    
    if [[ "$NO_COV" == "true" ]]; then
        cmd="$cmd --no-cov"
    fi
    
    eval $cmd
    print_success "Performance tests completed"
}

# Function to run integration tests
run_integration_tests() {
    print_status "Running integration tests with external resources"
    
    local cmd="$PYTEST_CMD -v -m integration"
    
    if [[ "$PARALLEL" == "true" ]]; then
        cmd="$cmd -n auto"
    fi
    
    if [[ "$NO_COV" == "true" ]]; then
        cmd="$cmd --no-cov"
    fi
    
    eval $cmd
    print_success "Integration tests completed"
}

# Function to run property-based tests
run_property_tests() {
    print_status "Running property-based tests with Hypothesis"
    
    local profile=${HYPOTHESIS_PROFILE:-"dev"}
    print_status "Using Hypothesis profile: $profile"
    
    local cmd="$PYTEST_CMD -v -m property --hypothesis-profile=$profile"
    
    if [[ "$PARALLEL" == "true" ]]; then
        cmd="$cmd -n auto"
    fi
    
    if [[ "$NO_COV" == "true" ]]; then
        cmd="$cmd --no-cov"
    fi
    
    eval $cmd
    print_success "Property-based tests completed"
}

# Function to run unit tests
run_unit_tests() {
    print_status "Running unit tests"
    
    local cmd="$PYTEST_CMD -v -m \"unit or (not integration and not e2e and not performance)\""
    
    if [[ "$PARALLEL" == "true" ]]; then
        cmd="$cmd -n auto"
    fi
    
    if [[ "$NO_COV" == "true" ]]; then
        cmd="$cmd --no-cov"
    fi
    
    eval $cmd
    print_success "Unit tests completed"
}

# Function to run end-to-end tests
run_e2e_tests() {
    print_status "Running end-to-end tests"
    
    local cmd="$PYTEST_CMD -v -m e2e"
    
    if [[ "$NO_COV" == "true" ]]; then
        cmd="$cmd --no-cov"
    fi
    
    eval $cmd
    print_success "End-to-end tests completed"
}

# Function to run all tests except performance
run_all_tests() {
    print_status "Running all tests except performance"
    
    local cmd="$PYTEST_CMD -v -m \"not performance\""
    
    if [[ "$PARALLEL" == "true" ]]; then
        cmd="$cmd -n auto"
    fi
    
    if [[ "$NO_COV" == "true" ]]; then
        cmd="$cmd --no-cov"
    fi
    
    eval $cmd
    print_success "All tests completed"
}

# Function to run CI test suite
run_ci_tests() {
    print_status "Running CI/CD test suite with comprehensive coverage"
    
    # Run tests in stages for better CI feedback
    print_status "Stage 1: Fast tests"
    run_fast_tests
    
    print_status "Stage 2: Unit tests"
    run_unit_tests
    
    print_status "Stage 3: Property-based tests (CI profile)"
    HYPOTHESIS_PROFILE="ci" run_property_tests
    
    print_status "Stage 4: Integration tests"
    run_integration_tests
    
    print_status "Stage 5: Slow tests"
    run_slow_tests
    
    print_success "CI test suite completed successfully"
}

# Function to generate coverage report
generate_coverage() {
    print_status "Generating comprehensive coverage report"
    
    $PYTEST_CMD --cov=ambient --cov=server --cov-report=html:htmlcov --cov-report=term-missing --cov-report=xml
    
    print_success "Coverage report generated in htmlcov/"
    
    if command -v open &> /dev/null; then
        print_status "Opening coverage report in browser"
        open htmlcov/index.html
    elif command -v xdg-open &> /dev/null; then
        print_status "Opening coverage report in browser"
        xdg-open htmlcov/index.html
    fi
}

# Function to clean test artifacts
clean_artifacts() {
    print_status "Cleaning test artifacts and cache"
    
    # Remove pytest cache
    rm -rf .pytest_cache
    
    # Remove coverage files
    rm -rf htmlcov
    rm -f .coverage
    rm -f coverage.xml
    
    # Remove Python cache
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    
    # Remove Hypothesis cache
    rm -rf .hypothesis
    
    print_success "Test artifacts cleaned"
}

# Parse command line arguments
COMMAND=""
HYPOTHESIS_PROFILE="dev"
PARALLEL="false"
VERBOSE="false"
NO_COV="false"
FAIL_FAST="false"

while [[ $# -gt 0 ]]; do
    case $1 in
        fast|slow|performance|integration|property|unit|e2e|all|ci|coverage|clean)
            COMMAND="$1"
            shift
            ;;
        --profile)
            HYPOTHESIS_PROFILE="$2"
            shift 2
            ;;
        --parallel)
            PARALLEL="true"
            shift
            ;;
        --verbose)
            VERBOSE="true"
            shift
            ;;
        --no-cov)
            NO_COV="true"
            shift
            ;;
        --fail-fast)
            FAIL_FAST="true"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Check for required tools
check_uv

# Execute command
case $COMMAND in
    fast)
        run_fast_tests
        ;;
    slow)
        run_slow_tests
        ;;
    performance)
        run_performance_tests
        ;;
    integration)
        run_integration_tests
        ;;
    property)
        run_property_tests
        ;;
    unit)
        run_unit_tests
        ;;
    e2e)
        run_e2e_tests
        ;;
    all)
        run_all_tests
        ;;
    ci)
        run_ci_tests
        ;;
    coverage)
        generate_coverage
        ;;
    clean)
        clean_artifacts
        ;;
    "")
        print_error "No command specified"
        show_help
        exit 1
        ;;
    *)
        print_error "Unknown command: $COMMAND"
        show_help
        exit 1
        ;;
esac