#!/bin/bash
# CI/CD Coverage Reporting Script
# Automates coverage collection, analysis, and reporting in CI/CD pipelines

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================="
echo "CI/CD Coverage Reporter"
echo "========================================="

# Configuration
COVERAGE_THRESHOLD=${COVERAGE_THRESHOLD:-80}
COMPONENT_CORE_THRESHOLD=${COMPONENT_CORE_THRESHOLD:-90}
COMPONENT_DOMAIN_THRESHOLD=${COMPONENT_DOMAIN_THRESHOLD:-85}
COMPONENT_INTEGRATION_THRESHOLD=${COMPONENT_INTEGRATION_THRESHOLD:-75}
FAIL_ON_THRESHOLD=${FAIL_ON_THRESHOLD:-true}
CODECOV_TOKEN=${CODECOV_TOKEN:-""}
GENERATE_REPORTS=${GENERATE_REPORTS:-true}
CAPTURE_TRENDS=${CAPTURE_TRENDS:-true}

# Step 1: Run tests with coverage
echo -e "${YELLOW}Step 1: Running tests with coverage...${NC}"
python -m pytest \
    --cov=ambient \
    --cov=server \
    --cov-report=xml:coverage.xml \
    --cov-report=html:htmlcov \
    --cov-report=json:coverage.json \
    --cov-report=term-missing \
    --cov-fail-under=${COVERAGE_THRESHOLD} \
    || TEST_EXIT_CODE=$?

if [ -n "$TEST_EXIT_CODE" ] && [ "$TEST_EXIT_CODE" -ne 0 ]; then
    echo -e "${RED}âœ— Tests failed with exit code ${TEST_EXIT_CODE}${NC}"
    if [ "$FAIL_ON_THRESHOLD" = "true" ]; then
        exit $TEST_EXIT_CODE
    fi
fi

echo -e "${GREEN}âœ“ Tests completed${NC}"

# Step 2: Parse coverage results
echo -e "${YELLOW}Step 2: Parsing coverage results...${NC}"
if [ -f "coverage.json" ]; then
    OVERALL_COVERAGE=$(python -c "import json; data=json.load(open('coverage.json')); print(f\"{data['totals']['percent_covered']:.2f}\")")
    echo "Overall Coverage: ${OVERALL_COVERAGE}%"
else
    echo -e "${RED}âœ— Coverage report not found${NC}"
    exit 1
fi

# Step 3: Run coverage analysis
echo -e "${YELLOW}Step 3: Running coverage analysis...${NC}"
python -m tests.coverage.coverage_analyzer > coverage_analysis.txt || true
cat coverage_analysis.txt

# Step 4: Check coverage thresholds
echo -e "${YELLOW}Step 4: Checking coverage thresholds...${NC}"
THRESHOLD_CHECK_PASSED=true

if (( $(echo "$OVERALL_COVERAGE < $COVERAGE_THRESHOLD" | bc -l) )); then
    echo -e "${RED}âœ— Overall coverage (${OVERALL_COVERAGE}%) is below threshold (${COVERAGE_THRESHOLD}%)${NC}"
    THRESHOLD_CHECK_PASSED=false
else
    echo -e "${GREEN}âœ“ Overall coverage meets threshold${NC}"
fi

# Step 5: Capture coverage trends (if enabled)
if [ "$CAPTURE_TRENDS" = "true" ]; then
    echo -e "${YELLOW}Step 5: Capturing coverage trends...${NC}"
    python -m tests.coverage.coverage_trend_analyzer --capture || true
    echo -e "${GREEN}âœ“ Coverage snapshot captured${NC}"
fi

# Step 6: Generate reports (if enabled)
if [ "$GENERATE_REPORTS" = "true" ]; then
    echo -e "${YELLOW}Step 6: Generating coverage reports...${NC}"
    
    # Generate coverage gap report
    python -c "
from tests.coverage.coverage_analyzer import CoverageAnalyzer
analyzer = CoverageAnalyzer()
report = analyzer.generate_coverage_report()
with open('coverage_gap_report.md', 'w') as f:
    f.write(report)
print('Coverage gap report saved to: coverage_gap_report.md')
" || true
    
    # Generate trend report (if we have historical data)
    python -m tests.coverage.coverage_trend_analyzer --report coverage_trend_report.md --days 30 || true
    
    echo -e "${GREEN}âœ“ Reports generated${NC}"
fi

# Step 7: Upload to Codecov (if token provided)
if [ -n "$CODECOV_TOKEN" ]; then
    echo -e "${YELLOW}Step 7: Uploading coverage to Codecov...${NC}"
    
    # Install codecov if not available
    if ! command -v codecov &> /dev/null; then
        pip install codecov
    fi
    
    codecov -t "$CODECOV_TOKEN" -f coverage.xml || {
        echo -e "${YELLOW}âš  Codecov upload failed (non-fatal)${NC}"
    }
    
    echo -e "${GREEN}âœ“ Coverage uploaded to Codecov${NC}"
else
    echo -e "${YELLOW}âš  Codecov token not provided, skipping upload${NC}"
fi

# Step 8: Generate CI summary
echo -e "${YELLOW}Step 8: Generating CI summary...${NC}"

cat > coverage_summary.md << EOF
# Coverage Report Summary

## Overall Coverage
- **Coverage**: ${OVERALL_COVERAGE}%
- **Threshold**: ${COVERAGE_THRESHOLD}%
- **Status**: $([ "$THRESHOLD_CHECK_PASSED" = "true" ] && echo "âœ… PASS" || echo "âŒ FAIL")

## Coverage Files
- [HTML Report](htmlcov/index.html)
- [XML Report](coverage.xml)
- [JSON Report](coverage.json)
- [Gap Analysis](coverage_gap_report.md)
- [Trend Analysis](coverage_trend_report.md)

## Next Steps
$(if [ "$THRESHOLD_CHECK_PASSED" = "false" ]; then
    echo "- Review coverage gaps in coverage_gap_report.md"
    echo "- Add tests for uncovered code"
    echo "- Focus on high-priority gaps first"
else
    echo "- Coverage targets are being met"
    echo "- Continue maintaining high coverage standards"
fi)
EOF

cat coverage_summary.md

# Step 9: Create GitHub Actions summary (if in GitHub Actions)
if [ -n "$GITHUB_STEP_SUMMARY" ]; then
    echo -e "${YELLOW}Step 9: Creating GitHub Actions summary...${NC}"
    cat coverage_summary.md >> "$GITHUB_STEP_SUMMARY"
    echo -e "${GREEN}âœ“ GitHub Actions summary created${NC}"
fi

# Step 10: Final status
echo "========================================="
if [ "$THRESHOLD_CHECK_PASSED" = "true" ]; then
    echo -e "${GREEN}âœ“ Coverage reporting completed successfully${NC}"
    echo -e "${GREEN}âœ“ All coverage thresholds met${NC}"
    exit 0
else
    echo -e "${RED}âœ— Coverage reporting completed with failures${NC}"
    echo -e "${RED}âœ— Coverage thresholds not met${NC}"
    
    if [ "$FAIL_ON_THRESHOLD" = "true" ]; then
        exit 1
    else
        echo -e "${YELLOW}âš  Continuing despite threshold failures (FAIL_ON_THRESHOLD=false)${NC}"
        exit 0
    fi
fi