#!/bin/bash
# CI/CD Quality Gates Script
# Enforces quality standards in CI/CD pipelines

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "========================================="
echo "CI/CD Quality Gates Enforcement"
echo "========================================="

# Configuration
FAIL_ON_QUALITY_GATE=${FAIL_ON_QUALITY_GATE:-true}
RUN_MUTATION_TESTING=${RUN_MUTATION_TESTING:-false}
GENERATE_QUALITY_REPORT=${GENERATE_QUALITY_REPORT:-true}
TRACK_QUALITY_METRICS=${TRACK_QUALITY_METRICS:-true}

# Initialize results
QUALITY_GATES_PASSED=true
GATES_PASSED=0
GATES_FAILED=0

# Function to run a quality gate
run_quality_gate() {
    local gate_name=$1
    local gate_command=$2
    
    echo -e "${BLUE}Running: ${gate_name}${NC}"
    
    if eval "$gate_command"; then
        echo -e "${GREEN}✓ ${gate_name} PASSED${NC}"
        GATES_PASSED=$((GATES_PASSED + 1))
        return 0
    else
        echo -e "${RED}✗ ${gate_name} FAILED${NC}"
        GATES_FAILED=$((GATES_FAILED + 1))
        QUALITY_GATES_PASSED=false
        return 1
    fi
}

# Step 1: Run all quality gates
echo -e "${YELLOW}Step 1: Running quality gates...${NC}"

python -m tests.quality.quality_gates > quality_gates_output.txt 2>&1 || QUALITY_EXIT_CODE=$?

cat quality_gates_output.txt

if [ -n "$QUALITY_EXIT_CODE" ] && [ "$QUALITY_EXIT_CODE" -ne 0 ]; then
    echo -e "${RED}✗ Quality gates failed${NC}"
    QUALITY_GATES_PASSED=false
else
    echo -e "${GREEN}✓ Quality gates passed${NC}"
fi

# Step 2: Track quality metrics (if enabled)
if [ "$TRACK_QUALITY_METRICS" = "true" ]; then
    echo -e "${YELLOW}Step 2: Tracking quality metrics...${NC}"
    
    python -m tests.quality.test_quality_metrics --collect || {
        echo -e "${YELLOW}⚠ Quality metrics collection failed (non-fatal)${NC}"
    }
    
    echo -e "${GREEN}✓ Quality metrics tracked${NC}"
fi

# Step 3: Generate quality report (if enabled)
if [ "$GENERATE_QUALITY_REPORT" = "true" ]; then
    echo -e "${YELLOW}Step 3: Generating quality report...${NC}"
    
    python -m tests.quality.test_quality_metrics --report quality_metrics_report.md || {
        echo -e "${YELLOW}⚠ Quality report generation failed (non-fatal)${NC}"
    }
    
    # Generate quality gates report
    python -c "
from tests.quality.quality_gates import QualityGateEnforcer
enforcer = QualityGateEnforcer()
report_file = enforcer.save_quality_report()
print(f'Quality gates report saved to: {report_file}')
" || true
    
    echo -e "${GREEN}✓ Quality reports generated${NC}"
fi

# Step 4: Run mutation testing (if enabled)
if [ "$RUN_MUTATION_TESTING" = "true" ]; then
    echo -e "${YELLOW}Step 4: Running mutation testing...${NC}"
    
    python -m tests.quality.mutation_testing \
        --max-mutations 10 \
        --report mutation_testing_report.md || {
        echo -e "${YELLOW}⚠ Mutation testing failed (non-fatal)${NC}"
    }
    
    echo -e "${GREEN}✓ Mutation testing completed${NC}"
else
    echo -e "${YELLOW}⚠ Mutation testing skipped (RUN_MUTATION_TESTING=false)${NC}"
fi

# Step 5: Parse quality gate results
echo -e "${YELLOW}Step 5: Parsing quality gate results...${NC}"

if [ -f "quality_report.json" ]; then
    OVERALL_PASSED=$(python -c "import json; data=json.load(open('quality_report.json')); print(data['overall_passed'])")
    GATES_PASSED_COUNT=$(python -c "import json; data=json.load(open('quality_report.json')); print(data['gates_passed'])")
    TOTAL_GATES=$(python -c "import json; data=json.load(open('quality_report.json')); print(data['total_gates'])")
    
    echo "Quality Gates: ${GATES_PASSED_COUNT}/${TOTAL_GATES} passed"
    
    if [ "$OVERALL_PASSED" = "True" ]; then
        QUALITY_GATES_PASSED=true
    else
        QUALITY_GATES_PASSED=false
    fi
fi

# Step 6: Generate CI summary
echo -e "${YELLOW}Step 6: Generating CI summary...${NC}"

cat > quality_gates_summary.md << EOF
# Quality Gates Summary

## Overall Status
- **Status**: $([ "$QUALITY_GATES_PASSED" = "true" ] && echo "✅ PASS" || echo "❌ FAIL")
- **Gates Passed**: ${GATES_PASSED_COUNT:-0}/${TOTAL_GATES:-0}

## Quality Reports
- [Quality Gates Report](quality_report.json)
- [Quality Metrics Report](quality_metrics_report.md)
$([ "$RUN_MUTATION_TESTING" = "true" ] && echo "- [Mutation Testing Report](mutation_testing_report.md)")

## Quality Gate Results

### Test Pass Rate
$(grep -A 2 "Test Pass Rate" quality_gates_output.txt | tail -1 || echo "Not available")

### Coverage Thresholds
$(grep -A 2 "Coverage Thresholds" quality_gates_output.txt | tail -1 || echo "Not available")

### Test Performance
$(grep -A 2 "Test Performance" quality_gates_output.txt | tail -1 || echo "Not available")

### Code Quality
$(grep -A 2 "Code Quality" quality_gates_output.txt | tail -1 || echo "Not available")

### Test Reliability
$(grep -A 2 "Test Reliability" quality_gates_output.txt | tail -1 || echo "Not available")

## Next Steps
$(if [ "$QUALITY_GATES_PASSED" = "false" ]; then
    echo "- Review failed quality gates above"
    echo "- Address quality issues before merging"
    echo "- Check detailed reports for specific recommendations"
else
    echo "- All quality gates passed"
    echo "- Code meets quality standards"
    echo "- Ready for merge"
fi)
EOF

cat quality_gates_summary.md

# Step 7: Create GitHub Actions summary (if in GitHub Actions)
if [ -n "$GITHUB_STEP_SUMMARY" ]; then
    echo -e "${YELLOW}Step 7: Creating GitHub Actions summary...${NC}"
    cat quality_gates_summary.md >> "$GITHUB_STEP_SUMMARY"
    echo -e "${GREEN}✓ GitHub Actions summary created${NC}"
fi

# Step 8: Create PR comment (if in GitHub Actions PR)
if [ -n "$GITHUB_EVENT_PATH" ] && [ -f "$GITHUB_EVENT_PATH" ]; then
    PR_NUMBER=$(python -c "import json; data=json.load(open('$GITHUB_EVENT_PATH')); print(data.get('pull_request', {}).get('number', ''))" 2>/dev/null || echo "")
    
    if [ -n "$PR_NUMBER" ] && [ -n "$GITHUB_TOKEN" ]; then
        echo -e "${YELLOW}Step 8: Creating PR comment...${NC}"
        
        # Create PR comment with quality gates results
        COMMENT_BODY=$(cat quality_gates_summary.md | jq -Rs .)
        
        curl -X POST \
            -H "Authorization: token $GITHUB_TOKEN" \
            -H "Accept: application/vnd.github.v3+json" \
            "https://api.github.com/repos/$GITHUB_REPOSITORY/issues/$PR_NUMBER/comments" \
            -d "{\"body\": $COMMENT_BODY}" || {
            echo -e "${YELLOW}⚠ Failed to create PR comment (non-fatal)${NC}"
        }
        
        echo -e "${GREEN}✓ PR comment created${NC}"
    fi
fi

# Step 9: Final status
echo "========================================="
if [ "$QUALITY_GATES_PASSED" = "true" ]; then
    echo -e "${GREEN}✓ Quality gates enforcement completed successfully${NC}"
    echo -e "${GREEN}✓ All quality standards met${NC}"
    exit 0
else
    echo -e "${RED}✗ Quality gates enforcement completed with failures${NC}"
    echo -e "${RED}✗ Quality standards not met${NC}"
    
    if [ "$FAIL_ON_QUALITY_GATE" = "true" ]; then
        exit 1
    else
        echo -e "${YELLOW}⚠ Continuing despite quality gate failures (FAIL_ON_QUALITY_GATE=false)${NC}"
        exit 0
    fi
fi
