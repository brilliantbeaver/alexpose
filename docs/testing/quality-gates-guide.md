# Quality Gates Guide

## Overview

This guide provides comprehensive documentation for the quality gates system implemented in the AlexPose Gait Analysis System. Quality gates enforce testing standards, track quality metrics, and ensure code quality throughout the development lifecycle.

## Table of Contents

1. [Quality Gates Overview](#quality-gates-overview)
2. [Coverage Analysis](#coverage-analysis)
3. [Quality Metrics Tracking](#quality-metrics-tracking)
4. [Coverage Trend Analysis](#coverage-trend-analysis)
5. [Mutation Testing](#mutation-testing)
6. [CI/CD Integration](#cicd-integration)
7. [Local Development](#local-development)
8. [Troubleshooting](#troubleshooting)

## Quality Gates Overview

### What are Quality Gates?

Quality gates are automated checks that enforce quality standards before code can be merged or deployed. They ensure:

- **Test Coverage**: Minimum coverage thresholds are met
- **Test Pass Rate**: Tests pass consistently (>99%)
- **Test Performance**: Tests execute within acceptable time limits
- **Code Quality**: Code meets quality standards
- **Test Reliability**: Tests are stable and not flaky

### Quality Gate Components

The quality gates system consists of five main components:

1. **Coverage Analyzer** (`tests/coverage/coverage_analyzer.py`)
   - Analyzes test coverage by component
   - Identifies coverage gaps
   - Generates recommendations

2. **Quality Gate Enforcer** (`tests/quality/quality_gates.py`)
   - Enforces quality standards
   - Runs all quality checks
   - Fails builds when standards not met

3. **Quality Metrics Tracker** (`tests/quality/test_quality_metrics.py`)
   - Tracks quality metrics over time
   - Monitors trends
   - Identifies quality regressions

4. **Coverage Trend Analyzer** (`tests/coverage/coverage_trend_analyzer.py`)
   - Tracks coverage trends over time
   - Identifies coverage regressions
   - Generates trend visualizations

5. **Mutation Tester** (`tests/quality/mutation_testing.py`)
   - Tests test suite effectiveness
   - Identifies test gaps
   - Validates test quality

## Coverage Analysis

### Running Coverage Analysis

```bash
# Run coverage analysis
python -m tests.coverage.coverage_analyzer

# Generate coverage report
python -c "
from tests.coverage.coverage_analyzer import CoverageAnalyzer
analyzer = CoverageAnalyzer()
report_file = analyzer.save_coverage_report()
print(f'Report saved to: {report_file}')
"
```

### Coverage Targets

| Component | Minimum | Target |
|-----------|---------|--------|
| Overall | 80% | 85% |
| Core | 90% | 95% |
| Domain | 85% | 90% |
| Integration | 75% | 85% |

### Coverage Gap Analysis

The coverage analyzer identifies gaps and prioritizes them:

- **High Priority**: <50% coverage or critical components
- **Medium Priority**: 50-70% coverage or important components
- **Low Priority**: 70-85% coverage or non-critical components

### Example Output

```
# Coverage Analysis Report
Overall Coverage: 82.5%

## Component Coverage
✅ core: 92.3% (min: 90.0%, target: 95.0%)
✅ domain: 87.1% (min: 85.0%, target: 90.0%)
✅ integration: 78.4% (min: 75.0%, target: 85.0%)

## Coverage Gaps
- ambient/analysis/gait_analyzer.py: 65.2% (45 uncovered lines) [high priority]
- ambient/classification/llm_classifier.py: 72.8% (23 uncovered lines) [medium priority]

## Recommendations
- Address 2 high-priority coverage gaps
- Improve domain coverage from 87.1% to target 90.0%
```

## Quality Metrics Tracking

### Collecting Quality Metrics

```bash
# Collect current metrics
python -m tests.quality.test_quality_metrics --collect

# Generate quality report
python -m tests.quality.test_quality_metrics --report quality_report.md

# Analyze trends
python -m tests.quality.test_quality_metrics --trends 30
```

### Tracked Metrics

1. **Test Pass Rate**: Percentage of tests passing
2. **Coverage Percentage**: Overall test coverage
3. **Flaky Tests**: Number of unreliable tests
4. **Slow Tests**: Number of tests >1 second
5. **Test Reliability Score**: Overall test reliability (0-100)
6. **Code Quality Score**: Code quality assessment (0-100)

### Quality Metrics Database

Metrics are stored in `test_quality_metrics.db` with:

- Historical metrics data
- Test failure tracking
- Flaky test identification
- Trend analysis data

### Example Output

```
# Test Quality Metrics Report
Generated: 2026-01-02 10:30:00

## Current Metrics
- Total Tests: 156
- Pass Rate: 99.4%
- Coverage: 82.5%
- Execution Time: 45.2s
- Flaky Tests: 2
- Slow Tests: 8
- Reliability Score: 97.3/100
- Code Quality Score: 92.1/100

## Trends (Last 30 Days)
- Pass Rate: 📈 +1.2%
- Coverage: 📈 +3.4%
- Reliability: ➡️ +0.5%

## Recommendations
- Address 2 flaky tests
- Optimize 8 slow tests
```

## Coverage Trend Analysis

### Capturing Coverage Snapshots

```bash
# Capture current coverage snapshot
python -m tests.coverage.coverage_trend_analyzer --capture

# Generate trend report
python -m tests.coverage.coverage_trend_analyzer --report coverage_trends.md --days 30

# Generate trend chart
python -m tests.coverage.coverage_trend_analyzer --chart coverage_trends.png --days 90
```

### Coverage Trends Database

Coverage snapshots are stored in `coverage_trends.db` with:

- Overall coverage over time
- Component-specific coverage trends
- File-level coverage history
- Regression detection

### Trend Analysis Features

1. **Trend Detection**: Identifies improving or declining coverage
2. **Velocity Calculation**: Coverage change per day
3. **Regression Identification**: Detects coverage drops
4. **File-Level Tracking**: Tracks individual file coverage
5. **Visualization**: Generates trend charts

### Example Output

```
# Coverage Trend Analysis Report
Period: Last 30 days
Data Points: 15

## Current Coverage
- Overall: 82.5%
- Core: 92.3%
- Domain: 87.1%
- Integration: 78.4%
- Lines: 4,523/5,482

## Trends
- Overall: 📈 +3.2%
- Core: ➡️ +0.5%
- Domain: 📈 +2.8%
- Integration: 📈 +4.1%
- Velocity: +0.11% per day

## Statistics
- Average: 80.3%
- Minimum: 78.1%
- Maximum: 82.5%
- Range: 4.4%

## Recommendations
- Coverage trends are positive! Keep up the good work.
```

## Mutation Testing

### Running Mutation Testing

```bash
# Run mutation testing on critical files
python -m tests.quality.mutation_testing --max-mutations 10 --report mutation_report.md

# Test specific files
python -m tests.quality.mutation_testing --files ambient/core/frame.py ambient/analysis/gait_analyzer.py
```

### Mutation Operators

1. **Arithmetic Operators**: +, -, *, /, %
2. **Comparison Operators**: <, >, <=, >=, ==, !=
3. **Boolean Operators**: and, or
4. **Constants**: Numbers, booleans

### Mutation Score

Mutation score = (Killed mutations / Total mutations) × 100

- **>80%**: Excellent test quality
- **60-80%**: Good test quality
- **40-60%**: Moderate test quality
- **<40%**: Poor test quality

### Example Output

```
# Mutation Testing Report

Mutation Testing Results:
  Total mutations: 45
  Killed: 38
  Survived: 7
  Mutation score: 84.4%

## Survived Mutations (Potential Test Gaps)
- ambient/core/frame.py:125
  Original: `if x > 0`
  Mutated:  `if x >= 0`

## Mutation Operator Effectiveness
- Arithmetic: 15/18 killed (83.3%)
- Comparison: 12/14 killed (85.7%)
- Boolean: 8/9 killed (88.9%)
- Constant: 3/4 killed (75.0%)
```

## CI/CD Integration

### GitHub Actions Workflows

#### Quality Gates Workflow

The main quality gates workflow (`.github/workflows/quality-gates.yml`) runs on:

- **Push to main/develop**: Full quality gates
- **Pull requests**: Full quality gates with PR comments
- **Daily schedule**: Trend tracking and metrics collection

#### Workflow Jobs

1. **quality-gates**: Main quality enforcement
   - Runs coverage analysis
   - Enforces quality gates
   - Uploads reports
   - Comments on PRs

2. **mutation-testing**: Weekly mutation testing
   - Runs on schedule or manual trigger
   - Tests critical components
   - Generates mutation reports

3. **coverage-trends**: Coverage trend tracking
   - Runs on main branch only
   - Captures coverage snapshots
   - Generates trend reports and charts

4. **quality-metrics**: Quality metrics tracking
   - Runs on main branch only
   - Collects quality metrics
   - Tracks trends over time

### CI Scripts

#### Coverage Reporter Script

```bash
# Run coverage reporter in CI
bash scripts/ci-coverage-reporter.sh
```

Environment variables:
- `COVERAGE_THRESHOLD`: Minimum coverage (default: 80)
- `FAIL_ON_THRESHOLD`: Fail build if below threshold (default: true)
- `CODECOV_TOKEN`: Codecov upload token
- `GENERATE_REPORTS`: Generate detailed reports (default: true)
- `CAPTURE_TRENDS`: Capture coverage trends (default: true)

#### Quality Gates Script

```bash
# Run quality gates in CI
bash scripts/ci-quality-gates.sh
```

Environment variables:
- `FAIL_ON_QUALITY_GATE`: Fail build on quality gate failure (default: true)
- `RUN_MUTATION_TESTING`: Run mutation testing (default: false)
- `GENERATE_QUALITY_REPORT`: Generate quality reports (default: true)
- `TRACK_QUALITY_METRICS`: Track quality metrics (default: true)

### PR Comments

Quality gates automatically comment on PRs with:

- Coverage summary
- Quality gates results
- Recommendations
- Links to detailed reports

### Artifacts

All workflows upload artifacts:

- **coverage-reports**: Coverage XML, HTML, JSON, and analysis reports
- **quality-reports**: Quality gates and metrics reports
- **mutation-testing-report**: Mutation testing results
- **coverage-trends**: Trend reports and charts
- **quality-metrics**: Quality metrics reports

## Local Development

### Running Quality Gates Locally

```bash
# Run all quality gates
python -m tests.quality.quality_gates

# Run without failing on errors
python -m tests.quality.quality_gates --no-fail

# Generate quality report
python -m tests.quality.quality_gates --report quality_report.json
```

### Running Coverage Analysis Locally

```bash
# Run coverage analysis
python -m tests.coverage.coverage_analyzer

# Generate coverage report
python -c "
from tests.coverage.coverage_analyzer import CoverageAnalyzer
analyzer = CoverageAnalyzer()
report_file = analyzer.save_coverage_report()
print(f'Report saved to: {report_file}')
"
```

### Pre-commit Checks

Add quality gates to pre-commit hooks:

```yaml
# .pre-commit-config.yaml
- repo: local
  hooks:
    - id: quality-gates
      name: Quality Gates
      entry: python -m tests.quality.quality_gates --no-fail
      language: system
      pass_filenames: false
```

### IDE Integration

Configure your IDE to run quality gates:

**VS Code** (`tasks.json`):
```json
{
  "label": "Run Quality Gates",
  "type": "shell",
  "command": "python -m tests.quality.quality_gates",
  "problemMatcher": []
}
```

## Troubleshooting

### Common Issues

#### 1. Coverage Below Threshold

**Problem**: Coverage is below the minimum threshold

**Solution**:
```bash
# Identify coverage gaps
python -m tests.coverage.coverage_analyzer

# Review gap report
cat coverage_gap_report.md

# Add tests for high-priority gaps
```

#### 2. Flaky Tests

**Problem**: Tests fail intermittently

**Solution**:
```bash
# Identify flaky tests
python -m tests.quality.test_quality_metrics --collect

# Review flaky tests in database
sqlite3 test_quality_metrics.db "SELECT * FROM flaky_tests WHERE flakiness_score > 0.2"

# Fix or mark flaky tests
```

#### 3. Slow Tests

**Problem**: Tests take too long to execute

**Solution**:
```bash
# Identify slow tests
pytest --durations=10

# Optimize slow tests:
# - Use fixtures instead of setup/teardown
# - Mock external dependencies
# - Use smaller test data
# - Parallelize with pytest-xdist
```

#### 4. Quality Gates Failing in CI

**Problem**: Quality gates pass locally but fail in CI

**Solution**:
```bash
# Run with same configuration as CI
export COVERAGE_THRESHOLD=80
export FAIL_ON_THRESHOLD=true
bash scripts/ci-quality-gates.sh

# Check for environment-specific issues
# - Different Python versions
# - Missing dependencies
# - Different test data
```

#### 5. Mutation Testing Too Slow

**Problem**: Mutation testing takes too long

**Solution**:
```bash
# Limit mutations per file
python -m tests.quality.mutation_testing --max-mutations 5

# Test specific files only
python -m tests.quality.mutation_testing --files ambient/core/frame.py

# Run mutation testing weekly instead of on every commit
```

### Debug Mode

Enable debug output for troubleshooting:

```bash
# Coverage analyzer debug
COVERAGE_DEBUG=1 python -m tests.coverage.coverage_analyzer

# Quality gates debug
QUALITY_GATES_DEBUG=1 python -m tests.quality.quality_gates

# Verbose pytest output
pytest -vv --tb=long
```

### Getting Help

If you encounter issues:

1. Check the [Testing Documentation](./developer-guidelines.md)
2. Review [Debugging Guide](./debugging-guide.md)
3. Check CI logs for detailed error messages
4. Review quality reports for specific recommendations

## Best Practices

### 1. Regular Monitoring

- Review quality metrics weekly
- Track coverage trends monthly
- Run mutation testing quarterly

### 2. Incremental Improvement

- Set realistic coverage targets
- Improve coverage gradually
- Focus on high-priority gaps first

### 3. Test Quality

- Write meaningful tests, not just for coverage
- Use property-based testing for complex logic
- Validate tests with mutation testing

### 4. CI/CD Integration

- Enforce quality gates on all PRs
- Track trends on main branch
- Generate reports for visibility

### 5. Team Collaboration

- Share quality reports with team
- Discuss quality trends in retrospectives
- Celebrate quality improvements

## Summary

The quality gates system provides comprehensive quality enforcement and monitoring:

- **Automated Enforcement**: Quality gates run automatically in CI/CD
- **Trend Tracking**: Monitor quality metrics and coverage over time
- **Gap Identification**: Identify and prioritize quality improvements
- **Test Validation**: Mutation testing validates test effectiveness
- **Continuous Improvement**: Track progress and celebrate improvements

By following this guide and leveraging the quality gates system, you can maintain high code quality and testing standards throughout the development lifecycle.
