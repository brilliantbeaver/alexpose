# Quality Gates System

## Overview

The Quality Gates system provides comprehensive quality enforcement and monitoring for the AlexPose Gait Analysis System. It ensures code quality, test coverage, and test reliability through automated checks and trend analysis.

## Components

### 1. Quality Gate Enforcer (`quality_gates.py`)

Enforces quality standards through automated checks:

- **Test Pass Rate**: Ensures tests pass consistently (>99%)
- **Coverage Thresholds**: Validates coverage meets targets
- **Test Performance**: Checks test execution time
- **Code Quality**: Validates code quality standards
- **Test Reliability**: Detects flaky tests

**Usage**:
```bash
# Run all quality gates
python -m tests.quality.quality_gates

# Run without failing on errors
python -m tests.quality.quality_gates --no-fail

# Generate quality report
python -m tests.quality.quality_gates --report quality_report.json
```

### 2. Quality Metrics Tracker (`test_quality_metrics.py`)

Tracks quality metrics over time:

- Test pass rate trends
- Coverage trends
- Flaky test detection
- Slow test identification
- Reliability scoring
- Code quality scoring

**Usage**:
```bash
# Collect current metrics
python -m tests.quality.test_quality_metrics --collect

# Generate quality report
python -m tests.quality.test_quality_metrics --report quality_report.md

# Analyze trends
python -m tests.quality.test_quality_metrics --trends 30
```

### 3. Mutation Tester (`mutation_testing.py`)

Validates test effectiveness through mutation testing:

- Arithmetic operator mutations
- Comparison operator mutations
- Boolean operator mutations
- Constant mutations

**Usage**:
```bash
# Run mutation testing
python -m tests.quality.mutation_testing --max-mutations 10 --report mutation_report.md

# Test specific files
python -m tests.quality.mutation_testing --files ambient/core/frame.py
```

## Quality Standards

### Coverage Targets

| Component | Minimum | Target |
|-----------|---------|--------|
| Overall | 80% | 85% |
| Core | 90% | 95% |
| Domain | 85% | 90% |
| Integration | 75% | 85% |

### Quality Thresholds

- **Test Pass Rate**: ≥99%
- **Test Reliability**: ≥90%
- **Code Quality Score**: ≥80/100
- **Fast Test Performance**: <30 seconds
- **Mutation Score**: ≥80% (target)

## CI/CD Integration

### GitHub Actions Workflow

The quality gates workflow (`.github/workflows/quality-gates.yml`) runs:

1. **On Push/PR**: Full quality gates enforcement
2. **Daily**: Trend tracking and metrics collection
3. **Weekly**: Mutation testing (scheduled)

### CI Scripts

#### Coverage Reporter
```bash
bash scripts/ci-coverage-reporter.sh
```

Environment variables:
- `COVERAGE_THRESHOLD`: Minimum coverage (default: 80)
- `FAIL_ON_THRESHOLD`: Fail build if below threshold
- `CODECOV_TOKEN`: Codecov upload token

#### Quality Gates
```bash
bash scripts/ci-quality-gates.sh
```

Environment variables:
- `FAIL_ON_QUALITY_GATE`: Fail build on quality gate failure
- `RUN_MUTATION_TESTING`: Run mutation testing
- `GENERATE_QUALITY_REPORT`: Generate quality reports

## Local Development

### Quick Start

```bash
# Run all quality checks
python -m tests.quality.quality_gates

# Check coverage
python -m tests.coverage.coverage_analyzer

# Track quality metrics
python -m tests.quality.test_quality_metrics --collect
```

### Pre-commit Integration

Add to `.pre-commit-config.yaml`:
```yaml
- repo: local
  hooks:
    - id: quality-gates
      name: Quality Gates
      entry: python -m tests.quality.quality_gates --no-fail
      language: system
      pass_filenames: false
```

## Databases

### Quality Metrics Database (`test_quality_metrics.db`)

Stores:
- Historical quality metrics
- Test failure tracking
- Flaky test identification
- Trend analysis data

### Coverage Trends Database (`coverage_trends.db`)

Stores:
- Coverage snapshots over time
- Component-specific coverage
- File-level coverage history
- Regression detection data

## Reports

### Quality Gates Report (`quality_report.json`)

Contains:
- Overall pass/fail status
- Individual gate results
- Detailed metrics
- Recommendations

### Quality Metrics Report (`quality_metrics_report.md`)

Contains:
- Current quality metrics
- Trend analysis
- Issues detected
- Recommendations

### Mutation Testing Report (`mutation_testing_report.md`)

Contains:
- Mutation score
- Survived mutations
- Operator effectiveness
- Test gap identification

## Best Practices

1. **Run quality gates before committing**
   ```bash
   python -m tests.quality.quality_gates
   ```

2. **Monitor trends regularly**
   ```bash
   python -m tests.quality.test_quality_metrics --trends 30
   ```

3. **Address high-priority gaps first**
   - Review coverage gap report
   - Focus on critical components
   - Add meaningful tests

4. **Validate test quality**
   ```bash
   python -m tests.quality.mutation_testing --max-mutations 10
   ```

5. **Track progress**
   - Review quality metrics weekly
   - Celebrate improvements
   - Address regressions promptly

## Troubleshooting

### Quality Gates Failing

1. Check which gate failed:
   ```bash
   python -m tests.quality.quality_gates
   ```

2. Review detailed output for specific issues

3. Address the failing gate:
   - **Test Pass Rate**: Fix failing tests
   - **Coverage**: Add tests for uncovered code
   - **Performance**: Optimize slow tests
   - **Code Quality**: Address code quality issues
   - **Reliability**: Fix flaky tests

### Coverage Below Threshold

1. Identify gaps:
   ```bash
   python -m tests.coverage.coverage_analyzer
   ```

2. Review gap report for priorities

3. Add tests for high-priority gaps

### Flaky Tests

1. Identify flaky tests:
   ```bash
   python -m tests.quality.test_quality_metrics --collect
   ```

2. Review flaky tests in database:
   ```bash
   sqlite3 test_quality_metrics.db "SELECT * FROM flaky_tests"
   ```

3. Fix or mark flaky tests

## Documentation

For detailed documentation, see:

- [Quality Gates Guide](../../docs/testing/quality-gates-guide.md)
- [Developer Guidelines](../../docs/testing/developer-guidelines.md)
- [Debugging Guide](../../docs/testing/debugging-guide.md)

## Support

For issues or questions:

1. Check the documentation
2. Review CI logs
3. Check quality reports
4. Contact the testing team

## Summary

The Quality Gates system provides:

- ✅ Automated quality enforcement
- ✅ Comprehensive metrics tracking
- ✅ Trend analysis and visualization
- ✅ Test effectiveness validation
- ✅ CI/CD integration
- ✅ Detailed reporting and recommendations

By leveraging this system, you can maintain high code quality and testing standards throughout the development lifecycle.
