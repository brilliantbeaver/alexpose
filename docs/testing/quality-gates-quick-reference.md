# Quality Gates Quick Reference

## Quick Commands

### Run All Quality Gates
```bash
python -m tests.quality.quality_gates
```

### Check Coverage
```bash
python -m tests.coverage.coverage_analyzer
```

### Track Quality Metrics
```bash
python -m tests.quality.test_quality_metrics --collect
```

### Analyze Coverage Trends
```bash
python -m tests.coverage.coverage_trend_analyzer --capture
```

### Run Mutation Testing
```bash
python -m tests.quality.mutation_testing --max-mutations 10
```

## Quality Thresholds

| Metric | Threshold | Target |
|--------|-----------|--------|
| Overall Coverage | 80% | 85% |
| Core Coverage | 90% | 95% |
| Domain Coverage | 85% | 90% |
| Integration Coverage | 75% | 85% |
| Test Pass Rate | 99% | 100% |
| Test Reliability | 90% | 95% |
| Code Quality Score | 80/100 | 90/100 |
| Mutation Score | 70% | 80% |

## CI/CD Scripts

### Coverage Reporter
```bash
bash scripts/ci-coverage-reporter.sh
```

### Quality Gates
```bash
bash scripts/ci-quality-gates.sh
```

## Environment Variables

### Coverage Reporter
- `COVERAGE_THRESHOLD=80` - Minimum coverage
- `FAIL_ON_THRESHOLD=true` - Fail build if below threshold
- `CODECOV_TOKEN=<token>` - Codecov upload token
- `GENERATE_REPORTS=true` - Generate detailed reports
- `CAPTURE_TRENDS=true` - Capture coverage trends

### Quality Gates
- `FAIL_ON_QUALITY_GATE=true` - Fail build on quality gate failure
- `RUN_MUTATION_TESTING=false` - Run mutation testing
- `GENERATE_QUALITY_REPORT=true` - Generate quality reports
- `TRACK_QUALITY_METRICS=true` - Track quality metrics

## Common Issues

### Coverage Below Threshold
```bash
# Identify gaps
python -m tests.coverage.coverage_analyzer

# Review gap report
cat coverage_gap_report.md
```

### Flaky Tests
```bash
# Identify flaky tests
python -m tests.quality.test_quality_metrics --collect

# Review flaky tests
sqlite3 test_quality_metrics.db "SELECT * FROM flaky_tests WHERE flakiness_score > 0.2"
```

### Slow Tests
```bash
# Identify slow tests
pytest --durations=10

# Run only fast tests
pytest -m fast
```

### Quality Gates Failing
```bash
# Run with detailed output
python -m tests.quality.quality_gates

# Run without failing
python -m tests.quality.quality_gates --no-fail
```

## Report Files

| Report | Location | Description |
|--------|----------|-------------|
| Coverage Report | `coverage_report.md` | Coverage analysis and gaps |
| Coverage Trends | `coverage_trend_report.md` | Coverage trends over time |
| Quality Report | `quality_report.json` | Quality gates results |
| Quality Metrics | `quality_metrics_report.md` | Quality metrics and trends |
| Mutation Report | `mutation_testing_report.md` | Mutation testing results |

## Database Files

| Database | Location | Purpose |
|----------|----------|---------|
| Quality Metrics | `test_quality_metrics.db` | Quality metrics history |
| Coverage Trends | `coverage_trends.db` | Coverage snapshots |

## GitHub Actions Workflows

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| Quality Gates | Push/PR | Enforce quality standards |
| Mutation Testing | Weekly | Validate test effectiveness |
| Coverage Trends | Daily | Track coverage trends |
| Quality Metrics | Daily | Track quality metrics |

## Pre-commit Hook

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

## IDE Integration

### VS Code Task
```json
{
  "label": "Run Quality Gates",
  "type": "shell",
  "command": "python -m tests.quality.quality_gates",
  "problemMatcher": []
}
```

### PyCharm External Tool
- Program: `python`
- Arguments: `-m tests.quality.quality_gates`
- Working directory: `$ProjectFileDir$`

## Best Practices

1. ✅ Run quality gates before committing
2. ✅ Monitor trends weekly
3. ✅ Address high-priority gaps first
4. ✅ Fix flaky tests immediately
5. ✅ Optimize slow tests
6. ✅ Track progress over time
7. ✅ Celebrate improvements

## Getting Help

- [Quality Gates Guide](./quality-gates-guide.md)
- [Developer Guidelines](./developer-guidelines.md)
- [Debugging Guide](./debugging-guide.md)
- [Test Data Management](./test-data-management.md)

## Summary

The Quality Gates system ensures:
- ✅ High test coverage (>80%)
- ✅ Reliable tests (>99% pass rate)
- ✅ Fast feedback (<30s for fast tests)
- ✅ Quality code (>80 quality score)
- ✅ Effective tests (>70% mutation score)

Run `python -m tests.quality.quality_gates` to check all quality standards!
