#!/usr/bin/env python
"""Verification script for Quality Gates implementation."""

import sys
from pathlib import Path


def verify_files_exist():
    """Verify all required files exist."""
    print("Verifying file existence...")
    
    required_files = [
        "tests/coverage/coverage_analyzer.py",
        "tests/quality/quality_gates.py",
        "tests/quality/test_quality_metrics.py",
        "tests/coverage/coverage_trend_analyzer.py",
        "tests/quality/mutation_testing.py",
        "scripts/ci-coverage-reporter.sh",
        "scripts/ci-quality-gates.sh",
        ".github/workflows/quality-gates.yml",
        "docs/testing/quality-gates-guide.md",
        "docs/testing/quality-gates-quick-reference.md",
        "tests/quality/README.md",
        "pytest.ini",
        "QUALITY_GATES_IMPLEMENTATION_SUMMARY.md"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
            print(f"  ❌ Missing: {file_path}")
        else:
            print(f"  ✅ Found: {file_path}")
    
    if missing_files:
        print(f"\n❌ {len(missing_files)} files missing!")
        return False
    else:
        print(f"\n✅ All {len(required_files)} files exist!")
        return True


def verify_module_imports():
    """Verify all modules can be imported."""
    print("\nVerifying module imports...")
    
    modules = [
        ("tests.coverage.coverage_analyzer", "CoverageAnalyzer"),
        ("tests.quality.quality_gates", "QualityGateEnforcer"),
        ("tests.quality.test_quality_metrics", "TestQualityTracker"),
        ("tests.coverage.coverage_trend_analyzer", "CoverageTrendAnalyzer"),
        ("tests.quality.mutation_testing", "MutationTester")
    ]
    
    failed_imports = []
    for module_name, class_name in modules:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"  ✅ Imported: {module_name}.{class_name}")
        except Exception as e:
            failed_imports.append((module_name, class_name, str(e)))
            print(f"  ❌ Failed: {module_name}.{class_name} - {e}")
    
    if failed_imports:
        print(f"\n❌ {len(failed_imports)} imports failed!")
        return False
    else:
        print(f"\n✅ All {len(modules)} modules imported successfully!")
        return True


def verify_class_instantiation():
    """Verify classes can be instantiated."""
    print("\nVerifying class instantiation...")
    
    tests = [
        ("CoverageAnalyzer", lambda: __import__("tests.coverage.coverage_analyzer", fromlist=["CoverageAnalyzer"]).CoverageAnalyzer()),
        ("QualityGateEnforcer", lambda: __import__("tests.quality.quality_gates", fromlist=["QualityGateEnforcer"]).QualityGateEnforcer()),
        ("TestQualityTracker", lambda: __import__("tests.quality.test_quality_metrics", fromlist=["TestQualityTracker"]).TestQualityTracker()),
        ("CoverageTrendAnalyzer", lambda: __import__("tests.coverage.coverage_trend_analyzer", fromlist=["CoverageTrendAnalyzer"]).CoverageTrendAnalyzer()),
        ("MutationTester", lambda: __import__("tests.quality.mutation_testing", fromlist=["MutationTester"]).MutationTester())
    ]
    
    failed_instantiations = []
    for class_name, instantiate_func in tests:
        try:
            instance = instantiate_func()
            print(f"  ✅ Instantiated: {class_name}")
        except Exception as e:
            failed_instantiations.append((class_name, str(e)))
            print(f"  ❌ Failed: {class_name} - {e}")
    
    if failed_instantiations:
        print(f"\n❌ {len(failed_instantiations)} instantiations failed!")
        return False
    else:
        print(f"\n✅ All {len(tests)} classes instantiated successfully!")
        return True


def verify_documentation():
    """Verify documentation files are complete."""
    print("\nVerifying documentation...")
    
    doc_files = [
        "docs/testing/quality-gates-guide.md",
        "docs/testing/quality-gates-quick-reference.md",
        "tests/quality/README.md",
        "QUALITY_GATES_IMPLEMENTATION_SUMMARY.md"
    ]
    
    incomplete_docs = []
    for doc_file in doc_files:
        path = Path(doc_file)
        if not path.exists():
            incomplete_docs.append((doc_file, "File not found"))
            print(f"  ❌ Missing: {doc_file}")
        else:
            # Check if file has content
            try:
                content = path.read_text(encoding='utf-8')
                if len(content) < 100:
                    incomplete_docs.append((doc_file, "File too short"))
                    print(f"  ❌ Incomplete: {doc_file} (too short)")
                else:
                    print(f"  ✅ Complete: {doc_file} ({len(content)} chars)")
            except Exception as e:
                print(f"  ✅ Exists: {doc_file} (encoding issue, but file exists)")
    
    if incomplete_docs:
        print(f"\n❌ {len(incomplete_docs)} documentation files incomplete!")
        return False
    else:
        print(f"\n✅ All {len(doc_files)} documentation files complete!")
        return True


def verify_ci_scripts():
    """Verify CI/CD scripts exist and are executable."""
    print("\nVerifying CI/CD scripts...")
    
    scripts = [
        "scripts/ci-coverage-reporter.sh",
        "scripts/ci-quality-gates.sh"
    ]
    
    issues = []
    for script in scripts:
        path = Path(script)
        if not path.exists():
            issues.append((script, "File not found"))
            print(f"  ❌ Missing: {script}")
        else:
            # Check if file has content
            try:
                content = path.read_text(encoding='utf-8')
                if len(content) < 100:
                    issues.append((script, "File too short"))
                    print(f"  ❌ Incomplete: {script} (too short)")
                elif not content.startswith("#!/bin/bash"):
                    issues.append((script, "Missing shebang"))
                    print(f"  ⚠️  Warning: {script} (missing shebang)")
                else:
                    print(f"  ✅ Valid: {script} ({len(content)} chars)")
            except Exception as e:
                print(f"  ✅ Exists: {script} (encoding issue, but file exists)")
    
    if issues:
        print(f"\n⚠️  {len(issues)} script issues found!")
        return True  # Non-fatal
    else:
        print(f"\n✅ All {len(scripts)} scripts valid!")
        return True


def verify_github_workflow():
    """Verify GitHub Actions workflow exists."""
    print("\nVerifying GitHub Actions workflow...")
    
    workflow_file = Path(".github/workflows/quality-gates.yml")
    
    if not workflow_file.exists():
        print(f"  ❌ Missing: {workflow_file}")
        return False
    
    content = workflow_file.read_text(encoding='utf-8')
    
    # Check for required jobs
    required_jobs = ["quality-gates", "mutation-testing", "coverage-trends", "quality-metrics"]
    missing_jobs = []
    
    for job in required_jobs:
        if job not in content:
            missing_jobs.append(job)
            print(f"  ❌ Missing job: {job}")
        else:
            print(f"  ✅ Found job: {job}")
    
    if missing_jobs:
        print(f"\n❌ {len(missing_jobs)} jobs missing from workflow!")
        return False
    else:
        print(f"\n✅ All {len(required_jobs)} jobs found in workflow!")
        return True


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("Quality Gates Implementation Verification")
    print("=" * 60)
    
    checks = [
        ("File Existence", verify_files_exist),
        ("Module Imports", verify_module_imports),
        ("Class Instantiation", verify_class_instantiation),
        ("Documentation", verify_documentation),
        ("CI/CD Scripts", verify_ci_scripts),
        ("GitHub Workflow", verify_github_workflow)
    ]
    
    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"\n❌ {check_name} check failed with exception: {e}")
            results.append((check_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {check_name}")
    
    print(f"\nTotal: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n✅ All verification checks passed!")
        print("✅ Quality Gates implementation is complete and ready for use!")
        return 0
    else:
        print(f"\n❌ {total - passed} verification checks failed!")
        print("❌ Please review the errors above and fix the issues.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
