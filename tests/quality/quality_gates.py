"""Quality gates for enforcing testing standards."""

import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json
import time

from tests.coverage.coverage_analyzer import CoverageAnalyzer


@dataclass
class QualityGateResult:
    """Result of a quality gate check."""
    gate_name: str
    passed: bool
    score: float
    threshold: float
    message: str
    details: Dict[str, Any] = None


@dataclass
class QualityGateConfig:
    """Configuration for quality gates."""
    coverage_threshold: float = 80.0
    component_coverage_thresholds: Dict[str, float] = None
    test_pass_rate_threshold: float = 99.0
    performance_regression_threshold: float = 10.0  # percent
    max_test_duration: float = 300.0  # seconds
    
    def __post_init__(self):
        if self.component_coverage_thresholds is None:
            self.component_coverage_thresholds = {
                "core": 90.0,
                "domain": 85.0,
                "integration": 75.0
            }


class QualityGateEnforcer:
    """Enforce quality gates for testing standards."""
    
    def __init__(self, config: Optional[QualityGateConfig] = None):
        self.config = config or QualityGateConfig()
        self.coverage_analyzer = CoverageAnalyzer()
        self.results: List[QualityGateResult] = []
    
    def run_all_quality_gates(self) -> Dict[str, Any]:
        """Run all quality gates and return comprehensive results."""
        print("Running quality gates...")
        
        # Clear previous results
        self.results = []
        
        # Run individual gates
        gates = [
            self._check_test_pass_rate,
            self._check_coverage_thresholds,
            self._check_test_performance,
            self._check_code_quality,
            self._check_test_reliability
        ]
        
        for gate_func in gates:
            try:
                result = gate_func()
                self.results.append(result)
                status = "✅ PASS" if result.passed else "❌ FAIL"
                print(f"{status} {result.gate_name}: {result.message}")
            except Exception as e:
                error_result = QualityGateResult(
                    gate_name=gate_func.__name__.replace("_check_", ""),
                    passed=False,
                    score=0.0,
                    threshold=0.0,
                    message=f"Gate execution failed: {str(e)}"
                )
                self.results.append(error_result)
                print(f"❌ ERROR {error_result.gate_name}: {error_result.message}")
        
        # Calculate overall result
        passed_gates = [r for r in self.results if r.passed]
        overall_passed = len(passed_gates) == len(self.results)
        
        return {
            "overall_passed": overall_passed,
            "gates_passed": len(passed_gates),
            "total_gates": len(self.results),
            "pass_rate": len(passed_gates) / len(self.results) if self.results else 0,
            "results": [self._result_to_dict(r) for r in self.results],
            "summary": self._generate_summary()
        }
    
    def _check_test_pass_rate(self) -> QualityGateResult:
        """Check that test pass rate meets threshold."""
        try:
            # Run tests and capture results
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                "--tb=no", "-q", "--json-report", "--json-report-file=test_results.json"
            ], capture_output=True, text=True, timeout=300)
            
            # Parse test results
            test_results_file = Path("test_results.json")
            if test_results_file.exists():
                with open(test_results_file, 'r') as f:
                    test_data = json.load(f)
                
                total_tests = test_data.get("summary", {}).get("total", 0)
                passed_tests = test_data.get("summary", {}).get("passed", 0)
                
                if total_tests > 0:
                    pass_rate = (passed_tests / total_tests) * 100
                else:
                    pass_rate = 0.0
            else:
                # Fallback: parse from stdout
                lines = result.stdout.split('\n')
                for line in lines:
                    if "passed" in line and "failed" in line:
                        # Parse line like "5 failed, 71 passed in 2.34s"
                        parts = line.split()
                        passed = failed = 0
                        for i, part in enumerate(parts):
                            if part == "passed":
                                passed = int(parts[i-1])
                            elif part == "failed":
                                failed = int(parts[i-1])
                        
                        total_tests = passed + failed
                        pass_rate = (passed / total_tests) * 100 if total_tests > 0 else 0.0
                        break
                else:
                    pass_rate = 100.0 if result.returncode == 0 else 0.0
            
            passed = pass_rate >= self.config.test_pass_rate_threshold
            
            return QualityGateResult(
                gate_name="Test Pass Rate",
                passed=passed,
                score=pass_rate,
                threshold=self.config.test_pass_rate_threshold,
                message=f"{pass_rate:.1f}% pass rate (threshold: {self.config.test_pass_rate_threshold:.1f}%)",
                details={"total_tests": total_tests, "passed_tests": passed_tests} if 'total_tests' in locals() else None
            )
            
        except subprocess.TimeoutExpired:
            return QualityGateResult(
                gate_name="Test Pass Rate",
                passed=False,
                score=0.0,
                threshold=self.config.test_pass_rate_threshold,
                message="Test execution timed out"
            )
        except Exception as e:
            return QualityGateResult(
                gate_name="Test Pass Rate",
                passed=False,
                score=0.0,
                threshold=self.config.test_pass_rate_threshold,
                message=f"Failed to check test pass rate: {str(e)}"
            )
    
    def _check_coverage_thresholds(self) -> QualityGateResult:
        """Check that coverage meets all thresholds."""
        try:
            analysis = self.coverage_analyzer.run_coverage_analysis()
            
            if not analysis["success"]:
                return QualityGateResult(
                    gate_name="Coverage Thresholds",
                    passed=False,
                    score=0.0,
                    threshold=self.config.coverage_threshold,
                    message=f"Coverage analysis failed: {analysis['error']}"
                )
            
            overall_coverage = analysis["overall_coverage"]
            component_analysis = analysis["component_analysis"]
            
            # Check overall coverage
            overall_passed = overall_coverage >= self.config.coverage_threshold
            
            # Check component coverage
            component_failures = []
            for component, target in component_analysis.items():
                if component in self.config.component_coverage_thresholds:
                    threshold = self.config.component_coverage_thresholds[component]
                    if target.current_coverage < threshold:
                        component_failures.append(
                            f"{component}: {target.current_coverage:.1f}% < {threshold:.1f}%"
                        )
            
            all_passed = overall_passed and len(component_failures) == 0
            
            if all_passed:
                message = f"All coverage thresholds met (overall: {overall_coverage:.1f}%)"
            else:
                failures = []
                if not overall_passed:
                    failures.append(f"overall: {overall_coverage:.1f}% < {self.config.coverage_threshold:.1f}%")
                failures.extend(component_failures)
                message = f"Coverage failures: {', '.join(failures)}"
            
            return QualityGateResult(
                gate_name="Coverage Thresholds",
                passed=all_passed,
                score=overall_coverage,
                threshold=self.config.coverage_threshold,
                message=message,
                details={
                    "overall_coverage": overall_coverage,
                    "component_coverage": {comp: target.current_coverage for comp, target in component_analysis.items()},
                    "failures": component_failures
                }
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name="Coverage Thresholds",
                passed=False,
                score=0.0,
                threshold=self.config.coverage_threshold,
                message=f"Failed to check coverage: {str(e)}"
            )
    
    def _check_test_performance(self) -> QualityGateResult:
        """Check that test execution performance is acceptable."""
        try:
            start_time = time.time()
            
            # Run fast tests only for performance check
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                "-m", "fast", "--tb=no", "-q"
            ], capture_output=True, text=True, timeout=self.config.max_test_duration)
            
            execution_time = time.time() - start_time
            
            # Performance targets
            fast_test_target = 30.0  # seconds
            passed = execution_time <= fast_test_target
            
            return QualityGateResult(
                gate_name="Test Performance",
                passed=passed,
                score=execution_time,
                threshold=fast_test_target,
                message=f"Fast tests completed in {execution_time:.1f}s (target: <{fast_test_target:.1f}s)",
                details={"execution_time": execution_time, "return_code": result.returncode}
            )
            
        except subprocess.TimeoutExpired:
            return QualityGateResult(
                gate_name="Test Performance",
                passed=False,
                score=self.config.max_test_duration,
                threshold=30.0,
                message=f"Test execution exceeded {self.config.max_test_duration:.1f}s timeout"
            )
        except Exception as e:
            return QualityGateResult(
                gate_name="Test Performance",
                passed=False,
                score=0.0,
                threshold=30.0,
                message=f"Failed to check test performance: {str(e)}"
            )
    
    def _check_code_quality(self) -> QualityGateResult:
        """Check code quality metrics."""
        try:
            # Check for basic code quality issues
            issues = []
            
            # Check for TODO/FIXME comments in test files
            test_files = list(Path("tests").rglob("*.py"))
            todo_count = 0
            
            for test_file in test_files:
                try:
                    with open(test_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        todo_count += content.upper().count("TODO")
                        todo_count += content.upper().count("FIXME")
                except Exception:
                    continue
            
            if todo_count > 10:
                issues.append(f"{todo_count} TODO/FIXME comments in test files")
            
            # Check for test file naming conventions
            non_conforming_files = []
            for test_file in test_files:
                if not (test_file.name.startswith("test_") or test_file.name == "__init__.py" or test_file.name == "conftest.py"):
                    non_conforming_files.append(test_file.name)
            
            if non_conforming_files:
                issues.append(f"{len(non_conforming_files)} test files don't follow naming convention")
            
            # Overall quality score
            quality_score = max(0, 100 - len(issues) * 10 - todo_count)
            passed = quality_score >= 80
            
            message = "Code quality checks passed" if passed else f"Quality issues: {', '.join(issues)}"
            
            return QualityGateResult(
                gate_name="Code Quality",
                passed=passed,
                score=quality_score,
                threshold=80.0,
                message=message,
                details={
                    "todo_count": todo_count,
                    "non_conforming_files": non_conforming_files,
                    "issues": issues
                }
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name="Code Quality",
                passed=False,
                score=0.0,
                threshold=80.0,
                message=f"Failed to check code quality: {str(e)}"
            )
    
    def _check_test_reliability(self) -> QualityGateResult:
        """Check test reliability and stability."""
        try:
            # Run tests multiple times to check for flakiness
            reliability_runs = 3
            results = []
            
            for run in range(reliability_runs):
                result = subprocess.run([
                    sys.executable, "-m", "pytest", 
                    "-m", "fast", "--tb=no", "-q"
                ], capture_output=True, text=True, timeout=60)
                
                results.append(result.returncode == 0)
            
            # Calculate reliability
            successful_runs = sum(results)
            reliability_rate = (successful_runs / reliability_runs) * 100
            
            passed = reliability_rate >= 90.0  # 90% reliability threshold
            
            return QualityGateResult(
                gate_name="Test Reliability",
                passed=passed,
                score=reliability_rate,
                threshold=90.0,
                message=f"{successful_runs}/{reliability_runs} runs successful ({reliability_rate:.1f}% reliability)",
                details={
                    "successful_runs": successful_runs,
                    "total_runs": reliability_runs,
                    "results": results
                }
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name="Test Reliability",
                passed=False,
                score=0.0,
                threshold=90.0,
                message=f"Failed to check test reliability: {str(e)}"
            )
    
    def _result_to_dict(self, result: QualityGateResult) -> Dict[str, Any]:
        """Convert QualityGateResult to dictionary."""
        return {
            "gate_name": result.gate_name,
            "passed": result.passed,
            "score": result.score,
            "threshold": result.threshold,
            "message": result.message,
            "details": result.details
        }
    
    def _generate_summary(self) -> str:
        """Generate a summary of quality gate results."""
        passed_gates = [r for r in self.results if r.passed]
        failed_gates = [r for r in self.results if not r.passed]
        
        summary = []
        summary.append(f"Quality Gates: {len(passed_gates)}/{len(self.results)} passed")
        
        if failed_gates:
            summary.append("Failed gates:")
            for gate in failed_gates:
                summary.append(f"  - {gate.gate_name}: {gate.message}")
        else:
            summary.append("All quality gates passed! 🎉")
        
        return "\n".join(summary)
    
    def enforce_quality_gates(self, fail_on_error: bool = True) -> bool:
        """Enforce quality gates and optionally fail the build."""
        results = self.run_all_quality_gates()
        
        if not results["overall_passed"]:
            print(f"\n❌ Quality gates failed: {results['gates_passed']}/{results['total_gates']} passed")
            print(results["summary"])
            
            if fail_on_error:
                sys.exit(1)
            return False
        else:
            print(f"\n✅ All quality gates passed: {results['gates_passed']}/{results['total_gates']}")
            return True
    
    def save_quality_report(self, output_file: Path = None) -> Path:
        """Save quality gate results to file."""
        output_file = output_file or Path("quality_report.json")
        
        results = self.run_all_quality_gates()
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return output_file


def main():
    """Run quality gates from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run quality gates")
    parser.add_argument("--no-fail", action="store_true", help="Don't fail on quality gate errors")
    parser.add_argument("--report", type=str, help="Save report to file")
    
    args = parser.parse_args()
    
    enforcer = QualityGateEnforcer()
    
    if args.report:
        report_file = enforcer.save_quality_report(Path(args.report))
        print(f"Quality report saved to: {report_file}")
    
    success = enforcer.enforce_quality_gates(fail_on_error=not args.no_fail)
    
    if not success and not args.no_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()