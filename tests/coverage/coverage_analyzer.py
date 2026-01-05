"""Coverage analysis and gap reporting tools."""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import xml.etree.ElementTree as ET


@dataclass
class CoverageTarget:
    """Coverage target for a component."""
    component: str
    minimum_coverage: float
    target_coverage: float
    current_coverage: float = 0.0
    
    @property
    def meets_minimum(self) -> bool:
        """Check if current coverage meets minimum requirement."""
        return self.current_coverage >= self.minimum_coverage
    
    @property
    def meets_target(self) -> bool:
        """Check if current coverage meets target."""
        return self.current_coverage >= self.target_coverage


@dataclass
class CoverageGap:
    """Represents a coverage gap that needs attention."""
    file_path: str
    uncovered_lines: List[int]
    coverage_percentage: float
    component: str
    priority: str  # 'high', 'medium', 'low'


class CoverageAnalyzer:
    """Analyze test coverage and identify gaps."""
    
    def __init__(self, coverage_file: Optional[Path] = None):
        self.coverage_file = coverage_file or Path("htmlcov/coverage.xml")
        self.coverage_targets = self._define_coverage_targets()
        self.coverage_data = {}
    
    def _define_coverage_targets(self) -> Dict[str, CoverageTarget]:
        """Define coverage targets for different components."""
        return {
            "core": CoverageTarget("core", 90.0, 95.0),
            "domain": CoverageTarget("domain", 85.0, 90.0),
            "integration": CoverageTarget("integration", 75.0, 85.0),
            "overall": CoverageTarget("overall", 80.0, 85.0)
        }
    
    def run_coverage_analysis(self) -> Dict[str, Any]:
        """Run comprehensive coverage analysis."""
        try:
            # Run coverage with XML output
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                "--cov=ambient", "--cov=server",
                "--cov-report=xml:coverage.xml",
                "--cov-report=html:htmlcov",
                "--cov-report=term-missing",
                "-q"
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                return {
                    "success": False,
                    "error": f"Coverage analysis failed: {result.stderr}",
                    "stdout": result.stdout
                }
            
            # Parse coverage results
            coverage_data = self._parse_coverage_xml()
            
            # Analyze by component
            component_analysis = self._analyze_by_component(coverage_data)
            
            # Identify coverage gaps
            coverage_gaps = self._identify_coverage_gaps(coverage_data)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(component_analysis, coverage_gaps)
            
            return {
                "success": True,
                "overall_coverage": coverage_data.get("overall_coverage", 0.0),
                "component_analysis": component_analysis,
                "coverage_gaps": coverage_gaps,
                "recommendations": recommendations,
                "targets_met": self._check_targets_met(component_analysis)
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Coverage analysis timed out after 5 minutes"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Coverage analysis error: {str(e)}"
            }
    
    def _parse_coverage_xml(self) -> Dict[str, Any]:
        """Parse coverage XML file."""
        coverage_file = Path("coverage.xml")
        if not coverage_file.exists():
            return {"overall_coverage": 0.0, "files": {}}
        
        try:
            tree = ET.parse(coverage_file)
            root = tree.getroot()
            
            # Get overall coverage
            overall_coverage = 0.0
            if root.attrib.get("line-rate"):
                overall_coverage = float(root.attrib["line-rate"]) * 100
            
            # Parse file-level coverage
            files_coverage = {}
            for package in root.findall(".//package"):
                for class_elem in package.findall("classes/class"):
                    filename = class_elem.attrib.get("filename", "")
                    line_rate = float(class_elem.attrib.get("line-rate", 0))
                    
                    # Get uncovered lines
                    uncovered_lines = []
                    for line in class_elem.findall("lines/line"):
                        if line.attrib.get("hits") == "0":
                            uncovered_lines.append(int(line.attrib["number"]))
                    
                    files_coverage[filename] = {
                        "coverage": line_rate * 100,
                        "uncovered_lines": uncovered_lines
                    }
            
            return {
                "overall_coverage": overall_coverage,
                "files": files_coverage
            }
            
        except Exception as e:
            print(f"Error parsing coverage XML: {e}")
            return {"overall_coverage": 0.0, "files": {}}
    
    def _analyze_by_component(self, coverage_data: Dict[str, Any]) -> Dict[str, CoverageTarget]:
        """Analyze coverage by component."""
        files_coverage = coverage_data.get("files", {})
        
        # Categorize files by component
        component_files = {
            "core": [],
            "domain": [],
            "integration": [],
            "other": []
        }
        
        for file_path, file_data in files_coverage.items():
            if any(component in file_path.lower() for component in ["frame", "config", "interfaces", "core"]):
                component_files["core"].append(file_data["coverage"])
            elif any(component in file_path.lower() for component in ["pose", "gait", "classification", "analysis"]):
                component_files["domain"].append(file_data["coverage"])
            elif any(component in file_path.lower() for component in ["api", "video", "storage", "server"]):
                component_files["integration"].append(file_data["coverage"])
            else:
                component_files["other"].append(file_data["coverage"])
        
        # Calculate component averages
        component_analysis = {}
        for component, target in self.coverage_targets.items():
            if component == "overall":
                target.current_coverage = coverage_data.get("overall_coverage", 0.0)
            else:
                coverages = component_files.get(component, [])
                if coverages:
                    target.current_coverage = sum(coverages) / len(coverages)
                else:
                    target.current_coverage = 0.0
            
            component_analysis[component] = target
        
        return component_analysis
    
    def _identify_coverage_gaps(self, coverage_data: Dict[str, Any]) -> List[CoverageGap]:
        """Identify significant coverage gaps."""
        gaps = []
        files_coverage = coverage_data.get("files", {})
        
        for file_path, file_data in files_coverage.items():
            coverage_pct = file_data["coverage"]
            uncovered_lines = file_data["uncovered_lines"]
            
            # Determine component
            component = "other"
            if any(comp in file_path.lower() for comp in ["frame", "config", "interfaces", "core"]):
                component = "core"
            elif any(comp in file_path.lower() for comp in ["pose", "gait", "classification", "analysis"]):
                component = "domain"
            elif any(comp in file_path.lower() for comp in ["api", "video", "storage", "server"]):
                component = "integration"
            
            # Determine priority based on coverage level and component importance
            if coverage_pct < 50:
                priority = "high"
            elif coverage_pct < 70:
                priority = "medium"
            elif coverage_pct < 85 and component in ["core", "domain"]:
                priority = "medium"
            else:
                priority = "low"
            
            # Only include significant gaps
            if coverage_pct < 85 or len(uncovered_lines) > 10:
                gaps.append(CoverageGap(
                    file_path=file_path,
                    uncovered_lines=uncovered_lines,
                    coverage_percentage=coverage_pct,
                    component=component,
                    priority=priority
                ))
        
        # Sort by priority and coverage percentage
        priority_order = {"high": 0, "medium": 1, "low": 2}
        gaps.sort(key=lambda g: (priority_order[g.priority], g.coverage_percentage))
        
        return gaps
    
    def _generate_recommendations(self, component_analysis: Dict[str, CoverageTarget], 
                                coverage_gaps: List[CoverageGap]) -> List[str]:
        """Generate recommendations for improving coverage."""
        recommendations = []
        
        # Component-level recommendations
        for component, target in component_analysis.items():
            if not target.meets_minimum:
                recommendations.append(
                    f"CRITICAL: {component.title()} coverage ({target.current_coverage:.1f}%) "
                    f"is below minimum requirement ({target.minimum_coverage:.1f}%)"
                )
            elif not target.meets_target:
                recommendations.append(
                    f"Improve {component} coverage from {target.current_coverage:.1f}% "
                    f"to target {target.target_coverage:.1f}%"
                )
        
        # File-level recommendations
        high_priority_gaps = [g for g in coverage_gaps if g.priority == "high"]
        if high_priority_gaps:
            recommendations.append(
                f"Address {len(high_priority_gaps)} high-priority coverage gaps"
            )
            
            for gap in high_priority_gaps[:3]:  # Top 3 high-priority gaps
                recommendations.append(
                    f"  - {gap.file_path}: {gap.coverage_percentage:.1f}% coverage, "
                    f"{len(gap.uncovered_lines)} uncovered lines"
                )
        
        # General recommendations
        medium_priority_gaps = [g for g in coverage_gaps if g.priority == "medium"]
        if len(medium_priority_gaps) > 5:
            recommendations.append(
                f"Consider addressing {len(medium_priority_gaps)} medium-priority coverage gaps"
            )
        
        if not recommendations:
            recommendations.append("Coverage targets are being met! Consider raising targets for continuous improvement.")
        
        return recommendations
    
    def _check_targets_met(self, component_analysis: Dict[str, CoverageTarget]) -> Dict[str, bool]:
        """Check which targets are being met."""
        return {
            component: target.meets_minimum 
            for component, target in component_analysis.items()
        }
    
    def generate_coverage_report(self) -> str:
        """Generate a comprehensive coverage report."""
        analysis = self.run_coverage_analysis()
        
        if not analysis["success"]:
            return f"Coverage analysis failed: {analysis['error']}"
        
        report = []
        report.append("# Coverage Analysis Report")
        report.append(f"Overall Coverage: {analysis['overall_coverage']:.1f}%")
        report.append("")
        
        # Component analysis
        report.append("## Component Coverage")
        for component, target in analysis["component_analysis"].items():
            status = "✅" if target.meets_minimum else "❌"
            report.append(
                f"{status} {component.title()}: {target.current_coverage:.1f}% "
                f"(min: {target.minimum_coverage:.1f}%, target: {target.target_coverage:.1f}%)"
            )
        report.append("")
        
        # Coverage gaps
        if analysis["coverage_gaps"]:
            report.append("## Coverage Gaps")
            for gap in analysis["coverage_gaps"][:10]:  # Top 10 gaps
                report.append(
                    f"- {gap.file_path}: {gap.coverage_percentage:.1f}% "
                    f"({len(gap.uncovered_lines)} uncovered lines) [{gap.priority} priority]"
                )
            report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        for rec in analysis["recommendations"]:
            report.append(f"- {rec}")
        
        return "\n".join(report)
    
    def save_coverage_report(self, output_file: Path = None) -> Path:
        """Save coverage report to file."""
        output_file = output_file or Path("coverage_report.md")
        
        report = self.generate_coverage_report()
        
        with open(output_file, 'w') as f:
            f.write(report)
        
        return output_file


def main():
    """Run coverage analysis from command line."""
    analyzer = CoverageAnalyzer()
    
    print("Running coverage analysis...")
    analysis = analyzer.run_coverage_analysis()
    
    if analysis["success"]:
        print(f"Overall Coverage: {analysis['overall_coverage']:.1f}%")
        
        # Check if targets are met
        targets_met = analysis["targets_met"]
        if all(targets_met.values()):
            print("✅ All coverage targets are being met!")
        else:
            print("❌ Some coverage targets are not being met:")
            for component, met in targets_met.items():
                if not met:
                    target = analysis["component_analysis"][component]
                    print(f"  - {component}: {target.current_coverage:.1f}% < {target.minimum_coverage:.1f}%")
        
        # Save detailed report
        report_file = analyzer.save_coverage_report()
        print(f"Detailed report saved to: {report_file}")
        
    else:
        print(f"Coverage analysis failed: {analysis['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()