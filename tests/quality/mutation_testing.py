"""Mutation testing for critical components."""

import ast
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import random
import time


@dataclass
class MutationResult:
    """Result of a mutation test."""
    mutant_id: str
    file_path: str
    line_number: int
    original_code: str
    mutated_code: str
    killed: bool  # True if tests detected the mutation
    test_output: str
    execution_time: float


class MutationOperator:
    """Base class for mutation operators."""
    
    def can_mutate(self, node: ast.AST) -> bool:
        """Check if this operator can mutate the given AST node."""
        raise NotImplementedError
    
    def mutate(self, node: ast.AST) -> List[ast.AST]:
        """Generate mutations for the given AST node."""
        raise NotImplementedError
    
    def get_description(self) -> str:
        """Get description of this mutation operator."""
        raise NotImplementedError


class ArithmeticOperatorMutation(MutationOperator):
    """Mutate arithmetic operators (+, -, *, /, %)."""
    
    MUTATIONS = {
        ast.Add: [ast.Sub, ast.Mult],
        ast.Sub: [ast.Add, ast.Mult],
        ast.Mult: [ast.Add, ast.Sub, ast.Div],
        ast.Div: [ast.Mult, ast.Sub],
        ast.Mod: [ast.Mult, ast.Add]
    }
    
    def can_mutate(self, node: ast.AST) -> bool:
        return isinstance(node, ast.BinOp) and type(node.op) in self.MUTATIONS
    
    def mutate(self, node: ast.AST) -> List[ast.AST]:
        if not self.can_mutate(node):
            return []
        
        mutations = []
        original_op = type(node.op)
        
        for new_op_class in self.MUTATIONS[original_op]:
            mutated = ast.copy_location(
                ast.BinOp(left=node.left, op=new_op_class(), right=node.right),
                node
            )
            mutations.append(mutated)
        
        return mutations
    
    def get_description(self) -> str:
        return "Arithmetic operator mutation (+, -, *, /, %)"


class ComparisonOperatorMutation(MutationOperator):
    """Mutate comparison operators (<, >, <=, >=, ==, !=)."""
    
    MUTATIONS = {
        ast.Lt: [ast.Gt, ast.LtE, ast.GtE],
        ast.Gt: [ast.Lt, ast.LtE, ast.GtE],
        ast.LtE: [ast.Lt, ast.Gt, ast.GtE],
        ast.GtE: [ast.Lt, ast.Gt, ast.LtE],
        ast.Eq: [ast.NotEq],
        ast.NotEq: [ast.Eq]
    }
    
    def can_mutate(self, node: ast.AST) -> bool:
        return isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in self.MUTATIONS
    
    def mutate(self, node: ast.AST) -> List[ast.AST]:
        if not self.can_mutate(node):
            return []
        
        mutations = []
        original_op = type(node.ops[0])
        
        for new_op_class in self.MUTATIONS[original_op]:
            mutated = ast.copy_location(
                ast.Compare(left=node.left, ops=[new_op_class()], comparators=node.comparators),
                node
            )
            mutations.append(mutated)
        
        return mutations
    
    def get_description(self) -> str:
        return "Comparison operator mutation (<, >, <=, >=, ==, !=)"


class BooleanOperatorMutation(MutationOperator):
    """Mutate boolean operators (and, or)."""
    
    def can_mutate(self, node: ast.AST) -> bool:
        return isinstance(node, ast.BoolOp)
    
    def mutate(self, node: ast.AST) -> List[ast.AST]:
        if not self.can_mutate(node):
            return []
        
        mutations = []
        
        if isinstance(node.op, ast.And):
            mutated = ast.copy_location(
                ast.BoolOp(op=ast.Or(), values=node.values),
                node
            )
            mutations.append(mutated)
        elif isinstance(node.op, ast.Or):
            mutated = ast.copy_location(
                ast.BoolOp(op=ast.And(), values=node.values),
                node
            )
            mutations.append(mutated)
        
        return mutations
    
    def get_description(self) -> str:
        return "Boolean operator mutation (and <-> or)"


class ConstantMutation(MutationOperator):
    """Mutate constants (numbers, booleans)."""
    
    def can_mutate(self, node: ast.AST) -> bool:
        return isinstance(node, (ast.Constant, ast.Num, ast.NameConstant))
    
    def mutate(self, node: ast.AST) -> List[ast.AST]:
        if not self.can_mutate(node):
            return []
        
        mutations = []
        
        # Handle different constant types
        if isinstance(node, ast.Constant):
            value = node.value
        elif isinstance(node, ast.Num):
            value = node.n
        elif isinstance(node, ast.NameConstant):
            value = node.value
        else:
            return []
        
        # Mutate based on type
        if isinstance(value, bool):
            mutated = ast.copy_location(ast.Constant(value=not value), node)
            mutations.append(mutated)
        elif isinstance(value, (int, float)):
            if value == 0:
                mutations.append(ast.copy_location(ast.Constant(value=1), node))
            elif value == 1:
                mutations.append(ast.copy_location(ast.Constant(value=0), node))
            else:
                mutations.append(ast.copy_location(ast.Constant(value=0), node))
                mutations.append(ast.copy_location(ast.Constant(value=value + 1), node))
                mutations.append(ast.copy_location(ast.Constant(value=value - 1), node))
        
        return mutations
    
    def get_description(self) -> str:
        return "Constant mutation (numbers, booleans)"


class MutationTester:
    """Perform mutation testing on critical components."""
    
    def __init__(self, target_files: List[Path] = None):
        self.target_files = target_files or []
        self.operators = [
            ArithmeticOperatorMutation(),
            ComparisonOperatorMutation(),
            BooleanOperatorMutation(),
            ConstantMutation()
        ]
        self.results: List[MutationResult] = []
    
    def discover_critical_files(self) -> List[Path]:
        """Discover critical files for mutation testing."""
        if self.target_files:
            return self.target_files
        
        # Focus on core components
        critical_patterns = [
            "ambient/core/*.py",
            "ambient/analysis/*.py",
            "ambient/classification/*.py"
        ]
        
        critical_files = []
        for pattern in critical_patterns:
            critical_files.extend(Path(".").glob(pattern))
        
        # Filter out __init__.py and test files
        critical_files = [
            f for f in critical_files 
            if f.name != "__init__.py" and not f.name.startswith("test_")
        ]
        
        return critical_files[:5]  # Limit to 5 files for performance
    
    def generate_mutations(self, file_path: Path) -> List[Tuple[ast.AST, int, str, str]]:
        """Generate mutations for a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            tree = ast.parse(source_code)
            mutations = []
            
            for node in ast.walk(tree):
                for operator in self.operators:
                    if operator.can_mutate(node):
                        mutated_nodes = operator.mutate(node)
                        for mutated_node in mutated_nodes:
                            # Get original code snippet
                            original_code = ast.unparse(node) if hasattr(ast, 'unparse') else str(node)
                            mutated_code = ast.unparse(mutated_node) if hasattr(ast, 'unparse') else str(mutated_node)
                            
                            mutations.append((
                                mutated_node,
                                getattr(node, 'lineno', 0),
                                original_code,
                                mutated_code
                            ))
            
            return mutations
            
        except Exception as e:
            print(f"Error generating mutations for {file_path}: {e}")
            return []
    
    def apply_mutation(self, file_path: Path, original_node: ast.AST, mutated_node: ast.AST) -> Path:
        """Apply a mutation to a file and return the mutated file path."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            tree = ast.parse(source_code)
            
            # Replace the node in the AST
            class MutationTransformer(ast.NodeTransformer):
                def __init__(self, target_node, replacement_node):
                    self.target_node = target_node
                    self.replacement_node = replacement_node
                    self.replaced = False
                
                def visit(self, node):
                    if (not self.replaced and 
                        type(node) == type(self.target_node) and
                        getattr(node, 'lineno', None) == getattr(self.target_node, 'lineno', None)):
                        self.replaced = True
                        return self.replacement_node
                    return self.generic_visit(node)
            
            transformer = MutationTransformer(original_node, mutated_node)
            mutated_tree = transformer.visit(tree)
            
            # Generate mutated source code
            if hasattr(ast, 'unparse'):
                mutated_source = ast.unparse(mutated_tree)
            else:
                # Fallback for older Python versions
                import astor
                mutated_source = astor.to_source(mutated_tree)
            
            # Write to temporary file
            temp_file = tempfile.NamedTemporaryFile(
                mode='w', suffix='.py', delete=False, encoding='utf-8'
            )
            temp_file.write(mutated_source)
            temp_file.close()
            
            return Path(temp_file.name)
            
        except Exception as e:
            print(f"Error applying mutation to {file_path}: {e}")
            return None
    
    def run_tests_against_mutant(self, mutant_file: Path, original_file: Path) -> Tuple[bool, str, float]:
        """Run tests against a mutant and check if the mutation is killed."""
        try:
            # Backup original file
            backup_file = original_file.with_suffix('.backup')
            shutil.copy2(original_file, backup_file)
            
            # Replace original with mutant
            shutil.copy2(mutant_file, original_file)
            
            # Run tests
            start_time = time.time()
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                f"tests/", "-x", "--tb=no", "-q"
            ], capture_output=True, text=True, timeout=60)
            execution_time = time.time() - start_time
            
            # Mutation is "killed" if tests fail (return code != 0)
            killed = result.returncode != 0
            
            return killed, result.stdout + result.stderr, execution_time
            
        except subprocess.TimeoutExpired:
            return False, "Test execution timed out", 60.0
        except Exception as e:
            return False, f"Error running tests: {str(e)}", 0.0
        finally:
            # Restore original file
            if backup_file.exists():
                shutil.copy2(backup_file, original_file)
                backup_file.unlink()
            
            # Clean up mutant file
            if mutant_file.exists():
                mutant_file.unlink()
    
    def run_mutation_testing(self, max_mutations_per_file: int = 10) -> Dict[str, Any]:
        """Run mutation testing on critical files."""
        print("Starting mutation testing...")
        
        critical_files = self.discover_critical_files()
        if not critical_files:
            return {
                "success": False,
                "error": "No critical files found for mutation testing"
            }
        
        total_mutations = 0
        killed_mutations = 0
        
        for file_path in critical_files:
            print(f"Testing mutations in {file_path}...")
            
            mutations = self.generate_mutations(file_path)
            if not mutations:
                continue
            
            # Limit mutations per file for performance
            selected_mutations = random.sample(
                mutations, 
                min(len(mutations), max_mutations_per_file)
            )
            
            for i, (mutated_node, line_no, original_code, mutated_code) in enumerate(selected_mutations):
                mutant_id = f"{file_path.stem}_{line_no}_{i}"
                
                # Apply mutation
                mutant_file = self.apply_mutation(file_path, mutations[0][0], mutated_node)
                if not mutant_file:
                    continue
                
                # Test the mutant
                killed, test_output, exec_time = self.run_tests_against_mutant(mutant_file, file_path)
                
                result = MutationResult(
                    mutant_id=mutant_id,
                    file_path=str(file_path),
                    line_number=line_no,
                    original_code=original_code,
                    mutated_code=mutated_code,
                    killed=killed,
                    test_output=test_output,
                    execution_time=exec_time
                )
                
                self.results.append(result)
                total_mutations += 1
                if killed:
                    killed_mutations += 1
                
                status = "KILLED" if killed else "SURVIVED"
                print(f"  Mutant {mutant_id}: {status}")
        
        # Calculate mutation score
        mutation_score = (killed_mutations / total_mutations * 100) if total_mutations > 0 else 0
        
        return {
            "success": True,
            "total_mutations": total_mutations,
            "killed_mutations": killed_mutations,
            "survived_mutations": total_mutations - killed_mutations,
            "mutation_score": mutation_score,
            "results": [self._result_to_dict(r) for r in self.results],
            "summary": self._generate_mutation_summary()
        }
    
    def _result_to_dict(self, result: MutationResult) -> Dict[str, Any]:
        """Convert MutationResult to dictionary."""
        return {
            "mutant_id": result.mutant_id,
            "file_path": result.file_path,
            "line_number": result.line_number,
            "original_code": result.original_code,
            "mutated_code": result.mutated_code,
            "killed": result.killed,
            "execution_time": result.execution_time
        }
    
    def _generate_mutation_summary(self) -> str:
        """Generate summary of mutation testing results."""
        if not self.results:
            return "No mutations tested"
        
        total = len(self.results)
        killed = len([r for r in self.results if r.killed])
        survived = total - killed
        score = (killed / total * 100) if total > 0 else 0
        
        summary = []
        summary.append(f"Mutation Testing Results:")
        summary.append(f"  Total mutations: {total}")
        summary.append(f"  Killed: {killed}")
        summary.append(f"  Survived: {survived}")
        summary.append(f"  Mutation score: {score:.1f}%")
        
        if survived > 0:
            summary.append(f"  Survived mutations indicate potential test gaps")
        
        return "\n".join(summary)
    
    def generate_mutation_report(self) -> str:
        """Generate detailed mutation testing report."""
        if not self.results:
            return "No mutation testing results available"
        
        report = []
        report.append("# Mutation Testing Report")
        report.append("")
        report.append(self._generate_mutation_summary())
        report.append("")
        
        # Survived mutations (potential test gaps)
        survived = [r for r in self.results if not r.killed]
        if survived:
            report.append("## Survived Mutations (Potential Test Gaps)")
            for result in survived:
                report.append(f"- {result.file_path}:{result.line_number}")
                report.append(f"  Original: `{result.original_code}`")
                report.append(f"  Mutated:  `{result.mutated_code}`")
                report.append("")
        
        # Operator effectiveness
        operator_stats = {}
        for result in self.results:
            # Simple heuristic to categorize mutations
            if any(op in result.mutated_code for op in ['+', '-', '*', '/']):
                op_type = "arithmetic"
            elif any(op in result.mutated_code for op in ['<', '>', '==', '!=']):
                op_type = "comparison"
            elif any(op in result.mutated_code for op in ['and', 'or']):
                op_type = "boolean"
            else:
                op_type = "constant"
            
            if op_type not in operator_stats:
                operator_stats[op_type] = {"total": 0, "killed": 0}
            
            operator_stats[op_type]["total"] += 1
            if result.killed:
                operator_stats[op_type]["killed"] += 1
        
        if operator_stats:
            report.append("## Mutation Operator Effectiveness")
            for op_type, stats in operator_stats.items():
                score = (stats["killed"] / stats["total"] * 100) if stats["total"] > 0 else 0
                report.append(f"- {op_type.title()}: {stats['killed']}/{stats['total']} killed ({score:.1f}%)")
        
        return "\n".join(report)


def main():
    """Run mutation testing from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run mutation testing")
    parser.add_argument("--files", nargs="+", help="Specific files to test")
    parser.add_argument("--max-mutations", type=int, default=10, help="Max mutations per file")
    parser.add_argument("--report", type=str, help="Save report to file")
    
    args = parser.parse_args()
    
    target_files = [Path(f) for f in args.files] if args.files else None
    tester = MutationTester(target_files)
    
    results = tester.run_mutation_testing(max_mutations_per_file=args.max_mutations)
    
    if results["success"]:
        print(f"\nMutation Score: {results['mutation_score']:.1f}%")
        print(f"Mutations: {results['killed_mutations']}/{results['total_mutations']} killed")
        
        if args.report:
            report = tester.generate_mutation_report()
            with open(args.report, 'w') as f:
                f.write(report)
            print(f"Report saved to: {args.report}")
    else:
        print(f"Mutation testing failed: {results['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()