#!/usr/bin/env python3
# ABOUTME: Contract test report generator and analyzer
# ABOUTME: Provides comprehensive analysis of contract test coverage and results

import subprocess
from pathlib import Path
from typing import Dict, Any
from datetime import datetime


class ContractTestReporter:
    """Generate comprehensive reports for contract test results."""

    def __init__(self):
        self.test_results = {}
        self.coverage_data = {}
        self.performance_data = {}

    def run_contract_tests(self) -> Dict[str, Any]:
        """Run all contract tests and collect results."""
        try:
            # Run basic contract tests
            basic_result = subprocess.run(
                [
                    "uv",
                    "run",
                    "pytest",
                    "tests/contract/",
                    "-v",
                    "--tb=short",
                    "-m",
                    "contract",
                    "--json-report",
                    "--json-report-file=basic_contract_results.json",
                ],
                capture_output=True,
                text=True,
                cwd=".",
            )

            # Run enhanced contract tests
            enhanced_result = subprocess.run(
                [
                    "uv",
                    "run",
                    "pytest",
                    "tests/contract/",
                    "-v",
                    "--tb=short",
                    "-k",
                    "enhanced",
                    "--json-report",
                    "--json-report-file=enhanced_contract_results.json",
                ],
                capture_output=True,
                text=True,
                cwd=".",
            )

            return {
                "basic": {
                    "exit_code": basic_result.returncode,
                    "stdout": basic_result.stdout,
                    "stderr": basic_result.stderr,
                },
                "enhanced": {
                    "exit_code": enhanced_result.returncode,
                    "stdout": enhanced_result.stdout,
                    "stderr": enhanced_result.stderr,
                },
            }

        except Exception as e:
            return {"error": str(e)}

    def analyze_interface_coverage(self) -> Dict[str, Any]:
        """Analyze contract test coverage across interfaces."""
        interfaces_dir = Path("core/interfaces")
        contract_tests_dir = Path("tests/contract")

        coverage_analysis = {
            "total_interfaces": 0,
            "interfaces_with_basic_tests": 0,
            "interfaces_with_enhanced_tests": 0,
            "coverage_by_category": {},
            "missing_tests": [],
        }

        # Count interfaces
        interface_files = []
        for category_dir in interfaces_dir.iterdir():
            if category_dir.is_dir() and category_dir.name != "__pycache__":
                for file in category_dir.glob("*.py"):
                    if file.name != "__init__.py":
                        interface_files.append((category_dir.name, file.stem))

        coverage_analysis["total_interfaces"] = len(interface_files)

        # Check for corresponding tests
        for category, interface_name in interface_files:
            basic_test_file = contract_tests_dir / category / f"test_{interface_name}_contract.py"
            enhanced_test_file = contract_tests_dir / category / f"test_{interface_name}_enhanced_contract.py"

            if basic_test_file.exists():
                coverage_analysis["interfaces_with_basic_tests"] += 1

            if enhanced_test_file.exists():
                coverage_analysis["interfaces_with_enhanced_tests"] += 1

            if not basic_test_file.exists():
                coverage_analysis["missing_tests"].append(f"{category}/{interface_name}")

            # Track by category
            if category not in coverage_analysis["coverage_by_category"]:
                coverage_analysis["coverage_by_category"][category] = {
                    "total": 0,
                    "basic_tests": 0,
                    "enhanced_tests": 0,
                }

            coverage_analysis["coverage_by_category"][category]["total"] += 1
            if basic_test_file.exists():
                coverage_analysis["coverage_by_category"][category]["basic_tests"] += 1
            if enhanced_test_file.exists():
                coverage_analysis["coverage_by_category"][category]["enhanced_tests"] += 1

        return coverage_analysis

    def generate_report(self) -> str:
        """Generate a comprehensive contract test report."""
        print("🔍 Generating Contract Test Report...")

        # Run tests
        test_results = self.run_contract_tests()

        # Analyze coverage
        coverage_analysis = self.analyze_interface_coverage()

        # Generate report
        report = f"""
╭─ Contract Test Report ─────────────────────────────────────────────────────────────────────╮
│                                                                                            │
│                           📊 Trading Chart Core - Contract Test Analysis                   │
│                                    Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}                        │
│                                                                                            │
├─ 🎯 Interface Coverage Analysis ──────────────────────────────────────────────────────────┤
│                                                                                            │
│  Total Interfaces: {coverage_analysis["total_interfaces"]:>3}                                                                   │
│  With Basic Tests: {coverage_analysis["interfaces_with_basic_tests"]:>3} ({coverage_analysis["interfaces_with_basic_tests"] / coverage_analysis["total_interfaces"] * 100:>5.1f}%)                                                │
│  With Enhanced Tests: {coverage_analysis["interfaces_with_enhanced_tests"]:>2} ({coverage_analysis["interfaces_with_enhanced_tests"] / coverage_analysis["total_interfaces"] * 100:>5.1f}%)                                               │
│                                                                                            │
├─ 📈 Coverage by Category ─────────────────────────────────────────────────────────────────┤
│                                                                                            │"""

        for category, data in coverage_analysis["coverage_by_category"].items():
            basic_pct = (data["basic_tests"] / data["total"] * 100) if data["total"] > 0 else 0
            enhanced_pct = (data["enhanced_tests"] / data["total"] * 100) if data["total"] > 0 else 0

            report += f"""
│  {category.capitalize():>12}: {data["total"]:>2} interfaces, {data["basic_tests"]:>2} basic ({basic_pct:>5.1f}%), {data["enhanced_tests"]:>2} enhanced ({enhanced_pct:>5.1f}%)     │"""

        report += """
│                                                                                            │
├─ 🚨 Missing Contract Tests ───────────────────────────────────────────────────────────────┤
│                                                                                            │"""

        if coverage_analysis["missing_tests"]:
            for missing in coverage_analysis["missing_tests"]:
                report += f"""
│  ❌ {missing:<75} │"""
        else:
            report += """
│  ✅ All interfaces have contract tests!                                                   │"""

        report += """
│                                                                                            │
├─ 🧪 Test Execution Results ───────────────────────────────────────────────────────────────┤
│                                                                                            │"""

        if "basic" in test_results:
            basic_status = "✅ PASSED" if test_results["basic"]["exit_code"] == 0 else "❌ FAILED"
            report += f"""
│  Basic Contract Tests: {basic_status}                                                     │"""

        if "enhanced" in test_results:
            enhanced_status = "✅ PASSED" if test_results["enhanced"]["exit_code"] == 0 else "❌ FAILED"
            report += f"""
│  Enhanced Contract Tests: {enhanced_status}                                               │"""

        report += """
│                                                                                            │
├─ 🎯 Recommendations ──────────────────────────────────────────────────────────────────────┤
│                                                                                            │"""

        # Generate recommendations
        basic_coverage = coverage_analysis["interfaces_with_basic_tests"] / coverage_analysis["total_interfaces"]
        enhanced_coverage = coverage_analysis["interfaces_with_enhanced_tests"] / coverage_analysis["total_interfaces"]

        if basic_coverage < 1.0:
            report += """
│  📝 Create basic contract tests for missing interfaces                                    │"""

        if enhanced_coverage < 0.5:
            report += """
│  🚀 Add enhanced contract tests for better behavior verification                          │"""

        if basic_coverage >= 1.0 and enhanced_coverage >= 0.8:
            report += """
│  🎉 Excellent contract test coverage! Consider adding performance benchmarks             │"""

        report += """
│                                                                                            │
╰────────────────────────────────────────────────────────────────────────────────────────────╯
"""

        return report


def main():
    """Generate and display contract test report."""
    reporter = ContractTestReporter()
    report = reporter.generate_report()
    print(report)


if __name__ == "__main__":
    main()
