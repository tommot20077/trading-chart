#!/usr/bin/env python3
# ABOUTME: Script to run benchmark tests with detailed reporting
# ABOUTME: Uses pytest-benchmark for accurate performance measurements

import subprocess
import sys
import argparse
from pathlib import Path


def run_benchmarks(
    save_baseline: bool = False,
    compare_baseline: str = None,
    output_format: str = "table",
    benchmark_only: bool = True
):
    """Run benchmark tests with various options."""
    
    print("🚀 Starting Benchmark Test Suite")
    print("=" * 50)
    
    # Change to project root
    project_root = Path(__file__).parent.parent
    
    # Base command
    cmd = [
        "uv", "run", "pytest",
        "src/core/tests/benchmark/",
        "-v"
    ]
    
    # Add benchmark-specific options
    if benchmark_only:
        cmd.extend(["-m", "benchmark"])
    
    # Benchmark reporting options
    benchmark_args = []
    
    if save_baseline:
        benchmark_args.extend(["--benchmark-save=baseline"])
        print("💾 Saving benchmark results as baseline")
    
    if compare_baseline:
        benchmark_args.extend([f"--benchmark-compare={compare_baseline}"])
        print(f"📊 Comparing against baseline: {compare_baseline}")
    
    # Output format
    if output_format == "json":
        benchmark_args.extend(["--benchmark-json=benchmark_results.json"])
        print("📄 Saving results to benchmark_results.json")
    elif output_format == "histogram":
        benchmark_args.extend(["--benchmark-histogram=benchmark_histogram"])
        print("📈 Generating histogram: benchmark_histogram.svg")
    
    # Add benchmark configuration
    benchmark_args.extend([
        "--benchmark-min-rounds=5",
        "--benchmark-max-time=2.0",
        "--benchmark-warmup=on",
        "--benchmark-sort=mean"
    ])
    
    cmd.extend(benchmark_args)
    
    print(f"🔧 Running command: {' '.join(cmd)}")
    print("-" * 50)
    
    try:
        result = subprocess.run(
            cmd,
            cwd=project_root,
            text=True
        )
        
        if result.returncode == 0:
            print("\n✅ Benchmark tests completed successfully!")
            
            if output_format == "json":
                print("📄 Results saved to benchmark_results.json")
            elif output_format == "histogram":
                print("📈 Histogram saved to benchmark_histogram.svg")
                
        else:
            print("\n❌ Benchmark tests failed!")
            return 1
            
    except Exception as e:
        print(f"💥 Error running benchmarks: {e}")
        return 1
    
    return 0


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Run benchmark tests for trading-chart-core"
    )
    
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="Save benchmark results as baseline for future comparisons"
    )
    
    parser.add_argument(
        "--compare",
        type=str,
        help="Compare against a saved baseline (e.g., 'baseline')"
    )
    
    parser.add_argument(
        "--format",
        choices=["table", "json", "histogram"],
        default="table",
        help="Output format for benchmark results"
    )
    
    parser.add_argument(
        "--all-tests",
        action="store_true",
        help="Run all tests, not just benchmark tests"
    )
    
    args = parser.parse_args()
    
    return run_benchmarks(
        save_baseline=args.save_baseline,
        compare_baseline=args.compare,
        output_format=args.format,
        benchmark_only=not args.all_tests
    )


if __name__ == "__main__":
    sys.exit(main())