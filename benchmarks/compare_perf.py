#!/usr/bin/env python3
"""
Compare MUSA benchmark results with NVIDIA baseline data.

Usage:
    python compare_perf.py <nvidia_baseline.json> <musa_result.json>
    python compare_perf.py --batch <nvidia_baseline.json> <musa_results_dir/>

Examples:
    # Compare single result
    python compare_perf.py results/nvidia_baseline.json musa_result.json

    # Compare all JSON files in a directory
    python compare_perf.py --batch results/nvidia_baseline.json results/musa/
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Any


def load_json_file(path: str) -> Any:
    """Load JSON from file."""
    with open(path, 'r') as f:
        return json.load(f)


def save_json_file(path: str, data: Any) -> None:
    """Save data to JSON file."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_nvidia_baseline(json_path: str) -> Dict[str, Dict]:
    """Load NVIDIA baseline results as a dict keyed by benchmark name."""
    data = load_json_file(json_path)

    # Handle both single object and array formats
    if isinstance(data, dict):
        items = [data]
    else:
        items = data

    # Create lookup dict by benchmark name and type
    result = {}
    for item in items:
        key = f"{item['benchmark']}:{item['type']}"
        result[key] = item

    return result


def compare_benchmark(nvidia_baseline: Dict[str, Dict],
                      musa_result: Dict) -> Optional[Dict]:
    """Compare a single benchmark result with NVIDIA baseline."""
    benchmark = musa_result.get("benchmark", "")
    type_name = musa_result.get("type", "")

    # Create lookup key
    key = f"{benchmark}:{type_name}"

    if key not in nvidia_baseline:
        # Try without type for backward compatibility
        if benchmark in nvidia_baseline:
            nvidia = nvidia_baseline[benchmark]
        else:
            return None
    else:
        nvidia = nvidia_baseline[key]

    # Calculate comparison metrics
    musa_throughput = musa_result.get("throughput_gb_s", 0)
    nvidia_throughput = nvidia.get("throughput_gb_s", 0)

    if nvidia_throughput > 0:
        throughput_ratio = musa_throughput / nvidia_throughput
    else:
        throughput_ratio = 0.0

    nvidia_time = nvidia.get("avg_time_ms", 0)
    musa_time = musa_result.get("avg_time_ms", 0)

    if musa_time > 0:
        speedup = nvidia_time / musa_time
    else:
        speedup = 0.0

    return {
        "benchmark": benchmark,
        "type": type_name,
        "elements": musa_result.get("elements", 0),
        "nvidia_throughput_gbs": round(nvidia_throughput, 2),
        "musa_throughput_gbs": round(musa_throughput, 2),
        "nvidia_time_ms": round(nvidia_time, 3),
        "musa_time_ms": round(musa_time, 3),
        "throughput_ratio": round(throughput_ratio, 3),
        "speedup": round(speedup, 3),
    }


def print_report(comparisons: List[Dict], show_details: bool = True) -> None:
    """Print comparison report."""
    # Filter out None results
    valid_comparisons = [c for c in comparisons if c is not None]

    if not valid_comparisons:
        print("No matching benchmarks found for comparison.")
        return

    print("\n" + "=" * 90)
    print("NVIDIA vs MUSA Performance Comparison Report")
    print("=" * 90 + "\n")

    if show_details:
        # Detailed table
        print(f"{'Benchmark':<40} {'Type':<10} {'NVIDIA GB/s':>12} {'MUSA GB/s':>12} {'Ratio':>8}")
        print("-" * 90)

        for c in sorted(valid_comparisons, key=lambda x: x["benchmark"]):
            ratio_str = f"{c['throughput_ratio']:.1%}"
            print(f"{c['benchmark']:<40} {c['type']:<10} "
                  f"{c['nvidia_throughput_gbs']:>12.2f} {c['musa_throughput_gbs']:>12.2f} {ratio_str:>8}")

        print("-" * 90)

    # Summary statistics
    total_ratio = sum(c["throughput_ratio"] for c in valid_comparisons)
    avg_ratio = total_ratio / len(valid_comparisons)

    min_ratio = min(c["throughput_ratio"] for c in valid_comparisons)
    max_ratio = max(c["throughput_ratio"] for c in valid_comparisons)

    print(f"\n{'Summary Statistics':^90}")
    print("-" * 90)
    print(f"  Total Benchmarks Compared: {len(valid_comparisons)}")
    print(f"  Average Throughput Ratio:  {avg_ratio:.1%}")
    print(f"  Min Throughput Ratio:      {min_ratio:.1%}")
    print(f"  Max Throughput Ratio:      {max_ratio:.1%}")

    # Performance categories
    excellent = sum(1 for c in valid_comparisons if c["throughput_ratio"] >= 0.8)
    good = sum(1 for c in valid_comparisons if 0.5 <= c["throughput_ratio"] < 0.8)
    needs_work = sum(1 for c in valid_comparisons if c["throughput_ratio"] < 0.5)

    print(f"\n  Performance Categories:")
    print(f"    >= 80% (Excellent):  {excellent} benchmarks")
    print(f"    50-80% (Good):       {good} benchmarks")
    print(f"    < 50% (Needs Work):  {needs_work} benchmarks")
    print()


def print_csv_report(comparisons: List[Dict]) -> None:
    """Print comparison as CSV for easy import into spreadsheets."""
    valid_comparisons = [c for c in comparisons if c is not None]

    # Header
    print("benchmark,type,elements,nvidia_throughput_gbs,musa_throughput_gbs,"
          "nvidia_time_ms,musa_time_ms,throughput_ratio")

    # Data rows
    for c in sorted(valid_comparisons, key=lambda x: x["benchmark"]):
        print(f"{c['benchmark']},{c['type']},{c['elements']},"
              f"{c['nvidia_throughput_gbs']},{c['musa_throughput_gbs']},"
              f"{c['nvidia_time_ms']},{c['musa_time_ms']},{c['throughput_ratio']}")


def run_batch_comparison(nvidia_path: str, musa_dir: str,
                         output_json: Optional[str] = None) -> List[Dict]:
    """Compare all JSON files in a directory against NVIDIA baseline."""
    nvidia_baseline = load_nvidia_baseline(nvidia_path)
    comparisons = []

    musa_path = Path(musa_dir)
    json_files = list(musa_path.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {musa_dir}")
        return []

    print(f"Found {len(json_files)} JSON files to compare...")

    for json_file in sorted(json_files):
        try:
            musa_result = load_json_file(str(json_file))

            # Handle both single object and array
            if isinstance(musa_result, dict):
                musa_results = [musa_result]
            else:
                musa_results = musa_result

            for result in musa_results:
                comparison = compare_benchmark(nvidia_baseline, result)
                if comparison:
                    comparisons.append(comparison)
        except Exception as e:
            print(f"Warning: Failed to process {json_file}: {e}")

    if output_json:
        save_json_file(output_json, comparisons)
        print(f"Results saved to {output_json}")

    return comparisons


def main():
    args = sys.argv[1:]

    if not args:
        print(__doc__)
        sys.exit(1)

    # Parse arguments
    output_json = None
    csv_output = False
    batch_mode = False

    i = 0
    while i < len(args):
        arg = args[i]

        if arg == "-h" or arg == "--help":
            print(__doc__)
            sys.exit(0)
        elif arg == "-o" or arg == "--output":
            output_json = args[i + 1]
            i += 2
        elif arg == "--csv":
            csv_output = True
            i += 1
        elif arg == "--batch":
            batch_mode = True
            i += 1
        else:
            break

    remaining_args = args[i:]

    if batch_mode:
        if len(remaining_args) < 2:
            print("Error: --batch requires <nvidia_baseline.json> <musa_results_dir>")
            sys.exit(1)

        nvidia_path = remaining_args[0]
        musa_dir = remaining_args[1]

        comparisons = run_batch_comparison(nvidia_path, musa_dir, output_json)

        if csv_output:
            print_csv_report(comparisons)
        else:
            print_report(comparisons)

    else:
        if len(remaining_args) < 2:
            print("Error: Need <nvidia_baseline.json> <musa_result.json>")
            sys.exit(1)

        nvidia_path = remaining_args[0]
        musa_path = remaining_args[1]

        nvidia_baseline = load_nvidia_baseline(nvidia_path)
        musa_result = load_json_file(musa_path)

        # Handle both single object and array
        if isinstance(musa_result, dict):
            musa_results = [musa_result]
        else:
            musa_results = musa_result

        comparisons = [compare_benchmark(nvidia_baseline, r) for r in musa_results]

        if output_json:
            save_json_file(output_json, [c for c in comparisons if c])

        if csv_output:
            print_csv_report(comparisons)
        else:
            print_report(comparisons)


if __name__ == "__main__":
    main()