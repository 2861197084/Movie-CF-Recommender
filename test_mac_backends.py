#!/usr/bin/env python
"""Test script for Mac backend performance comparison."""

import time
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.platform_utils import print_system_info, get_optimal_settings, check_backend_availability
from config import cfg


def run_backend_test(backend: str, device: str, dataset: str = "ml-latest-small"):
    """Run a test with specified backend and device."""
    import subprocess

    print(f"\n{'='*60}")
    print(f"Testing: backend={backend}, device={device}")
    print('='*60)

    cmd = [
        "python", "main.py",
        "--dataset", dataset,
        "--backend", backend,
        "--device", device,
        "--quick-test",
        "--experiment-name", f"test-{backend}-{device}"
    ]

    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        elapsed = time.time() - start_time

        # Extract key metrics from output
        lines = result.stdout.split('\n')
        for line in lines[-20:]:  # Check last 20 lines for results
            if "MAE" in line or "RMSE" in line or "Time" in line:
                print(line)

        print(f"\nTotal execution time: {elapsed:.2f} seconds")
        return elapsed

    except subprocess.CalledProcessError as e:
        print(f"Error running test: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Test different backends on Mac")
    parser.add_argument("--dataset", default="ml-latest-small",
                        help="Dataset to test with")
    parser.add_argument("--all", action="store_true",
                        help="Test all available backends")
    args = parser.parse_args()

    # Print system information
    print_system_info()

    # Check available backends
    backends = check_backend_availability()

    # Get optimal settings
    optimal_backend, optimal_device = get_optimal_settings()

    # Test configurations
    test_configs = []

    if args.all:
        # Test all available configurations
        test_configs.append(("numpy", "cpu"))

        if backends["torch"]:
            test_configs.append(("torch", "cpu"))

            if backends["mps"]:
                test_configs.append(("torch", "mps"))

            if backends["cuda"]:
                test_configs.append(("torch", "cuda"))
    else:
        # Test only optimal configuration
        test_configs = [
            ("numpy", "cpu"),  # Baseline
            (optimal_backend, optimal_device)  # Optimal
        ]

    # Run tests
    results = {}
    for backend, device in test_configs:
        if backend == "torch" and not backends["torch"]:
            print(f"\nSkipping {backend}/{device} - PyTorch not installed")
            continue

        if device == "mps" and not backends["mps"]:
            print(f"\nSkipping {backend}/{device} - MPS not available")
            continue

        if device == "cuda" and not backends["cuda"]:
            print(f"\nSkipping {backend}/{device} - CUDA not available")
            continue

        elapsed = run_backend_test(backend, device, args.dataset)
        if elapsed is not None:
            results[f"{backend}/{device}"] = elapsed

    # Print summary
    if results:
        print(f"\n{'='*60}")
        print("PERFORMANCE SUMMARY")
        print('='*60)

        baseline = results.get("numpy/cpu", 1.0)
        for config, elapsed in sorted(results.items(), key=lambda x: x[1]):
            speedup = baseline / elapsed if elapsed > 0 else 0
            print(f"{config:20}: {elapsed:8.2f}s (speedup: {speedup:.2f}x)")

        # Highlight recommendation
        print(f"\n{'='*60}")
        if f"{optimal_backend}/{optimal_device}" in results:
            print(f"Recommended: {optimal_backend}/{optimal_device}")
            optimal_time = results[f"{optimal_backend}/{optimal_device}"]
            optimal_speedup = baseline / optimal_time if optimal_time > 0 else 0
            print(f"Expected speedup: {optimal_speedup:.2f}x over NumPy CPU")
        print('='*60)


if __name__ == "__main__":
    main()