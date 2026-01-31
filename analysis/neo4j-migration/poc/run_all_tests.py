#!/usr/bin/env python3
"""
Run all Neo4j POC tests in sequence.

This script runs all 5 POC tests and generates a summary report.
"""

import subprocess
import sys
import time
from pathlib import Path


def run_test(test_file):
    """Run a single test and return result."""
    test_path = Path(__file__).parent / test_file
    print(f"\n{'='*60}")
    print(f"Running: {test_file}")
    print(f"{'='*60}\n")

    start_time = time.time()
    result = subprocess.run(
        [sys.executable, str(test_path)],
        capture_output=False
    )
    elapsed = time.time() - start_time

    return {
        "test": test_file,
        "passed": result.returncode == 0,
        "duration": elapsed
    }


def main():
    """Run all POC tests."""
    print("="*60)
    print("Neo4j POC Test Suite")
    print("="*60)
    print("\nRunning all 5 POC tests to validate Neo4j integration...")
    print("Expected duration: 2-3 hours")

    tests = [
        "test_01_connection.py",
        "test_02_write_entities.py",
        "test_03_gds_community.py",
        "test_04_dataframe_export.py",
        "test_05_full_integration.py",
    ]

    results = []
    for test in tests:
        result = run_test(test)
        results.append(result)

        # Stop on first failure
        if not result["passed"]:
            print("\n" + "="*60)
            print("❌ TEST SUITE FAILED - Stopping at first failure")
            print("="*60)
            break

    # Print summary
    print("\n" + "="*60)
    print("POC Test Suite Summary")
    print("="*60)

    total_duration = sum(r["duration"] for r in results)
    passed_count = sum(1 for r in results if r["passed"])
    total_count = len(results)

    for i, result in enumerate(results, 1):
        status = "✅ PASSED" if result["passed"] else "❌ FAILED"
        print(f"\nTest {i}: {result['test']}")
        print(f"  Status: {status}")
        print(f"  Duration: {result['duration']:.1f}s")

    print(f"\n{'='*60}")
    print(f"Total: {passed_count}/{len(tests)} tests passed")
    print(f"Duration: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")
    print(f"{'='*60}")

    if passed_count == len(tests):
        print("\n✅ ALL POC TESTS PASSED")
        print("\nNext Steps:")
        print("  1. Document results in POC_RESULTS.md")
        print("  2. Commit POC tests: git commit -m 'POC: Neo4j integration tests'")
        print("  3. Tag POC: git tag neo4j-poc-complete")
        print("  4. Start Phase 1 implementation")
        return 0
    else:
        print("\n❌ POC TESTS FAILED")
        print("\nNext Steps:")
        print("  1. Review failure logs above")
        print("  2. Fix blockers")
        print("  3. Re-run POC tests")
        print("  4. Document blockers in POC_RESULTS.md")
        return 1


if __name__ == "__main__":
    sys.exit(main())
