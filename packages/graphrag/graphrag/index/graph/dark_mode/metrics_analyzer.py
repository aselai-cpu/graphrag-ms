# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Tool for analyzing dark mode metrics logs."""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MetricsAnalysis:
    """Analysis results from dark mode metrics."""

    # Summary statistics
    total_operations: int
    operations_by_type: dict[str, int]
    shadow_errors: int
    shadow_error_rate: float
    comparisons_passed: int
    comparisons_failed: int
    comparison_pass_rate: float

    # Performance metrics
    avg_primary_duration_ms: float
    avg_shadow_duration_ms: float
    avg_latency_ratio: float
    p50_latency_ratio: float
    p95_latency_ratio: float
    p99_latency_ratio: float

    # Cutover readiness
    ready_for_cutover: bool
    blocking_reasons: list[str]

    # Detailed breakdowns
    operation_details: dict[str, dict[str, Any]]
    failure_details: list[dict[str, Any]]


class MetricsAnalyzer:
    """Analyzes dark mode metrics from JSON lines log files."""

    def __init__(self, log_path: str | Path):
        """Initialize metrics analyzer.

        Parameters
        ----------
        log_path : str | Path
            Path to metrics log file (JSON lines format)
        """
        self.log_path = Path(log_path)
        self.metrics: list[dict[str, Any]] = []

    def load_metrics(self) -> None:
        """Load metrics from log file."""
        if not self.log_path.exists():
            msg = f"Metrics log file not found: {self.log_path}"
            raise FileNotFoundError(msg)

        self.metrics = []
        with open(self.log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        self.metrics.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning("Failed to parse metrics line: %s", e)

        logger.info("Loaded %d metrics from %s", len(self.metrics), self.log_path)

    def analyze(
        self,
        *,
        min_operations: int = 1000,
        max_error_rate: float = 0.01,
        min_pass_rate: float = 0.95,
        max_latency_ratio: float = 2.0,
    ) -> MetricsAnalysis:
        """Analyze loaded metrics.

        Parameters
        ----------
        min_operations : int, optional
            Minimum operations for cutover (default: 1000)
        max_error_rate : float, optional
            Maximum shadow error rate (default: 0.01)
        min_pass_rate : float, optional
            Minimum comparison pass rate (default: 0.95)
        max_latency_ratio : float, optional
            Maximum latency ratio (default: 2.0)

        Returns
        -------
        MetricsAnalysis
            Comprehensive analysis results
        """
        if not self.metrics:
            self.load_metrics()

        # Count operations by type
        operations_by_type = defaultdict(int)
        for m in self.metrics:
            operations_by_type[m["operation"]] += 1

        # Count errors and comparisons
        shadow_errors = sum(1 for m in self.metrics if m.get("shadow_error"))
        comparisons = [m for m in self.metrics if m.get("comparison")]
        comparisons_passed = sum(
            1 for m in comparisons if m["comparison"].get("passed")
        )
        comparisons_failed = len(comparisons) - comparisons_passed

        # Calculate rates
        total_ops = len(self.metrics)
        shadow_error_rate = shadow_errors / total_ops if total_ops > 0 else 0.0
        comparison_pass_rate = (
            comparisons_passed / len(comparisons) if comparisons else 1.0
        )

        # Performance metrics
        primary_durations = [m["primary_duration_ms"] for m in self.metrics]
        shadow_durations = [
            m["shadow_duration_ms"]
            for m in self.metrics
            if m.get("shadow_duration_ms") is not None
        ]

        avg_primary = sum(primary_durations) / len(primary_durations) if primary_durations else 0
        avg_shadow = sum(shadow_durations) / len(shadow_durations) if shadow_durations else 0

        # Calculate latency ratios
        latency_ratios = []
        for m in self.metrics:
            if m.get("shadow_duration_ms") and m["primary_duration_ms"] > 0:
                ratio = m["shadow_duration_ms"] / m["primary_duration_ms"]
                latency_ratios.append(ratio)

        avg_latency_ratio = sum(latency_ratios) / len(latency_ratios) if latency_ratios else 0

        # Calculate percentiles
        if latency_ratios:
            latency_df = pd.Series(latency_ratios)
            p50 = float(latency_df.quantile(0.50))
            p95 = float(latency_df.quantile(0.95))
            p99 = float(latency_df.quantile(0.99))
        else:
            p50 = p95 = p99 = 0.0

        # Operation-level details
        operation_details = {}
        for op_type in operations_by_type.keys():
            op_metrics = [m for m in self.metrics if m["operation"] == op_type]
            op_errors = sum(1 for m in op_metrics if m.get("shadow_error"))
            op_comparisons = [m for m in op_metrics if m.get("comparison")]
            op_passed = sum(
                1 for m in op_comparisons if m["comparison"].get("passed")
            )

            operation_details[op_type] = {
                "count": len(op_metrics),
                "shadow_errors": op_errors,
                "error_rate": op_errors / len(op_metrics) if op_metrics else 0,
                "comparisons_passed": op_passed,
                "comparisons_failed": len(op_comparisons) - op_passed,
                "pass_rate": op_passed / len(op_comparisons) if op_comparisons else 1.0,
            }

        # Failure details
        failure_details = []
        for m in self.metrics:
            if m.get("shadow_error"):
                failure_details.append({
                    "operation": m["operation"],
                    "timestamp": m["timestamp"],
                    "error": m["shadow_error"],
                })
            elif m.get("comparison") and not m["comparison"].get("passed"):
                failure_details.append({
                    "operation": m["operation"],
                    "timestamp": m["timestamp"],
                    "differences": m["comparison"].get("differences", []),
                })

        # Check cutover readiness
        blocking_reasons = []
        if total_ops < min_operations:
            blocking_reasons.append(
                f"Not enough operations: {total_ops} < {min_operations}"
            )
        if shadow_error_rate > max_error_rate:
            blocking_reasons.append(
                f"Shadow error rate too high: {shadow_error_rate:.2%} > {max_error_rate:.2%}"
            )
        if comparison_pass_rate < min_pass_rate:
            blocking_reasons.append(
                f"Comparison pass rate too low: {comparison_pass_rate:.2%} < {min_pass_rate:.2%}"
            )
        if avg_latency_ratio > max_latency_ratio:
            blocking_reasons.append(
                f"Latency ratio too high: {avg_latency_ratio:.2f}x > {max_latency_ratio:.2f}x"
            )

        ready_for_cutover = len(blocking_reasons) == 0

        return MetricsAnalysis(
            total_operations=total_ops,
            operations_by_type=dict(operations_by_type),
            shadow_errors=shadow_errors,
            shadow_error_rate=shadow_error_rate,
            comparisons_passed=comparisons_passed,
            comparisons_failed=comparisons_failed,
            comparison_pass_rate=comparison_pass_rate,
            avg_primary_duration_ms=avg_primary,
            avg_shadow_duration_ms=avg_shadow,
            avg_latency_ratio=avg_latency_ratio,
            p50_latency_ratio=p50,
            p95_latency_ratio=p95,
            p99_latency_ratio=p99,
            ready_for_cutover=ready_for_cutover,
            blocking_reasons=blocking_reasons,
            operation_details=operation_details,
            failure_details=failure_details,
        )

    def print_summary(self, analysis: MetricsAnalysis | None = None) -> None:
        """Print human-readable summary of analysis.

        Parameters
        ----------
        analysis : MetricsAnalysis | None, optional
            Analysis to print. If None, runs analysis first.
        """
        if analysis is None:
            analysis = self.analyze()

        print("=" * 80)
        print("DARK MODE METRICS ANALYSIS")
        print("=" * 80)
        print()

        print("📊 OPERATIONS SUMMARY")
        print(f"  Total operations: {analysis.total_operations}")
        for op_type, count in analysis.operations_by_type.items():
            print(f"    - {op_type}: {count}")
        print()

        print("✅ CORRECTNESS METRICS")
        print(f"  Shadow errors: {analysis.shadow_errors} ({analysis.shadow_error_rate:.2%})")
        print(f"  Comparisons passed: {analysis.comparisons_passed}")
        print(f"  Comparisons failed: {analysis.comparisons_failed}")
        print(f"  Pass rate: {analysis.comparison_pass_rate:.2%}")
        print()

        print("⚡ PERFORMANCE METRICS")
        print(f"  Avg primary duration: {analysis.avg_primary_duration_ms:.1f}ms")
        print(f"  Avg shadow duration: {analysis.avg_shadow_duration_ms:.1f}ms")
        print(f"  Avg latency ratio: {analysis.avg_latency_ratio:.2f}x")
        print(f"  P50 latency ratio: {analysis.p50_latency_ratio:.2f}x")
        print(f"  P95 latency ratio: {analysis.p95_latency_ratio:.2f}x")
        print(f"  P99 latency ratio: {analysis.p99_latency_ratio:.2f}x")
        print()

        print("📈 OPERATION BREAKDOWN")
        for op_type, details in analysis.operation_details.items():
            print(f"  {op_type}:")
            print(f"    Operations: {details['count']}")
            print(f"    Shadow errors: {details['shadow_errors']} ({details['error_rate']:.2%})")
            print(f"    Pass rate: {details['pass_rate']:.2%}")
        print()

        print("🚀 CUTOVER READINESS")
        if analysis.ready_for_cutover:
            print("  ✅ READY FOR CUTOVER")
            print("  All criteria met - shadow backend validated successfully!")
        else:
            print("  ❌ NOT READY FOR CUTOVER")
            print("  Blocking reasons:")
            for reason in analysis.blocking_reasons:
                print(f"    - {reason}")
        print()

        if analysis.failure_details:
            print(f"⚠️  RECENT FAILURES ({len(analysis.failure_details)} total)")
            for i, failure in enumerate(analysis.failure_details[:5]):
                print(f"  {i+1}. {failure['operation']} at {failure['timestamp']}")
                if "error" in failure:
                    print(f"     Error: {failure['error']}")
                elif "differences" in failure:
                    for diff in failure["differences"][:2]:
                        print(f"     - {diff}")
            if len(analysis.failure_details) > 5:
                print(f"  ... and {len(analysis.failure_details) - 5} more")
            print()

        print("=" * 80)

    def export_to_csv(self, output_path: str | Path) -> None:
        """Export metrics to CSV for further analysis.

        Parameters
        ----------
        output_path : str | Path
            Path to write CSV file
        """
        if not self.metrics:
            self.load_metrics()

        df = pd.DataFrame(self.metrics)
        df.to_csv(output_path, index=False)
        logger.info("Exported %d metrics to %s", len(df), output_path)
