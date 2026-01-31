# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Metrics collection for dark mode execution."""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from graphrag.index.graph.dark_mode.comparison_framework import ComparisonResult

logger = logging.getLogger(__name__)


@dataclass
class OperationMetrics:
    """Metrics for a single operation execution.

    Attributes
    ----------
    operation : str
        Name of the operation
    timestamp : str
        ISO timestamp of operation
    primary_duration_ms : float
        Duration of primary backend execution (milliseconds)
    shadow_duration_ms : float | None
        Duration of shadow backend execution (milliseconds), None if failed
    shadow_error : str | None
        Error message if shadow backend failed
    comparison : ComparisonResult | None
        Comparison result if shadow succeeded
    """

    operation: str
    timestamp: str
    primary_duration_ms: float
    shadow_duration_ms: float | None
    shadow_error: str | None
    comparison: ComparisonResult | None


class MetricsCollector:
    """Collects and persists metrics from dark mode execution.

    Writes metrics to JSON lines file for analysis.
    """

    def __init__(self, log_path: str | Path | None = None):
        """Initialize metrics collector.

        Parameters
        ----------
        log_path : str | Path | None, optional
            Path to write metrics log file. If None, logs to memory only.
        """
        self.log_path = Path(log_path) if log_path else None
        self.metrics: list[OperationMetrics] = []

        if self.log_path:
            # Create parent directory if needed
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Metrics collector will write to: %s", self.log_path)

    def record_operation(self, metrics: OperationMetrics) -> None:
        """Record metrics for an operation.

        Parameters
        ----------
        metrics : OperationMetrics
            Metrics to record
        """
        self.metrics.append(metrics)

        # Write to log file if configured
        if self.log_path:
            self._write_to_log(metrics)

        # Log summary
        if metrics.shadow_error:
            logger.warning(
                "Operation %s: shadow failed with error: %s",
                metrics.operation,
                metrics.shadow_error,
            )
        elif metrics.comparison:
            logger.info(
                "Operation %s: comparison %s (primary: %.1fms, shadow: %.1fms)",
                metrics.operation,
                "PASSED" if metrics.comparison.passed else "FAILED",
                metrics.primary_duration_ms,
                metrics.shadow_duration_ms or 0,
            )

    def _write_to_log(self, metrics: OperationMetrics) -> None:
        """Write metrics to JSON lines log file."""
        try:
            # Convert to dict for JSON serialization
            metrics_dict = self._metrics_to_dict(metrics)

            # Append to file
            with open(self.log_path, "a") as f:
                f.write(json.dumps(metrics_dict) + "\n")

        except Exception as e:
            logger.error("Failed to write metrics to log: %s", e)

    def _metrics_to_dict(self, metrics: OperationMetrics) -> dict[str, Any]:
        """Convert OperationMetrics to JSON-serializable dict."""
        result = {
            "operation": metrics.operation,
            "timestamp": metrics.timestamp,
            "primary_duration_ms": metrics.primary_duration_ms,
            "shadow_duration_ms": metrics.shadow_duration_ms,
            "shadow_error": metrics.shadow_error,
        }

        if metrics.comparison:
            # Convert numpy booleans to Python booleans for JSON serialization
            comparison_metrics = {}
            for key, value in metrics.comparison.metrics.items():
                if hasattr(value, "item"):  # numpy scalar
                    comparison_metrics[key] = value.item()
                elif isinstance(value, dict):
                    # Recursively convert nested dicts
                    comparison_metrics[key] = {
                        k: v.item() if hasattr(v, "item") else v for k, v in value.items()
                    }
                else:
                    comparison_metrics[key] = value

            result["comparison"] = {
                "operation": metrics.comparison.operation,
                "passed": bool(metrics.comparison.passed),  # Ensure Python bool
                "metrics": comparison_metrics,
                "differences": metrics.comparison.differences,
            }

        return result

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics of all recorded metrics.

        Returns
        -------
        dict[str, Any]
            Summary statistics including:
            - total_operations: Total operations executed
            - shadow_errors: Number of shadow errors
            - comparisons_passed: Number of passing comparisons
            - comparisons_failed: Number of failing comparisons
            - avg_primary_duration_ms: Average primary execution time
            - avg_shadow_duration_ms: Average shadow execution time
            - avg_latency_ratio: Average shadow/primary latency ratio
        """
        total = len(self.metrics)
        shadow_errors = sum(1 for m in self.metrics if m.shadow_error)
        comparisons = [m.comparison for m in self.metrics if m.comparison]
        passed = sum(1 for c in comparisons if c.passed)
        failed = len(comparisons) - passed

        primary_durations = [m.primary_duration_ms for m in self.metrics]
        shadow_durations = [
            m.shadow_duration_ms for m in self.metrics if m.shadow_duration_ms
        ]

        avg_primary = (
            sum(primary_durations) / len(primary_durations) if primary_durations else 0
        )
        avg_shadow = (
            sum(shadow_durations) / len(shadow_durations) if shadow_durations else 0
        )
        avg_ratio = avg_shadow / avg_primary if avg_primary > 0 else 0

        return {
            "total_operations": total,
            "shadow_errors": shadow_errors,
            "shadow_error_rate": shadow_errors / total if total > 0 else 0,
            "comparisons_passed": passed,
            "comparisons_failed": failed,
            "comparison_pass_rate": passed / len(comparisons) if comparisons else 1.0,  # 1.0 if no comparisons yet
            "avg_primary_duration_ms": avg_primary,
            "avg_shadow_duration_ms": avg_shadow,
            "avg_latency_ratio": avg_ratio,
        }

    def check_cutover_criteria(
        self,
        *,
        min_operations: int = 1000,
        max_error_rate: float = 0.01,
        min_pass_rate: float = 0.95,
        max_latency_ratio: float = 2.0,
    ) -> tuple[bool, list[str]]:
        """Check if cutover criteria are met.

        Parameters
        ----------
        min_operations : int, optional
            Minimum number of operations to validate (default: 1000)
        max_error_rate : float, optional
            Maximum shadow error rate (default: 0.01 = 1%)
        min_pass_rate : float, optional
            Minimum comparison pass rate (default: 0.95 = 95%)
        max_latency_ratio : float, optional
            Maximum shadow/primary latency ratio (default: 2.0)

        Returns
        -------
        tuple[bool, list[str]]
            (ready_for_cutover, list_of_reasons_if_not_ready)
        """
        summary = self.get_summary()
        reasons = []

        if summary["total_operations"] < min_operations:
            reasons.append(
                f"Not enough operations: {summary['total_operations']} < {min_operations}"
            )

        if summary["shadow_error_rate"] > max_error_rate:
            reasons.append(
                f"Shadow error rate too high: {summary['shadow_error_rate']:.2%} > {max_error_rate:.2%}"
            )

        if summary["comparison_pass_rate"] < min_pass_rate:
            reasons.append(
                f"Comparison pass rate too low: {summary['comparison_pass_rate']:.2%} < {min_pass_rate:.2%}"
            )

        if summary["avg_latency_ratio"] > max_latency_ratio:
            reasons.append(
                f"Latency ratio too high: {summary['avg_latency_ratio']:.2f}x > {max_latency_ratio:.2f}x"
            )

        ready = len(reasons) == 0

        if ready:
            logger.info("✅ Cutover criteria MET - ready for production cutover")
        else:
            logger.warning(
                "❌ Cutover criteria NOT MET:\n  - " + "\n  - ".join(reasons)
            )

        return ready, reasons
