# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Dark mode orchestrator for parallel graph backend execution."""

import logging
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from graphrag.index.graph.dark_mode.comparison_framework import ComparisonFramework
from graphrag.index.graph.dark_mode.metrics_collector import (
    MetricsCollector,
    OperationMetrics,
)
from graphrag.index.graph.graph_backend import Communities, GraphBackend

logger = logging.getLogger(__name__)


class DarkModeOrchestrator:
    """Orchestrates parallel execution of primary and shadow graph backends.

    In dark mode:
    - Primary backend serves production traffic (results returned to user)
    - Shadow backend runs identical operations in parallel
    - Shadow failures do NOT affect primary results
    - All results are compared and logged for validation
    """

    def __init__(
        self,
        primary: GraphBackend,
        shadow: GraphBackend,
        *,
        comparison_framework: ComparisonFramework | None = None,
        metrics_collector: MetricsCollector | None = None,
        log_path: str | Path | None = None,
    ):
        """Initialize dark mode orchestrator.

        Parameters
        ----------
        primary : GraphBackend
            Primary backend (serves production)
        shadow : GraphBackend
            Shadow backend (for validation)
        comparison_framework : ComparisonFramework | None, optional
            Framework for comparing results. If None, uses default.
        metrics_collector : MetricsCollector | None, optional
            Collector for logging metrics. If None, creates with log_path.
        log_path : str | Path | None, optional
            Path for metrics log file (used if metrics_collector is None)
        """
        self.primary = primary
        self.shadow = shadow

        self.comparison_framework = comparison_framework or ComparisonFramework()
        self.metrics_collector = metrics_collector or MetricsCollector(log_path=log_path)

        logger.info(
            "DarkModeOrchestrator initialized with primary=%s, shadow=%s",
            type(primary).__name__,
            type(shadow).__name__,
        )

    def load_graph(
        self,
        entities: pd.DataFrame,
        relationships: pd.DataFrame,
        **kwargs,
    ) -> None:
        """Load graph into both primary and shadow backends.

        Primary execution is synchronous and errors propagate.
        Shadow execution is best-effort and errors are logged.

        Parameters
        ----------
        entities : pd.DataFrame
            Entity nodes
        relationships : pd.DataFrame
            Relationship edges
        **kwargs
            Additional arguments passed to both backends
        """
        operation = "load_graph"
        timestamp = datetime.utcnow().isoformat()

        # Execute on primary (synchronous, errors propagate)
        logger.info("Loading graph on PRIMARY backend...")
        start = time.time()
        # Copy DataFrames to avoid modifying caller's data
        # (some backends like NetworkX modify DataFrames inplace with set_index())
        self.primary.load_graph(entities.copy(), relationships.copy(), **kwargs)
        primary_duration = (time.time() - start) * 1000  # Convert to ms

        logger.info(
            "PRIMARY load complete: %d entities, %d relationships in %.1fms",
            len(entities),
            len(relationships),
            primary_duration,
        )

        # Execute on shadow (best-effort, errors caught)
        shadow_duration = None
        shadow_error = None
        comparison = None

        try:
            logger.info("Loading graph on SHADOW backend...")
            start = time.time()
            # Copy DataFrames to avoid primary backend modifications affecting shadow
            # (e.g., set_index() called inplace by some backends)
            self.shadow.load_graph(entities.copy(), relationships.copy(), **kwargs)
            shadow_duration = (time.time() - start) * 1000

            logger.info(
                "SHADOW load complete: %d entities, %d relationships in %.1fms",
                len(entities),
                len(relationships),
                shadow_duration,
            )

            # Compare results (node counts, edge counts)
            primary_nodes = self.primary.node_count()
            shadow_nodes = self.shadow.node_count()
            primary_edges = self.primary.edge_count()
            shadow_edges = self.shadow.edge_count()

            differences = []
            if primary_nodes != shadow_nodes:
                differences.append(
                    f"Node count mismatch: primary={primary_nodes}, shadow={shadow_nodes}"
                )
            if primary_edges != shadow_edges:
                differences.append(
                    f"Edge count mismatch: primary={primary_edges}, shadow={shadow_edges}"
                )

            from graphrag.index.graph.dark_mode.comparison_framework import (
                ComparisonResult,
            )

            comparison = ComparisonResult(
                operation="load_graph",
                passed=len(differences) == 0,
                metrics={
                    "primary_nodes": primary_nodes,
                    "shadow_nodes": shadow_nodes,
                    "primary_edges": primary_edges,
                    "shadow_edges": shadow_edges,
                },
                differences=differences,
            )

        except Exception as e:
            shadow_error = str(e)
            logger.warning("SHADOW load failed: %s", shadow_error, exc_info=True)

        # Record metrics
        metrics = OperationMetrics(
            operation=operation,
            timestamp=timestamp,
            primary_duration_ms=primary_duration,
            shadow_duration_ms=shadow_duration,
            shadow_error=shadow_error,
            comparison=comparison,
        )
        self.metrics_collector.record_operation(metrics)

    def detect_communities(self, **kwargs) -> Communities:
        """Run community detection on both backends and compare results.

        Returns primary backend results. Shadow results are compared and logged.

        Parameters
        ----------
        **kwargs
            Arguments passed to detect_communities (max_cluster_size, use_lcc, seed)

        Returns
        -------
        Communities
            Community detection results from primary backend
        """
        operation = "detect_communities"
        timestamp = datetime.utcnow().isoformat()

        # Execute on primary
        logger.info("Running community detection on PRIMARY backend...")
        start = time.time()
        primary_communities = self.primary.detect_communities(**kwargs)
        primary_duration = (time.time() - start) * 1000

        logger.info(
            "PRIMARY communities: %d detected in %.1fms",
            len(primary_communities),
            primary_duration,
        )

        # Execute on shadow
        shadow_duration = None
        shadow_error = None
        comparison = None

        try:
            logger.info("Running community detection on SHADOW backend...")
            start = time.time()
            shadow_communities = self.shadow.detect_communities(**kwargs)
            shadow_duration = (time.time() - start) * 1000

            logger.info(
                "SHADOW communities: %d detected in %.1fms",
                len(shadow_communities),
                shadow_duration,
            )

            # Compare community structures
            comparison = self.comparison_framework.compare_communities(
                primary_communities, shadow_communities
            )

        except Exception as e:
            shadow_error = str(e)
            logger.warning(
                "SHADOW community detection failed: %s", shadow_error, exc_info=True
            )

        # Record metrics
        metrics = OperationMetrics(
            operation=operation,
            timestamp=timestamp,
            primary_duration_ms=primary_duration,
            shadow_duration_ms=shadow_duration,
            shadow_error=shadow_error,
            comparison=comparison,
        )
        self.metrics_collector.record_operation(metrics)

        # Return primary results
        return primary_communities

    def compute_node_degrees(self) -> pd.DataFrame:
        """Compute node degrees on both backends and compare results.

        Returns primary backend results. Shadow results are compared and logged.

        Returns
        -------
        pd.DataFrame
            Node degrees from primary backend
        """
        operation = "compute_node_degrees"
        timestamp = datetime.utcnow().isoformat()

        # Execute on primary
        logger.info("Computing node degrees on PRIMARY backend...")
        start = time.time()
        primary_degrees = self.primary.compute_node_degrees()
        primary_duration = (time.time() - start) * 1000

        logger.info(
            "PRIMARY degrees: %d nodes in %.1fms", len(primary_degrees), primary_duration
        )

        # Execute on shadow
        shadow_duration = None
        shadow_error = None
        comparison = None

        try:
            logger.info("Computing node degrees on SHADOW backend...")
            start = time.time()
            shadow_degrees = self.shadow.compute_node_degrees()
            shadow_duration = (time.time() - start) * 1000

            logger.info(
                "SHADOW degrees: %d nodes in %.1fms", len(shadow_degrees), shadow_duration
            )

            # Compare degrees
            comparison = self.comparison_framework.compare_node_degrees(
                primary_degrees, shadow_degrees
            )

        except Exception as e:
            shadow_error = str(e)
            logger.warning("SHADOW degree computation failed: %s", shadow_error, exc_info=True)

        # Record metrics
        metrics = OperationMetrics(
            operation=operation,
            timestamp=timestamp,
            primary_duration_ms=primary_duration,
            shadow_duration_ms=shadow_duration,
            shadow_error=shadow_error,
            comparison=comparison,
        )
        self.metrics_collector.record_operation(metrics)

        # Return primary results
        return primary_degrees

    def export_graph(self, **kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Export graph from both backends and compare results.

        Returns primary backend results. Shadow results are compared and logged.

        Parameters
        ----------
        **kwargs
            Arguments passed to export_graph

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            (entities, relationships) from primary backend
        """
        operation = "export_graph"
        timestamp = datetime.utcnow().isoformat()

        # Execute on primary
        logger.info("Exporting graph from PRIMARY backend...")
        start = time.time()
        primary_entities, primary_relationships = self.primary.export_graph(**kwargs)
        primary_duration = (time.time() - start) * 1000

        logger.info(
            "PRIMARY export: %d entities, %d relationships in %.1fms",
            len(primary_entities),
            len(primary_relationships),
            primary_duration,
        )

        # Execute on shadow
        shadow_duration = None
        shadow_error = None
        comparison = None

        try:
            logger.info("Exporting graph from SHADOW backend...")
            start = time.time()
            shadow_entities, shadow_relationships = self.shadow.export_graph(**kwargs)
            shadow_duration = (time.time() - start) * 1000

            logger.info(
                "SHADOW export: %d entities, %d relationships in %.1fms",
                len(shadow_entities),
                len(shadow_relationships),
                shadow_duration,
            )

            # Compare entities
            comparison = self.comparison_framework.compare_entities(
                primary_entities, shadow_entities
            )

        except Exception as e:
            shadow_error = str(e)
            logger.warning("SHADOW export failed: %s", shadow_error, exc_info=True)

        # Record metrics
        metrics = OperationMetrics(
            operation=operation,
            timestamp=timestamp,
            primary_duration_ms=primary_duration,
            shadow_duration_ms=shadow_duration,
            shadow_error=shadow_error,
            comparison=comparison,
        )
        self.metrics_collector.record_operation(metrics)

        # Return primary results
        return primary_entities, primary_relationships

    def clear(self) -> None:
        """Clear both backends."""
        logger.info("Clearing PRIMARY backend...")
        self.primary.clear()

        try:
            logger.info("Clearing SHADOW backend...")
            self.shadow.clear()
        except Exception as e:
            logger.warning("SHADOW clear failed: %s", e, exc_info=True)

    def node_count(self) -> int:
        """Return node count from primary backend."""
        return self.primary.node_count()

    def edge_count(self) -> int:
        """Return edge count from primary backend."""
        return self.primary.edge_count()

    def get_metrics_summary(self) -> dict:
        """Get summary of dark mode execution metrics.

        Returns
        -------
        dict
            Summary statistics from metrics collector
        """
        return self.metrics_collector.get_summary()

    def check_cutover_readiness(self, **kwargs) -> tuple[bool, list[str]]:
        """Check if ready for cutover to shadow backend.

        Parameters
        ----------
        **kwargs
            Cutover criteria parameters (min_operations, max_error_rate, etc.)

        Returns
        -------
        tuple[bool, list[str]]
            (ready, list_of_blocking_reasons)
        """
        return self.metrics_collector.check_cutover_criteria(**kwargs)
