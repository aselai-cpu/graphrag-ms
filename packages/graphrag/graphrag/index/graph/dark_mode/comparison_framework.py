# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Comparison framework for validating shadow backend results against primary."""

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from graphrag.index.graph.graph_backend import Communities

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Result of comparing primary and shadow backend outputs.

    Attributes
    ----------
    operation : str
        Name of the operation being compared
    passed : bool
        Whether the comparison passed validation thresholds
    metrics : dict[str, Any]
        Detailed comparison metrics
    differences : list[str]
        List of differences found
    """

    operation: str
    passed: bool
    metrics: dict[str, Any]
    differences: list[str]


class ComparisonFramework:
    """Framework for comparing results between primary and shadow backends.

    This compares entities, relationships, communities, and node degrees to
    validate that the shadow backend produces equivalent results.
    """

    def __init__(
        self,
        *,
        entity_match_threshold: float = 0.99,
        community_match_threshold: float = 0.95,
        degree_tolerance: float = 0.1,
    ):
        """Initialize comparison framework.

        Parameters
        ----------
        entity_match_threshold : float, optional
            Minimum entity F1 score to pass (default: 0.99)
        community_match_threshold : float, optional
            Minimum community assignment match rate (default: 0.95)
        degree_tolerance : float, optional
            Maximum relative difference in node degrees (default: 0.1)
        """
        self.entity_match_threshold = entity_match_threshold
        self.community_match_threshold = community_match_threshold
        self.degree_tolerance = degree_tolerance

    def compare_entities(
        self,
        primary_entities: pd.DataFrame,
        shadow_entities: pd.DataFrame,
        *,
        id_col: str = "title",
    ) -> ComparisonResult:
        """Compare entity DataFrames between primary and shadow.

        Parameters
        ----------
        primary_entities : pd.DataFrame
            Entities from primary backend
        shadow_entities : pd.DataFrame
            Entities from shadow backend
        id_col : str, optional
            Column name for entity ID (default: "title")

        Returns
        -------
        ComparisonResult
            Comparison results with metrics and pass/fail status
        """
        differences = []

        # Count comparison
        primary_count = len(primary_entities)
        shadow_count = len(shadow_entities)
        count_match = primary_count == shadow_count

        if not count_match:
            differences.append(
                f"Entity count mismatch: primary={primary_count}, shadow={shadow_count}"
            )

        # ID overlap analysis
        primary_ids = set(primary_entities[id_col].tolist())
        shadow_ids = set(shadow_entities[id_col].tolist())

        overlap = primary_ids & shadow_ids
        missing_in_shadow = primary_ids - shadow_ids
        extra_in_shadow = shadow_ids - primary_ids

        if missing_in_shadow:
            differences.append(
                f"{len(missing_in_shadow)} entities missing in shadow: "
                f"{list(missing_in_shadow)[:5]}..."
            )

        if extra_in_shadow:
            differences.append(
                f"{len(extra_in_shadow)} extra entities in shadow: "
                f"{list(extra_in_shadow)[:5]}..."
            )

        # Calculate precision, recall, F1
        precision = len(overlap) / len(shadow_ids) if shadow_ids else 0.0
        recall = len(overlap) / len(primary_ids) if primary_ids else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        metrics = {
            "primary_count": primary_count,
            "shadow_count": shadow_count,
            "count_match": count_match,
            "overlap_count": len(overlap),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "missing_in_shadow": len(missing_in_shadow),
            "extra_in_shadow": len(extra_in_shadow),
        }

        passed = f1 >= self.entity_match_threshold

        logger.info(
            "Entity comparison: F1=%.4f (threshold=%.2f), passed=%s",
            f1,
            self.entity_match_threshold,
            passed,
        )

        return ComparisonResult(
            operation="compare_entities",
            passed=passed,
            metrics=metrics,
            differences=differences,
        )

    def compare_communities(
        self,
        primary_communities: Communities,
        shadow_communities: Communities,
    ) -> ComparisonResult:
        """Compare community detection results.

        Note: Community IDs may differ, but the clustering of nodes should be similar.
        This compares the actual node assignments, not the community IDs themselves.

        Parameters
        ----------
        primary_communities : Communities
            Communities from primary backend
        shadow_communities : Communities
            Communities from shadow backend

        Returns
        -------
        ComparisonResult
            Comparison results with clustering similarity metrics
        """
        differences = []

        # Extract node assignments at each level
        primary_assignments = self._extract_node_assignments(primary_communities)
        shadow_assignments = self._extract_node_assignments(shadow_communities)

        # Compare number of levels
        primary_levels = set(primary_assignments.keys())
        shadow_levels = set(shadow_assignments.keys())

        if primary_levels != shadow_levels:
            differences.append(
                f"Level count mismatch: primary={len(primary_levels)}, "
                f"shadow={len(shadow_levels)}"
            )

        # Compare clustering similarity at each level
        level_similarities = {}
        for level in primary_levels & shadow_levels:
            similarity = self._compute_clustering_similarity(
                primary_assignments[level], shadow_assignments[level]
            )
            level_similarities[level] = similarity

            if similarity < self.community_match_threshold:
                differences.append(
                    f"Level {level} similarity {similarity:.4f} below threshold "
                    f"{self.community_match_threshold:.2f}"
                )

        # Overall match rate (average across levels)
        avg_similarity = (
            sum(level_similarities.values()) / len(level_similarities)
            if level_similarities
            else 0.0
        )

        metrics = {
            "primary_level_count": len(primary_levels),
            "shadow_level_count": len(shadow_levels),
            "common_levels": len(primary_levels & shadow_levels),
            "level_similarities": level_similarities,
            "avg_similarity": avg_similarity,
        }

        passed = avg_similarity >= self.community_match_threshold

        logger.info(
            "Community comparison: avg_similarity=%.4f (threshold=%.2f), passed=%s",
            avg_similarity,
            self.community_match_threshold,
            passed,
        )

        return ComparisonResult(
            operation="compare_communities",
            passed=passed,
            metrics=metrics,
            differences=differences,
        )

    def compare_node_degrees(
        self,
        primary_degrees: pd.DataFrame,
        shadow_degrees: pd.DataFrame,
        *,
        id_col: str = "title",
        degree_col: str = "degree",
    ) -> ComparisonResult:
        """Compare node degree computations.

        Parameters
        ----------
        primary_degrees : pd.DataFrame
            Node degrees from primary backend
        shadow_degrees : pd.DataFrame
            Node degrees from shadow backend
        id_col : str, optional
            Column name for node ID (default: "title")
        degree_col : str, optional
            Column name for degree value (default: "degree")

        Returns
        -------
        ComparisonResult
            Comparison results with degree difference metrics
        """
        differences = []

        # Merge on node ID
        merged = primary_degrees.merge(
            shadow_degrees,
            on=id_col,
            how="outer",
            suffixes=("_primary", "_shadow"),
        )

        primary_degree_col = f"{degree_col}_primary"
        shadow_degree_col = f"{degree_col}_shadow"

        # Find nodes with mismatched degrees
        mismatches = merged[
            (merged[primary_degree_col].notna())
            & (merged[shadow_degree_col].notna())
            & (merged[primary_degree_col] != merged[shadow_degree_col])
        ]

        if len(mismatches) > 0:
            differences.append(
                f"{len(mismatches)} nodes have different degrees: "
                f"{mismatches[[id_col, primary_degree_col, shadow_degree_col]].head().to_dict('records')}"
            )

        # Calculate relative differences
        merged["relative_diff"] = (
            abs(merged[primary_degree_col] - merged[shadow_degree_col])
            / merged[primary_degree_col].clip(lower=1)
        )

        max_relative_diff = merged["relative_diff"].max()
        avg_relative_diff = merged["relative_diff"].mean()

        metrics = {
            "node_count": len(merged),
            "mismatch_count": len(mismatches),
            "max_relative_diff": float(max_relative_diff),
            "avg_relative_diff": float(avg_relative_diff),
            "exact_match_rate": 1.0 - (len(mismatches) / len(merged)),
        }

        passed = bool(max_relative_diff <= self.degree_tolerance)

        logger.info(
            "Degree comparison: max_diff=%.4f, avg_diff=%.4f (tolerance=%.2f), passed=%s",
            max_relative_diff,
            avg_relative_diff,
            self.degree_tolerance,
            passed,
        )

        return ComparisonResult(
            operation="compare_node_degrees",
            passed=passed,
            metrics=metrics,
            differences=differences,
        )

    def _extract_node_assignments(
        self, communities: Communities
    ) -> dict[int, dict[str, int]]:
        """Extract node-to-community assignments at each level.

        Returns dict mapping level -> {node_id: community_id}
        """
        assignments = {}
        for community in communities:
            level = community.level
            if level not in assignments:
                assignments[level] = {}

            for node_id in community.node_ids:
                assignments[level][node_id] = community.cluster_id

        return assignments

    def _compute_clustering_similarity(
        self, primary_assignment: dict[str, int], shadow_assignment: dict[str, int]
    ) -> float:
        """Compute clustering similarity using Adjusted Rand Index.

        Since community IDs may differ, we use ARI which is invariant to permutation.
        For simplicity, we use a pairwise agreement metric here.

        Returns
        -------
        float
            Similarity score between 0 and 1
        """
        # Get common nodes
        common_nodes = set(primary_assignment.keys()) & set(shadow_assignment.keys())

        if len(common_nodes) < 2:
            return 1.0 if len(common_nodes) == 1 else 0.0

        # Count pairwise agreements
        # Two nodes are in same cluster in primary iff they're in same cluster in shadow
        agreements = 0
        total_pairs = 0

        node_list = list(common_nodes)
        for i, node_a in enumerate(node_list):
            for node_b in node_list[i + 1 :]:
                primary_same = (
                    primary_assignment[node_a] == primary_assignment[node_b]
                )
                shadow_same = shadow_assignment[node_a] == shadow_assignment[node_b]

                if primary_same == shadow_same:
                    agreements += 1
                total_pairs += 1

        return agreements / total_pairs if total_pairs > 0 else 1.0
