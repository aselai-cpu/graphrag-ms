# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Unit tests for ComparisonFramework."""

import pandas as pd
import pytest

from graphrag.index.graph.dark_mode import ComparisonFramework
from graphrag.index.graph.graph_backend import CommunityResult


@pytest.fixture
def sample_entities():
    """Create sample entities DataFrame."""
    return pd.DataFrame([
        {"title": "Entity A", "type": "org", "description": "First"},
        {"title": "Entity B", "type": "person", "description": "Second"},
        {"title": "Entity C", "type": "org", "description": "Third"},
        {"title": "Entity D", "type": "person", "description": "Fourth"},
    ])


@pytest.fixture
def sample_degrees():
    """Create sample node degrees DataFrame."""
    return pd.DataFrame([
        {"title": "Entity A", "degree": 2},
        {"title": "Entity B", "degree": 3},
        {"title": "Entity C", "degree": 2},
        {"title": "Entity D", "degree": 1},
    ])


@pytest.fixture
def sample_communities():
    """Create sample communities list."""
    return [
        CommunityResult(
            level=0,
            cluster_id=0,
            parent_cluster_id=-1,
            node_ids=["Entity A", "Entity B"],
        ),
        CommunityResult(
            level=0,
            cluster_id=1,
            parent_cluster_id=-1,
            node_ids=["Entity C", "Entity D"],
        ),
    ]


def test_comparison_framework_initialization():
    """Test ComparisonFramework can be initialized."""
    framework = ComparisonFramework()

    assert framework.entity_match_threshold == 0.99
    assert framework.community_match_threshold == 0.95
    assert framework.degree_tolerance == 0.1


def test_comparison_framework_custom_thresholds():
    """Test ComparisonFramework with custom thresholds."""
    framework = ComparisonFramework(
        entity_match_threshold=0.95,
        community_match_threshold=0.90,
        degree_tolerance=0.2,
    )

    assert framework.entity_match_threshold == 0.95
    assert framework.community_match_threshold == 0.90
    assert framework.degree_tolerance == 0.2


def test_compare_entities_exact_match(sample_entities):
    """Test entity comparison with exact match."""
    framework = ComparisonFramework()

    result = framework.compare_entities(sample_entities, sample_entities.copy())

    assert result.operation == "compare_entities"
    assert result.passed is True
    assert result.metrics["primary_count"] == 4
    assert result.metrics["shadow_count"] == 4
    assert result.metrics["count_match"] is True
    assert result.metrics["overlap_count"] == 4
    assert result.metrics["precision"] == 1.0
    assert result.metrics["recall"] == 1.0
    assert result.metrics["f1"] == 1.0
    assert result.metrics["missing_in_shadow"] == 0
    assert result.metrics["extra_in_shadow"] == 0
    assert len(result.differences) == 0


def test_compare_entities_missing_in_shadow(sample_entities):
    """Test entity comparison with entities missing in shadow."""
    framework = ComparisonFramework()

    # Shadow missing one entity
    shadow_entities = sample_entities[sample_entities["title"] != "Entity D"].copy()

    result = framework.compare_entities(sample_entities, shadow_entities)

    assert result.operation == "compare_entities"
    assert result.metrics["primary_count"] == 4
    assert result.metrics["shadow_count"] == 3
    assert result.metrics["count_match"] is False
    assert result.metrics["overlap_count"] == 3
    assert result.metrics["missing_in_shadow"] == 1
    assert result.metrics["extra_in_shadow"] == 0
    assert result.metrics["precision"] == 1.0  # All shadow entities are in primary
    assert result.metrics["recall"] == 0.75  # 3 out of 4 primary entities in shadow
    assert result.metrics["f1"] < 1.0
    assert len(result.differences) > 0


def test_compare_entities_extra_in_shadow(sample_entities):
    """Test entity comparison with extra entities in shadow."""
    framework = ComparisonFramework()

    # Shadow has extra entity
    shadow_entities = pd.concat([
        sample_entities,
        pd.DataFrame([{"title": "Entity E", "type": "org", "description": "Extra"}])
    ], ignore_index=True)

    result = framework.compare_entities(sample_entities, shadow_entities)

    assert result.metrics["primary_count"] == 4
    assert result.metrics["shadow_count"] == 5
    assert result.metrics["count_match"] is False
    assert result.metrics["overlap_count"] == 4
    assert result.metrics["missing_in_shadow"] == 0
    assert result.metrics["extra_in_shadow"] == 1
    assert result.metrics["precision"] == 0.8  # 4 out of 5 shadow entities in primary
    assert result.metrics["recall"] == 1.0  # All primary entities in shadow
    assert result.metrics["f1"] < 1.0


def test_compare_entities_complete_mismatch():
    """Test entity comparison with no overlap."""
    framework = ComparisonFramework()

    primary = pd.DataFrame([
        {"title": "A", "type": "org"},
        {"title": "B", "type": "person"},
    ])
    shadow = pd.DataFrame([
        {"title": "X", "type": "org"},
        {"title": "Y", "type": "person"},
    ])

    result = framework.compare_entities(primary, shadow)

    assert result.passed is False
    assert result.metrics["overlap_count"] == 0
    assert result.metrics["precision"] == 0.0
    assert result.metrics["recall"] == 0.0
    assert result.metrics["f1"] == 0.0
    assert result.metrics["missing_in_shadow"] == 2
    assert result.metrics["extra_in_shadow"] == 2


def test_compare_entities_empty_shadow():
    """Test entity comparison with empty shadow."""
    framework = ComparisonFramework()

    primary = pd.DataFrame([{"title": "A", "type": "org"}])
    shadow = pd.DataFrame(columns=["title", "type"])

    result = framework.compare_entities(primary, shadow)

    assert result.passed is False
    assert result.metrics["primary_count"] == 1
    assert result.metrics["shadow_count"] == 0
    assert result.metrics["precision"] == 0.0
    assert result.metrics["recall"] == 0.0


def test_compare_communities_exact_match(sample_communities):
    """Test community comparison with exact match."""
    framework = ComparisonFramework()

    result = framework.compare_communities(sample_communities, sample_communities)

    assert result.operation == "compare_communities"
    assert result.passed is True
    assert result.metrics["primary_level_count"] == 1
    assert result.metrics["shadow_level_count"] == 1
    assert result.metrics["common_levels"] == 1
    assert result.metrics["avg_similarity"] == 1.0
    assert len(result.differences) == 0


def test_compare_communities_different_cluster_ids():
    """Test community comparison with different cluster IDs but same clustering."""
    framework = ComparisonFramework()

    primary = [
        CommunityResult(level=0, cluster_id=0, parent_cluster_id=-1, node_ids=["A", "B"]),
        CommunityResult(level=0, cluster_id=1, parent_cluster_id=-1, node_ids=["C", "D"]),
    ]

    # Same clustering but different IDs
    shadow = [
        CommunityResult(level=0, cluster_id=99, parent_cluster_id=-1, node_ids=["A", "B"]),
        CommunityResult(level=0, cluster_id=100, parent_cluster_id=-1, node_ids=["C", "D"]),
    ]

    result = framework.compare_communities(primary, shadow)

    # Should match perfectly since clustering is the same
    assert result.passed is True
    assert result.metrics["avg_similarity"] == 1.0


def test_compare_communities_different_clustering():
    """Test community comparison with different clustering."""
    framework = ComparisonFramework()

    primary = [
        CommunityResult(level=0, cluster_id=0, parent_cluster_id=-1, node_ids=["A", "B"]),
        CommunityResult(level=0, cluster_id=1, parent_cluster_id=-1, node_ids=["C", "D"]),
    ]

    # Different clustering
    shadow = [
        CommunityResult(level=0, cluster_id=0, parent_cluster_id=-1, node_ids=["A", "C"]),
        CommunityResult(level=0, cluster_id=1, parent_cluster_id=-1, node_ids=["B", "D"]),
    ]

    result = framework.compare_communities(primary, shadow)

    # Should not match well
    assert result.metrics["avg_similarity"] < 1.0


def test_compare_communities_different_levels():
    """Test community comparison with different level counts."""
    framework = ComparisonFramework()

    primary = [
        CommunityResult(level=0, cluster_id=0, parent_cluster_id=-1, node_ids=["A", "B"]),
    ]

    shadow = [
        CommunityResult(level=0, cluster_id=0, parent_cluster_id=-1, node_ids=["A", "B"]),
        CommunityResult(level=1, cluster_id=1, parent_cluster_id=0, node_ids=["A"]),
        CommunityResult(level=1, cluster_id=2, parent_cluster_id=0, node_ids=["B"]),
    ]

    result = framework.compare_communities(primary, shadow)

    assert result.metrics["primary_level_count"] == 1
    assert result.metrics["shadow_level_count"] == 2
    assert "Level count mismatch" in result.differences[0]


def test_compare_node_degrees_exact_match(sample_degrees):
    """Test degree comparison with exact match."""
    framework = ComparisonFramework()

    result = framework.compare_node_degrees(sample_degrees, sample_degrees.copy())

    assert result.operation == "compare_node_degrees"
    assert result.passed is True
    assert result.metrics["node_count"] == 4
    assert result.metrics["mismatch_count"] == 0
    assert result.metrics["max_relative_diff"] == 0.0
    assert result.metrics["avg_relative_diff"] == 0.0
    assert result.metrics["exact_match_rate"] == 1.0
    assert len(result.differences) == 0


def test_compare_node_degrees_small_differences():
    """Test degree comparison with small differences."""
    framework = ComparisonFramework(degree_tolerance=0.15)

    primary = pd.DataFrame([
        {"title": "A", "degree": 10},
        {"title": "B", "degree": 20},
    ])

    # Shadow has slightly different degrees (within 10% tolerance)
    shadow = pd.DataFrame([
        {"title": "A", "degree": 11},  # 10% difference
        {"title": "B", "degree": 20},  # Exact match
    ])

    result = framework.compare_node_degrees(primary, shadow)

    assert result.metrics["node_count"] == 2
    assert result.metrics["mismatch_count"] == 1
    assert result.metrics["max_relative_diff"] == 0.1  # 10% difference
    assert result.passed is True  # Within 15% tolerance


def test_compare_node_degrees_large_differences():
    """Test degree comparison with large differences."""
    framework = ComparisonFramework(degree_tolerance=0.1)

    primary = pd.DataFrame([
        {"title": "A", "degree": 10},
        {"title": "B", "degree": 20},
    ])

    # Shadow has large differences (beyond 10% tolerance)
    shadow = pd.DataFrame([
        {"title": "A", "degree": 15},  # 50% difference
        {"title": "B", "degree": 20},
    ])

    result = framework.compare_node_degrees(primary, shadow)

    assert result.passed is False  # Beyond 10% tolerance
    assert result.metrics["max_relative_diff"] == 0.5


def test_compare_node_degrees_missing_nodes():
    """Test degree comparison with missing nodes."""
    framework = ComparisonFramework()

    primary = pd.DataFrame([
        {"title": "A", "degree": 10},
        {"title": "B", "degree": 20},
        {"title": "C", "degree": 15},
    ])

    shadow = pd.DataFrame([
        {"title": "A", "degree": 10},
        {"title": "B", "degree": 20},
    ])

    result = framework.compare_node_degrees(primary, shadow)

    # Should handle missing nodes gracefully
    assert result.metrics["node_count"] == 3  # Outer join includes all nodes


def test_clustering_similarity_empty_communities():
    """Test clustering similarity with single node."""
    framework = ComparisonFramework()

    primary = [
        CommunityResult(level=0, cluster_id=0, parent_cluster_id=-1, node_ids=["A"]),
    ]

    shadow = [
        CommunityResult(level=0, cluster_id=0, parent_cluster_id=-1, node_ids=["A"]),
    ]

    result = framework.compare_communities(primary, shadow)

    # Single node should match
    assert result.metrics["avg_similarity"] == 1.0


def test_clustering_similarity_two_nodes():
    """Test clustering similarity with two nodes."""
    framework = ComparisonFramework()

    # Both put A and B in same cluster
    primary = [
        CommunityResult(level=0, cluster_id=0, parent_cluster_id=-1, node_ids=["A", "B"]),
    ]
    shadow = [
        CommunityResult(level=0, cluster_id=99, parent_cluster_id=-1, node_ids=["A", "B"]),
    ]

    result = framework.compare_communities(primary, shadow)
    assert result.metrics["avg_similarity"] == 1.0

    # Primary splits A and B, shadow keeps them together
    primary2 = [
        CommunityResult(level=0, cluster_id=0, parent_cluster_id=-1, node_ids=["A"]),
        CommunityResult(level=0, cluster_id=1, parent_cluster_id=-1, node_ids=["B"]),
    ]
    shadow2 = [
        CommunityResult(level=0, cluster_id=0, parent_cluster_id=-1, node_ids=["A", "B"]),
    ]

    result2 = framework.compare_communities(primary2, shadow2)
    assert result2.metrics["avg_similarity"] == 0.0  # Complete disagreement
