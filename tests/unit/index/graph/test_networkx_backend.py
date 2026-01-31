# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Unit tests for NetworkX graph backend."""

import pandas as pd
import pytest

from graphrag.index.graph import NetworkXBackend


@pytest.fixture
def sample_entities():
    """Create sample entities DataFrame."""
    return pd.DataFrame([
        {
            "title": "Entity A",
            "type": "organization",
            "description": "First entity",
        },
        {
            "title": "Entity B",
            "type": "person",
            "description": "Second entity",
        },
        {
            "title": "Entity C",
            "type": "organization",
            "description": "Third entity",
        },
    ])


@pytest.fixture
def sample_relationships():
    """Create sample relationships DataFrame."""
    return pd.DataFrame([
        {
            "source": "Entity A",
            "target": "Entity B",
            "weight": 0.8,
            "description": "works for",
        },
        {
            "source": "Entity B",
            "target": "Entity C",
            "weight": 0.6,
            "description": "collaborates with",
        },
    ])


def test_networkx_backend_initialization():
    """Test NetworkX backend can be initialized."""
    backend = NetworkXBackend()
    assert backend is not None
    assert backend.node_count() == 0
    assert backend.edge_count() == 0


def test_networkx_backend_load_graph(sample_entities, sample_relationships):
    """Test loading entities and relationships."""
    backend = NetworkXBackend()
    backend.load_graph(
        entities=sample_entities,
        relationships=sample_relationships,
        edge_attributes=["weight"],
    )

    assert backend.node_count() == 3
    assert backend.edge_count() == 2


def test_networkx_backend_compute_degrees(sample_entities, sample_relationships):
    """Test computing node degrees."""
    backend = NetworkXBackend()
    backend.load_graph(
        entities=sample_entities,
        relationships=sample_relationships,
    )

    degrees = backend.compute_node_degrees()

    assert len(degrees) == 3
    assert "title" in degrees.columns
    assert "degree" in degrees.columns

    # Check specific degrees
    entity_a_degree = degrees[degrees["title"] == "Entity A"]["degree"].values[0]
    entity_b_degree = degrees[degrees["title"] == "Entity B"]["degree"].values[0]

    assert entity_a_degree == 1  # Connected to Entity B
    assert entity_b_degree == 2  # Connected to both Entity A and Entity C


def test_networkx_backend_detect_communities(sample_entities, sample_relationships):
    """Test community detection."""
    backend = NetworkXBackend()
    backend.load_graph(
        entities=sample_entities,
        relationships=sample_relationships,
    )

    communities = backend.detect_communities(
        max_cluster_size=10,
        use_lcc=False,
        seed=42,
    )

    assert len(communities) > 0
    assert all(hasattr(c, "level") for c in communities)
    assert all(hasattr(c, "cluster_id") for c in communities)
    assert all(hasattr(c, "parent_cluster_id") for c in communities)
    assert all(hasattr(c, "node_ids") for c in communities)


def test_networkx_backend_export_graph(sample_entities, sample_relationships):
    """Test exporting graph to DataFrames."""
    backend = NetworkXBackend()
    backend.load_graph(
        entities=sample_entities,
        relationships=sample_relationships,
    )

    entities_df, relationships_df = backend.export_graph()

    assert len(entities_df) == 3
    assert len(relationships_df) == 2
    assert "title" in entities_df.columns


def test_networkx_backend_clear(sample_entities, sample_relationships):
    """Test clearing the graph."""
    backend = NetworkXBackend()
    backend.load_graph(
        entities=sample_entities,
        relationships=sample_relationships,
    )

    assert backend.node_count() == 3
    backend.clear()
    assert backend.node_count() == 0


def test_networkx_backend_empty_graph():
    """Test operations on empty graph."""
    backend = NetworkXBackend()

    # Should not raise errors
    assert backend.node_count() == 0
    assert backend.edge_count() == 0

    # These should raise ValueError
    with pytest.raises(ValueError, match="Graph not loaded"):
        backend.detect_communities()

    with pytest.raises(ValueError, match="Graph not loaded"):
        backend.compute_node_degrees()

    with pytest.raises(ValueError, match="Graph not loaded"):
        backend.export_graph()
