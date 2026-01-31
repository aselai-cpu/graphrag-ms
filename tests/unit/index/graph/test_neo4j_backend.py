# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Unit tests for Neo4j graph backend."""

import os

import pandas as pd
import pytest

# Skip all tests if neo4j package not installed
pytest.importorskip("neo4j")

from graphrag.index.graph.neo4j_backend import Neo4jBackend


@pytest.fixture
def neo4j_config():
    """Get Neo4j configuration from environment."""
    password = os.environ.get("NEO4J_PASSWORD", "speedkg123")
    return {
        "uri": "bolt://localhost:7687",
        "username": "neo4j",
        "password": password,
        "database": "neo4j",
        "node_label": "TestEntity",  # Use specific label for tests
        "relationship_type": "TEST_RELATED",
    }


@pytest.fixture
def sample_entities():
    """Create sample entities DataFrame."""
    return pd.DataFrame([
        {
            "title": "Neo4j Test A",
            "type": "organization",
            "description": "First test entity",
        },
        {
            "title": "Neo4j Test B",
            "type": "person",
            "description": "Second test entity",
        },
        {
            "title": "Neo4j Test C",
            "type": "organization",
            "description": "Third test entity",
        },
        {
            "title": "Neo4j Test D",
            "type": "person",
            "description": "Fourth test entity",
        },
    ])


@pytest.fixture
def sample_relationships():
    """Create sample relationships DataFrame."""
    return pd.DataFrame([
        {
            "source": "Neo4j Test A",
            "target": "Neo4j Test B",
            "weight": 0.8,
            "description": "works for",
        },
        {
            "source": "Neo4j Test B",
            "target": "Neo4j Test C",
            "weight": 0.6,
            "description": "collaborates with",
        },
        {
            "source": "Neo4j Test C",
            "target": "Neo4j Test D",
            "weight": 0.7,
            "description": "manages",
        },
    ])


@pytest.mark.skipif(
    not os.environ.get("NEO4J_AVAILABLE", "").lower() == "true",
    reason="Neo4j not available (set NEO4J_AVAILABLE=true to enable)",
)
class TestNeo4jBackend:
    """Test Neo4j backend integration."""

    def test_neo4j_backend_initialization(self, neo4j_config):
        """Test Neo4j backend can be initialized and connects."""
        backend = Neo4jBackend(**neo4j_config)
        assert backend is not None
        backend.close()

    def test_neo4j_backend_load_graph(self, neo4j_config, sample_entities, sample_relationships):
        """Test loading entities and relationships into Neo4j."""
        backend = Neo4jBackend(**neo4j_config)
        try:
            backend.load_graph(
                entities=sample_entities,
                relationships=sample_relationships,
                edge_attributes=["weight"],
            )

            assert backend.node_count() == 4
            assert backend.edge_count() == 3
        finally:
            backend.clear()
            backend.close()

    def test_neo4j_backend_compute_degrees(self, neo4j_config, sample_entities, sample_relationships):
        """Test computing node degrees in Neo4j."""
        backend = Neo4jBackend(**neo4j_config)
        try:
            backend.load_graph(
                entities=sample_entities,
                relationships=sample_relationships,
            )

            degrees = backend.compute_node_degrees()

            assert len(degrees) == 4
            assert "title" in degrees.columns
            assert "degree" in degrees.columns

            # Check specific degrees
            entity_a_degree = degrees[degrees["title"] == "Neo4j Test A"]["degree"].values[0]
            entity_b_degree = degrees[degrees["title"] == "Neo4j Test B"]["degree"].values[0]

            assert entity_a_degree == 1  # Connected to Test B
            assert entity_b_degree == 2  # Connected to Test A and Test C
        finally:
            backend.clear()
            backend.close()

    def test_neo4j_backend_detect_communities(self, neo4j_config, sample_entities, sample_relationships):
        """Test Neo4j GDS community detection."""
        backend = Neo4jBackend(**neo4j_config)
        try:
            backend.load_graph(
                entities=sample_entities,
                relationships=sample_relationships,
                edge_attributes=["weight"],
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

            # Verify all nodes are assigned to communities
            all_node_ids = set()
            for community in communities:
                all_node_ids.update(community.node_ids)

            expected_nodes = set(sample_entities["title"].tolist())
            assert all_node_ids == expected_nodes

        finally:
            backend.clear()
            backend.close()

    def test_neo4j_backend_export_graph(self, neo4j_config, sample_entities, sample_relationships):
        """Test exporting Neo4j graph to DataFrames."""
        backend = Neo4jBackend(**neo4j_config)
        try:
            backend.load_graph(
                entities=sample_entities,
                relationships=sample_relationships,
            )

            entities_df, relationships_df = backend.export_graph()

            assert len(entities_df) == 4
            assert len(relationships_df) == 3
            assert "title" in entities_df.columns

        finally:
            backend.clear()
            backend.close()

    def test_neo4j_backend_clear(self, neo4j_config, sample_entities, sample_relationships):
        """Test clearing Neo4j graph."""
        backend = Neo4jBackend(**neo4j_config)
        try:
            backend.load_graph(
                entities=sample_entities,
                relationships=sample_relationships,
            )

            assert backend.node_count() == 4
            backend.clear()
            assert backend.node_count() == 0

        finally:
            backend.close()


# Manual test (not run by pytest by default)
def manual_test_neo4j_backend():
    """Manual test for Neo4j backend - run with: python -c 'from tests.unit.index.graph.test_neo4j_backend import manual_test_neo4j_backend; manual_test_neo4j_backend()'"""
    password = os.environ.get("NEO4J_PASSWORD", "speedkg123")

    backend = Neo4jBackend(
        uri="bolt://localhost:7687",
        username="neo4j",
        password=password,
        database="neo4j",
        node_label="ManualTestEntity",
    )

    print("✓ Connected to Neo4j")

    # Create sample data
    entities = pd.DataFrame([
        {"title": "Alice", "type": "person", "description": "Engineer"},
        {"title": "Bob", "type": "person", "description": "Designer"},
        {"title": "Acme Corp", "type": "organization", "description": "Company"},
    ])

    relationships = pd.DataFrame([
        {"source": "Alice", "target": "Acme Corp", "weight": 0.9, "description": "works at"},
        {"source": "Bob", "target": "Acme Corp", "weight": 0.8, "description": "works at"},
    ])

    print(f"✓ Loading {len(entities)} entities, {len(relationships)} relationships")

    backend.load_graph(entities, relationships, edge_attributes=["weight"])

    print(f"✓ Loaded: {backend.node_count()} nodes, {backend.edge_count()} edges")

    communities = backend.detect_communities(seed=42)

    print(f"✓ Detected {len(communities)} communities")
    for comm in communities:
        print(f"  Level {comm.level}, Cluster {comm.cluster_id}: {comm.node_ids}")

    backend.clear()
    backend.close()

    print("✓ Manual test completed successfully")


if __name__ == "__main__":
    # Run manual test
    manual_test_neo4j_backend()
