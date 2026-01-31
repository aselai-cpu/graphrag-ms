# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Unit tests for DarkModeOrchestrator."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from graphrag.index.graph import NetworkXBackend
from graphrag.index.graph.dark_mode import DarkModeOrchestrator


@pytest.fixture
def sample_entities():
    """Create sample entities DataFrame."""
    df = pd.DataFrame([
        {"title": "Entity A", "type": "org", "description": "First"},
        {"title": "Entity B", "type": "person", "description": "Second"},
        {"title": "Entity C", "type": "org", "description": "Third"},
        {"title": "Entity D", "type": "person", "description": "Fourth"},
    ])
    # Don't set index - keep title as a column
    return df


@pytest.fixture
def sample_relationships():
    """Create sample relationships DataFrame."""
    return pd.DataFrame([
        {"source": "Entity A", "target": "Entity B", "weight": 0.8},
        {"source": "Entity B", "target": "Entity C", "weight": 0.6},
        {"source": "Entity C", "target": "Entity D", "weight": 0.7},
    ])


def test_dark_mode_orchestrator_initialization():
    """Test DarkModeOrchestrator can be initialized."""
    primary = NetworkXBackend()
    shadow = NetworkXBackend()

    orchestrator = DarkModeOrchestrator(primary, shadow)

    assert orchestrator.primary is primary
    assert orchestrator.shadow is shadow
    assert orchestrator.comparison_framework is not None
    assert orchestrator.metrics_collector is not None


def test_dark_mode_load_graph_identical_backends(sample_entities, sample_relationships):
    """Test load_graph with two identical backends (should match perfectly)."""
    primary = NetworkXBackend()
    shadow = NetworkXBackend()

    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "metrics.jsonl"
        orchestrator = DarkModeOrchestrator(primary, shadow, log_path=log_path)

        # Load graph - should succeed on both
        orchestrator.load_graph(sample_entities, sample_relationships, edge_attributes=["weight"])

        # Check both backends loaded correctly
        assert primary.node_count() == 4
        assert shadow.node_count() == 4
        assert primary.edge_count() == 3
        assert shadow.edge_count() == 3

        # Check metrics were recorded
        summary = orchestrator.get_metrics_summary()
        assert summary["total_operations"] == 1
        assert summary["shadow_errors"] == 0
        assert summary["comparisons_passed"] == 1


def test_dark_mode_detect_communities(sample_entities, sample_relationships):
    """Test community detection in dark mode."""
    primary = NetworkXBackend()
    shadow = NetworkXBackend()

    orchestrator = DarkModeOrchestrator(primary, shadow)

    # Load graph first
    orchestrator.load_graph(sample_entities, sample_relationships)

    # Detect communities - should work on both
    communities = orchestrator.detect_communities(max_cluster_size=10, seed=42)

    # Check communities returned from primary
    assert len(communities) > 0
    assert all(hasattr(c, "level") for c in communities)

    # Check metrics
    summary = orchestrator.get_metrics_summary()
    assert summary["total_operations"] == 2  # load_graph + detect_communities


def test_dark_mode_compute_degrees(sample_entities, sample_relationships):
    """Test degree computation in dark mode."""
    primary = NetworkXBackend()
    shadow = NetworkXBackend()

    orchestrator = DarkModeOrchestrator(primary, shadow)

    # Load graph first
    orchestrator.load_graph(sample_entities, sample_relationships)

    # Compute degrees
    degrees = orchestrator.compute_node_degrees()

    # Check degrees returned
    assert len(degrees) == 4
    assert "title" in degrees.columns
    assert "degree" in degrees.columns

    # Check metrics
    summary = orchestrator.get_metrics_summary()
    assert summary["total_operations"] == 2  # load_graph + compute_node_degrees


def test_dark_mode_export_graph(sample_entities, sample_relationships):
    """Test graph export in dark mode."""
    primary = NetworkXBackend()
    shadow = NetworkXBackend()

    orchestrator = DarkModeOrchestrator(primary, shadow)

    # Load graph first
    orchestrator.load_graph(sample_entities, sample_relationships)

    # Export graph
    entities, relationships = orchestrator.export_graph()

    # Check exported data
    assert len(entities) == 4
    assert len(relationships) == 3

    # Check metrics
    summary = orchestrator.get_metrics_summary()
    assert summary["total_operations"] == 2  # load_graph + export_graph


def test_dark_mode_shadow_failure_does_not_affect_primary(sample_entities, sample_relationships):
    """Test that shadow backend failures don't affect primary results."""

    class FailingShadowBackend(NetworkXBackend):
        """Shadow backend that always fails."""

        def detect_communities(self, **kwargs):
            raise RuntimeError("Shadow backend intentionally failed")

    primary = NetworkXBackend()
    shadow = FailingShadowBackend()

    orchestrator = DarkModeOrchestrator(primary, shadow)

    # Load graph (should succeed)
    orchestrator.load_graph(sample_entities, sample_relationships)

    # Detect communities - primary should succeed despite shadow failure
    communities = orchestrator.detect_communities(seed=42)

    assert len(communities) > 0  # Primary succeeded

    # Check metrics show shadow error
    summary = orchestrator.get_metrics_summary()
    assert summary["shadow_errors"] == 1  # detect_communities failed
    assert summary["shadow_error_rate"] == 0.5  # 1 out of 2 operations


def test_dark_mode_metrics_collection(sample_entities, sample_relationships):
    """Test metrics are properly collected."""
    primary = NetworkXBackend()
    shadow = NetworkXBackend()

    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "metrics.jsonl"
        orchestrator = DarkModeOrchestrator(primary, shadow, log_path=log_path)

        # Run several operations
        orchestrator.load_graph(sample_entities, sample_relationships)
        orchestrator.detect_communities(seed=42)
        orchestrator.compute_node_degrees()

        # Check summary
        summary = orchestrator.get_metrics_summary()
        assert summary["total_operations"] == 3
        assert summary["shadow_errors"] == 0
        assert summary["comparisons_passed"] >= 1  # At least load_graph
        assert summary["avg_primary_duration_ms"] > 0
        assert summary["avg_shadow_duration_ms"] > 0

        # Check log file was created and has content
        assert log_path.exists()
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 3  # One line per operation


def test_dark_mode_cutover_criteria():
    """Test cutover readiness checks."""
    primary = NetworkXBackend()
    shadow = NetworkXBackend()

    orchestrator = DarkModeOrchestrator(primary, shadow)

    # Not enough operations yet
    ready, reasons = orchestrator.check_cutover_readiness(min_operations=1000)
    assert not ready
    assert any("Not enough operations" in r for r in reasons)

    # With minimal operations requirement
    ready, reasons = orchestrator.check_cutover_readiness(min_operations=0)
    assert ready  # No operations yet, but threshold is 0
    assert len(reasons) == 0


def test_dark_mode_clear():
    """Test clearing both backends."""
    primary = NetworkXBackend()
    shadow = NetworkXBackend()

    orchestrator = DarkModeOrchestrator(primary, shadow)

    # Load some data
    entities = pd.DataFrame([{"title": "A"}, {"title": "B"}])
    relationships = pd.DataFrame([{"source": "A", "target": "B", "weight": 1.0}])

    orchestrator.load_graph(entities, relationships)

    assert orchestrator.node_count() == 2

    # Clear
    orchestrator.clear()

    assert orchestrator.node_count() == 0


def test_dark_mode_full_workflow(sample_entities, sample_relationships):
    """Test complete dark mode workflow."""
    primary = NetworkXBackend()
    shadow = NetworkXBackend()

    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "metrics.jsonl"
        orchestrator = DarkModeOrchestrator(primary, shadow, log_path=log_path)

        # Full workflow
        orchestrator.load_graph(sample_entities, sample_relationships, edge_attributes=["weight"])
        communities = orchestrator.detect_communities(max_cluster_size=10, seed=42)
        degrees = orchestrator.compute_node_degrees()
        entities, relationships = orchestrator.export_graph()

        # Verify all operations completed
        assert len(communities) > 0
        assert len(degrees) == 4
        assert len(entities) == 4
        assert len(relationships) == 3

        # Check comprehensive metrics
        summary = orchestrator.get_metrics_summary()
        assert summary["total_operations"] == 4
        assert summary["shadow_errors"] == 0
        assert summary["comparison_pass_rate"] == 1.0  # All comparisons passed
        assert summary["avg_latency_ratio"] > 0  # Shadow has some latency

        # Check cutover readiness with lenient criteria
        ready, reasons = orchestrator.check_cutover_readiness(
            min_operations=4,
            max_error_rate=0.01,
            min_pass_rate=0.95,
            max_latency_ratio=10.0,  # Lenient for NetworkX-NetworkX
        )
        assert ready
        assert len(reasons) == 0
