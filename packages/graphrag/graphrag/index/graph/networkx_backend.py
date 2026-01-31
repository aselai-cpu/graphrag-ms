# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""NetworkX graph backend implementation."""

import logging

import networkx as nx
import pandas as pd

from graphrag.index.graph.graph_backend import Communities, CommunityResult, GraphBackend
from graphrag.index.operations.cluster_graph import cluster_graph
from graphrag.index.operations.compute_degree import compute_degree
from graphrag.index.operations.create_graph import create_graph
from graphrag.index.operations.graph_to_dataframes import graph_to_dataframes

logger = logging.getLogger(__name__)


class NetworkXBackend(GraphBackend):
    """NetworkX-based graph backend implementation.

    This is the default backend that uses NetworkX for all graph operations.
    It wraps existing GraphRAG NetworkX operations into the GraphBackend interface.
    """

    def __init__(self):
        """Initialize the NetworkX backend."""
        self._graph: nx.Graph | None = None
        self._entity_id_col: str = "title"

    def load_graph(
        self,
        entities: pd.DataFrame,
        relationships: pd.DataFrame,
        *,
        entity_id_col: str = "title",
        source_col: str = "source",
        target_col: str = "target",
        edge_attributes: list[str] | None = None,
    ) -> None:
        """Load entities and relationships into NetworkX graph."""
        self._entity_id_col = entity_id_col

        # Create graph from relationships (existing operation)
        self._graph = create_graph(
            edges=relationships,
            edge_attr=edge_attributes,
            nodes=entities if not entities.empty else None,
            node_id=entity_id_col,
        )

        logger.info(
            "Loaded graph: %d nodes, %d edges",
            len(self._graph.nodes),
            len(self._graph.edges),
        )

    def detect_communities(
        self,
        *,
        max_cluster_size: int = 10,
        use_lcc: bool = True,
        seed: int | None = None,
    ) -> Communities:
        """Run Leiden community detection on NetworkX graph."""
        if self._graph is None:
            raise ValueError("Graph not loaded. Call load_graph() first.")

        # Run existing cluster_graph operation
        raw_communities = cluster_graph(
            graph=self._graph,
            max_cluster_size=max_cluster_size,
            use_lcc=use_lcc,
            seed=seed,
        )

        # Convert to CommunityResult objects
        communities = [
            CommunityResult(
                level=level,
                cluster_id=cluster_id,
                parent_cluster_id=parent_id,
                node_ids=node_ids,
            )
            for level, cluster_id, parent_id, node_ids in raw_communities
        ]

        logger.info("Detected %d communities", len(communities))
        return communities

    def compute_node_degrees(self) -> pd.DataFrame:
        """Compute degree centrality using NetworkX."""
        if self._graph is None:
            raise ValueError("Graph not loaded. Call load_graph() first.")

        # Use existing compute_degree operation
        degrees_df = compute_degree(self._graph)

        # Ensure correct column name
        if self._entity_id_col != "title" and "title" in degrees_df.columns:
            degrees_df = degrees_df.rename(columns={"title": self._entity_id_col})

        return degrees_df

    def export_graph(
        self,
        *,
        entity_id_col: str = "title",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Export NetworkX graph to DataFrames."""
        if self._graph is None:
            raise ValueError("Graph not loaded. Call load_graph() first.")

        # Use existing graph_to_dataframes utility
        entities_df, relationships_df = graph_to_dataframes(
            graph=self._graph,
            node_id=entity_id_col,
        )

        return entities_df, relationships_df

    def clear(self) -> None:
        """Clear the NetworkX graph."""
        self._graph = None
        logger.info("Cleared NetworkX graph")

    def node_count(self) -> int:
        """Return number of nodes."""
        if self._graph is None:
            return 0
        return len(self._graph.nodes)

    def edge_count(self) -> int:
        """Return number of edges."""
        if self._graph is None:
            return 0
        return len(self._graph.edges)
