# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Abstract base class for graph backend implementations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass
class CommunityResult:
    """Result from community detection algorithm.

    Attributes
    ----------
    level : int
        Hierarchy level (0 = leaf communities)
    cluster_id : int
        Unique cluster/community identifier
    parent_cluster_id : int
        Parent cluster ID (-1 for root)
    node_ids : list[str]
        List of node IDs in this community
    """

    level: int
    cluster_id: int
    parent_cluster_id: int
    node_ids: list[str]


Communities = list[CommunityResult]


class GraphBackend(ABC):
    """Abstract base class for graph backend implementations.

    This defines the interface that graph backends (NetworkX, Neo4j, etc.)
    must implement to be compatible with GraphRAG indexing pipeline.
    """

    @abstractmethod
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
        """Load entities and relationships into the graph backend.

        Parameters
        ----------
        entities : pd.DataFrame
            DataFrame containing entity nodes with columns: id, title, type, description, etc.
        relationships : pd.DataFrame
            DataFrame containing relationships with columns: source, target, weight, description, etc.
        entity_id_col : str, optional
            Column name to use as entity identifier, by default "title"
        source_col : str, optional
            Column name for relationship source, by default "source"
        target_col : str, optional
            Column name for relationship target, by default "target"
        edge_attributes : list[str] | None, optional
            List of edge attribute columns to include (e.g., ["weight"]), by default None
        """
        ...

    @abstractmethod
    def detect_communities(
        self,
        *,
        max_cluster_size: int = 10,
        use_lcc: bool = True,
        seed: int | None = None,
    ) -> Communities:
        """Run community detection algorithm on the graph.

        Parameters
        ----------
        max_cluster_size : int, optional
            Maximum community size, by default 10
        use_lcc : bool, optional
            Use largest connected component only, by default True
        seed : int | None, optional
            Random seed for reproducibility, by default None

        Returns
        -------
        Communities
            List of CommunityResult objects with hierarchy information
        """
        ...

    @abstractmethod
    def compute_node_degrees(self) -> pd.DataFrame:
        """Compute degree centrality for all nodes.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: [entity_id_col, degree]
            where degree is the number of connections
        """
        ...

    @abstractmethod
    def export_graph(
        self,
        *,
        entity_id_col: str = "title",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Export the graph back to DataFrames.

        Parameters
        ----------
        entity_id_col : str, optional
            Column name for entity identifier, by default "title"

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            (entities_df, relationships_df)
        """
        ...

    @abstractmethod
    def clear(self) -> None:
        """Clear all data from the graph backend."""
        ...

    @abstractmethod
    def node_count(self) -> int:
        """Return the number of nodes in the graph."""
        ...

    @abstractmethod
    def edge_count(self) -> int:
        """Return the number of edges in the graph."""
        ...
