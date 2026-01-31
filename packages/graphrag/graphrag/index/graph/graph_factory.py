# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Factory for creating graph backend instances."""

import logging
from typing import Any

from graphrag.index.graph.graph_backend import GraphBackend
from graphrag.index.graph.networkx_backend import NetworkXBackend

logger = logging.getLogger(__name__)


def create_graph_backend(backend_type: str = "networkx", **kwargs: Any) -> GraphBackend:
    """Create a graph backend instance based on type.

    Parameters
    ----------
    backend_type : str, optional
        Type of backend to create ("networkx" or "neo4j"), by default "networkx"
    **kwargs : Any
        Backend-specific configuration parameters

    Returns
    -------
    GraphBackend
        Configured graph backend instance

    Raises
    ------
    ValueError
        If backend_type is not recognized

    Examples
    --------
    Create NetworkX backend (default):
    >>> backend = create_graph_backend("networkx")

    Create Neo4j backend:
    >>> backend = create_graph_backend(
    ...     "neo4j",
    ...     uri="bolt://localhost:7687",
    ...     username="neo4j",
    ...     password="password",
    ...     database="graphrag"
    ... )
    """
    backend_type = backend_type.lower()

    if backend_type == "networkx":
        logger.info("Creating NetworkX graph backend")
        return NetworkXBackend()

    elif backend_type == "neo4j":
        logger.info("Creating Neo4j graph backend")
        # Import here to avoid requiring neo4j package if not used
        try:
            from graphrag.index.graph.neo4j_backend import Neo4jBackend
        except ImportError as e:
            raise ImportError(
                "Neo4j backend requires 'neo4j' package. Install with: pip install neo4j"
            ) from e

        # Extract Neo4j configuration
        required_params = ["uri", "username", "password"]
        missing_params = [p for p in required_params if p not in kwargs]
        if missing_params:
            raise ValueError(
                f"Neo4j backend requires parameters: {missing_params}. "
                f"Provide them in the configuration."
            )

        return Neo4jBackend(**kwargs)

    else:
        raise ValueError(
            f"Unknown backend type: {backend_type}. "
            f"Supported types: 'networkx', 'neo4j'"
        )
