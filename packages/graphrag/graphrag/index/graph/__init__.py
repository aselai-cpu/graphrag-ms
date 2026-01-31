# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Graph backend abstraction for GraphRAG indexing."""

from graphrag.index.graph.graph_backend import (
    Communities,
    CommunityResult,
    GraphBackend,
)
from graphrag.index.graph.graph_factory import (
    create_graph_backend,
    create_graph_backend_with_dark_mode,
)
from graphrag.index.graph.networkx_backend import NetworkXBackend

__all__ = [
    "Communities",
    "CommunityResult",
    "GraphBackend",
    "NetworkXBackend",
    "create_graph_backend",
    "create_graph_backend_with_dark_mode",
]
