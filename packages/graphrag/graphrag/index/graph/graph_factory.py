# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Factory for creating graph backend instances."""

import logging
from pathlib import Path
from typing import Any

from graphrag.config.models.dark_mode_config import DarkModeConfig
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


def create_graph_backend_with_dark_mode(
    dark_mode_config: DarkModeConfig,
    primary_backend_kwargs: dict[str, Any] | None = None,
    shadow_backend_kwargs: dict[str, Any] | None = None,
) -> GraphBackend:
    """Create a graph backend with optional dark mode support.

    If dark mode is enabled, returns a DarkModeOrchestrator that wraps
    primary and shadow backends. Otherwise, returns just the primary backend.

    Parameters
    ----------
    dark_mode_config : DarkModeConfig
        Dark mode configuration
    primary_backend_kwargs : dict[str, Any] | None, optional
        Additional kwargs for primary backend
    shadow_backend_kwargs : dict[str, Any] | None, optional
        Additional kwargs for shadow backend (only used if dark mode enabled)

    Returns
    -------
    GraphBackend
        Either a single backend or a DarkModeOrchestrator wrapping both backends

    Raises
    ------
    ValueError
        If backend types are not supported

    Examples
    --------
    Create backend with dark mode disabled:
    >>> config = DarkModeConfig(enabled=False, primary_backend="networkx")
    >>> backend = create_graph_backend_with_dark_mode(config)

    Create backend with dark mode enabled:
    >>> config = DarkModeConfig(
    ...     enabled=True,
    ...     primary_backend="networkx",
    ...     shadow_backend="neo4j",
    ...     log_path="metrics.jsonl"
    ... )
    >>> backend = create_graph_backend_with_dark_mode(
    ...     config,
    ...     shadow_backend_kwargs={"uri": "bolt://localhost:7687", ...}
    ... )
    """
    primary_backend_kwargs = primary_backend_kwargs or {}
    shadow_backend_kwargs = shadow_backend_kwargs or {}

    if not dark_mode_config.enabled:
        # Dark mode disabled - just return primary backend
        logger.info("Dark mode disabled, using %s backend", dark_mode_config.primary_backend)
        return create_graph_backend(
            dark_mode_config.primary_backend,
            **primary_backend_kwargs,
        )

    # Dark mode enabled - create orchestrator with both backends
    logger.info(
        "Dark mode enabled: primary=%s, shadow=%s",
        dark_mode_config.primary_backend,
        dark_mode_config.shadow_backend,
    )

    # Import here to avoid circular dependency
    from graphrag.index.graph.dark_mode import (
        ComparisonFramework,
        DarkModeOrchestrator,
        MetricsCollector,
    )

    primary_backend = create_graph_backend(
        dark_mode_config.primary_backend,
        **primary_backend_kwargs,
    )

    # Try to create shadow backend - if it fails, provide helpful error message
    try:
        shadow_backend = create_graph_backend(
            dark_mode_config.shadow_backend,
            **shadow_backend_kwargs,
        )
    except ConnectionError as e:
        logger.error(
            "Failed to initialize shadow backend (%s). "
            "Dark mode requires shadow backend to be accessible. "
            "Error: %s",
            dark_mode_config.shadow_backend,
            e,
        )
        raise RuntimeError(
            f"Dark mode enabled but shadow backend ({dark_mode_config.shadow_backend}) "
            f"cannot be initialized. Please either:\n"
            f"  1. Ensure {dark_mode_config.shadow_backend} is running and accessible\n"
            f"  2. Disable dark mode: Set dark_mode.enabled: false in settings.yaml\n"
            f"Original error: {e}"
        ) from e
    except Exception as e:
        logger.error(
            "Unexpected error initializing shadow backend (%s): %s",
            dark_mode_config.shadow_backend,
            e,
        )
        raise

    # Create comparison framework with configured thresholds
    comparison_framework = ComparisonFramework(
        entity_match_threshold=dark_mode_config.comparison.entity_match_threshold,
        community_match_threshold=dark_mode_config.comparison.community_match_threshold,
        degree_tolerance=dark_mode_config.comparison.degree_tolerance,
    )

    # Create metrics collector with log path
    log_path = Path(dark_mode_config.log_path) if dark_mode_config.log_path else None
    metrics_collector = MetricsCollector(log_path=log_path)

    # Create and return orchestrator
    orchestrator = DarkModeOrchestrator(
        primary=primary_backend,
        shadow=shadow_backend,
        comparison_framework=comparison_framework,
        metrics_collector=metrics_collector,
    )

    logger.info("Dark mode orchestrator created successfully")

    return orchestrator
