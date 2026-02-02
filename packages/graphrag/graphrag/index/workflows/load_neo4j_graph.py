# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Workflow to load complete graph into Neo4j."""

import logging

from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.index.typing.context import PipelineRunContext
from graphrag.index.typing.workflow import WorkflowFunctionOutput
from graphrag.utils.storage import load_table_from_storage

logger = logging.getLogger(__name__)


async def run_workflow(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """Load entities, relationships, text units, communities, and claims into Neo4j.

    This workflow loads the complete knowledge graph into Neo4j:
    - Entities with multi-type labels (Person, Organization, Geo)
    - Relationships between entities
    - Documents with HAS_TEXT_UNIT relationships
    - Text units with MENTIONS relationships
    - Communities with CONTAINS and PARENT_OF relationships
    - Claims with ABOUT, INVOLVES, and EXTRACTED_FROM relationships
    """
    logger.info("Workflow started: load_neo4j_graph")

    # Check if Neo4j is configured
    if config.neo4j is None:
        logger.info("Neo4j not configured, skipping graph loading")
        return WorkflowFunctionOutput(result=None)

    logger.info("Loading complete graph into Neo4j")

    try:
        from graphrag.index.graph.neo4j_backend import Neo4jBackend

        # Create Neo4j backend
        backend = Neo4jBackend(
            uri=config.neo4j.uri,
            username=config.neo4j.username,
            password=config.neo4j.password,
            database=config.neo4j.database,
            node_label=config.neo4j.node_label,
            relationship_type=config.neo4j.relationship_type,
            use_entity_type_labels=config.neo4j.use_entity_type_labels,
            gds_library=config.neo4j.gds_library,
        )

        # Load entities and relationships
        logger.info("Loading entities and relationships")
        entities = await load_table_from_storage("entities", context.output_storage)
        relationships = await load_table_from_storage("relationships", context.output_storage)

        if entities is not None and not entities.empty:
            logger.info("Loading %d entities into Neo4j", len(entities))
            if relationships is not None and not relationships.empty:
                logger.info("Loading %d relationships into Neo4j", len(relationships))
                backend.load_graph(
                    entities=entities,
                    relationships=relationships,
                    edge_attributes=["weight"],
                )
            else:
                logger.warning("No relationships found, loading entities only")
                backend.load_graph(
                    entities=entities,
                    relationships=relationships if relationships is not None else None,
                )
        else:
            logger.warning("No entities found in storage")

        # Load text units first (needed for document relationships)
        logger.info("Loading text units")
        text_units = await load_table_from_storage("text_units", context.output_storage)
        if text_units is not None and not text_units.empty:
            logger.info("Loading %d text units into Neo4j", len(text_units))
            backend.load_text_units(text_units)
        else:
            logger.warning("No text units found in storage")

        # Load documents (creates relationships to text units)
        logger.info("Loading documents")
        documents = await load_table_from_storage("documents", context.output_storage)
        if documents is not None and not documents.empty:
            logger.info("Loading %d documents into Neo4j", len(documents))
            backend.load_documents(documents)
        else:
            logger.warning("No documents found in storage")

        # Load communities and community reports
        logger.info("Loading communities")
        communities = await load_table_from_storage("communities", context.output_storage)
        community_reports = await load_table_from_storage("community_reports", context.output_storage)

        if communities is not None and not communities.empty:
            logger.info("Loading %d communities into Neo4j", len(communities))
            if community_reports is not None and not community_reports.empty:
                logger.info("Community reports available: %d", len(community_reports))
            backend.load_communities(communities, community_reports)
        else:
            logger.warning("No communities found in storage")

        # Load claims/covariates
        logger.info("Loading claims")
        covariates = await load_table_from_storage("covariates", context.output_storage)
        if covariates is not None and not covariates.empty:
            logger.info("Loading %d claims into Neo4j", len(covariates))
            backend.load_claims(covariates)
        else:
            logger.info("No claims found in storage (claims extraction may be disabled)")

        # Close connection
        backend.close()

        logger.info("Successfully loaded complete graph into Neo4j")
        logger.info("Workflow completed: load_neo4j_graph")

        return WorkflowFunctionOutput(result={"status": "success"})

    except Exception as e:
        logger.error("Error loading graph into Neo4j: %s", e, exc_info=True)
        # Don't fail the pipeline, just log the error
        logger.warning("Continuing pipeline despite Neo4j loading error")
        return WorkflowFunctionOutput(result={"status": "error", "message": str(e)})
