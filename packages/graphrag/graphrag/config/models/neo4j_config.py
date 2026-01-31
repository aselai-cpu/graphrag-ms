# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Configuration for Neo4j graph backend."""

from pydantic import BaseModel, Field


class Neo4jConfig(BaseModel):
    """Configuration for Neo4j graph backend.

    Required when using Neo4j as primary or shadow backend.
    """

    uri: str = Field(
        description="Neo4j connection URI (e.g., bolt://localhost:7687)",
        default="bolt://localhost:7687",
    )
    username: str = Field(
        description="Neo4j username",
        default="neo4j",
    )
    password: str = Field(
        description="Neo4j password (use environment variable: ${NEO4J_PASSWORD})",
    )
    database: str = Field(
        description="Neo4j database name",
        default="neo4j",
    )
    gds_library: bool = Field(
        description="Whether to use Neo4j Graph Data Science library for community detection",
        default=True,
    )
    node_label: str = Field(
        description="Base label for entity nodes in Neo4j",
        default="Entity",
    )
    relationship_type: str = Field(
        description="Type for entity relationships in Neo4j",
        default="RELATED_TO",
    )
    use_entity_type_labels: bool = Field(
        description="Whether to add entity type as additional node label (e.g., :Entity:Person). Requires APOC plugin.",
        default=True,
    )
