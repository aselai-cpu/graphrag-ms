# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Neo4j graph backend implementation with GDS support."""

import logging
from typing import Any

import neo4j
import pandas as pd

from graphrag.index.graph.graph_backend import Communities, CommunityResult, GraphBackend

logger = logging.getLogger(__name__)


class Neo4jBackend(GraphBackend):
    """Neo4j-based graph backend implementation using Graph Data Science (GDS).

    This backend stores entities and relationships as native Neo4j nodes and edges,
    and uses Neo4j GDS for community detection (Louvain algorithm).

    Attributes
    ----------
    driver : neo4j.Driver
        Neo4j driver instance
    database : str
        Neo4j database name
    batch_size : int
        Batch size for bulk operations
    node_label : str
        Label for entity nodes
    relationship_type : str
        Type for entity relationships
    """

    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        database: str = "neo4j",
        batch_size: int = 1000,
        node_label: str = "Entity",
        relationship_type: str = "RELATED_TO",
        gds_library: bool = True,
        **kwargs: Any,
    ):
        """Initialize Neo4j backend.

        Parameters
        ----------
        uri : str
            Neo4j connection URI (e.g., "bolt://localhost:7687")
        username : str
            Neo4j username
        password : str
            Neo4j password
        database : str, optional
            Database name, by default "neo4j"
        batch_size : int, optional
            Batch size for bulk operations, by default 1000
        node_label : str, optional
            Label for entity nodes, by default "Entity"
        relationship_type : str, optional
            Type for relationships, by default "RELATED_TO"
        gds_library : bool, optional
            Whether to use Neo4j GDS library for community detection, by default True
        **kwargs : Any
            Additional Neo4j driver configuration (e.g., max_connection_pool_size)
        """
        self.uri = uri
        self.database = database
        self.batch_size = batch_size
        self.node_label = node_label
        self.relationship_type = relationship_type
        self.gds_library = gds_library
        self.driver = None  # Initialize to None for safe cleanup

        try:
            # Create Neo4j driver
            logger.info("Connecting to Neo4j at %s (database: %s)", uri, database)
            # Only pass valid driver config to neo4j.GraphDatabase.driver()
            # Filter out our custom parameters
            driver_kwargs = {
                k: v for k, v in kwargs.items()
                if k not in ["gds_library"]  # Our custom params that shouldn't go to driver
            }
            self.driver = neo4j.GraphDatabase.driver(
                uri,
                auth=(username, password),
                **driver_kwargs,
            )

            # Verify connectivity
            self.driver.verify_connectivity()
            logger.info("Successfully connected to Neo4j at %s", uri)

        except neo4j.exceptions.ServiceUnavailable as e:
            logger.error(
                "Failed to connect to Neo4j at %s. Is Neo4j running? Error: %s",
                uri, e
            )
            raise ConnectionError(
                f"Cannot connect to Neo4j at {uri}. "
                f"Please ensure Neo4j is running and accessible. "
                f"Original error: {e}"
            ) from e
        except neo4j.exceptions.AuthError as e:
            logger.error(
                "Authentication failed for Neo4j at %s. Check username/password. Error: %s",
                uri, e
            )
            raise ValueError(
                f"Neo4j authentication failed at {uri}. "
                f"Please check NEO4J_PASSWORD environment variable. "
                f"Original error: {e}"
            ) from e
        except Exception as e:
            logger.error("Unexpected error connecting to Neo4j: %s", e)
            raise

        # Track entity ID column name
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
        """Load entities and relationships into Neo4j.

        This creates nodes with the configured label and relationships with the
        configured type. Uses batch operations (UNWIND) for performance.
        """
        self._entity_id_col = entity_id_col

        with self.driver.session(database=self.database) as session:
            # Clear existing data for this label
            logger.info("Clearing existing %s nodes", self.node_label)
            session.run(f"MATCH (n:{self.node_label}) DETACH DELETE n")

            # Create index on entity ID for performance
            logger.info("Creating index on %s.%s", self.node_label, entity_id_col)
            session.run(
                f"CREATE INDEX {self.node_label.lower()}_id IF NOT EXISTS "
                f"FOR (e:{self.node_label}) ON (e.{entity_id_col})"
            )

            # Write entities in batches
            entity_count = 0
            for i in range(0, len(entities), self.batch_size):
                batch = entities.iloc[i : i + self.batch_size]

                # Prepare batch data (convert NaN to None)
                batch_data = batch.fillna("").to_dict("records")

                result = session.run(
                    f"""
                    UNWIND $entities AS entity
                    CREATE (e:{self.node_label})
                    SET e = entity
                    RETURN count(e) AS created
                    """,
                    entities=batch_data,
                )
                count = result.single()["created"]
                entity_count += count
                logger.debug("Batch %d: Created %d entities", i // self.batch_size + 1, count)

            logger.info("Created %d entity nodes", entity_count)

            # Write relationships in batches
            if not relationships.empty:
                rel_count = 0
                edge_attr_str = ""
                if edge_attributes:
                    # Build property assignment for edge attributes
                    edge_attr_assignments = [
                        f"r.{attr} = rel.{attr}" for attr in edge_attributes
                    ]
                    edge_attr_str = ", " + ", ".join(edge_attr_assignments)

                for i in range(0, len(relationships), self.batch_size):
                    batch = relationships.iloc[i : i + self.batch_size]

                    # Prepare batch data
                    batch_data = batch.fillna("").to_dict("records")

                    result = session.run(
                        f"""
                        UNWIND $relationships AS rel
                        MATCH (source:{self.node_label} {{{entity_id_col}: rel.{source_col}}})
                        MATCH (target:{self.node_label} {{{entity_id_col}: rel.{target_col}}})
                        CREATE (source)-[r:{self.relationship_type}]->(target)
                        SET r.description = rel.description{edge_attr_str}
                        RETURN count(r) AS created
                        """,
                        relationships=batch_data,
                    )
                    count = result.single()["created"]
                    rel_count += count
                    logger.debug(
                        "Batch %d: Created %d relationships", i // self.batch_size + 1, count
                    )

                logger.info("Created %d relationships", rel_count)

    def detect_communities(
        self,
        *,
        max_cluster_size: int = 10,
        use_lcc: bool = True,
        seed: int | None = None,
    ) -> Communities:
        """Run Louvain community detection using Neo4j GDS.

        This creates a graph projection, runs Louvain algorithm,
        writes community assignments back to nodes, and returns
        the community hierarchy.
        """
        with self.driver.session(database=self.database) as session:
            # Drop existing graph projection if exists
            graph_name = f"{self.node_label.lower()}_graph"
            try:
                session.run(f"CALL gds.graph.drop('{graph_name}', false)")
                logger.debug("Dropped existing graph projection")
            except Exception:
                pass  # Graph doesn't exist yet

            # Create graph projection
            logger.info("Creating GDS graph projection")
            result = session.run(
                f"""
                CALL gds.graph.project(
                    $graphName,
                    $nodeLabel,
                    {{
                        {self.relationship_type}: {{
                            orientation: 'UNDIRECTED',
                            properties: 'weight'
                        }}
                    }}
                )
                YIELD nodeCount, relationshipCount
                RETURN nodeCount, relationshipCount
                """,
                graphName=graph_name,
                nodeLabel=self.node_label,
            )
            record = result.single()
            logger.info(
                "Graph projection: %d nodes, %d relationships",
                record["nodeCount"],
                record["relationshipCount"],
            )

            # Run Louvain with includeIntermediateCommunities for hierarchy
            logger.info("Running Louvain community detection")
            louvain_config = {
                "relationshipWeightProperty": "weight",
                "includeIntermediateCommunities": True,
            }
            # Note: GDS Louvain doesn't support random seed like NetworkX Leiden
            # It uses seedProperty which requires a pre-existing node property
            # For deterministic results, would need to set a seed property on nodes first
            if seed is not None:
                logger.warning(
                    "Neo4j GDS Louvain does not support random seed parameter. "
                    "Results may vary between runs."
                )
            if max_cluster_size:
                louvain_config["maxLevels"] = 10  # Allow up to 10 hierarchy levels

            # Run in stream mode to get hierarchy
            result = session.run(
                f"""
                CALL gds.louvain.stream($graphName, $config)
                YIELD nodeId, communityId, intermediateCommunityIds
                RETURN gds.util.asNode(nodeId).{self._entity_id_col} AS entityId,
                       communityId,
                       intermediateCommunityIds
                ORDER BY communityId
                """,
                graphName=graph_name,
                config=louvain_config,
            )

            # Build community structure with hierarchy
            community_map: dict[int, dict[int, list[str]]] = {}  # level -> {community_id -> [nodes]}
            parent_map: dict[int, int] = {}  # community_id -> parent_id

            for record in result:
                entity_id = record["entityId"]
                final_community = record["communityId"]
                intermediate = record["intermediateCommunityIds"] or []

                # Add to each level
                for level, community_id in enumerate(intermediate):
                    if level not in community_map:
                        community_map[level] = {}
                    if community_id not in community_map[level]:
                        community_map[level][community_id] = []
                    community_map[level][community_id].append(entity_id)

                    # Track parent relationships
                    if level > 0:
                        parent_map[community_id] = intermediate[level - 1]
                    else:
                        parent_map[community_id] = -1

                # Add final community
                final_level = len(intermediate)
                if final_level not in community_map:
                    community_map[final_level] = {}
                if final_community not in community_map[final_level]:
                    community_map[final_level][final_community] = []
                community_map[final_level][final_community].append(entity_id)

                if final_level > 0 and intermediate:
                    parent_map[final_community] = intermediate[-1]
                else:
                    parent_map[final_community] = -1

            # Write community assignments back to nodes
            logger.info("Writing community assignments to nodes")
            result = session.run(
                f"""
                CALL gds.louvain.write($graphName, {{
                    relationshipWeightProperty: 'weight',
                    writeProperty: 'community'
                }})
                YIELD communityCount, modularity
                RETURN communityCount, modularity
                """,
                graphName=graph_name,
            )
            record = result.single()
            logger.info(
                "Communities: %d, Modularity: %.4f",
                record["communityCount"],
                record["modularity"],
            )

            # Drop graph projection (cleanup)
            session.run(f"CALL gds.graph.drop('{graph_name}')")

            # Convert to CommunityResult format
            communities: Communities = []
            for level in sorted(community_map.keys()):
                for cluster_id, node_ids in community_map[level].items():
                    communities.append(
                        CommunityResult(
                            level=level,
                            cluster_id=cluster_id,
                            parent_cluster_id=parent_map.get(cluster_id, -1),
                            node_ids=node_ids,
                        )
                    )

            logger.info("Detected %d communities across %d levels", len(communities), len(community_map))
            return communities

    def compute_node_degrees(self) -> pd.DataFrame:
        """Compute node degrees from Neo4j graph."""
        with self.driver.session(database=self.database) as session:
            # Use COUNT{} for Neo4j 5.x (size() is deprecated)
            result = session.run(
                f"""
                MATCH (e:{self.node_label})
                RETURN e.{self._entity_id_col} AS {self._entity_id_col},
                       COUNT {{(e)--()}} AS degree
                ORDER BY {self._entity_id_col}
                """
            )

            degrees = pd.DataFrame([dict(record) for record in result])
            logger.info("Computed degrees for %d nodes", len(degrees))
            return degrees

    def export_graph(
        self,
        *,
        entity_id_col: str = "title",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Export Neo4j graph to DataFrames."""
        with self.driver.session(database=self.database) as session:
            # Export entities
            result = session.run(
                f"""
                MATCH (e:{self.node_label})
                RETURN properties(e) AS props
                ORDER BY e.{entity_id_col}
                """
            )
            entities = pd.DataFrame([record["props"] for record in result])

            # Export relationships
            result = session.run(
                f"""
                MATCH (source:{self.node_label})-[r:{self.relationship_type}]->(target:{self.node_label})
                RETURN source.{entity_id_col} AS source,
                       target.{entity_id_col} AS target,
                       properties(r) AS props
                """
            )
            relationships_data = []
            for record in result:
                row = {"source": record["source"], "target": record["target"]}
                row.update(record["props"])
                relationships_data.append(row)

            relationships = pd.DataFrame(relationships_data)

            logger.info("Exported %d entities, %d relationships", len(entities), len(relationships))
            return entities, relationships

    def clear(self) -> None:
        """Clear all entities and relationships from Neo4j."""
        with self.driver.session(database=self.database) as session:
            session.run(f"MATCH (n:{self.node_label}) DETACH DELETE n")
            logger.info("Cleared all %s nodes from Neo4j", self.node_label)

    def node_count(self) -> int:
        """Return number of nodes in Neo4j."""
        with self.driver.session(database=self.database) as session:
            result = session.run(f"MATCH (e:{self.node_label}) RETURN count(e) AS count")
            return result.single()["count"]

    def edge_count(self) -> int:
        """Return number of edges in Neo4j."""
        with self.driver.session(database=self.database) as session:
            result = session.run(
                f"MATCH ()-[r:{self.relationship_type}]->() RETURN count(r) AS count"
            )
            return result.single()["count"]

    def close(self) -> None:
        """Close Neo4j driver connection."""
        if hasattr(self, "driver") and self.driver:
            self.driver.close()
            logger.info("Closed Neo4j connection")

    def __del__(self):
        """Clean up on deletion."""
        try:
            self.close()
        except Exception:
            # Silently ignore errors during cleanup
            pass
