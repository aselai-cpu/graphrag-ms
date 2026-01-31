# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Neo4j graph backend implementation with GDS support."""

import logging
from typing import Any

import neo4j
import pandas as pd

from graphrag.index.graph.graph_backend import Communities, CommunityResult, GraphBackend

logger = logging.getLogger(__name__)


def sanitize_neo4j_label(label: str | None) -> str | None:
    """Sanitize entity type to valid Neo4j label name.

    Neo4j label requirements:
    - Must start with letter or underscore
    - Can contain letters, numbers, underscores
    - Case-sensitive
    - Cannot be empty

    Examples
    --------
        "person" -> "Person"
        "ORGANIZATION" -> "Organization"
        "geo location" -> "GeoLocation"
        "" -> None
        None -> None

    Parameters
    ----------
    label : str | None
        Entity type string to sanitize

    Returns
    -------
    str | None
        Sanitized label name or None if invalid
    """
    import re

    if not label or not label.strip():
        return None

    # Remove leading/trailing whitespace
    label = label.strip()

    # Replace spaces and special chars with underscores
    label = re.sub(r'[^a-zA-Z0-9_]', '_', label)

    # Remove leading numbers/underscores
    label = re.sub(r'^[0-9_]+', '', label)

    # If empty after sanitization, return None
    if not label:
        return None

    # Capitalize first letter of each word (PascalCase)
    parts = label.split('_')
    label = ''.join(part.capitalize() for part in parts if part)

    return label if label else None


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
        use_entity_type_labels: bool = True,
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
        use_entity_type_labels : bool, optional
            Whether to add entity type as additional node label (requires APOC), by default True
        **kwargs : Any
            Additional Neo4j driver configuration (e.g., max_connection_pool_size)
        """
        self.uri = uri
        self.database = database
        self.batch_size = batch_size
        self.node_label = node_label
        self.relationship_type = relationship_type
        self.gds_library = gds_library
        self.use_entity_type_labels = use_entity_type_labels
        self.driver = None  # Initialize to None for safe cleanup

        try:
            # Create Neo4j driver
            logger.info("Connecting to Neo4j at %s (database: %s)", uri, database)
            # Only pass valid driver config to neo4j.GraphDatabase.driver()
            # Filter out our custom parameters
            driver_kwargs = {
                k: v for k, v in kwargs.items()
                if k not in ["gds_library", "use_entity_type_labels", "node_label", "relationship_type"]  # Our custom params
            }
            self.driver = neo4j.GraphDatabase.driver(
                uri,
                auth=(username, password),
                **driver_kwargs,
            )

            # Verify connectivity
            self.driver.verify_connectivity()
            logger.info("Successfully connected to Neo4j at %s", uri)

            # Verify APOC is available if using entity type labels
            if self.use_entity_type_labels:
                try:
                    with self.driver.session(database=self.database) as session:
                        result = session.run("RETURN apoc.version() AS version")
                        apoc_version = result.single()["version"]
                        logger.info("APOC plugin available, version: %s", apoc_version)
                except Exception as e:
                    logger.error(
                        "APOC library not found but use_entity_type_labels=True. Error: %s", e
                    )
                    raise RuntimeError(
                        "APOC library required for use_entity_type_labels=True. "
                        "Please install APOC (https://neo4j.com/labs/apoc/) "
                        "or set use_entity_type_labels=False in configuration."
                    ) from e

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

            # Create type-specific indexes if using type labels
            if self.use_entity_type_labels and "type" in entities.columns:
                unique_types = entities["type"].dropna().unique()
                for entity_type in unique_types:
                    sanitized_type = sanitize_neo4j_label(entity_type)
                    if sanitized_type:
                        logger.debug("Creating index for type: %s", sanitized_type)
                        session.run(
                            f"CREATE INDEX {sanitized_type.lower()}_id IF NOT EXISTS "
                            f"FOR (e:{sanitized_type}) ON (e.{entity_id_col})"
                        )

            # Write entities in batches
            entity_count = 0
            for i in range(0, len(entities), self.batch_size):
                batch = entities.iloc[i : i + self.batch_size]

                # Prepare batch data with sanitized type labels
                batch_data = []
                for _, row in batch.iterrows():
                    entity_dict = row.fillna("").to_dict()

                    # Add sanitized label if using type labels
                    if self.use_entity_type_labels and "type" in entity_dict:
                        sanitized_type = sanitize_neo4j_label(entity_dict.get("type"))
                        entity_dict["_label"] = sanitized_type  # Store for Cypher
                    else:
                        entity_dict["_label"] = None

                    batch_data.append(entity_dict)

                # Use APOC to create nodes with dynamic labels
                if self.use_entity_type_labels:
                    result = session.run(
                        f"""
                        UNWIND $entities AS entity
                        CALL apoc.create.node(
                            CASE
                                WHEN entity._label IS NOT NULL
                                THEN ['{self.node_label}'] + [entity._label]
                                ELSE ['{self.node_label}']
                            END,
                            apoc.map.removeKey(entity, '_label')
                        )
                        YIELD node
                        RETURN count(node) AS created
                        """,
                        entities=batch_data,
                    )
                else:
                    # Fallback to simple CREATE if not using type labels
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

    def load_text_units(
        self,
        text_units: pd.DataFrame,
        *,
        entity_id_col: str = "title",
    ) -> None:
        """Load text units into Neo4j and create relationships to entities.

        Creates TextUnit nodes and MENTIONS relationships to entities.

        Parameters
        ----------
        text_units : pd.DataFrame
            DataFrame with columns: id, text, n_tokens, entity_ids, relationship_ids
        entity_id_col : str, optional
            Column name for entity identifier, by default "title"
        """
        if text_units.empty:
            logger.info("No text units to load")
            return

        with self.driver.session(database=self.database) as session:
            # Clear existing TextUnit nodes
            logger.info("Clearing existing TextUnit nodes")
            session.run("MATCH (t:TextUnit) DETACH DELETE t")

            # Create TextUnit nodes in batches
            text_unit_count = 0
            for i in range(0, len(text_units), self.batch_size):
                batch = text_units.iloc[i : i + self.batch_size]

                # Prepare batch data
                batch_data = []
                for _, row in batch.iterrows():
                    unit_dict = {
                        "id": row["id"],
                        "text": row.get("text", ""),
                        "n_tokens": int(row.get("n_tokens", 0)) if pd.notna(row.get("n_tokens")) else 0,
                    }
                    batch_data.append(unit_dict)

                # Create TextUnit nodes
                result = session.run(
                    """
                    UNWIND $units AS unit
                    CREATE (t:TextUnit)
                    SET t = unit
                    RETURN count(t) AS created
                    """,
                    units=batch_data,
                )
                count = result.single()["created"]
                text_unit_count += count
                logger.debug("Batch %d: Created %d text units", i // self.batch_size + 1, count)

            logger.info("Created %d TextUnit nodes", text_unit_count)

            # Create MENTIONS relationships to entities
            mentions_count = 0
            for _, row in text_units.iterrows():
                text_unit_id = row["id"]
                entity_ids = row.get("entity_ids", [])

                # Check if entity_ids is empty or null
                if entity_ids is None or (isinstance(entity_ids, float) and pd.isna(entity_ids)):
                    continue

                # Convert to list if needed
                if isinstance(entity_ids, str):
                    import ast
                    entity_ids = ast.literal_eval(entity_ids)

                # Check if list is empty
                if isinstance(entity_ids, list) and len(entity_ids) == 0:
                    continue

                # Create relationships in batch
                result = session.run(
                    f"""
                    MATCH (t:TextUnit {{id: $text_unit_id}})
                    UNWIND $entity_ids AS entity_id
                    MATCH (e:{self.node_label} {{id: entity_id}})
                    CREATE (t)-[:MENTIONS]->(e)
                    RETURN count(*) AS created
                    """,
                    text_unit_id=text_unit_id,
                    entity_ids=entity_ids,
                )
                if result.peek():
                    mentions_count += result.single()["created"]

            logger.info("Created %d MENTIONS relationships", mentions_count)

    def load_communities(
        self,
        communities: pd.DataFrame,
        community_reports: pd.DataFrame | None = None,
    ) -> None:
        """Load communities into Neo4j and create hierarchical relationships.

        Creates Community nodes with summaries and creates:
        - CONTAINS relationships to entities
        - PARENT_OF relationships for hierarchy

        Parameters
        ----------
        communities : pd.DataFrame
            DataFrame with columns: id, title, level, parent, entity_ids, text_unit_ids
        community_reports : pd.DataFrame | None, optional
            DataFrame with community summaries and reports
        """
        if communities.empty:
            logger.info("No communities to load")
            return

        with self.driver.session(database=self.database) as session:
            # Clear existing Community nodes
            logger.info("Clearing existing Community nodes")
            session.run("MATCH (c:Community) DETACH DELETE c")

            # Merge community reports if provided
            communities_with_reports = communities.copy()
            if community_reports is not None and not community_reports.empty:
                # Merge on community ID
                communities_with_reports = communities.merge(
                    community_reports[["community", "summary", "findings", "full_content", "rank"]],
                    on="community",
                    how="left",
                    suffixes=("", "_report"),
                )

            # Create Community nodes in batches
            community_count = 0
            for i in range(0, len(communities_with_reports), self.batch_size):
                batch = communities_with_reports.iloc[i : i + self.batch_size]

                # Prepare batch data
                batch_data = []
                for _, row in batch.iterrows():
                    comm_dict = {
                        "id": row["id"],
                        "community": int(row["community"]) if pd.notna(row["community"]) else -1,
                        "title": row.get("title", ""),
                        "level": int(row["level"]) if pd.notna(row["level"]) else 0,
                        "size": int(row.get("size", 0)) if pd.notna(row.get("size")) else 0,
                    }

                    # Add optional fields
                    if "summary" in row and pd.notna(row["summary"]):
                        comm_dict["summary"] = row["summary"]
                    if "rank" in row and pd.notna(row["rank"]):
                        comm_dict["rank"] = float(row["rank"])

                    batch_data.append(comm_dict)

                # Create Community nodes
                result = session.run(
                    """
                    UNWIND $communities AS comm
                    CREATE (c:Community)
                    SET c = comm
                    RETURN count(c) AS created
                    """,
                    communities=batch_data,
                )
                count = result.single()["created"]
                community_count += count
                logger.debug("Batch %d: Created %d communities", i // self.batch_size + 1, count)

            logger.info("Created %d Community nodes", community_count)

            # Create CONTAINS relationships to entities
            contains_count = 0
            for _, row in communities_with_reports.iterrows():
                community_id = row["id"]
                entity_ids = row.get("entity_ids", [])

                # Check if entity_ids is empty or null
                if entity_ids is None or (isinstance(entity_ids, float) and pd.isna(entity_ids)):
                    continue

                # Convert to list if needed
                if isinstance(entity_ids, str):
                    import ast
                    entity_ids = ast.literal_eval(entity_ids)

                # Check if list is empty
                if isinstance(entity_ids, list) and len(entity_ids) == 0:
                    continue

                # Create relationships in batch
                result = session.run(
                    f"""
                    MATCH (c:Community {{id: $community_id}})
                    UNWIND $entity_ids AS entity_id
                    MATCH (e:{self.node_label} {{id: entity_id}})
                    CREATE (c)-[:CONTAINS]->(e)
                    RETURN count(*) AS created
                    """,
                    community_id=community_id,
                    entity_ids=entity_ids,
                )
                if result.peek():
                    contains_count += result.single()["created"]

            logger.info("Created %d CONTAINS relationships", contains_count)

            # Create PARENT_OF relationships for hierarchy
            parent_count = 0
            for _, row in communities_with_reports.iterrows():
                if "parent" not in row or pd.isna(row["parent"]) or row["parent"] == "":
                    continue

                child_community = int(row["community"])
                parent_community = int(row["parent"])

                # Find parent community node by community number
                result = session.run(
                    """
                    MATCH (parent:Community {community: $parent_community})
                    MATCH (child:Community {community: $child_community})
                    CREATE (parent)-[:PARENT_OF]->(child)
                    RETURN count(*) AS created
                    """,
                    parent_community=parent_community,
                    child_community=child_community,
                )
                if result.peek():
                    parent_count += result.single()["created"]

            logger.info("Created %d PARENT_OF relationships", parent_count)

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
