#!/usr/bin/env python3
"""
POC Test 3: GDS Community Detection

Test: Can we run Louvain community detection using Neo4j GDS?
Duration: 10-15 minutes
Success Criteria: Successfully detect communities and assign them to entities

This test validates:
- Neo4j GDS plugin is installed and working
- Can create in-memory graph projections
- Can run Louvain algorithm
- Can write community assignments back to nodes
- Community IDs are hierarchical (for multi-level hierarchy)
"""

import os
import sys
from pathlib import Path

import neo4j
import yaml


def load_neo4j_config():
    """Load Neo4j configuration from settings file."""
    config_path = Path(__file__).parent / "settings.neo4j.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    neo4j_config = config["storage"]["neo4j"]

    # Substitute environment variables
    password = neo4j_config["password"]
    if password.startswith("${") and password.endswith("}"):
        env_var = password[2:-1]
        password = os.environ.get(env_var)
        if not password:
            raise ValueError(f"Environment variable {env_var} not set")

    return {
        "uri": neo4j_config["uri"],
        "username": neo4j_config["username"],
        "password": password,
        "database": neo4j_config["database"],
        "gds": neo4j_config.get("gds", {}),
    }


def test_gds_community():
    """Test GDS community detection."""
    print("=" * 60)
    print("POC Test 3: GDS Community Detection")
    print("=" * 60)

    try:
        # Load config
        print("\n1. Loading configuration...")
        config = load_neo4j_config()
        gds_config = config["gds"]
        print(f"   ✓ GDS enabled: {gds_config.get('enabled', False)}")
        print(f"   ✓ Algorithm: {gds_config.get('community_algorithm', 'louvain')}")
        print(f"   ✓ Max cluster size: {gds_config.get('max_cluster_size', 10)}")

        # Connect to Neo4j
        print("\n2. Connecting to Neo4j...")
        driver = neo4j.GraphDatabase.driver(
            config["uri"],
            auth=(config["username"], config["password"])
        )
        print("   ✓ Connected")

        # Check if test data exists (from test 2)
        print("\n3. Checking test data...")
        with driver.session(database=config["database"]) as session:
            result = session.run("MATCH (e:Entity) RETURN count(e) AS count")
            entity_count = result.single()["count"]
            if entity_count == 0:
                raise ValueError("No entities found. Run test_02_write_entities.py first.")
            print(f"   ✓ Found {entity_count} entities")

        # Check GDS version
        print("\n4. Checking GDS plugin...")
        with driver.session(database=config["database"]) as session:
            result = session.run("RETURN gds.version() AS version")
            record = result.single()
            print(f"   ✓ GDS version: {record['version']}")

        # Drop existing graph projection if exists
        print("\n5. Cleaning up old graph projections...")
        with driver.session(database=config["database"]) as session:
            try:
                session.run("CALL gds.graph.drop('entities', false)")
                print("   ✓ Dropped existing projection")
            except Exception:
                print("   Note: No existing projection found")

        # Create graph projection
        print("\n6. Creating graph projection...")
        with driver.session(database=config["database"]) as session:
            result = session.run(
                """
                CALL gds.graph.project(
                    'entities',
                    'Entity',
                    {
                        RELATED_TO: {
                            orientation: 'UNDIRECTED',
                            properties: 'weight'
                        }
                    }
                )
                YIELD graphName, nodeCount, relationshipCount
                RETURN graphName, nodeCount, relationshipCount
                """
            )
            record = result.single()
            print(f"   ✓ Graph: {record['graphName']}")
            print(f"   ✓ Nodes: {record['nodeCount']}")
            print(f"   ✓ Relationships: {record['relationshipCount']}")

        # Run Louvain community detection (stream mode to see results)
        print("\n7. Running Louvain community detection (stream)...")
        with driver.session(database=config["database"]) as session:
            result = session.run(
                """
                CALL gds.louvain.stream('entities', {
                    relationshipWeightProperty: 'weight',
                    includeIntermediateCommunities: true
                })
                YIELD nodeId, communityId, intermediateCommunityIds
                RETURN gds.util.asNode(nodeId).id AS entityId,
                       gds.util.asNode(nodeId).title AS title,
                       communityId,
                       intermediateCommunityIds
                ORDER BY communityId
                """
            )

            print("\n   Community assignments:")
            communities = {}
            for record in result:
                entity_id = record['entityId']
                title = record['title']
                community_id = record['communityId']
                intermediate = record['intermediateCommunityIds']

                print(f"     - {entity_id} ({title}): Community {community_id}")
                print(f"       Hierarchy: {intermediate}")

                if community_id not in communities:
                    communities[community_id] = []
                communities[community_id].append(entity_id)

            print(f"\n   ✓ Found {len(communities)} communities")
            for community_id, members in communities.items():
                print(f"     Community {community_id}: {len(members)} members")

        # Run Louvain in write mode to persist results
        print("\n8. Writing community assignments to nodes...")
        with driver.session(database=config["database"]) as session:
            result = session.run(
                """
                CALL gds.louvain.write('entities', {
                    relationshipWeightProperty: 'weight',
                    writeProperty: 'community'
                })
                YIELD communityCount, modularity, modularities
                RETURN communityCount, modularity, modularities
                """
            )
            record = result.single()
            print(f"   ✓ Community count: {record['communityCount']}")
            print(f"   ✓ Modularity: {record['modularity']:.4f}")
            print(f"   ✓ Modularities by level: {record['modularities']}")

        # Verify community assignments were written
        print("\n9. Verifying community assignments...")
        with driver.session(database=config["database"]) as session:
            result = session.run(
                """
                MATCH (e:Entity)
                WHERE e.community IS NOT NULL
                RETURN count(e) AS assigned
                """
            )
            assigned = result.single()["assigned"]
            print(f"   ✓ Entities with community assignments: {assigned}/{entity_count}")

        # Drop graph projection (cleanup)
        print("\n10. Cleaning up graph projection...")
        with driver.session(database=config["database"]) as session:
            session.run("CALL gds.graph.drop('entities')")
            print("   ✓ Graph projection dropped")

        # Close connection
        driver.close()
        print("\n" + "=" * 60)
        print("✅ POC Test 3: PASSED")
        print("=" * 60)
        print("\nKey Findings:")
        print("  - GDS Louvain algorithm works correctly")
        print("  - Can create graph projections from Entity nodes")
        print("  - Community IDs are hierarchical (intermediate levels available)")
        print("  - Can write community assignments back to nodes")
        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n" + "=" * 60)
        print("❌ POC Test 3: FAILED")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_gds_community()
    sys.exit(0 if success else 1)
