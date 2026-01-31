#!/usr/bin/env python3
"""
POC Test 2: Write Entities to Neo4j

Test: Can we write GraphRAG entities and relationships to Neo4j?
Duration: 10-15 minutes
Success Criteria: Successfully create nodes and relationships in Neo4j

This test validates:
- Creating entity nodes with properties
- Creating relationship edges
- Batch operations (important for performance)
- Property types (strings, floats, lists)
"""

import os
import sys
from pathlib import Path

import neo4j
import pandas as pd
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
        "batch_size": neo4j_config.get("batch_size", 1000),
    }


def create_sample_entities():
    """Create sample GraphRAG entities for testing."""
    entities = pd.DataFrame([
        {
            "id": "entity_001",
            "title": "Microsoft Corporation",
            "type": "organization",
            "description": "A multinational technology corporation",
            "degree": 5,
            "community": None,  # Will be assigned by GDS
        },
        {
            "id": "entity_002",
            "title": "Satya Nadella",
            "type": "person",
            "description": "CEO of Microsoft",
            "degree": 3,
            "community": None,
        },
        {
            "id": "entity_003",
            "title": "Azure",
            "type": "organization",
            "description": "Cloud computing platform",
            "degree": 4,
            "community": None,
        },
        {
            "id": "entity_004",
            "title": "GraphRAG",
            "type": "organization",
            "description": "Graph-based retrieval augmented generation",
            "degree": 2,
            "community": None,
        },
    ])
    return entities


def create_sample_relationships():
    """Create sample relationships for testing."""
    relationships = pd.DataFrame([
        {
            "source": "entity_002",
            "target": "entity_001",
            "weight": 0.9,
            "description": "leads",
            "text_unit_ids": ["text_001"],
        },
        {
            "source": "entity_003",
            "target": "entity_001",
            "weight": 0.8,
            "description": "owned by",
            "text_unit_ids": ["text_002"],
        },
        {
            "source": "entity_004",
            "target": "entity_003",
            "weight": 0.7,
            "description": "runs on",
            "text_unit_ids": ["text_003"],
        },
    ])
    return relationships


def test_write_entities():
    """Test writing entities and relationships to Neo4j."""
    print("=" * 60)
    print("POC Test 2: Write Entities to Neo4j")
    print("=" * 60)

    try:
        # Load config
        print("\n1. Loading configuration...")
        config = load_neo4j_config()
        print(f"   ✓ Batch size: {config['batch_size']}")

        # Connect to Neo4j
        print("\n2. Connecting to Neo4j...")
        driver = neo4j.GraphDatabase.driver(
            config["uri"],
            auth=(config["username"], config["password"])
        )
        print("   ✓ Connected")

        # Clear existing test data
        print("\n3. Clearing existing test data...")
        with driver.session(database=config["database"]) as session:
            session.run("MATCH (n:Entity) DETACH DELETE n")
        print("   ✓ Test data cleared")

        # Create sample data
        print("\n4. Creating sample data...")
        entities = create_sample_entities()
        relationships = create_sample_relationships()
        print(f"   ✓ {len(entities)} entities")
        print(f"   ✓ {len(relationships)} relationships")

        # Write entities
        print("\n5. Writing entities to Neo4j...")
        with driver.session(database=config["database"]) as session:
            # Create entities using UNWIND for batch insert
            result = session.run(
                """
                UNWIND $entities AS entity
                CREATE (e:Entity {
                    id: entity.id,
                    title: entity.title,
                    type: entity.type,
                    description: entity.description,
                    degree: entity.degree
                })
                RETURN count(e) AS created
                """,
                entities=entities.to_dict('records')
            )
            count = result.single()["created"]
            print(f"   ✓ Created {count} entity nodes")

        # Create index on entity ID for performance
        print("\n6. Creating index on entity ID...")
        with driver.session(database=config["database"]) as session:
            try:
                session.run("CREATE INDEX entity_id IF NOT EXISTS FOR (e:Entity) ON (e.id)")
                print("   ✓ Index created")
            except Exception as e:
                print(f"   Note: Index may already exist: {e}")

        # Write relationships
        print("\n7. Writing relationships to Neo4j...")
        with driver.session(database=config["database"]) as session:
            # Create relationships using UNWIND
            result = session.run(
                """
                UNWIND $relationships AS rel
                MATCH (source:Entity {id: rel.source})
                MATCH (target:Entity {id: rel.target})
                CREATE (source)-[r:RELATED_TO {
                    weight: rel.weight,
                    description: rel.description,
                    text_unit_ids: rel.text_unit_ids
                }]->(target)
                RETURN count(r) AS created
                """,
                relationships=relationships.to_dict('records')
            )
            count = result.single()["created"]
            print(f"   ✓ Created {count} relationships")

        # Verify data
        print("\n8. Verifying data...")
        with driver.session(database=config["database"]) as session:
            # Count entities
            result = session.run("MATCH (e:Entity) RETURN count(e) AS count")
            entity_count = result.single()["count"]
            print(f"   ✓ Total entities: {entity_count}")

            # Count relationships
            result = session.run("MATCH ()-[r:RELATED_TO]->() RETURN count(r) AS count")
            rel_count = result.single()["count"]
            print(f"   ✓ Total relationships: {rel_count}")

            # Show sample data
            result = session.run(
                """
                MATCH (e:Entity)
                RETURN e.id AS id, e.title AS title, e.type AS type
                LIMIT 3
                """
            )
            print("\n   Sample entities:")
            for record in result:
                print(f"     - {record['id']}: {record['title']} ({record['type']})")

        # Close connection
        driver.close()
        print("\n" + "=" * 60)
        print("✅ POC Test 2: PASSED")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n" + "=" * 60)
        print("❌ POC Test 2: FAILED")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_write_entities()
    sys.exit(0 if success else 1)
