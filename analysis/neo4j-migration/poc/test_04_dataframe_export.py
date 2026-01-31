#!/usr/bin/env python3
"""
POC Test 4: DataFrame Export

Test: Can we read data from Neo4j back as pandas DataFrames?
Duration: 5-10 minutes
Success Criteria: Successfully convert Neo4j results to DataFrames matching GraphRAG format

This test validates:
- Can export entities as DataFrame with correct schema
- Can export relationships as DataFrame with correct schema
- Data types match GraphRAG expectations
- Community assignments are included
- Compatible with GraphRAG query operations
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
    }


def test_dataframe_export():
    """Test exporting Neo4j data as pandas DataFrames."""
    print("=" * 60)
    print("POC Test 4: DataFrame Export")
    print("=" * 60)

    try:
        # Load config
        print("\n1. Loading configuration...")
        config = load_neo4j_config()

        # Connect to Neo4j
        print("\n2. Connecting to Neo4j...")
        driver = neo4j.GraphDatabase.driver(
            config["uri"],
            auth=(config["username"], config["password"])
        )
        print("   ✓ Connected")

        # Export entities as DataFrame
        print("\n3. Exporting entities as DataFrame...")
        with driver.session(database=config["database"]) as session:
            result = session.run(
                """
                MATCH (e:Entity)
                RETURN e.id AS id,
                       e.title AS title,
                       e.type AS type,
                       e.description AS description,
                       e.degree AS degree,
                       e.community AS community
                ORDER BY e.id
                """
            )

            # Convert to DataFrame
            entities_df = pd.DataFrame([dict(record) for record in result])

            print(f"   ✓ Exported {len(entities_df)} entities")
            print(f"\n   DataFrame shape: {entities_df.shape}")
            print(f"   Columns: {list(entities_df.columns)}")
            print(f"   Data types:\n{entities_df.dtypes}")

            print("\n   Sample data:")
            print(entities_df.head())

        # Verify DataFrame schema matches GraphRAG expectations
        print("\n4. Verifying DataFrame schema...")
        required_columns = ["id", "title", "type", "description"]
        missing_columns = set(required_columns) - set(entities_df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        print(f"   ✓ All required columns present: {required_columns}")

        # Check data types
        print("\n   Checking data types...")
        assert entities_df["id"].dtype == object, "id should be string"
        print("   ✓ id: string")
        assert entities_df["title"].dtype == object, "title should be string"
        print("   ✓ title: string")
        assert entities_df["type"].dtype == object, "type should be string"
        print("   ✓ type: string")
        assert entities_df["description"].dtype == object, "description should be string"
        print("   ✓ description: string")

        # Export relationships as DataFrame
        print("\n5. Exporting relationships as DataFrame...")
        with driver.session(database=config["database"]) as session:
            result = session.run(
                """
                MATCH (source:Entity)-[r:RELATED_TO]->(target:Entity)
                RETURN source.id AS source,
                       target.id AS target,
                       r.weight AS weight,
                       r.description AS description,
                       r.text_unit_ids AS text_unit_ids
                ORDER BY source.id, target.id
                """
            )

            # Convert to DataFrame
            relationships_df = pd.DataFrame([dict(record) for record in result])

            print(f"   ✓ Exported {len(relationships_df)} relationships")
            print(f"\n   DataFrame shape: {relationships_df.shape}")
            print(f"   Columns: {list(relationships_df.columns)}")
            print(f"   Data types:\n{relationships_df.dtypes}")

            print("\n   Sample data:")
            print(relationships_df.head())

        # Verify relationships schema
        print("\n6. Verifying relationships schema...")
        required_rel_columns = ["source", "target", "weight"]
        missing_rel_columns = set(required_rel_columns) - set(relationships_df.columns)
        if missing_rel_columns:
            raise ValueError(f"Missing required columns: {missing_rel_columns}")
        print(f"   ✓ All required columns present: {required_rel_columns}")

        # Test filtering operations (common in GraphRAG queries)
        print("\n7. Testing DataFrame operations...")

        # Filter by entity type
        org_entities = entities_df[entities_df["type"] == "organization"]
        print(f"   ✓ Filter by type: {len(org_entities)} organizations")

        # Filter by community
        if "community" in entities_df.columns and entities_df["community"].notna().any():
            community_0 = entities_df[entities_df["community"] == entities_df["community"].min()]
            print(f"   ✓ Filter by community: {len(community_0)} entities in first community")

        # Join entities with relationships
        relationships_with_titles = relationships_df.merge(
            entities_df[["id", "title"]],
            left_on="source",
            right_on="id"
        ).rename(columns={"title": "source_title"}).drop(columns=["id"])

        print(f"   ✓ Join entities with relationships: {len(relationships_with_titles)} rows")

        # Test aggregations
        entities_per_type = entities_df.groupby("type").size()
        print(f"\n   Entities per type:")
        for entity_type, count in entities_per_type.items():
            print(f"     {entity_type}: {count}")

        # Test GraphRAG-style queries
        print("\n8. Testing GraphRAG-style queries...")

        # Get all entities connected to a specific entity
        target_id = entities_df.iloc[0]["id"]
        connected = relationships_df[
            (relationships_df["source"] == target_id) |
            (relationships_df["target"] == target_id)
        ]
        print(f"   ✓ Connected entities to {target_id}: {len(connected)}")

        # Get entities by degree (hub detection)
        high_degree = entities_df[entities_df["degree"] >= entities_df["degree"].median()]
        print(f"   ✓ High-degree entities (hubs): {len(high_degree)}")

        # Close connection
        driver.close()
        print("\n" + "=" * 60)
        print("✅ POC Test 4: PASSED")
        print("=" * 60)
        print("\nKey Findings:")
        print("  - Can export entities and relationships as DataFrames")
        print("  - Schema matches GraphRAG expectations")
        print("  - Data types are correct")
        print("  - DataFrame operations work as expected")
        print("  - Compatible with GraphRAG query patterns")
        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n" + "=" * 60)
        print("❌ POC Test 4: FAILED")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_dataframe_export()
    sys.exit(0 if success else 1)
