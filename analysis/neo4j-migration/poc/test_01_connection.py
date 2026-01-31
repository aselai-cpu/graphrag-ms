#!/usr/bin/env python3
"""
POC Test 1: Neo4j Connection from Settings

Test: Can we connect to Neo4j using configuration values?
Duration: 5-10 minutes
Success Criteria: Successfully connect and query Neo4j

This test validates:
- Neo4j driver connection works
- Environment variable substitution (NEO4J_PASSWORD)
- Basic query execution
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
    }


def test_connection():
    """Test Neo4j connection."""
    print("=" * 60)
    print("POC Test 1: Neo4j Connection from Settings")
    print("=" * 60)

    try:
        # Load config
        print("\n1. Loading configuration from settings.neo4j.yaml...")
        config = load_neo4j_config()
        print(f"   ✓ URI: {config['uri']}")
        print(f"   ✓ Username: {config['username']}")
        print(f"   ✓ Database: {config['database']}")
        print(f"   ✓ Password: {'*' * 8} (from environment)")

        # Connect to Neo4j
        print("\n2. Connecting to Neo4j...")
        driver = neo4j.GraphDatabase.driver(
            config["uri"],
            auth=(config["username"], config["password"])
        )
        print("   ✓ Driver created successfully")

        # Verify connection
        print("\n3. Verifying connection...")
        driver.verify_connectivity()
        print("   ✓ Connection verified")

        # Run simple query
        print("\n4. Running test query...")
        with driver.session(database=config["database"]) as session:
            result = session.run("RETURN 1 AS test, 'Hello Neo4j!' AS message")
            record = result.single()
            print(f"   ✓ Query result: test={record['test']}, message={record['message']}")

        # Check Neo4j version
        print("\n5. Checking Neo4j version...")
        with driver.session(database=config["database"]) as session:
            result = session.run("CALL dbms.components() YIELD name, versions, edition")
            for record in result:
                print(f"   ✓ {record['name']}: {record['versions'][0]} ({record['edition']})")

        # Check GDS plugin
        print("\n6. Checking GDS plugin...")
        with driver.session(database=config["database"]) as session:
            try:
                result = session.run("RETURN gds.version() AS version")
                record = result.single()
                print(f"   ✓ GDS version: {record['version']}")
            except Exception as e:
                print(f"   ✗ GDS not available: {e}")
                print("   Note: GDS plugin needs to be installed and configured")

        # Close connection
        driver.close()
        print("\n" + "=" * 60)
        print("✅ POC Test 1: PASSED")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n" + "=" * 60)
        print("❌ POC Test 1: FAILED")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)
