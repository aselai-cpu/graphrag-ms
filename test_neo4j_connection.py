#!/usr/bin/env python3
"""Quick test script to validate Neo4j connection and dark mode setup."""

import os
import sys

def test_neo4j_connection():
    """Test Neo4j connection with current configuration."""
    print("🔍 Testing Neo4j Connection...")
    print()

    # Check environment variable
    password = os.getenv("NEO4J_PASSWORD")
    if not password:
        print("❌ NEO4J_PASSWORD environment variable not set")
        print("   Set it with: export NEO4J_PASSWORD=your_password")
        return False
    else:
        print("✅ NEO4J_PASSWORD is set")

    # Try to import neo4j
    try:
        import neo4j
        print("✅ neo4j package installed")
    except ImportError:
        print("❌ neo4j package not installed")
        print("   Install with: pip install neo4j")
        return False

    # Try to connect
    try:
        from graphrag.index.graph.neo4j_backend import Neo4jBackend
        print("✅ Neo4jBackend imports successfully")

        print("\n📡 Attempting connection to Neo4j...")
        backend = Neo4jBackend(
            uri="bolt://localhost:7687",
            username="neo4j",
            password=password,
            database="neo4j",
            gds_library=True,
        )
        print("✅ Successfully connected to Neo4j!")

        # Get counts
        node_count = backend.node_count()
        edge_count = backend.edge_count()

        print(f"\n📊 Neo4j Database Status:")
        print(f"   Nodes: {node_count}")
        print(f"   Relationships: {edge_count}")

        backend.close()

        if node_count == 0 and edge_count == 0:
            print("\n💡 Database is empty - this is expected before running indexing")
            print("   Run: uv poe run index")
        else:
            print("\n✅ Database has data!")

        return True

    except ConnectionError as e:
        print(f"❌ Connection failed: {e}")
        print("\n💡 Make sure Neo4j is running:")
        print("   docker run -d --name neo4j -p 7474:7474 -p 7687:7687 \\")
        print("     -e NEO4J_AUTH=neo4j/your_password neo4j:latest")
        return False
    except ValueError as e:
        print(f"❌ Authentication failed: {e}")
        print("\n💡 Check your password:")
        print("   export NEO4J_PASSWORD=your_actual_password")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dark_mode_config():
    """Test dark mode configuration."""
    print("\n🔍 Testing Dark Mode Configuration...")
    print()

    try:
        from graphrag.config.models.graph_rag_config import GraphRagConfig
        import yaml

        # Load settings
        with open("settings.yaml") as f:
            settings = yaml.safe_load(f)

        dark_mode = settings.get("dark_mode", {})
        neo4j_config = settings.get("neo4j", {})

        # Check dark mode enabled
        if dark_mode.get("enabled"):
            print("✅ Dark mode is enabled")
        else:
            print("❌ Dark mode is disabled")
            print("   Enable in settings.yaml: dark_mode.enabled: true")
            return False

        # Check backends
        primary = dark_mode.get("primary_backend")
        shadow = dark_mode.get("shadow_backend")
        print(f"✅ Primary backend: {primary}")
        print(f"✅ Shadow backend: {shadow}")

        # Check Neo4j config
        if shadow == "neo4j":
            if neo4j_config:
                print("✅ Neo4j configuration present")
                uri = neo4j_config.get("uri", "bolt://localhost:7687")
                database = neo4j_config.get("database", "neo4j")
                print(f"   URI: {uri}")
                print(f"   Database: {database}")
            else:
                print("❌ Neo4j configuration missing in settings.yaml")
                return False

        # Check metrics path
        log_path = dark_mode.get("log_path")
        if log_path:
            print(f"✅ Metrics will be logged to: {log_path}")

        return True

    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return False

if __name__ == "__main__":
    print("=" * 70)
    print("Neo4j & Dark Mode Validation Script")
    print("=" * 70)
    print()

    config_ok = test_dark_mode_config()
    connection_ok = test_neo4j_connection()

    print()
    print("=" * 70)
    if config_ok and connection_ok:
        print("✅ All checks passed! Ready to run indexing with dark mode.")
        print()
        print("Next steps:")
        print("  1. Run: uv poe run index")
        print("  2. Check Neo4j Browser: http://localhost:7474")
        print("  3. Analyze metrics: graphrag dark-mode analyze output/dark_mode_metrics.jsonl")
        sys.exit(0)
    else:
        print("❌ Some checks failed. Fix the issues above and try again.")
        sys.exit(1)
