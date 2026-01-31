#!/usr/bin/env python3
"""Test script to load text units and communities into Neo4j."""

import os
import pandas as pd
from graphrag.index.graph.neo4j_backend import Neo4jBackend

# Configuration
password = os.getenv("NEO4J_PASSWORD", "speedkg123")

# Create Neo4j backend
backend = Neo4jBackend(
    uri="bolt://localhost:7687",
    username="neo4j",
    password=password,
    database="neo4j",
    node_label="Entity",
    relationship_type="RELATED_TO",
    use_entity_type_labels=True,
)

print("✅ Connected to Neo4j")

# Load text units
print("\n📄 Loading text units...")
text_units = pd.read_parquet("output/text_units.parquet")
print(f"Found {len(text_units)} text units")
backend.load_text_units(text_units)
print("✅ Text units loaded")

# Load communities and community reports
print("\n🏘️  Loading communities...")
communities = pd.read_parquet("output/communities.parquet")
community_reports = pd.read_parquet("output/community_reports.parquet")
print(f"Found {len(communities)} communities")
print(f"Found {len(community_reports)} community reports")
backend.load_communities(communities, community_reports)
print("✅ Communities loaded")

# Close connection
backend.close()

print("\n✅ All data loaded successfully!")
print("\nVerify in Neo4j Browser (http://localhost:7474):")
print("  MATCH (n) RETURN DISTINCT labels(n), count(*)")
