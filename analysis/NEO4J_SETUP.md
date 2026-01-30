# Neo4j Setup for GraphRAG Analysis

This Docker Compose setup provides a Neo4j instance with APOC and Graph Data Science (GDS) plugins for analyzing GraphRAG knowledge graphs.

## Features

- **Neo4j 5.16 Community Edition**
- **APOC Plugin**: Advanced procedures for graph operations
- **Graph Data Science (GDS)**: Community detection, centrality algorithms, similarity metrics
- **GraphML Import**: Pre-configured to access GraphRAG output

## Quick Start

### Step 1: Start Neo4j

```bash
cd analysis
docker-compose up -d
```

### Step 2: Wait for Startup

The first startup takes ~30-60 seconds to download plugins. Check status:

```bash
docker-compose logs -f neo4j
```

Wait until you see:
```
Remote interface available at http://localhost:7474/
```

Press `Ctrl+C` to exit the logs (container keeps running).

### Step 3: Access Neo4j Browser

Open your browser and navigate to:
```
http://localhost:7474
```

**Login Credentials**:
- Username: `neo4j`
- Password: `speedkg123` (or set via `NEO4J_PASSWORD` env variable)

### Step 4: Verify GraphML File is Accessible

In Neo4j Browser, run:

```cypher
// Check if file is accessible
CALL apoc.config.list()
YIELD key, value
WHERE key CONTAINS 'import'
RETURN key, value;
```

Then verify the file exists:

```bash
# In a separate terminal
docker-compose exec neo4j ls -la /graphrag-output/
```

You should see `graph.graphml` listed.

## Importing GraphRAG GraphML

The GraphRAG output directory is mounted at `/graphrag-output` in the container.

### ⚠️ Important: Check if GraphML File Exists

First, verify that GraphRAG has generated the GraphML file:

```bash
# Check on host machine
ls -la ../quickstart/output/

# Check inside container
docker-compose exec neo4j ls -la /graphrag-output/
```

**If `graph.graphml` doesn't exist**, you need to run GraphRAG indexing first with snapshots enabled:

```bash
# In your GraphRAG project root
cd ../quickstart

# Ensure snapshots are enabled in settings.yaml
# Add this to your settings.yaml:
# snapshots:
#   graphml: true

# Run indexing
graphrag index

# Verify output
ls -la output/graph.graphml
```

### GraphML Import Methods

### Option 1: Automated Import Script (Easiest) 🚀

Use the provided import script:

```bash
cd analysis
./import_graphml.sh
```

This script will:
1. ✅ Start Neo4j if not running
2. ✅ Wait for Neo4j to be ready
3. ✅ Verify GraphML file exists
4. ✅ Import the graph
5. ✅ Show verification statistics
6. ✅ Display sample entities

### Option 2: Import via Cypher (Manual)

```cypher
// Import the graph
CALL apoc.import.graphml(
  'file:///graphrag-output/graph.graphml',
  {
    batchSize: 1000,
    compression: 'NONE',
    source: {
      label: 'Entity'
    },
    target: {
      label: 'Entity'
    },
    relType: 'RELATED_TO'
  }
)
YIELD file, source, format, nodes, relationships, properties, time
RETURN file, nodes, relationships, time;
```

### Option 3: Copy to Import Directory

If you prefer to use the standard import directory:

```bash
# Copy GraphML to import directory
cp ../quickstart/output/graph.graphml ./data/

# Then in Neo4j Browser:
CALL apoc.import.graphml(
  'file:///import/graph.graphml',
  {batchSize: 1000}
)
```

### Option 4: Import via Command Line (Alternative)

You can also import using `cypher-shell` from the command line:

```bash
# First, ensure Neo4j is running
docker-compose up -d

# Wait for Neo4j to be ready
sleep 10

# Import via cypher-shell
docker-compose exec neo4j cypher-shell -u neo4j -p speedkg123 \
  "CALL apoc.import.graphml('file:///graphrag-output/graph.graphml', {batchSize: 1000})"
```

**Note**: `neo4j-admin import` does NOT support GraphML - it only supports CSV files. Always use APOC for GraphML imports.

## Verifying the Import

After import, verify the data:

```cypher
// Count nodes
MATCH (n)
RETURN count(n) as nodeCount;

// Count relationships
MATCH ()-[r]->()
RETURN count(r) as relationshipCount;

// View sample nodes
MATCH (n)
RETURN n
LIMIT 25;

// Check node labels
CALL db.labels()
YIELD label
RETURN label;

// Check relationship types
CALL db.relationshipTypes()
YIELD relationshipType
RETURN relationshipType;
```

## GraphRAG Schema Mapping

GraphRAG's GraphML typically exports with this structure:

| GraphML Attribute | Neo4j Mapping | Description |
|-------------------|---------------|-------------|
| Node `id` | Node ID | Unique identifier |
| Node `label` | `:Entity` label | Entity name |
| Node `type` | `type` property | Entity type (PERSON, ORG, etc.) |
| Node `description` | `description` property | Entity description |
| Node `degree` | `degree` property | Graph degree |
| Edge `source` | Relationship source | Source entity |
| Edge `target` | Relationship target | Target entity |
| Edge `weight` | `weight` property | Relationship strength |
| Edge `description` | `description` property | Relationship description |

### Example Query After Import

```cypher
// Find high-degree entities (important nodes)
MATCH (n:Entity)
WHERE n.degree > 10
RETURN n.label as entity, n.type as type, n.degree as connections
ORDER BY n.degree DESC
LIMIT 20;

// Find relationships with high weight
MATCH (source:Entity)-[r:RELATED_TO]->(target:Entity)
WHERE r.weight > 5
RETURN source.label, type(r), target.label, r.weight, r.description
ORDER BY r.weight DESC
LIMIT 20;

// Find entities of specific type
MATCH (n:Entity)
WHERE n.type = 'ORGANIZATION'
RETURN n.label as organization, n.description
LIMIT 10;
```

## Using Graph Data Science (GDS)

### 1. Create an In-Memory Graph Projection

```cypher
// Project graph into GDS
CALL gds.graph.project(
  'graphrag-graph',
  'Entity',
  'RELATED_TO',
  {
    nodeProperties: ['degree', 'type'],
    relationshipProperties: ['weight']
  }
)
YIELD graphName, nodeCount, relationshipCount;
```

### 2. Run Community Detection (Louvain)

```cypher
// Run Louvain community detection
CALL gds.louvain.stream('graphrag-graph', {
  relationshipWeightProperty: 'weight'
})
YIELD nodeId, communityId
WITH gds.util.asNode(nodeId) AS node, communityId
RETURN communityId,
       collect(node.label) as members,
       count(*) as size
ORDER BY size DESC
LIMIT 10;
```

### 3. Calculate PageRank (Entity Importance)

```cypher
// Calculate PageRank
CALL gds.pageRank.stream('graphrag-graph', {
  relationshipWeightProperty: 'weight'
})
YIELD nodeId, score
WITH gds.util.asNode(nodeId) AS node, score
RETURN node.label as entity,
       node.type as type,
       score
ORDER BY score DESC
LIMIT 20;
```

### 4. Find Similar Entities (Node Similarity)

```cypher
// Find similar entities based on relationships
CALL gds.nodeSimilarity.stream('graphrag-graph')
YIELD node1, node2, similarity
WITH gds.util.asNode(node1) AS entity1,
     gds.util.asNode(node2) AS entity2,
     similarity
WHERE similarity > 0.5
RETURN entity1.label, entity2.label, similarity
ORDER BY similarity DESC
LIMIT 20;
```

### 5. Detect Central Nodes (Betweenness Centrality)

```cypher
// Find bridge entities connecting different parts of the graph
CALL gds.betweenness.stream('graphrag-graph')
YIELD nodeId, score
WITH gds.util.asNode(nodeId) AS node, score
RETURN node.label as entity,
       node.type as type,
       score as centrality
ORDER BY score DESC
LIMIT 20;
```

### 6. Clean Up GDS Projection

```cypher
// Drop the projection when done
CALL gds.graph.drop('graphrag-graph');
```

## Advanced Analysis Examples

### Community Detection with Write-Back

```cypher
// Write community assignments to nodes
CALL gds.louvain.write('graphrag-graph', {
  writeProperty: 'community',
  relationshipWeightProperty: 'weight'
})
YIELD communityCount, modularity;

// Query by community
MATCH (n:Entity)
WHERE n.community = 0
RETURN n.label, n.type, n.description
LIMIT 20;
```

### Path Finding

```cypher
// Find shortest path between two entities
MATCH (start:Entity {label: 'Microsoft'}),
      (end:Entity {label: 'Bill Gates'}),
      path = shortestPath((start)-[*]-(end))
RETURN path;

// Find all paths up to length 3
MATCH path = (start:Entity {label: 'Microsoft'})-[*1..3]-(end:Entity)
RETURN path
LIMIT 10;
```

### Subgraph Extraction

```cypher
// Extract subgraph around an entity
MATCH (center:Entity {label: 'Microsoft'})-[r*1..2]-(neighbor:Entity)
RETURN center, r, neighbor;

// Extract subgraph by type
MATCH path = (n1:Entity)-[r]-(n2:Entity)
WHERE n1.type = 'ORGANIZATION' AND n2.type = 'PERSON'
RETURN path
LIMIT 50;
```

## Visualizing GraphRAG Communities

GraphRAG creates hierarchical communities. You can visualize them:

```cypher
// If GraphRAG community data is available
MATCH (n:Entity)
WHERE n.community IS NOT NULL
WITH n.community as communityId, collect(n) as members
RETURN communityId, members, size(members) as memberCount
ORDER BY memberCount DESC;

// Visualize a specific community
MATCH (n:Entity)
WHERE n.community = 5
MATCH path = (n)-[r]-(m:Entity)
WHERE m.community = 5
RETURN path;
```

## Exporting Results

### Export to CSV

```cypher
// Export high-degree entities
CALL apoc.export.csv.query(
  "MATCH (n:Entity) WHERE n.degree > 10 RETURN n.label, n.type, n.degree ORDER BY n.degree DESC",
  "high_degree_entities.csv",
  {}
)
```

### Export Subgraph to GraphML

```cypher
// Export a subgraph
CALL apoc.export.graphml.query(
  "MATCH (n:Entity)-[r]-(m:Entity) WHERE n.type = 'ORGANIZATION' RETURN n, r, m",
  "organizations_subgraph.graphml",
  {}
)
```

Exported files will be in `/import` directory (accessible at `./data/` on host).

## Troubleshooting

### GDS Plugin Not Loading

If GDS doesn't load, check logs:

```bash
docker-compose logs neo4j | grep -i gds
```

**Note**: GDS requires sufficient memory. Increase heap size if needed:

```yaml
environment:
  - NEO4J_dbms_memory_heap_max__size=8G  # Increase if you have RAM
```

### GraphML Import Fails

Check the file exists:

```bash
docker-compose exec neo4j ls -la /graphrag-output/
```

Verify APOC configuration:

```cypher
CALL apoc.config.list()
YIELD key, value
WHERE key CONTAINS 'import'
RETURN key, value;
```

### Out of Memory

For large graphs, increase container memory:

```yaml
deploy:
  resources:
    limits:
      memory: 8G
```

## Performance Tuning

For better performance with large GraphRAG graphs:

```yaml
environment:
  # Increase heap
  - NEO4J_dbms_memory_heap_max__size=8G

  # Increase page cache
  - NEO4J_server_memory_pagecache_size=4G

  # Increase transaction log size
  - NEO4J_db_logs_query_threshold=0

  # Parallel query execution
  - NEO4J_dbms_cypher_parallel_runtime_support=all
```

## Stopping Neo4j

```bash
# Stop container
docker-compose down

# Stop and remove volumes (deletes all data)
docker-compose down -v
```

## Useful Resources

- [Neo4j Browser Guide](https://neo4j.com/docs/browser-manual/current/)
- [APOC Documentation](https://neo4j.com/labs/apoc/5/)
- [GDS Documentation](https://neo4j.com/docs/graph-data-science/current/)
- [Cypher Query Language](https://neo4j.com/docs/cypher-manual/current/)

## GraphRAG-Specific Queries

### Entity Type Distribution

```cypher
MATCH (n:Entity)
RETURN n.type as entityType, count(*) as count
ORDER BY count DESC;
```

### Most Connected Entities by Type

```cypher
MATCH (n:Entity)
WITH n.type as type, n.label as entity, n.degree as degree
ORDER BY type, degree DESC
WITH type, collect({entity: entity, degree: degree})[0..5] as topEntities
RETURN type, topEntities;
```

### Relationship Weight Distribution

```cypher
MATCH ()-[r:RELATED_TO]->()
RETURN
  min(r.weight) as minWeight,
  max(r.weight) as maxWeight,
  avg(r.weight) as avgWeight,
  percentileDisc(r.weight, 0.5) as medianWeight,
  percentileDisc(r.weight, 0.95) as p95Weight;
```

### Find Isolated Entities

```cypher
MATCH (n:Entity)
WHERE NOT (n)-[]->()
RETURN n.label, n.type, n.description;
```

---

**Happy Graph Exploring! 🎯**
