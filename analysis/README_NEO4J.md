# Neo4j Graph Analysis for GraphRAG

Quick reference guide for analyzing GraphRAG knowledge graphs in Neo4j.

## 🚀 Quick Start (3 Steps)

### 1. Start Neo4j with GDS

```bash
cd analysis
docker-compose up -d
```

### 2. Import GraphML

```bash
./import_graphml.sh
```

### 3. Access Neo4j Browser

Open http://localhost:7474

- **Username**: `neo4j`
- **Password**: `speedkg123`

## 📊 What's Included

- ✅ **Neo4j 5.16 Community Edition**
- ✅ **APOC Plugin** - Advanced graph procedures
- ✅ **Graph Data Science (GDS)** - Community detection, PageRank, centrality algorithms
- ✅ **GraphML Import** - Direct access to GraphRAG output
- ✅ **Automated Import Script** - One-command setup

## 📁 Files

| File | Purpose |
|------|---------|
| `docker-compose.yml` | Neo4j configuration with GDS |
| `import_graphml.sh` | Automated GraphML import script |
| `NEO4J_SETUP.md` | Complete documentation and query examples |
| `README_NEO4J.md` | This quick reference |

## 🔍 Quick Queries

### View Sample Data

```cypher
// Show high-degree entities
MATCH (n:Entity)
WHERE n.degree > 10
RETURN n.label, n.type, n.degree
ORDER BY n.degree DESC
LIMIT 20;
```

### Run Community Detection

```cypher
// Create GDS projection
CALL gds.graph.project(
  'graphrag',
  'Entity',
  'RELATED_TO',
  {relationshipProperties: ['weight']}
);

// Run Louvain
CALL gds.louvain.stream('graphrag')
YIELD nodeId, communityId
WITH gds.util.asNode(nodeId) AS node, communityId
RETURN communityId, collect(node.label)[0..10] as sample, count(*) as size
ORDER BY size DESC
LIMIT 10;
```

### Calculate PageRank

```cypher
CALL gds.pageRank.stream('graphrag')
YIELD nodeId, score
WITH gds.util.asNode(nodeId) AS node, score
RETURN node.label, node.type, score
ORDER BY score DESC
LIMIT 20;
```

## 📚 Full Documentation

See [NEO4J_SETUP.md](./NEO4J_SETUP.md) for:
- Detailed import instructions
- Complete GDS algorithm examples
- Advanced query patterns
- Troubleshooting guide
- Performance tuning

## 🛠️ Common Commands

```bash
# Start Neo4j
docker-compose up -d

# Stop Neo4j
docker-compose down

# View logs
docker-compose logs -f neo4j

# Access cypher-shell
docker-compose exec neo4j cypher-shell -u neo4j -p speedkg123

# Check Neo4j status
docker-compose ps

# Restart Neo4j
docker-compose restart neo4j

# Delete all data and restart fresh
docker-compose down -v
docker-compose up -d
./import_graphml.sh
```

## 🎯 Use Cases

### 1. Entity Analysis
Find the most important entities in your knowledge graph:
```cypher
MATCH (n:Entity)
RETURN n.type as entityType,
       avg(n.degree) as avgConnections,
       max(n.degree) as maxConnections,
       count(*) as count
ORDER BY avgConnections DESC;
```

### 2. Relationship Patterns
Discover common relationship patterns:
```cypher
MATCH (s:Entity)-[r:RELATED_TO]->(t:Entity)
RETURN s.type + ' → ' + t.type as pattern,
       count(*) as frequency,
       avg(r.weight) as avgWeight
ORDER BY frequency DESC
LIMIT 20;
```

### 3. Subgraph Extraction
Extract relevant subgraphs for analysis:
```cypher
// Get neighborhood around an entity
MATCH path = (center:Entity {label: 'Microsoft'})-[*1..2]-(neighbor)
RETURN path;
```

### 4. Graph Statistics
Get overall graph statistics:
```cypher
MATCH (n:Entity)
WITH count(n) as nodeCount
MATCH ()-[r]->()
WITH nodeCount, count(r) as relCount
RETURN nodeCount as entities,
       relCount as relationships,
       round(toFloat(relCount) / nodeCount, 2) as avgDegree;
```

## 💡 Pro Tips

1. **Memory**: Increase heap size for large graphs:
   ```yaml
   # In docker-compose.yml
   - NEO4J_dbms_memory_heap_max__size=8G
   ```

2. **Indexes**: Create indexes for faster queries:
   ```cypher
   CREATE INDEX entity_label IF NOT EXISTS FOR (n:Entity) ON (n.label);
   CREATE INDEX entity_type IF NOT EXISTS FOR (n:Entity) ON (n.type);
   ```

3. **Export Results**: Save query results to CSV:
   ```cypher
   CALL apoc.export.csv.query(
     "MATCH (n:Entity) WHERE n.degree > 10 RETURN n.label, n.type, n.degree",
     "high_degree_entities.csv",
     {}
   )
   ```

4. **Visualize**: Use Neo4j Bloom for visual exploration (requires Neo4j Desktop)

## 🐛 Troubleshooting

### Import Fails
```bash
# Check if file exists
docker-compose exec neo4j ls -la /graphrag-output/

# Check APOC config
docker-compose exec neo4j cypher-shell -u neo4j -p speedkg123 \
  "CALL apoc.config.list() YIELD key, value WHERE key CONTAINS 'import' RETURN key, value;"
```

### GDS Not Available
```bash
# Verify plugins loaded
docker-compose exec neo4j cypher-shell -u neo4j -p speedkg123 \
  "CALL gds.version() YIELD gdsVersion RETURN gdsVersion;"
```

### Container Won't Start
```bash
# Check logs
docker-compose logs neo4j

# Reset everything
docker-compose down -v
docker-compose up -d
```

## 📖 Resources

- [Neo4j Cypher Manual](https://neo4j.com/docs/cypher-manual/current/)
- [APOC Documentation](https://neo4j.com/labs/apoc/)
- [GDS Documentation](https://neo4j.com/docs/graph-data-science/current/)
- [Neo4j Browser Guide](https://neo4j.com/docs/browser-manual/current/)

## 🙋 Questions?

Check the detailed [NEO4J_SETUP.md](./NEO4J_SETUP.md) for comprehensive examples and troubleshooting.

---

**Happy Graph Exploring! 🎉**
