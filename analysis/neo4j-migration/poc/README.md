# Neo4j POC Tests

This directory contains Proof-of-Concept (POC) tests to validate Neo4j integration with GraphRAG before full implementation.

## Philosophy

Following the key learning from the Claude implementation post-mortem:

> "**POC First, Plans Later**: Run 1-hour POC before 6 weeks of planning"

These POC tests are designed to validate ALL core assumptions in 2-3 hours, preventing wasted effort on incorrect assumptions.

## Test Suite

### Test 1: Connection from Settings (`test_01_connection.py`)
- **Duration**: 5-10 minutes
- **Purpose**: Validate Neo4j connection using configuration
- **Validates**:
  - Configuration loading from YAML
  - Environment variable substitution
  - Neo4j driver connection
  - GDS plugin availability

### Test 2: Write Entities (`test_02_write_entities.py`)
- **Duration**: 10-15 minutes
- **Purpose**: Write GraphRAG entities and relationships to Neo4j
- **Validates**:
  - Creating nodes with properties
  - Creating relationships
  - Batch operations
  - Property types (strings, floats, lists)

### Test 3: GDS Community Detection (`test_03_gds_community.py`)
- **Duration**: 10-15 minutes
- **Purpose**: Run Louvain community detection
- **Validates**:
  - Graph projection creation
  - Louvain algorithm execution
  - Hierarchical community IDs
  - Writing results back to nodes

### Test 4: DataFrame Export (`test_04_dataframe_export.py`)
- **Duration**: 5-10 minutes
- **Purpose**: Export Neo4j data as pandas DataFrames
- **Validates**:
  - Schema compatibility with GraphRAG
  - Data type correctness
  - DataFrame operations
  - Query patterns

### Test 5: Full Integration (`test_05_full_integration.py`)
- **Duration**: 15-20 minutes
- **Purpose**: End-to-end integration test
- **Validates**:
  - Processing real GraphRAG data
  - Full pipeline: extract → store → detect → retrieve
  - Performance with realistic volumes
  - Complete workflow compatibility

## Prerequisites

1. **Neo4j Running**: Neo4j must be running with GDS plugin
   ```bash
   cd analysis
   docker-compose up -d neo4j
   ```

2. **Environment Variables**: Set NEO4J_PASSWORD
   ```bash
   export NEO4J_PASSWORD=speedkg123
   # Or add to .env file in project root
   ```

3. **Python Dependencies**:
   ```bash
   pip install neo4j pandas pyyaml
   ```

## Running the Tests

### Run All Tests (Recommended)
```bash
cd analysis/neo4j-migration/poc
python run_all_tests.py
```

### Run Individual Tests
```bash
# Test 1: Connection
python test_01_connection.py

# Test 2: Write entities
python test_02_write_entities.py

# Test 3: GDS community detection
python test_03_gds_community.py

# Test 4: DataFrame export
python test_04_dataframe_export.py

# Test 5: Full integration
python test_05_full_integration.py
```

## Success Criteria

All 5 tests must pass for POC to be considered successful:
- ✅ Test 1: Connection works
- ✅ Test 2: Can write entities
- ✅ Test 3: GDS Louvain works
- ✅ Test 4: DataFrame export works
- ✅ Test 5: Full integration works

## Expected Outcome

### If ALL Tests Pass ✅
- **Conclusion**: Core assumptions validated
- **Next Step**: Proceed to Phase 1 implementation
- **Timeline**: On track for 2-3 week implementation

### If ANY Test Fails ❌
- **Conclusion**: Blocker identified
- **Next Step**: Fix blocker before implementation
- **Action**: Document blocker in POC_RESULTS.md

## Key Learnings to Validate

1. ✅ **Neo4j driver works with GraphRAG config**
2. ✅ **GDS plugin can run Louvain on GraphRAG graphs**
3. ✅ **Can export Neo4j results as DataFrames**
4. ✅ **Performance is acceptable for batch operations**
5. ✅ **Full integration with GraphRAG pipeline works**

## Configuration

POC uses `settings.neo4j.yaml` which contains:
- Neo4j connection settings (URI, credentials, database)
- Connection pool settings
- GDS algorithm settings
- Vector index settings
- Minimal GraphRAG settings required for testing

## Data Cleanup

Tests create test data with specific labels:
- Test 2-4: `Entity` label
- Test 5: `IntegrationEntity` label

To cleanup test data:
```cypher
// Cleanup test data
MATCH (n:Entity) DETACH DELETE n;
MATCH (n:IntegrationEntity) DETACH DELETE n;
```

## Troubleshooting

### Test 1 Fails (Connection)
- Check Neo4j is running: `docker ps | grep neo4j`
- Check NEO4J_PASSWORD environment variable
- Check URI is correct (bolt://localhost:7687)

### Test 3 Fails (GDS)
- Verify GDS plugin installed: `RETURN gds.version()`
- Check docker-compose.yml has GDS in NEO4J_PLUGINS
- Restart Neo4j container if needed

### Test 5 Fails (Integration)
- Check if GraphRAG index has been run
- Test will use mock data if no real data available
- Verify entities have required columns (id, title, type)

## Next Steps

After POC completion:
1. Document results in `POC_RESULTS.md`
2. If successful: Start Phase 1 implementation
3. If blocked: Fix blockers and re-run POC
4. Create git commits for POC milestone
5. Tag POC completion: `git tag neo4j-poc-complete`
