# Implementation Plan

**Document**: 06 - Implementation Plan
**Date**: 2026-01-29
**Status**: Complete

---

## Purpose

This document provides a detailed implementation roadmap for migrating GraphRAG from NetworkX to Neo4j, including phases, tasks, estimates, dependencies, and success criteria.

---

## Overview

### Total Timeline: 4-5 Months

**Phase 1: Foundation** (Weeks 1-4)
**Phase 2: Core Integration** (Weeks 5-10)
**Phase 3: Production Readiness** (Weeks 11-14)
**Phase 4: Rollout** (Weeks 15-20)

### Resource Requirements

- **Developers**: 1-2 full-time
- **Infrastructure**: Neo4j test environment
- **Budget**: Minimal (Community Edition free)

---

## Phase 1: Foundation (Weeks 1-4)

### Goals
- Create storage abstraction layer
- Implement basic Neo4j adapter
- Validate approach with proof-of-concept

### Tasks

#### Task 1.1: Storage Interface Design (Week 1)
**Estimate**: 3-5 days
**Owner**: Backend developer
**Dependencies**: None

**Deliverables**:
```python
# packages/graphrag-storage/graphrag_storage/graph_storage.py
class GraphStorage(ABC):
    @abstractmethod
    async def write_entities(self, entities: pd.DataFrame) -> None: ...
    @abstractmethod
    async def read_entities(self) -> pd.DataFrame: ...
    @abstractmethod
    async def calculate_degrees(self) -> None: ...
    @abstractmethod
    async def run_community_detection(...) -> pd.DataFrame: ...
    @abstractmethod
    async def write_embeddings(...) -> None: ...
```

**Acceptance Criteria**:
- [ ] Interface covers all storage operations
- [ ] Type hints complete
- [ ] Docstrings written
- [ ] Design review approved

#### Task 1.2: Refactor Existing Code to Use Interface (Week 1-2)
**Estimate**: 5-7 days
**Owner**: Backend developer
**Dependencies**: Task 1.1

**Changes Required**:
- `packages/graphrag/graphrag/index/workflows/finalize_graph.py`
- `packages/graphrag/graphrag/index/workflows/create_communities.py`
- `packages/graphrag/graphrag/index/workflows/generate_text_embeddings.py`

**Example Refactoring**:
```python
# Before
entities = await load_table_from_storage("entities", context.output_storage)
relationships = await load_table_from_storage("relationships", context.output_storage)
graph = create_graph(entities, relationships)
degrees = dict(graph.degree())

# After
graph_storage = create_graph_storage(config)
await graph_storage.calculate_degrees()
entities = await graph_storage.read_entities()
```

**Acceptance Criteria**:
- [ ] All workflows use GraphStorage interface
- [ ] Existing functionality unchanged
- [ ] All tests pass
- [ ] No performance regression

#### Task 1.3: Implement ParquetGraphStorage (Week 2)
**Estimate**: 3-5 days
**Owner**: Backend developer
**Dependencies**: Task 1.1

**Deliverables**:
```python
# packages/graphrag-storage/graphrag_storage/parquet_graph_storage.py
class ParquetGraphStorage(GraphStorage):
    # Wraps existing Parquet operations
    ...
```

**Acceptance Criteria**:
- [ ] All interface methods implemented
- [ ] Wraps existing Parquet logic
- [ ] Tests pass
- [ ] Documentation complete

#### Task 1.4: Neo4j Development Environment Setup (Week 2)
**Estimate**: 2-3 days
**Owner**: DevOps/Developer
**Dependencies**: None

**Deliverables**:
- Docker Compose configuration for Neo4j
- GDS plugin enabled
- Test data import scripts
- Connection testing utilities

**Files**:
```yaml
# docker-compose.dev.yml
services:
  neo4j-dev:
    image: neo4j:5.17.0
    environment:
      NEO4J_AUTH: neo4j/devpassword
      NEO4J_PLUGINS: '["graph-data-science"]'
    ports:
      - "7474:7474"
      - "7687:7687"
```

**Acceptance Criteria**:
- [ ] Neo4j starts successfully
- [ ] GDS plugin loaded
- [ ] Python driver can connect
- [ ] Sample queries work

#### Task 1.5: Implement Basic Neo4jGraphStorage (Week 3-4)
**Estimate**: 7-10 days
**Owner**: Backend developer
**Dependencies**: Task 1.1, Task 1.4

**Deliverables**:
```python
# packages/graphrag-storage/graphrag_storage/neo4j_graph_storage.py
class Neo4jGraphStorage(GraphStorage):
    async def write_entities(self, entities: pd.DataFrame) -> None:
        # Batch create entities with UNWIND
        ...

    async def calculate_degrees(self) -> None:
        # Use GDS degree algorithm
        ...

    async def run_community_detection(...) -> pd.DataFrame:
        # Use GDS Louvain algorithm
        ...
```

**Key Methods**:
1. Connection management (driver, session)
2. Entity write (batched with UNWIND)
3. Relationship write (batched)
4. Degree calculation (GDS)
5. Community detection (GDS Louvain)
6. Read operations (export to DataFrame)

**Acceptance Criteria**:
- [ ] All interface methods implemented
- [ ] Batching works correctly
- [ ] GDS operations succeed
- [ ] Error handling robust
- [ ] Unit tests written (mocked Neo4j)
- [ ] Integration tests written (real Neo4j)

#### Task 1.6: Proof-of-Concept End-to-End Test (Week 4)
**Estimate**: 3-5 days
**Owner**: Backend developer
**Dependencies**: Task 1.5

**Test Scenario**:
```python
# Run mini indexing pipeline with Neo4j
async def test_neo4j_poc():
    # 1. Load sample documents (Christmas Carol)
    documents = load_sample_documents()

    # 2. Extract entities and relationships
    entities, relationships = await extract_graph(documents, config)

    # 3. Write to Neo4j
    neo4j_storage = Neo4jGraphStorage(...)
    await neo4j_storage.write_entities(entities)
    await neo4j_storage.write_relationships(relationships)

    # 4. Calculate degrees
    await neo4j_storage.calculate_degrees()

    # 5. Run community detection
    communities = await neo4j_storage.run_community_detection()

    # 6. Verify results
    assert len(communities) > 0
    assert communities["community"].nunique() > 1
```

**Acceptance Criteria**:
- [ ] POC completes successfully
- [ ] Results match Parquet version
- [ ] Community structure reasonable
- [ ] Performance acceptable
- [ ] Code review approved

### Phase 1 Milestones

**Week 2 Checkpoint**:
- [ ] Storage interface defined
- [ ] Existing code refactored
- [ ] ParquetGraphStorage implemented

**Week 4 Completion**:
- [ ] Neo4jGraphStorage basic implementation done
- [ ] POC successful
- [ ] Go/No-Go decision for Phase 2

**Risk Assessment**: Low - Foundational work, can abort if POC fails

---

## Phase 2: Core Integration (Weeks 5-10)

### Goals
- Complete Neo4j adapter implementation
- Integrate with all indexing workflows
- Implement hybrid mode
- Comprehensive testing

### Tasks

#### Task 2.1: Complete Neo4j Schema Implementation (Week 5)
**Estimate**: 5-7 days
**Owner**: Backend developer
**Dependencies**: Phase 1 complete

**Schema Components**:
1. ✅ Entity nodes (done in Phase 1)
2. ✅ Relationship edges (done in Phase 1)
3. 🔲 Community nodes
4. 🔲 TextUnit nodes
5. 🔲 Document nodes
6. 🔲 Covariate nodes (if enabled)

**Indexes**:
```cypher
// Unique constraints
CREATE CONSTRAINT entity_id_unique FOR (e:Entity) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT community_id_unique FOR (c:Community) REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT text_unit_id_unique FOR (t:TextUnit) REQUIRE t.id IS UNIQUE;

// Vector indexes
CREATE VECTOR INDEX entity_description_vector FOR (e:Entity) ON e.description_embedding;
CREATE VECTOR INDEX community_summary_vector FOR (c:Community) ON c.summary_embedding;
CREATE VECTOR INDEX text_unit_vector FOR (t:TextUnit) ON t.text_embedding;
```

**Acceptance Criteria**:
- [ ] All node types created
- [ ] All relationships defined
- [ ] Indexes created
- [ ] Schema documented

#### Task 2.2: Implement Community Node Creation (Week 5-6)
**Estimate**: 3-5 days
**Owner**: Backend developer
**Dependencies**: Task 2.1

**Implementation**:
```python
async def _create_community_nodes(self, communities_df: pd.DataFrame):
    with self.driver.session() as session:
        # Create Community nodes
        session.execute_write(_create_communities, ...)

        # Create BELONGS_TO relationships
        session.execute_write(_create_memberships, ...)

        # Create PARENT_OF hierarchy
        session.execute_write(_create_hierarchy, ...)
```

**Acceptance Criteria**:
- [ ] Community nodes created correctly
- [ ] Hierarchy structure correct
- [ ] Entity membership links work
- [ ] Tests pass

#### Task 2.3: Implement TextUnit and Document Storage (Week 6)
**Estimate**: 3-5 days
**Owner**: Backend developer
**Dependencies**: Task 2.1

**Operations**:
1. Write Document nodes
2. Write TextUnit nodes
3. Create CONTAINS relationships
4. Create MENTIONS relationships (TextUnit → Entity)

**Acceptance Criteria**:
- [ ] Documents stored correctly
- [ ] Text units linked to documents
- [ ] Entity mentions tracked
- [ ] Bidirectional links verified

#### Task 2.4: Implement Vector Embedding Storage (Week 6-7)
**Estimate**: 3-5 days
**Owner**: Backend developer
**Dependencies**: Task 2.1

**Implementation**:
```python
async def write_embeddings(
    self,
    entity_embeddings: pd.DataFrame,
    community_embeddings: pd.DataFrame,
    text_unit_embeddings: pd.DataFrame
):
    # Batch update embedding properties
    # Use UNWIND for efficiency
    ...
```

**Acceptance Criteria**:
- [ ] Embeddings stored as LIST<FLOAT>
- [ ] Vector indexes created automatically
- [ ] Batch operations efficient (<10s for 10K vectors)
- [ ] Tests pass

#### Task 2.5: Implement Covariate Storage (Week 7)
**Estimate**: 2-3 days
**Owner**: Backend developer
**Dependencies**: Task 2.1

**Implementation**:
```python
async def write_covariates(self, covariates: pd.DataFrame):
    # Create Covariate nodes
    # Link to subject/object entities
    # Link to source text units
    ...
```

**Acceptance Criteria**:
- [ ] Covariates stored correctly
- [ ] Subject/object links work
- [ ] Temporal properties indexed
- [ ] Tests pass

#### Task 2.6: Update All Indexing Workflows (Week 7-8)
**Estimate**: 5-7 days
**Owner**: Backend developer
**Dependencies**: Tasks 2.1-2.5

**Workflows to Update**:
1. ✅ `finalize_graph.py` (done in Phase 1)
2. 🔲 `create_base_text_units.py`
3. 🔲 `create_final_documents.py`
4. 🔲 `extract_graph.py`
5. ✅ `create_communities.py` (done in Phase 1)
6. 🔲 `create_final_text_units.py`
7. 🔲 `create_community_reports.py`
8. 🔲 `generate_text_embeddings.py`
9. 🔲 `extract_covariates.py`

**Pattern**:
```python
# Each workflow:
async def run(config: GraphRagConfig, context: PipelineRunContext):
    # Get storage backend
    storage = create_graph_storage(config)

    # Do workflow logic
    ...

    # Write to storage
    await storage.write_X(data)
```

**Acceptance Criteria**:
- [ ] All workflows updated
- [ ] Parquet path still works
- [ ] Neo4j path works
- [ ] Tests updated
- [ ] All tests pass

#### Task 2.7: Implement Hybrid Mode (Week 8-9)
**Estimate**: 5-7 days
**Owner**: Backend developer
**Dependencies**: Task 2.6

**Implementation**:
```python
class HybridGraphStorage(GraphStorage):
    def __init__(self, parquet: ParquetGraphStorage, neo4j: Neo4jGraphStorage):
        self.parquet = parquet
        self.neo4j = neo4j

    async def write_entities(self, entities: pd.DataFrame):
        # Write to both in parallel
        await asyncio.gather(
            self.parquet.write_entities(entities),
            self.neo4j.write_entities(entities)
        )

    async def read_entities(self) -> pd.DataFrame:
        # Read from Neo4j (source of truth)
        return await self.neo4j.read_entities()
```

**Acceptance Criteria**:
- [ ] Writes go to both backends
- [ ] Reads prefer Neo4j
- [ ] Errors in one don't affect the other
- [ ] Configuration option works
- [ ] Tests pass

#### Task 2.8: Configuration Schema (Week 9)
**Estimate**: 2-3 days
**Owner**: Backend developer
**Dependencies**: None

**Configuration**:
```yaml
# settings.yaml
storage:
  type: neo4j  # Options: parquet, neo4j, hybrid

  neo4j:
    uri: "bolt://localhost:7687"
    username: "neo4j"
    password: "password"
    database: "neo4j"
    batch_size: 1000

    gds:
      enabled: true
      max_cluster_size: 10

    vector_index:
      enabled: true
      dimensions: 1536
      similarity_function: cosine
```

**Acceptance Criteria**:
- [ ] Configuration schema defined
- [ ] Validation works
- [ ] Defaults sensible
- [ ] Documentation written

#### Task 2.9: Integration Testing (Week 9-10)
**Estimate**: 5-7 days
**Owner**: Backend developer + QA
**Dependencies**: All Phase 2 tasks

**Test Suites**:

1. **Smoke Tests** (5 tests)
   - Basic indexing with Neo4j
   - Verify all node types created
   - Verify all relationships created
   - Verify embeddings stored
   - Verify queries work

2. **Workflow Tests** (10 tests)
   - Test each workflow in isolation
   - Test full pipeline end-to-end
   - Test with/without covariates
   - Test error handling

3. **Storage Backend Tests** (15 tests)
   - ParquetGraphStorage full coverage
   - Neo4jGraphStorage full coverage
   - HybridGraphStorage full coverage
   - Error cases

4. **Performance Tests** (5 tests)
   - Small graph (100 nodes)
   - Medium graph (1K nodes)
   - Large graph (10K nodes)
   - Compare NetworkX vs Neo4j
   - Memory usage profiling

**Acceptance Criteria**:
- [ ] All tests pass
- [ ] Test coverage > 80%
- [ ] Performance benchmarks documented
- [ ] No memory leaks
- [ ] Code review approved

### Phase 2 Milestones

**Week 7 Checkpoint**:
- [ ] All storage operations implemented
- [ ] Schema complete
- [ ] Basic workflows integrated

**Week 10 Completion**:
- [ ] All workflows integrated
- [ ] Hybrid mode working
- [ ] Integration tests passing
- [ ] Performance acceptable

**Risk Assessment**: Medium - Core implementation, must work correctly

---

## Phase 3: Production Readiness (Weeks 11-14)

### Goals
- Query operation updates
- Performance optimization
- Production deployment guides
- Migration tools

### Tasks

#### Task 3.1: Update Query Operations (Week 11-12)
**Estimate**: 7-10 days
**Owner**: Backend developer
**Dependencies**: Phase 2 complete

**Query Methods to Update**:
1. Global Search
2. Local Search
3. DRIFT Search
4. Basic Search (vector-only)

**Implementation**:
```python
# packages/graphrag/graphrag/query/engines/

class GlobalSearchEngine:
    async def search_neo4j(self, query: str, config: QueryConfig):
        # Use Neo4j vector index for community search
        query_embedding = await self.embed(query)

        with self.neo4j_driver.session() as session:
            results = session.run("""
                CALL db.index.vector.queryNodes(
                    'community_summary_vector',
                    $limit,
                    $query_embedding
                )
                YIELD node, score
                RETURN node.title, node.summary, node.full_content, score
            """, limit=config.community_limit, query_embedding=query_embedding)

            # MAP-REDUCE as before
            ...
```

**Hybrid Query Example** (Local Search):
```cypher
// Find similar entities + their neighborhoods
CALL db.index.vector.queryNodes('entity_description_vector', 20, $embedding)
YIELD node AS entity, score
MATCH (entity)-[:RELATED_TO*1..2]-(neighbor)
MATCH (t:TextUnit)-[:MENTIONS]->(entity)
RETURN entity, collect(DISTINCT neighbor), collect(DISTINCT t.text), score
```

**Acceptance Criteria**:
- [ ] All query methods work with Neo4j
- [ ] Hybrid queries implemented
- [ ] Performance comparable or better
- [ ] Tests updated
- [ ] Backward compatibility maintained (can still use Parquet)

#### Task 3.2: Performance Optimization (Week 12-13)
**Estimate**: 5-7 days
**Owner**: Backend developer
**Dependencies**: Task 3.1

**Optimization Areas**:

1. **Query Optimization**
   - Add query hints
   - Optimize Cypher patterns
   - Use indexes effectively

2. **Batch Size Tuning**
   - Test different batch sizes (500, 1000, 2000)
   - Find optimal for different graph sizes

3. **Connection Pooling**
   - Configure pool size
   - Test concurrent load

4. **Memory Configuration**
   - Tune Neo4j heap size
   - Configure page cache
   - GDS memory settings

**Benchmarking**:
```bash
# Run performance suite
pytest tests/performance/ --benchmark

# Results logged to:
# - Indexing time by graph size
# - Query latency percentiles
# - Memory usage
# - GDS projection time
```

**Acceptance Criteria**:
- [ ] Community detection ≤ NetworkX * 2 (target: faster)
- [ ] Query latency < 100ms for typical queries
- [ ] Memory usage reasonable
- [ ] Benchmarks documented

#### Task 3.3: Error Handling and Recovery (Week 13)
**Estimate**: 3-5 days
**Owner**: Backend developer
**Dependencies**: Phase 2 complete

**Error Scenarios**:

1. **Connection Failures**
   ```python
   # Retry logic with exponential backoff
   @retry(max_attempts=3, backoff=exponential)
   async def write_with_retry(...):
       ...
   ```

2. **Transaction Failures**
   ```python
   # Rollback and cleanup
   try:
       with session.begin_transaction() as tx:
           ...
           tx.commit()
   except Exception:
       tx.rollback()
       # Cleanup partial state
   ```

3. **Partial Writes**
   ```python
   # Track progress, allow resume
   checkpoint_file = "neo4j_import_checkpoint.json"
   # Save last successful batch
   ```

**Acceptance Criteria**:
- [ ] Transient errors handled gracefully
- [ ] Partial writes can be resumed
- [ ] Clear error messages
- [ ] Logging comprehensive
- [ ] Tests for error cases

#### Task 3.4: Documentation (Week 13-14)
**Estimate**: 5-7 days
**Owner**: Technical writer + Developer
**Dependencies**: Phase 2-3 complete

**Documentation Deliverables**:

1. **User Guide** (`docs/neo4j/user_guide.md`)
   - Why use Neo4j
   - When to use Neo4j vs Parquet
   - Configuration examples
   - Common patterns

2. **Setup Guide** (`docs/neo4j/setup.md`)
   - Docker installation
   - Native installation
   - Neo4j Aura (cloud)
   - Troubleshooting

3. **Migration Guide** (`docs/neo4j/migration.md`)
   - Migrating from Parquet
   - Using hybrid mode
   - Export/import tools
   - Rollback procedures

4. **Developer Guide** (`docs/neo4j/developer.md`)
   - Architecture overview
   - Storage interface
   - Adding new operations
   - Testing guidelines

5. **API Reference** (auto-generated)
   - GraphStorage interface
   - Neo4jGraphStorage class
   - Configuration schema
   - Cypher query examples

6. **Examples** (`examples/neo4j/`)
   - Basic indexing
   - Hybrid queries
   - Custom analytics
   - Production deployment

**Acceptance Criteria**:
- [ ] All guides written
- [ ] Examples tested
- [ ] Screenshots/diagrams included
- [ ] Review approved

#### Task 3.5: Migration Tools (Week 14)
**Estimate**: 3-5 days
**Owner**: Backend developer
**Dependencies**: Phase 2 complete

**Tools to Build**:

1. **Parquet → Neo4j Importer**
   ```bash
   graphrag import-to-neo4j \
     --input ./output \
     --neo4j-uri bolt://localhost:7687 \
     --neo4j-user neo4j \
     --neo4j-password password
   ```

2. **Neo4j → Parquet Exporter**
   ```bash
   graphrag export-from-neo4j \
     --output ./output \
     --neo4j-uri bolt://localhost:7687
   ```

3. **Validation Tool**
   ```bash
   graphrag validate-neo4j \
     --neo4j-uri bolt://localhost:7687
   # Checks: schema, indexes, data integrity
   ```

**Acceptance Criteria**:
- [ ] Import tool works correctly
- [ ] Export tool works correctly
- [ ] Validation tool comprehensive
- [ ] CLI documentation complete
- [ ] Error handling robust

### Phase 3 Milestones

**Week 12 Checkpoint**:
- [ ] Query operations updated
- [ ] Performance optimization done

**Week 14 Completion**:
- [ ] All documentation complete
- [ ] Migration tools ready
- [ ] Production deployment guides written
- [ ] Ready for beta release

**Risk Assessment**: Medium - Production concerns, need thorough testing

---

## Phase 4: Rollout (Weeks 15-20)

### Goals
- Beta release with hybrid mode
- User feedback and iteration
- Make Neo4j the default
- Deprecate Parquet (optional)

### Tasks

#### Task 4.1: Beta Release (Week 15)
**Estimate**: 2-3 days
**Owner**: Release manager
**Dependencies**: Phase 3 complete

**Release Checklist**:
- [ ] Version bump (e.g., v3.1.0-beta.1)
- [ ] Release notes written
- [ ] Migration guide published
- [ ] Beta announcement
- [ ] Support channels ready

**Configuration for Beta**:
```yaml
# Default: Parquet (backward compatible)
storage:
  type: parquet

# Opt-in: Neo4j (beta)
# storage:
#   type: neo4j
#   neo4j:
#     uri: "bolt://localhost:7687"

# Recommended: Hybrid (transition)
# storage:
#   type: hybrid
```

**Acceptance Criteria**:
- [ ] Beta release published
- [ ] Documentation live
- [ ] Examples work
- [ ] Support ready

#### Task 4.2: User Feedback Collection (Week 15-18)
**Estimate**: 3 weeks (continuous)
**Owner**: Product manager + Developer
**Dependencies**: Task 4.1

**Feedback Channels**:
- GitHub issues
- Discord/Slack support
- User interviews
- Bug reports

**Metrics to Track**:
- Adoption rate (% using Neo4j)
- Error rates
- Performance reports
- Feature requests
- User satisfaction

**Acceptance Criteria**:
- [ ] Feedback collected from 10+ users
- [ ] Critical bugs identified and fixed
- [ ] Performance validated on real datasets
- [ ] User satisfaction > 7/10

#### Task 4.3: Bug Fixes and Iterations (Week 16-18)
**Estimate**: Ongoing
**Owner**: Backend developer
**Dependencies**: Task 4.2

**Common Issues to Address**:
- Connection timeouts
- Memory issues with large graphs
- Configuration confusion
- Query performance
- Documentation gaps

**Release Cadence**: Weekly patches (v3.1.0-beta.2, beta.3, etc.)

**Acceptance Criteria**:
- [ ] Critical bugs fixed within 48 hours
- [ ] Patches released regularly
- [ ] User satisfaction improving

#### Task 4.4: Performance Validation (Week 17-18)
**Estimate**: 5-7 days
**Owner**: Backend developer + Users
**Dependencies**: Task 4.2

**Validation on Real Datasets**:
1. **Small Dataset** (1-10 documents)
   - Wikipedia articles
   - News articles

2. **Medium Dataset** (100-1000 documents)
   - Research papers
   - Company documentation

3. **Large Dataset** (10K+ documents)
   - Product documentation
   - Legal documents

**Benchmarks to Collect**:
- Indexing time
- Community detection time
- Query latency
- Memory usage
- Disk usage

**Acceptance Criteria**:
- [ ] Performance meets expectations
- [ ] No regressions vs Parquet
- [ ] User reports positive
- [ ] Benchmarks published

#### Task 4.5: Stable Release (Week 19)
**Estimate**: 2-3 days
**Owner**: Release manager
**Dependencies**: Tasks 4.2-4.4

**Release Checklist**:
- [ ] Version bump (e.g., v3.1.0)
- [ ] All beta feedback addressed
- [ ] Documentation finalized
- [ ] Examples updated
- [ ] Stable release announcement

**Configuration for Stable**:
```yaml
# Neo4j now recommended (but Parquet still supported)
storage:
  type: neo4j  # Recommended
  # type: parquet  # Still supported

  neo4j:
    uri: "bolt://localhost:7687"
    username: "neo4j"
    password: "${NEO4J_PASSWORD}"
```

**Acceptance Criteria**:
- [ ] Stable release published
- [ ] No critical bugs
- [ ] User satisfaction > 8/10
- [ ] Documentation complete

#### Task 4.6: Make Neo4j Default (Week 20)
**Estimate**: 2-3 days
**Owner**: Product manager + Developer
**Dependencies**: Task 4.5

**Changes**:
```yaml
# New projects default to Neo4j
# graphrag init --defaults

storage:
  type: neo4j  # Default (was: parquet)
```

**Documentation Updates**:
- Getting started guide uses Neo4j
- Examples use Neo4j by default
- Parquet moved to "Legacy" section

**Acceptance Criteria**:
- [ ] Default configuration uses Neo4j
- [ ] Documentation reflects new default
- [ ] Legacy Parquet docs preserved
- [ ] Migration path clear

#### Task 4.7: Deprecation Planning (Week 20)
**Estimate**: 1-2 days
**Owner**: Product manager
**Dependencies**: Task 4.6

**Deprecation Timeline** (if desired):
- **v3.1.0** (current): Neo4j default, Parquet supported
- **v3.2.0** (6 months): Parquet deprecated warning
- **v4.0.0** (12 months): Parquet removed (optional)

**Deprecation Notice**:
```python
# If using Parquet
warnings.warn(
    "Parquet storage is deprecated and will be removed in v4.0.0. "
    "Please migrate to Neo4j. See: docs/neo4j/migration.md",
    DeprecationWarning
)
```

**Acceptance Criteria**:
- [ ] Deprecation timeline decided
- [ ] Communication plan created
- [ ] Migration support planned

### Phase 4 Milestones

**Week 16 Checkpoint**:
- [ ] Beta feedback collected
- [ ] Major issues fixed

**Week 18 Checkpoint**:
- [ ] Performance validated
- [ ] User satisfaction high

**Week 20 Completion**:
- [ ] Stable release published
- [ ] Neo4j is default
- [ ] Project complete ✅

**Risk Assessment**: Low - User adoption, can always revert

---

## Success Metrics

### Technical Metrics

| Metric | Target | Critical |
|--------|--------|----------|
| **Community Detection Time** | ≤ NetworkX × 2 | Yes |
| **Query Latency** | < 100ms (p95) | Yes |
| **Test Coverage** | > 80% | Yes |
| **Memory Overhead** | < 2x NetworkX | No |
| **Disk Usage** | < 2x Parquet | No |

### User Metrics

| Metric | Target | Critical |
|--------|--------|----------|
| **Adoption Rate** (beta) | > 20% | No |
| **User Satisfaction** | > 8/10 | Yes |
| **Bug Report Rate** | < 5/week | Yes |
| **Support Requests** | < 10/week | No |

### Project Metrics

| Metric | Target | Critical |
|--------|--------|----------|
| **Timeline Adherence** | ±2 weeks | Yes |
| **Budget** | Within estimate | Yes |
| **Code Quality** | Review approved | Yes |
| **Documentation Quality** | Complete | Yes |

---

## Risk Management

### High Priority Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **Community detection quality** | Medium | High | Test on real data early (Week 4) |
| **Performance regression** | Medium | High | Continuous benchmarking |
| **User adoption resistance** | Medium | Medium | Maintain backward compatibility |
| **Timeline slippage** | Medium | Medium | Weekly check-ins, buffer time |

### Medium Priority Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **Neo4j licensing confusion** | Low | Medium | Clear docs on Community vs Enterprise |
| **Memory issues** | Low | Medium | Test with large graphs early |
| **Configuration complexity** | Medium | Low | Good defaults, validation |

### Contingency Plans

**If POC fails (Week 4)**:
- Re-evaluate approach
- Consider Louvain-only (without full Neo4j)
- Abort migration, keep NetworkX

**If performance unacceptable (Week 12)**:
- Profile and optimize queries
- Consider caching strategies
- Extend timeline for optimization

**If user adoption low (Week 18)**:
- Gather detailed feedback
- Address pain points
- Improve documentation
- Consider keeping Parquet as default longer

---

## Resource Plan

### Team Composition

**Phase 1-2** (Weeks 1-10):
- 1 Backend Developer (full-time)
- 0.5 DevOps (part-time, setup)

**Phase 3** (Weeks 11-14):
- 1 Backend Developer (full-time)
- 0.5 Technical Writer (part-time)
- 0.25 QA Engineer (part-time)

**Phase 4** (Weeks 15-20):
- 0.5 Backend Developer (part-time, bug fixes)
- 0.25 Product Manager (part-time)
- 0.25 Support Engineer (part-time)

**Total Effort**: ~12-15 person-weeks

### Infrastructure

**Development**:
- Local Neo4j instances (Docker)
- CI/CD Neo4j containers (GitHub Actions)
- Test data generation

**Testing**:
- Dedicated Neo4j test server
- Load testing environment
- Performance benchmarking server

**Production** (user responsibility):
- Neo4j deployment (Docker/Aura)
- Backup infrastructure
- Monitoring tools

### Budget

**Development** (internal):
- Personnel: ~$30,000-50,000 (12-18 weeks)
- Infrastructure: ~$500 (test servers)

**User Costs** (external):
- Neo4j Aura: $65-200/month (optional)
- Self-hosting: ~$90/month (compute)

---

## Quality Assurance

### Testing Strategy

**Unit Tests** (200+ tests):
- Storage interface implementations
- Individual operations
- Error handling
- Edge cases

**Integration Tests** (50+ tests):
- Full indexing pipeline
- Query operations
- Hybrid mode
- Migration tools

**Performance Tests** (10+ tests):
- Benchmark suite
- Memory profiling
- Scalability tests
- Regression detection

**End-to-End Tests** (20+ tests):
- Real dataset indexing
- Query workflows
- Error recovery
- User scenarios

### Code Review Process

**Requirements**:
- [ ] All PRs reviewed by 2+ developers
- [ ] Tests pass
- [ ] Documentation updated
- [ ] Performance validated
- [ ] Security reviewed

### Release Criteria

**Beta Release**:
- [ ] All Phase 3 tasks complete
- [ ] Critical tests pass
- [ ] Documentation complete
- [ ] No known critical bugs

**Stable Release**:
- [ ] Beta feedback addressed
- [ ] Performance validated
- [ ] User satisfaction > 8/10
- [ ] No known P0/P1 bugs

---

## Communication Plan

### Stakeholder Updates

**Weekly** (during implementation):
- Progress report
- Blockers/risks
- Next week's plan

**Monthly** (during rollout):
- Adoption metrics
- User feedback summary
- Roadmap updates

### User Communication

**Major Milestones**:
- Phase 1 complete: Internal blog post
- Beta release: Announcement, migration guide
- Stable release: Announcement, case studies
- Default change: Migration assistance

**Channels**:
- GitHub releases
- Blog posts
- Discord/Slack announcements
- Email newsletter (if available)

---

## Rollback Plan

### If Major Issues Arise

**Trigger Conditions**:
- Critical bugs affecting data integrity
- Performance > 3x worse than NetworkX
- User satisfaction < 5/10

**Rollback Steps**:
1. Revert default to Parquet
2. Mark Neo4j as "experimental"
3. Fix issues before re-release
4. Communicate clearly to users

**User Impact**:
- Minimal (Parquet still works)
- Users on Neo4j can export to Parquet
- No data loss

---

## Summary

### Timeline Summary

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| **Phase 1: Foundation** | 4 weeks | Storage interface, Neo4j adapter, POC |
| **Phase 2: Core Integration** | 6 weeks | Complete implementation, hybrid mode, tests |
| **Phase 3: Production Readiness** | 4 weeks | Query ops, optimization, docs, tools |
| **Phase 4: Rollout** | 6 weeks | Beta → Stable → Default |
| **Total** | **20 weeks** | **Neo4j migration complete** |

### Effort Summary

- **Development**: 12-15 person-weeks
- **Documentation**: 2-3 person-weeks
- **Testing/QA**: 2-3 person-weeks
- **Total**: **16-21 person-weeks** (~4-5 months for 1 FTE)

### Investment Summary

- **Cost**: $30,000-50,000 (personnel)
- **Timeline**: 4-5 months
- **Risk**: Medium (acceptable with mitigations)
- **Value**: High (performance + new capabilities)

### Next Steps

1. ✅ Get stakeholder approval
2. ✅ Allocate resources (1-2 developers)
3. ✅ Set up development environment
4. 🔲 Begin Phase 1: Foundation (Week 1)

---

**Status**: ✅ Complete
**Next Document**: `07_migration_strategy.md` - User migration approach
