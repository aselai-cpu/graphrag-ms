# Implementation Plan
## Dark Mode Migration Strategy

**Document**: 06 - Implementation Plan
**Date**: 2026-01-31
**Status**: Replanned for Dark Mode

---

## Purpose

This document provides a detailed implementation roadmap for migrating GraphRAG from NetworkX to Neo4j using a **dark mode parallel execution strategy**, including phases, tasks, estimates, dependencies, and success criteria.

---

## Overview

### Total Timeline: 5-6 Months (24 weeks)

**Phase 1: Foundation** (Weeks 1-4) - Storage interface, Neo4j adapter, POC
**Phase 2: Core Integration** (Weeks 5-10) - Complete Neo4j implementation, testing
**Phase 3: Dark Mode Framework** (Weeks 11-14) - Orchestrator, comparison, metrics
**Phase 4: Dark Mode Validation** (Weeks 15-18) - Production validation, 2-4 weeks
**Phase 5: Cutover & Stabilization** (Weeks 19-24) - Switch to Neo4j, monitor

### Resource Requirements

- **Developers**: 1-2 full-time
- **Infrastructure**: Neo4j test environment + production instance
- **Budget**: $40-60K (includes dark mode infrastructure)
  - Development: $30-50K
  - Dark mode framework: +$10K
  - Infrastructure: Minimal (Community Edition free)

### Dark Mode Premium

- **Additional Time**: +20% (4-6 weeks)
- **Additional Cost**: +$10K
- **Risk Reduction**: 80% (from Medium to Low)
- **ROI**: 5-20x on dark mode investment (prevents $50-200K in production issues)

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
import neo4j
from graphrag.config import GraphRagConfig

class Neo4jGraphStorage(GraphStorage):
    """Neo4j-based graph storage implementation."""

    def __init__(self, config: GraphRagConfig):
        """
        Initialize Neo4j storage from settings.yaml configuration.

        Args:
            config: GraphRagConfig object loaded from settings.yaml
        """
        # Extract Neo4j connection params from config
        neo4j_config = config.storage.neo4j

        # Create driver with config values
        self.driver = neo4j.GraphDatabase.driver(
            neo4j_config.uri,
            auth=(neo4j_config.username, neo4j_config.password),
            max_connection_pool_size=neo4j_config.max_connection_pool_size,
            connection_acquisition_timeout=neo4j_config.connection_acquisition_timeout
        )

        self.database = neo4j_config.database
        self.batch_size = neo4j_config.batch_size
        self.gds_enabled = neo4j_config.gds_enabled
        self.vector_index_enabled = neo4j_config.vector_index_enabled

    async def write_entities(self, entities: pd.DataFrame) -> None:
        # Batch create entities with UNWIND using self.batch_size
        ...

    async def calculate_degrees(self) -> None:
        # Use GDS degree algorithm (if self.gds_enabled)
        ...

    async def run_community_detection(...) -> pd.DataFrame:
        # Use GDS Louvain algorithm (if self.gds_enabled)
        ...

    def close(self):
        """Close Neo4j driver connection."""
        self.driver.close()
```

**Configuration Source** (settings.yaml):
```yaml
storage:
  type: neo4j_only
  neo4j:
    uri: "bolt://localhost:7687"           # From settings.yaml
    username: "neo4j"                       # From settings.yaml
    password: "${NEO4J_PASSWORD}"          # From environment variable
    database: "neo4j"                       # From settings.yaml
    batch_size: 1000                        # From settings.yaml
    max_connection_pool_size: 50            # From settings.yaml
    connection_acquisition_timeout: 60      # From settings.yaml
    gds_enabled: true                       # From settings.yaml
    vector_index_enabled: true              # From settings.yaml
```

**Key Methods**:
1. `__init__(config)`: Initialize from settings.yaml configuration
2. Connection management (driver, session) - uses config parameters
3. Entity write (batched with UNWIND) - uses config.batch_size
4. Relationship write (batched) - uses config.batch_size
5. Degree calculation (GDS) - checks config.gds_enabled
6. Community detection (GDS Louvain) - checks config.gds_enabled
7. Read operations (export to DataFrame)
8. `close()`: Cleanup driver connection

**Acceptance Criteria**:
- [ ] All interface methods implemented
- [ ] **Connection parameters read from settings.yaml config** ✅
- [ ] Environment variable substitution works (${NEO4J_PASSWORD})
- [ ] Batching uses config.batch_size
- [ ] GDS operations check config.gds_enabled
- [ ] Error handling robust (connection failures, auth errors)
- [ ] Unit tests written (mocked Neo4j)
- [ ] Integration tests written (real Neo4j with test config)

#### Task 1.6: Proof-of-Concept End-to-End Test (Week 4)
**Estimate**: 3-5 days
**Owner**: Backend developer
**Dependencies**: Task 1.5

**Test Configuration** (settings.yaml):
```yaml
# POC configuration - minimal Neo4j setup
storage:
  type: neo4j_only

  neo4j:
    uri: "bolt://localhost:7687"
    username: "neo4j"
    password: "${NEO4J_PASSWORD}"
    database: "neo4j"
    batch_size: 1000
    gds_enabled: true
```

**Test Scenario**:
```python
# Run mini indexing pipeline with Neo4j
async def test_neo4j_poc():
    # 1. Load configuration from settings.yaml
    from graphrag.config import load_config
    config = load_config("settings.yaml")

    # Verify Neo4j config loaded correctly
    assert config.storage.type == "neo4j_only"
    assert config.storage.neo4j.uri == "bolt://localhost:7687"

    # 2. Load sample documents (Christmas Carol)
    documents = load_sample_documents()

    # 3. Extract entities and relationships
    entities, relationships = await extract_graph(documents, config)

    # 4. Create Neo4j storage from config (not hardcoded!)
    neo4j_storage = Neo4jGraphStorage(config)

    try:
        # 5. Write to Neo4j
        await neo4j_storage.write_entities(entities)
        await neo4j_storage.write_relationships(relationships)

        # 6. Calculate degrees
        await neo4j_storage.calculate_degrees()

        # 7. Run community detection
        communities = await neo4j_storage.run_community_detection()

        # 8. Verify results
        assert len(communities) > 0
        assert communities["community"].nunique() > 1

        print(f"✅ POC Success: {len(entities)} entities, "
              f"{len(relationships)} relationships, "
              f"{communities['community'].nunique()} communities")

    finally:
        # 9. Cleanup
        neo4j_storage.close()
```

**Environment Setup**:
```bash
# Set Neo4j password via environment variable
export NEO4J_PASSWORD=your-secure-password

# Run POC
python test_neo4j_poc.py
```

**Acceptance Criteria**:
- [ ] POC completes successfully
- [ ] **Configuration loaded from settings.yaml** ✅
- [ ] **Environment variable substitution works** ✅
- [ ] Results match Parquet version
- [ ] Community structure reasonable
- [ ] Performance acceptable
- [ ] Connection cleanup works (driver closed)
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
- [ ] NetworkX and Neo4j backends complete
- [ ] Integration tests passing
- [ ] Performance acceptable
- [ ] Ready for dark mode framework

**Risk Assessment**: Medium - Core implementation, must work correctly

---

## Phase 3: Dark Mode Framework (Weeks 11-14)

### Goals
- Build dark mode orchestrator
- Implement comparison framework
- Create metrics collection infrastructure
- Prepare for production validation

### Overview

This phase builds the infrastructure for running NetworkX and Neo4j in parallel with automatic comparison and validation.

### Tasks

#### Task 3.1: Design Dark Mode Architecture (Week 11)
**Estimate**: 2-3 days
**Owner**: Backend developer + Architect
**Dependencies**: Phase 2 complete

**Deliverables**:
1. Execution coordinator design
2. Comparison framework design
3. Metrics collection design
4. Data models and schemas

**Architecture Components**:
```python
# packages/graphrag-storage/graphrag_storage/dark_mode/

├── orchestrator.py           # DarkModeOrchestrator
├── execution_coordinator.py  # ExecutionCoordinator
├── comparison_framework.py   # ComparisonFramework
├── metrics_collector.py      # MetricsCollector
├── models.py                 # Data models
└── report_generator.py       # DarkModeReport
```

**Acceptance Criteria**:
- [ ] Architecture design document complete
- [ ] Component interfaces defined
- [ ] Data models defined
- [ ] Design review approved

#### Task 3.2: Implement Execution Coordinator (Week 11)
**Estimate**: 3-4 days
**Owner**: Backend developer
**Dependencies**: Task 3.1

**Implementation**:
```python
class ExecutionCoordinator:
    """Coordinates parallel execution across backends."""

    async def execute_operation(
        self, operation: str, *args, **kwargs
    ) -> tuple[Any, OperationMetrics]:
        # Execute on primary (NetworkX) - MUST succeed
        primary_result, primary_metrics = await self._execute_primary(...)

        # Execute on shadow (Neo4j) - failures logged but not fatal
        shadow_result, shadow_metrics = await self._execute_shadow(...)

        # Compare results if both succeeded
        if shadow_result is not None:
            comparison = await self._compare(primary_result, shadow_result)

        return primary_result, metrics
```

**Key Features**:
- Async parallel execution
- Primary failures are fatal (propagate to user)
- Shadow failures are logged only (don't affect user)
- Timeout handling for shadow operations
- Metrics collection for both

**Acceptance Criteria**:
- [ ] Coordinator executes operations on both backends
- [ ] Primary failures propagate correctly
- [ ] Shadow failures logged but don't affect primary
- [ ] Metrics collected for both executions
- [ ] Unit tests pass (>90% coverage)

#### Task 3.3: Implement Comparison Framework (Week 11-12)
**Estimate**: 5-7 days
**Owner**: Backend developer
**Dependencies**: Task 3.2

**Comparison Types**:

1. **Entity Comparison**
   ```python
   def compare_entities(
       primary_df: pd.DataFrame,
       shadow_df: pd.DataFrame
   ) -> EntityComparison:
       # Count match
       # ID overlap (precision, recall, F1)
       # Missing/extra entities
       # Attribute differences
   ```

2. **Relationship Comparison**
   ```python
   def compare_relationships(
       primary_df: pd.DataFrame,
       shadow_df: pd.DataFrame
   ) -> RelationshipComparison:
       # Count match
       # Edge overlap
       # Weight differences
   ```

3. **Community Comparison**
   ```python
   def compare_communities(
       primary_df: pd.DataFrame,
       shadow_df: pd.DataFrame
   ) -> CommunityComparison:
       # Match rate (accounting for Louvain variance)
       # Hierarchy depth
       # Cluster size distributions
   ```

4. **Query Comparison**
   ```python
   def compare_query_results(
       primary_results: List[Dict],
       shadow_results: List[Dict]
   ) -> QueryComparison:
       # Result overlap (F1)
       # Ranking correlation
       # Score differences
   ```

**Acceptance Criteria**:
- [ ] All comparison types implemented
- [ ] Statistical metrics computed correctly
- [ ] Handles edge cases (empty results, missing IDs)
- [ ] Unit tests pass (>95% coverage)
- [ ] Performance acceptable (<100ms for typical comparisons)

#### Task 3.4: Implement Metrics Collector (Week 12)
**Estimate**: 3-4 days
**Owner**: Backend developer
**Dependencies**: Task 3.3

**Implementation**:
```python
class MetricsCollector:
    """Collects and persists dark mode metrics."""

    async def log_operation(self, metrics: OperationMetrics):
        # Buffer metrics
        self.buffer.append(metrics)

        # Flush periodically to disk
        if len(self.buffer) >= 100:
            await self._flush_to_disk()

        # Update aggregated stats
        await self._update_aggregates()
```

**Storage Format**:
```jsonl
// dark_mode_logs/comparison_metrics.jsonl
{"operation": "write_entities", "primary_latency": 0.5, "shadow_latency": 0.8, ...}
{"operation": "run_community_detection", "primary_latency": 30.2, "shadow_latency": 5.3, ...}
```

**Aggregated Metrics**:
```json
// dark_mode_logs/aggregated_metrics.json
{
  "validation_period": "2026-02-01 to 2026-02-15",
  "total_operations": 1523,
  "entity_match_rate": 0.998,
  "community_match_rate": 0.963,
  "avg_query_f1": 0.972,
  "avg_latency_ratio": 1.3,
  "shadow_error_rate": 0.002
}
```

**Acceptance Criteria**:
- [ ] Metrics buffered and flushed efficiently
- [ ] JSONL format for detailed logs
- [ ] Aggregated metrics updated in real-time
- [ ] Disk I/O optimized (async, batched)
- [ ] Tests pass

#### Task 3.5: Implement Report Generator (Week 12-13)
**Estimate**: 3-4 days
**Owner**: Backend developer
**Dependencies**: Task 3.4

**Report Features**:
1. **Validation Summary**
   - Validation period
   - Total operations processed
   - Operation breakdown by type

2. **Metric Analysis**
   - Entity/relationship match rates
   - Community match rates
   - Query F1 scores
   - Latency ratios (p50, p95, p99)
   - Error rates

3. **Cutover Readiness**
   - Check against cutover criteria
   - List blocking issues
   - Recommendation (GO/NO-GO)

4. **Visualizations**
   - Latency comparison charts
   - Match rate trends over time
   - Error distribution

**CLI Command**:
```bash
graphrag dark-mode-report \
  --log-path ./dark_mode_logs \
  --output report.html
```

**Acceptance Criteria**:
- [ ] Report generates from logs
- [ ] All metrics displayed
- [ ] Cutover criteria checked
- [ ] Visualizations rendered
- [ ] HTML output works
- [ ] CLI command works

#### Task 3.6: Update Configuration Schema (Week 13)
**Estimate**: 2-3 days
**Owner**: Backend developer
**Dependencies**: Tasks 3.1-3.5

**Purpose**: Define complete configuration schema in settings.yaml for all storage modes, with validation.

**Complete Configuration Schema** (settings.yaml):
```yaml
# GraphRAG Storage Configuration
# All Neo4j connection parameters are specified here

storage:
  type: dark_mode  # Options: networkx_only, neo4j_only, dark_mode

  # NetworkX configuration (for networkx_only and dark_mode)
  networkx:
    enabled: true
    cache_dir: ./cache
    vector_store:
      type: lancedb
      uri: ./output/lancedb

  # Neo4j configuration (for neo4j_only and dark_mode)
  # ALL Neo4j connection parameters read from here
  neo4j:
    enabled: true

    # Connection parameters (required)
    uri: "bolt://localhost:7687"           # Neo4j Bolt URI
    username: "neo4j"                       # Neo4j username
    password: "${NEO4J_PASSWORD}"          # From environment variable
    database: "neo4j"                       # Database name

    # Connection pool settings
    max_connection_pool_size: 50            # Max connections
    connection_acquisition_timeout: 60      # Timeout in seconds

    # Performance settings
    batch_size: 1000                        # Batch size for imports

    # Graph Data Science settings
    gds_enabled: true                       # Enable GDS algorithms
    gds_projection_prefix: "graphrag_"      # GDS projection name prefix
    max_cluster_size: 10                    # Community detection param

    # Vector index settings
    vector_index_enabled: true              # Enable vector indexes
    vector_dimensions: 1536                 # Vector size (384 for SentenceTransformer)
    vector_similarity_function: cosine      # Options: cosine, euclidean

  # Dark mode specific configuration (only when type=dark_mode)
  dark_mode:
    enabled: true
    primary_backend: networkx               # Production backend
    shadow_backend: neo4j                   # Validation backend

    comparison:
      enabled: true
      log_path: ./dark_mode_logs            # Where to save metrics
      log_format: jsonl                     # Format: jsonl or json
      flush_interval_seconds: 10            # How often to flush logs
      metrics:
        - entity_count
        - relationship_count
        - community_match_rate
        - query_f1
        - query_ranking_correlation
        - latency_ratio
        - error_rates

    error_handling:
      shadow_failure_action: log            # Options: log, alert, fail
      continue_on_shadow_error: true        # Don't fail on Neo4j errors

    cutover_criteria:
      validation_period_days: 14            # Min validation period
      min_operations: 1000                  # Min operations to compare
      entity_match_rate_threshold: 0.99     # 99% entity match required
      community_match_rate_threshold: 0.95  # 95% community match required
      query_f1_threshold: 0.95              # 95% query F1 required
      query_ranking_correlation_threshold: 0.90  # 0.90 correlation required
      latency_ratio_threshold: 2.0          # Neo4j < 2x NetworkX latency
      shadow_error_rate_threshold: 0.01     # Neo4j error rate < 1%
```

**Configuration Classes** (Pydantic):
```python
# packages/graphrag/graphrag/config/storage_config.py

from pydantic import BaseModel, Field, validator
from typing import Literal, Optional, List
from enum import Enum

class StorageType(str, Enum):
    NETWORKX_ONLY = "networkx_only"
    NEO4J_ONLY = "neo4j_only"
    DARK_MODE = "dark_mode"

class Neo4jConfig(BaseModel):
    """Neo4j connection and configuration - ALL from settings.yaml."""

    enabled: bool = Field(default=True)

    # Connection parameters (required)
    uri: str = Field(description="Neo4j Bolt URI")
    username: str = Field(default="neo4j")
    password: str = Field(description="Neo4j password (use ${ENV_VAR})")
    database: str = Field(default="neo4j")

    # Connection pool settings
    max_connection_pool_size: int = Field(default=50)
    connection_acquisition_timeout: int = Field(default=60)

    # Performance settings
    batch_size: int = Field(default=1000, ge=100, le=10000)

    # GDS settings
    gds_enabled: bool = Field(default=True)
    gds_projection_prefix: str = Field(default="graphrag_")
    max_cluster_size: int = Field(default=10)

    # Vector index settings
    vector_index_enabled: bool = Field(default=True)
    vector_dimensions: int = Field(default=1536, ge=128, le=4096)
    vector_similarity_function: Literal["cosine", "euclidean"] = Field(default="cosine")

    @validator('uri')
    def validate_uri(cls, v):
        if not v.startswith(('bolt://', 'neo4j://', 'bolt+s://', 'neo4j+s://')):
            raise ValueError('Neo4j URI must start with bolt:// or neo4j://')
        return v

    @validator('password')
    def validate_password(cls, v):
        if not v:
            raise ValueError('Neo4j password is required')
        return v

class DarkModeConfig(BaseModel):
    """Dark mode configuration - from settings.yaml."""

    enabled: bool = Field(default=True)
    primary_backend: Literal["networkx"] = Field(default="networkx")
    shadow_backend: Literal["neo4j"] = Field(default="neo4j")

    # Comparison settings
    comparison: dict = Field(default_factory=dict)

    # Error handling
    error_handling: dict = Field(default_factory=dict)

    # Cutover criteria
    cutover_criteria: dict = Field(default_factory=dict)

class StorageConfig(BaseModel):
    """Storage configuration - ALL from settings.yaml."""

    type: StorageType = Field(default=StorageType.NETWORKX_ONLY)

    # Backend configs
    networkx: Optional[dict] = None
    neo4j: Optional[Neo4jConfig] = None

    # Dark mode config
    dark_mode: Optional[DarkModeConfig] = None

    @validator('neo4j')
    def validate_neo4j_config(cls, v, values):
        """Ensure Neo4j config exists when needed."""
        storage_type = values.get('type')
        if storage_type in [StorageType.NEO4J_ONLY, StorageType.DARK_MODE]:
            if not v or not v.enabled:
                raise ValueError(f'Neo4j config required for storage type: {storage_type}')
        return v
```

**Configuration Validation**:
```python
# Load and validate config from settings.yaml
def load_and_validate_config(config_path: str) -> GraphRagConfig:
    """Load config from settings.yaml and validate."""

    # Load YAML
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    # Substitute environment variables
    config_dict = substitute_env_vars(config_dict)

    # Parse and validate with Pydantic
    try:
        config = GraphRagConfig(**config_dict)
        return config
    except ValidationError as e:
        print(f"❌ Configuration validation failed:")
        for error in e.errors():
            print(f"  - {'.'.join(str(x) for x in error['loc'])}: {error['msg']}")
        raise
```

**Acceptance Criteria**:
- [ ] **Complete configuration schema defined in settings.yaml** ✅
- [ ] **All Neo4j connection parameters configurable** ✅
- [ ] **Environment variable substitution works (${NEO4J_PASSWORD})** ✅
- [ ] Pydantic models validate all fields
- [ ] Validation errors provide clear messages
- [ ] Defaults sensible for each mode
- [ ] Configuration documentation written
- [ ] Example settings.yaml files provided for each mode:
  - [ ] `examples/settings.networkx_only.yaml`
  - [ ] `examples/settings.neo4j_only.yaml`
  - [ ] `examples/settings.dark_mode.yaml`

#### Task 3.7: Integration with Storage Factory (Week 13)
**Estimate**: 2-3 days
**Owner**: Backend developer
**Dependencies**: Task 3.6

**Implementation**:
```python
# packages/graphrag-storage/graphrag_storage/factory.py

from graphrag.config import GraphRagConfig
from graphrag_storage.graph_storage import GraphStorage
from graphrag_storage.networkx_graph_storage import NetworkXGraphStorage
from graphrag_storage.neo4j_graph_storage import Neo4jGraphStorage
from graphrag_storage.dark_mode import DarkModeGraphStorage, DarkModeOrchestrator

def create_graph_storage(config: GraphRagConfig) -> GraphStorage:
    """
    Create appropriate graph storage backend from settings.yaml configuration.

    Args:
        config: GraphRagConfig loaded from settings.yaml

    Returns:
        GraphStorage instance configured from settings.yaml
    """

    if config.storage.type == "networkx_only":
        # NetworkX storage - read config from settings.yaml
        return NetworkXGraphStorage(config)

    elif config.storage.type == "neo4j_only":
        # Neo4j storage - ALL connection params from settings.yaml
        return Neo4jGraphStorage(config)

    elif config.storage.type == "dark_mode":
        # Dark mode - create both backends from settings.yaml

        # Create NetworkX backend with config
        networkx = NetworkXGraphStorage(config)

        # Create Neo4j backend with config (uri, auth, etc. from settings.yaml)
        neo4j = Neo4jGraphStorage(config)

        # Create orchestrator with dark mode config from settings.yaml
        orchestrator = DarkModeOrchestrator(
            primary=networkx,
            shadow=neo4j,
            config=config.storage.dark_mode  # From settings.yaml
        )

        return DarkModeGraphStorage(orchestrator)

    else:
        raise ValueError(f"Unknown storage type: {config.storage.type}")
```

**Configuration Flow**:
```
settings.yaml
     ↓
GraphRagConfig (loaded via load_config)
     ↓
create_graph_storage(config)
     ↓
Neo4jGraphStorage(config)
     ↓
Reads: uri, username, password, database, batch_size, etc.
     ↓
neo4j.GraphDatabase.driver(config.neo4j.uri, auth=(...))
```

**Usage Example**:
```python
# In main.py or workflow
from graphrag.config import load_config
from graphrag_storage.factory import create_graph_storage

# Load config from settings.yaml
config = load_config("settings.yaml")

# Create storage from config - NO hardcoded values
storage = create_graph_storage(config)

# Use storage
await storage.write_entities(entities)

# Cleanup
if hasattr(storage, 'close'):
    storage.close()
```

**Acceptance Criteria**:
- [ ] Factory creates all three storage types from config
- [ ] **NetworkX backend reads config from settings.yaml** ✅
- [ ] **Neo4j backend reads ALL connection params from settings.yaml** ✅
- [ ] **Dark mode orchestrator reads config from settings.yaml** ✅
- [ ] No hardcoded connection strings or credentials
- [ ] Environment variable substitution works (${NEO4J_PASSWORD})
- [ ] Factory validates config before creating storage
- [ ] Tests pass with different config files

#### Task 3.8: Dark Mode Testing (Week 13-14)
**Estimate**: 5-7 days
**Owner**: Backend developer + QA
**Dependencies**: All Phase 3 tasks

**Test Suites**:

1. **Orchestrator Tests** (10 tests)
   - Primary success, shadow success
   - Primary success, shadow fail
   - Primary fail (should propagate)
   - Parallel execution works
   - Timeout handling

2. **Comparison Tests** (20 tests)
   - Entity comparison accuracy
   - Community comparison accuracy
   - Query comparison accuracy
   - Edge cases (empty, missing data)
   - Performance benchmarks

3. **Metrics Collection Tests** (8 tests)
   - Buffering works
   - Flushing works
   - Aggregation accurate
   - Disk I/O efficient

4. **End-to-End Tests** (5 tests)
   - Full indexing pipeline in dark mode
   - Query operations in dark mode
   - Report generation
   - Configuration modes
   - Rollback scenarios

**Acceptance Criteria**:
- [ ] All tests pass
- [ ] Test coverage > 85%
- [ ] Performance acceptable (dark mode overhead < 20%)
- [ ] Memory usage reasonable
- [ ] Code review approved

### Phase 3 Milestones

**Week 12 Checkpoint**:
- [ ] Orchestrator working
- [ ] Comparison framework complete
- [ ] Metrics collection working

**Week 14 Completion**:
- [ ] Dark mode framework fully implemented
- [ ] All tests passing
- [ ] Documentation complete
- [ ] Ready for production validation

**Risk Assessment**: Medium-High - New infrastructure, must be reliable

---

## Phase 4: Dark Mode Validation (Weeks 15-18)

### Goals
- Enable dark mode in production
- Collect 2-4 weeks of real validation data
- Analyze metrics against cutover criteria
- Build confidence for cutover decision

### Overview

This phase runs dark mode in the production environment with real user traffic. Neo4j runs in parallel but doesn't affect user results. We collect comprehensive comparison data to validate Neo4j is ready for cutover.

### Tasks

#### Task 4.1: Production Environment Setup (Week 15)
**Estimate**: 3-4 days
**Owner**: DevOps + Backend developer
**Dependencies**: Phase 3 complete

**Infrastructure**:
1. Deploy Neo4j instance
   - Docker or Neo4j Aura
   - Production sizing (based on data volume)
   - Backup configuration

2. Configure dark mode
   - Enable dark_mode in settings.yaml
   - Set up log storage
   - Configure metrics collection

3. Monitoring setup
   - Neo4j metrics (CPU, memory, disk)
   - Dark mode metrics dashboard
   - Alerts for errors

**Acceptance Criteria**:
- [ ] Neo4j deployed in production
- [ ] Dark mode enabled
- [ ] Monitoring configured
- [ ] Alerts working
- [ ] Runbook documented

#### Task 4.2: Enable Dark Mode (Week 15)
**Estimate**: 1 day
**Owner**: Backend developer + DevOps
**Dependencies**: Task 4.1

**Steps**:
1. Update configuration:
   ```yaml
   storage:
     type: dark_mode
   ```

2. Deploy configuration change

3. Verify dark mode active:
   ```bash
   # Check logs show both backends executing
   tail -f logs/graphrag.log | grep "dark_mode"

   # Check metrics being collected
   ls dark_mode_logs/comparison_metrics.jsonl
   ```

4. Monitor for errors

**Acceptance Criteria**:
- [ ] Dark mode active in production
- [ ] Both backends executing
- [ ] Metrics being collected
- [ ] No user-facing errors
- [ ] Performance acceptable

#### Task 4.3: Daily Monitoring (Weeks 15-18)
**Estimate**: 30 min/day × 20 days = 10 hours
**Owner**: Backend developer
**Dependencies**: Task 4.2

**Daily Tasks**:
1. Check dashboard
   - Entity match rates
   - Community match rates
   - Query F1 scores
   - Latency ratios
   - Error rates

2. Review error logs
   - Neo4j failures
   - Comparison discrepancies
   - Performance issues

3. Generate daily report:
   ```bash
   graphrag dark-mode-report --daily
   ```

4. Document issues
   - Any metric below threshold
   - Recurring errors
   - Performance anomalies

**Acceptance Criteria**:
- [ ] Daily monitoring performed
- [ ] Issues documented
- [ ] Trends tracked
- [ ] Stakeholders updated weekly

#### Task 4.4: Issue Resolution (Weeks 15-18)
**Estimate**: Variable (budget 5 days)
**Owner**: Backend developer
**Dependencies**: Task 4.3

**Common Issues**:
1. **Community match rate < 95%**
   - Investigate Louvain algorithm variance
   - Check random seed handling
   - Verify GDS configuration

2. **Query F1 < 95%**
   - Compare vector index results
   - Check embedding storage
   - Verify query logic

3. **Latency ratio > 2x**
   - Optimize Neo4j queries
   - Tune batch sizes
   - Check network latency

4. **Error rate > 1%**
   - Fix Neo4j connection issues
   - Handle edge cases
   - Improve error handling

**Acceptance Criteria**:
- [ ] Critical issues resolved
- [ ] Metrics meet cutover criteria
- [ ] Root causes documented
- [ ] Fixes tested

#### Task 4.5: Metrics Analysis (Week 18)
**Estimate**: 2-3 days
**Owner**: Backend developer + Architect
**Dependencies**: 2-4 weeks of validation data

**Analysis Steps**:
1. Generate comprehensive report:
   ```bash
   graphrag dark-mode-report \
     --start-date 2026-02-01 \
     --end-date 2026-02-28 \
     --output final_validation_report.html
   ```

2. Check cutover criteria:
   ```yaml
   Cutover Criteria               | Target  | Actual  | Pass
   -------------------------------|---------|---------|-----
   Entity match rate              | > 99%   | 99.8%   | ✅
   Relationship match rate        | > 99%   | 99.7%   | ✅
   Community match rate           | > 95%   | 96.3%   | ✅
   Avg query F1                   | > 95%   | 97.2%   | ✅
   Avg ranking correlation        | > 0.90  | 0.94    | ✅
   Latency ratio (p95)            | < 2.0x  | 1.3x    | ✅
   Shadow error rate              | < 1%    | 0.2%    | ✅
   Min operations                 | > 1000  | 2847    | ✅
   Validation period              | > 14d   | 28d     | ✅
   ```

3. Document findings
   - Summary of validation period
   - Key metrics
   - Issues encountered and resolved
   - Recommendation (GO/NO-GO for cutover)

**GO Decision Criteria**:
- ✅ ALL cutover criteria met
- ✅ No critical blocking issues
- ✅ Team confidence high
- ✅ Rollback plan tested

**Acceptance Criteria**:
- [ ] Comprehensive report generated
- [ ] Cutover criteria evaluated
- [ ] Recommendation documented
- [ ] Stakeholder review scheduled

#### Task 4.6: Cutover Decision (Week 18)
**Estimate**: 1 day
**Owner**: Engineering leadership + Team
**Dependencies**: Task 4.5

**Decision Meeting**:
1. Present validation report
2. Review metrics against criteria
3. Discuss any concerns
4. Make GO/NO-GO decision
5. If GO: Schedule cutover
6. If NO-GO: Document blockers, continue validation

**Acceptance Criteria**:
- [ ] Decision meeting held
- [ ] Decision documented
- [ ] If GO: Cutover date scheduled
- [ ] If NO-GO: Action plan created

### Phase 4 Milestones

**Week 16 Checkpoint**:
- [ ] Dark mode running in production for 1 week
- [ ] No critical issues
- [ ] Metrics trending positive

**Week 18 Completion**:
- [ ] 2-4 weeks validation data collected
- [ ] Metrics analyzed
- [ ] GO/NO-GO decision made
- [ ] Ready for cutover (if GO)

**Risk Assessment**: Low - Dark mode doesn't affect production, can be disabled anytime

---

## Phase 5: Cutover & Stabilization (Weeks 19-24)

### Goals
- Switch from dark_mode to neo4j_only
- Monitor for regressions
- Optimize performance
- Update documentation
- Stable release

### Overview

After successful dark mode validation, this phase switches production to Neo4j-only mode. NetworkX remains available as a fallback, but Neo4j becomes the primary system.

### Tasks

#### Task 5.1: Pre-Cutover Checklist (Week 19)
**Estimate**: 1-2 days
**Owner**: Backend developer + DevOps
**Dependencies**: Phase 4 GO decision

**Checklist**:
- [ ] All cutover criteria met
- [ ] Dark mode validation report approved
- [ ] Neo4j performance tuned
- [ ] Rollback plan documented and tested
- [ ] Monitoring dashboards ready
- [ ] On-call rotation scheduled
- [ ] Stakeholders notified of cutover date
- [ ] Backup of NetworkX data taken

**Acceptance Criteria**:
- [ ] All checklist items complete
- [ ] Team ready for cutover
- [ ] Rollback plan tested

#### Task 5.2: Cutover Execution (Week 19)
**Estimate**: 1 day
**Owner**: Backend developer + DevOps
**Dependencies**: Task 5.1

**Cutover Steps**:

1. **Backup current state**
   ```bash
   # Backup NetworkX/LanceDB data
   cp -r output/ output_backup_$(date +%Y%m%d)/
   cp -r cache/ cache_backup_$(date +%Y%m%d)/
   ```

2. **Update configuration**
   ```yaml
   storage:
     type: neo4j_only  # Was: dark_mode
     neo4j:
       uri: bolt://production-neo4j:7687
       username: neo4j
       password: ${NEO4J_PASSWORD}
   ```

3. **Deploy configuration change**
   ```bash
   # Deploy new config
   kubectl apply -f graphrag-config.yaml

   # Rolling restart
   kubectl rollout restart deployment/graphrag
   ```

4. **Verify cutover**
   ```bash
   # Check logs show Neo4j only
   kubectl logs -f deployment/graphrag | grep "storage_type"
   # Should show: storage_type=neo4j_only

   # Run smoke test
   graphrag query --method local "test query"
   ```

5. **Monitor closely**
   - Watch error rates
   - Check latency metrics
   - Monitor Neo4j resource usage

**Rollback Procedure** (if needed):
```bash
# Revert configuration
storage:
  type: networkx_only  # Instant rollback

# Deploy
kubectl apply -f graphrag-config-rollback.yaml
kubectl rollout restart deployment/graphrag

# Verify
kubectl logs -f deployment/graphrag | grep "storage_type"
# Should show: storage_type=networkx_only
```

**Acceptance Criteria**:
- [ ] Configuration deployed
- [ ] Neo4j only mode active
- [ ] Smoke tests pass
- [ ] No errors in logs
- [ ] Metrics nominal

#### Task 5.3: Post-Cutover Monitoring (Week 19-20)
**Estimate**: 2 hours/day × 10 days = 20 hours
**Owner**: Backend developer + DevOps
**Dependencies**: Task 5.2

**Monitoring Focus**:

1. **Day 1-2: Intensive monitoring**
   - Check dashboards every 2 hours
   - Review error logs continuously
   - Monitor latency (p50, p95, p99)
   - Track resource usage (CPU, memory, disk)

2. **Day 3-5: Active monitoring**
   - Check dashboards every 4 hours
   - Daily error log review
   - Track trends

3. **Day 6-10: Normal monitoring**
   - Daily dashboard check
   - Weekly trends analysis

**Key Metrics**:
- Query latency (should be < baseline * 1.5)
- Error rate (should be < 0.1%)
- Neo4j resource usage
- User-reported issues

**Acceptance Criteria**:
- [ ] No critical issues
- [ ] Performance within expected range
- [ ] Error rate acceptable
- [ ] User feedback positive

#### Task 5.4: Performance Optimization (Week 20-21)
**Estimate**: 5-7 days
**Owner**: Backend developer
**Dependencies**: Task 5.3

**Optimization Areas**:

1. **Query Optimization**
   - Profile slow queries
   - Add query hints
   - Optimize Cypher patterns
   - Use indexes effectively

2. **Neo4j Tuning**
   - Tune heap size
   - Configure page cache
   - GDS memory settings
   - Connection pool sizing

3. **Batch Size Tuning**
   - Test different batch sizes
   - Find optimal for workload

4. **Caching Strategy**
   - Identify frequently accessed data
   - Implement query result caching
   - Cache community reports

**Benchmarking**:
```bash
# Run performance suite
pytest tests/performance/ --benchmark --neo4j-only

# Compare with baseline
./scripts/compare_performance.sh \
  baseline_metrics.json \
  current_metrics.json
```

**Acceptance Criteria**:
- [ ] Query latency < 100ms (p95)
- [ ] Community detection ≤ NetworkX speed (ideally 6x faster)
- [ ] Memory usage reasonable
- [ ] Optimization gains documented

#### Task 5.5: Query Operations Update (Week 21)
**Estimate**: 3-5 days
**Owner**: Backend developer
**Dependencies**: Phase 2 complete

**Query Methods**:
1. Global Search (Neo4j vector index)
2. Local Search (hybrid: vector + graph traversal)
3. DRIFT Search
4. Basic Search (vector-only)

**New Capabilities** (Neo4j-specific):
```python
# Hybrid query: vector similarity + graph connectivity
async def hybrid_search(query: str, anchor_entity: str):
    """Find similar entities connected to anchor."""
    query_embedding = await embed(query)

    results = neo4j_session.run("""
        MATCH (anchor:Entity {title: $anchor})
        CALL db.index.vector.queryNodes(
            'entity_description_vector', 100, $embedding
        )
        YIELD node, score
        WHERE EXISTS {
            MATCH (anchor)-[:RELATED_TO*1..3]-(node)
        }
        RETURN node, score, shortestPath((anchor)-[:RELATED_TO*]-(node))
        ORDER BY score DESC
        LIMIT 10
    """, anchor=anchor_entity, embedding=query_embedding)
```

**Acceptance Criteria**:
- [ ] All query methods work with Neo4j
- [ ] Hybrid queries implemented
- [ ] Performance comparable or better
- [ ] Documentation updated
- [ ] Tests pass

#### Task 5.6: Documentation Updates (Week 21-22)
**Estimate**: 5-7 days
**Owner**: Technical writer + Developer
**Dependencies**: Cutover complete

**Documentation Deliverables**:

1. **User Guide** (`docs/neo4j/user_guide.md`)
   - Why Neo4j
   - Configuration guide
   - Query examples
   - Performance tuning
   - Troubleshooting

2. **Migration Guide** (`docs/neo4j/migration_guide.md`)
   - NetworkX → Neo4j migration
   - Dark mode validation process
   - Cutover procedure
   - Rollback procedure

3. **API Documentation**
   - Updated with Neo4j-specific methods
   - Hybrid query examples
   - Code samples

4. **Operations Guide** (`docs/neo4j/operations.md`)
   - Deployment (Docker, Aura)
   - Monitoring setup
   - Backup/restore procedures
   - Scaling guidelines
   - Troubleshooting common issues

5. **Dark Mode Guide** (`docs/neo4j/dark_mode.md`)
   - What is dark mode
   - When to use it
   - Configuration
   - Metrics interpretation
   - Best practices

**Acceptance Criteria**:
- [ ] All documentation complete
- [ ] Code examples tested
- [ ] Screenshots/diagrams included
- [ ] Reviewed and approved
- [ ] Published

#### Task 5.7: Release Preparation (Week 22-23)
**Estimate**: 5-7 days
**Owner**: Release manager + Team
**Dependencies**: All Phase 5 tasks

**Release Tasks**:

1. **Version Bump**
   - Update to v3.1.0
   - Update CHANGELOG.md

2. **Release Notes**
   - Neo4j integration highlights
   - Dark mode feature
   - Breaking changes (none - backward compatible)
   - Migration guide link
   - Performance improvements

3. **Backward Compatibility**
   - Ensure networkx_only still works
   - Provide clear migration path
   - Document support timeline

4. **Testing**
   - Full regression test suite
   - Integration tests
   - Performance tests
   - User acceptance testing

5. **Communication**
   - Blog post
   - Release announcement
   - User migration timeline

**Acceptance Criteria**:
- [ ] All tests pass
- [ ] Release notes complete
- [ ] Backward compatibility verified
- [ ] Communication materials ready
- [ ] Release approved

#### Task 5.8: Stable Release (Week 23-24)
**Estimate**: 3-5 days
**Owner**: Release manager
**Dependencies**: Task 5.7

**Release Steps**:

1. **Tag release**
   ```bash
   git tag -a v3.1.0 -m "Neo4j integration with dark mode validation"
   git push origin v3.1.0
   ```

2. **Publish packages**
   ```bash
   # Publish to PyPI
   python -m build
   twine upload dist/*
   ```

3. **Update documentation**
   - Publish updated docs
   - Update website

4. **Announce release**
   - GitHub release notes
   - Blog post
   - Social media
   - User mailing list

5. **Monitor adoption**
   - Track downloads
   - Monitor issue reports
   - Collect user feedback

**Acceptance Criteria**:
- [ ] Release published
- [ ] Documentation live
- [ ] Announcement sent
- [ ] Initial feedback positive

#### Task 5.9: Post-Release Support (Week 24+)
**Estimate**: Ongoing
**Owner**: Support team + Developers
**Dependencies**: Task 5.8

**Support Activities**:

1. **Issue Triage**
   - Monitor GitHub issues
   - Prioritize Neo4j-related issues
   - Respond within 24 hours

2. **User Support**
   - Answer questions
   - Help with migration
   - Troubleshoot issues

3. **Bug Fixes**
   - Fix critical bugs immediately
   - Plan patches for minor issues

4. **Feedback Collection**
   - User surveys
   - Usage analytics
   - Performance data

**Acceptance Criteria**:
- [ ] Support process established
- [ ] Issues responded to promptly
- [ ] User satisfaction high

### Phase 5 Milestones

**Week 19 Completion**:
- [ ] Cutover executed successfully
- [ ] Neo4j only mode active
- [ ] No critical issues

**Week 21 Checkpoint**:
- [ ] Performance optimized
- [ ] Query operations updated
- [ ] Stability proven

**Week 24 Completion**:
- [ ] v3.1.0 released
- [ ] Documentation complete
- [ ] User adoption starting
- [ ] Support process established

**Risk Assessment**: Low-Medium - Rollback available, dark mode provided confidence

---

## Summary

### Timeline Summary (Dark Mode Strategy)

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| **Phase 1: Foundation** | 4 weeks | Storage interface with mode support, Neo4j adapter, POC |
| **Phase 2: Core Integration** | 6 weeks | Complete Neo4j implementation, all workflows, tests |
| **Phase 3: Dark Mode Framework** | 4 weeks | Orchestrator, comparison framework, metrics collection |
| **Phase 4: Dark Mode Validation** | 4 weeks | Production validation, 2-4 weeks data collection, GO/NO-GO decision |
| **Phase 5: Cutover & Stabilization** | 6 weeks | Cutover, monitoring, optimization, stable release |
| **Total** | **24 weeks** | **Neo4j migration complete (risk-free)** |

### Effort Summary (Updated for Dark Mode)

**Development Effort**:
- Phase 1 (Foundation): 3-4 person-weeks
- Phase 2 (Core Integration): 6-7 person-weeks
- Phase 3 (Dark Mode Framework): 4-5 person-weeks
- Phase 4 (Dark Mode Validation): 1-2 person-weeks (mostly monitoring)
- Phase 5 (Cutover & Stabilization): 5-6 person-weeks
- **Total Development**: **19-24 person-weeks**

**Additional Effort**:
- Documentation: 2-3 person-weeks
- Testing/QA: 3-4 person-weeks
- **Total Effort**: **24-31 person-weeks** (~5-6 months for 1 FTE, ~3-4 months for 2 FTEs)

### Investment Summary (Dark Mode Strategy)

**Development Cost**:
- Core implementation (Phases 1-2): $30,000-40,000
- Dark mode infrastructure (Phase 3): $10,000-12,000
- Validation & cutover (Phases 4-5): $5,000-8,000
- **Total Development**: $45,000-60,000

**Operational Cost**:
- Neo4j Community Edition: Free
- Neo4j Aura (cloud): $65-200/month
- Dark mode validation (2-4 weeks): +$500-1,000 (temporary compute)
- **Net Increase**: $0-150/month (replaces separate vector store)

**Return on Investment**:
- Dark mode premium: +$10K
- Prevents production issues: $50-200K savings
- **ROI on dark mode**: 5-20x
- Overall timeline: 3-5 years positive ROI (primary value in new capabilities)

**Risk Level**: **Low** ✅
- Dark mode eliminates cutover risk (reduced from Medium → Low)
- Full validation before production impact
- Instant rollback capability
- Worth the 20% premium

### Configuration Management (settings.yaml)

**Core Principle**: All Neo4j connection parameters and settings come from `settings.yaml` configuration file. No hardcoded values.

**Configuration Sources**:
```
settings.yaml (YAML file)
     ↓
Environment variable substitution (${NEO4J_PASSWORD})
     ↓
GraphRagConfig (Pydantic model with validation)
     ↓
Neo4jGraphStorage(config) - receives all connection params
     ↓
neo4j.GraphDatabase.driver(config.neo4j.uri, auth=(...))
```

**Complete Neo4j Configuration from settings.yaml**:
```yaml
storage:
  neo4j:
    # Connection (required)
    uri: "bolt://localhost:7687"           # From settings.yaml
    username: "neo4j"                       # From settings.yaml
    password: "${NEO4J_PASSWORD}"          # From environment variable
    database: "neo4j"                       # From settings.yaml

    # Connection pool
    max_connection_pool_size: 50            # From settings.yaml
    connection_acquisition_timeout: 60      # From settings.yaml

    # Performance
    batch_size: 1000                        # From settings.yaml

    # GDS
    gds_enabled: true                       # From settings.yaml
    max_cluster_size: 10                    # From settings.yaml

    # Vector index
    vector_index_enabled: true              # From settings.yaml
    vector_dimensions: 1536                 # From settings.yaml
    vector_similarity_function: cosine      # From settings.yaml
```

**Benefits of Configuration-Driven Approach**:
1. ✅ **Security**: Credentials in environment variables, not code
2. ✅ **Flexibility**: Change connection without code changes
3. ✅ **Multi-environment**: Different settings.yaml for dev/staging/prod
4. ✅ **Validation**: Pydantic validates all parameters on load
5. ✅ **Documentation**: Settings.yaml serves as configuration reference
6. ✅ **Testing**: Easy to provide test configurations

**Implementation Examples**:

1. **Phase 1 (Task 1.5)**: Neo4jGraphStorage reads config
```python
class Neo4jGraphStorage(GraphStorage):
    def __init__(self, config: GraphRagConfig):
        # ALL connection params from config (loaded from settings.yaml)
        self.driver = neo4j.GraphDatabase.driver(
            config.storage.neo4j.uri,           # From settings.yaml
            auth=(
                config.storage.neo4j.username,   # From settings.yaml
                config.storage.neo4j.password    # From ${NEO4J_PASSWORD}
            ),
            max_connection_pool_size=config.storage.neo4j.max_connection_pool_size
        )
```

2. **Phase 3 (Task 3.7)**: Storage factory passes config
```python
def create_graph_storage(config: GraphRagConfig) -> GraphStorage:
    if config.storage.type == "neo4j_only":
        # Pass entire config, Neo4jGraphStorage extracts what it needs
        return Neo4jGraphStorage(config)
```

3. **Phase 4 (Task 4.1)**: Production deployment
```bash
# Production settings.yaml
storage:
  type: dark_mode
  neo4j:
    uri: "bolt://neo4j-prod.company.com:7687"  # Production URI
    password: "${NEO4J_PASSWORD}"               # From k8s secret

# Deploy
kubectl create secret generic graphrag-secrets \
  --from-literal=NEO4J_PASSWORD=<secure-password>
```

**Validation Example**:
```python
# Config validation catches errors early
try:
    config = load_config("settings.yaml")
except ValidationError as e:
    print("❌ Configuration errors:")
    print("  - neo4j.uri: URI must start with bolt:// or neo4j://")
    print("  - neo4j.password: Password is required")
    sys.exit(1)
```

### Comparison: Traditional vs Dark Mode Migration

| Aspect | Traditional | Dark Mode | Winner |
|--------|-------------|-----------|--------|
| **Timeline** | 16-20 weeks | 24 weeks | Traditional (faster) ⚠️ |
| **Cost** | $30-50K | $45-60K | Traditional (cheaper) ⚠️ |
| **Risk** | Medium | Low | **Dark Mode** ✅ |
| **Validation** | Sample testing | 100% real traffic | **Dark Mode** ✅ |
| **Confidence** | Moderate | High | **Dark Mode** ✅ |
| **Rollback** | Complex | Instant | **Dark Mode** ✅ |
| **Production Impact** | Possible | Zero (validated) | **Dark Mode** ✅ |
| **Issue Detection** | After cutover | Before cutover | **Dark Mode** ✅ |

**Verdict**: Dark mode strategy wins 6/8 categories. The 20% extra time and cost are worth the 80% risk reduction.

### Key Success Factors

**Technical**:
- ✅ Abstract storage interface cleanly separates backends
- ✅ Dark mode orchestrator handles parallel execution
- ✅ Comprehensive comparison framework validates correctness
- ✅ Neo4j GDS provides feature parity with NetworkX
- ✅ Vector index unifies graph and vector storage

**Process**:
- ✅ Phased approach reduces risk
- ✅ Dark mode enables full validation before impact
- ✅ Continuous testing throughout development
- ✅ Clear cutover criteria with objective metrics
- ✅ Easy rollback at any stage

**Organizational**:
- ✅ Stakeholder buy-in for 5-6 month project
- ✅ Resource allocation (1-2 developers)
- ✅ DevOps support for infrastructure
- ✅ User communication plan
- ✅ Post-release support commitment

### Critical Dependencies

**External**:
- Neo4j 5.17+ with GDS 2.6+
- Python neo4j driver 5.x
- Docker (for deployment)

**Internal**:
- GraphRAG v3.0+ codebase
- Claude 4.5 Sonnet + SentenceTransformer embeddings
- Existing test infrastructure
- CI/CD pipeline

**Infrastructure**:
- Neo4j test environment (Phase 1-3)
- Neo4j production instance (Phase 4-5)
- Dark mode log storage (~10-50GB)
- Monitoring dashboards

### Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Dark mode overhead too high | Low | Medium | Optimize orchestrator, acceptable during validation |
| Metrics don't meet criteria | Medium | High | Extend validation period, fix issues before cutover |
| Neo4j performance worse | Low | High | Detected in Phase 4, optimize or abort |
| Community detection differs | Medium | Medium | Louvain variance expected, validate > 95% match |
| Rollback needed after cutover | Low | Medium | Instant config change, zero data loss |
| Timeline slippage | Medium | Low | Buffer time included, weekly check-ins |

**Overall Risk**: **Low** with dark mode strategy (vs Medium for traditional)

### Decision Points

**Phase 1 (Week 4)**:
- **Decision**: GO/NO-GO for Phase 2
- **Criteria**: POC successful, storage interface working

**Phase 4 (Week 18)**:
- **Decision**: GO/NO-GO for cutover
- **Criteria**: All cutover metrics met, no blocking issues

**Phase 5 (Post-cutover)**:
- **Decision**: Rollback or continue
- **Criteria**: Performance acceptable, error rate < threshold

### Communication Plan

**Internal**:
- Weekly status updates to stakeholders
- Phase completion demos
- Dark mode validation report (Week 18)
- Post-cutover retrospective

**External**:
- Blog post: "Introducing Dark Mode Migration" (Phase 3 complete)
- Blog post: "Neo4j Integration Now Available" (v3.1.0 release)
- Documentation updates throughout
- User migration guide

**Timing**:
- Week 14: Announce dark mode validation starting
- Week 18: Share validation results
- Week 23: v3.1.0 release announcement
- Week 24+: Case studies, user feedback

### Next Steps

**Immediate** (This Week):
1. ⏳ Review this dark mode implementation plan
2. ⏳ Get stakeholder approval for 5-6 month project
3. ⏳ Allocate budget ($45-60K)
4. ⏳ Assign resources (1-2 developers)
5. ⏳ Set up project tracking

**Short-term** (Month 1):
1. ⏳ Set up development environment
2. ⏳ Begin Phase 1: Foundation
3. ⏳ Weekly check-ins established
4. ⏳ Technical design reviews scheduled

**Medium-term** (Months 2-4):
1. ⏳ Complete Phases 2-3
2. ⏳ Prepare for dark mode validation
3. ⏳ Production environment setup

**Long-term** (Months 5-6):
1. ⏳ Dark mode validation
2. ⏳ Cutover execution
3. ⏳ v3.1.0 release
4. ⏳ User adoption and support

---

## Appendix: Lessons from Claude Migration

### What Worked Well

From the recent Claude 4.5 Sonnet + SentenceTransformer migration, we learned:

1. **POC First**: Quick POC (2 hours) validated approach before analysis
2. **Incremental Commits**: Version-tagged commits (v3.1.0) for rollback safety
3. **Post-Mortem**: Reflection document captured learnings

### What We'll Do Differently

1. **Dark Mode Instead of Direct Cutover**: Learned from Claude migration that validation is critical
2. **Longer Validation Period**: 2-4 weeks vs immediate cutover
3. **Objective Metrics**: Cutover criteria instead of subjective assessment
4. **Automated Comparison**: Framework vs manual checking

### Applying Learnings

- ✅ POC in Phase 1 (Week 4)
- ✅ Dark mode validation (Phase 4)
- ✅ Objective cutover criteria
- ✅ Post-mortem planned (end of Phase 5)

---

**Status**: ✅ Complete - Dark Mode Strategy
**Date**: 2026-01-31
**Next Document**: `07_migration_strategy.md` - User migration approach with dark mode

