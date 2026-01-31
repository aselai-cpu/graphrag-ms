# Neo4j Integration Architecture Design
## Dark Mode Parallel Execution Strategy

**Document**: 03 - Architecture Design
**Date**: 2026-01-31
**Status**: Replanned for Dark Mode

---

## Purpose

This document designs the integration architecture for Neo4j as GraphRAG's unified graph and vector storage system using a **dark mode parallel execution strategy**. It defines:
- Three execution modes: networkx_only, dark_mode, neo4j_only
- Dark mode orchestrator architecture
- Comparison framework for validation
- Schema, data flow, and integration points
- Safe cutover strategy

---

## Design Principles

### 1. **Zero Production Risk (Dark Mode First)**
- Production system (NetworkX) remains authoritative during validation
- Shadow system (Neo4j) runs in parallel but doesn't affect results
- Failures in Neo4j don't impact users
- Full validation before cutover

### 2. **Configuration-Based Execution Modes**
- `networkx_only`: Current implementation
- `dark_mode`: Both run in parallel, NetworkX results returned
- `neo4j_only`: Neo4j only (target state)
- Single config change to switch modes

### 3. **Comprehensive Comparison**
- Log all operations (indexing + queries)
- Compare results: entities, relationships, communities, query results
- Track metrics: latency, accuracy, error rates
- Build confidence with real data

### 4. **Minimal Disruption**
- Preserve existing workflow structure
- Maintain current API interfaces where possible
- Support gradual migration

### 5. **Unified Storage (Neo4j Target)**
- Single source of truth for graph and vectors
- Eliminate data duplication between systems
- Enable hybrid queries

### 6. **Backward Compatibility**
- Continue supporting NetworkX mode
- Allow users to choose storage backend
- Provide migration tools
- Easy rollback

### 7. **Performance**
- Batch operations for efficiency
- Use native Neo4j operations where possible
- Minimize network round-trips
- Dark mode overhead acceptable (temporary)

---

## Overall Architecture - Three Execution Modes

### Mode 1: NetworkX Only (Current State)

```
┌───────────────────────────────────────┐
│   GraphRAG Indexing Pipeline          │
└───────────────────────────────────────┘
                  ↓
┌───────────────────────────────────────┐
│   NetworkX + LanceDB                  │
│   (In-memory graph + vector store)    │
└───────────────────────────────────────┘
                  ↓
┌───────────────────────────────────────┐
│   User Results ✅                      │
│   (Entities, Communities, Queries)    │
└───────────────────────────────────────┘
```

**Configuration**:
```yaml
storage:
  type: networkx_only
```

### Mode 2: Dark Mode (Parallel Validation)

```
┌───────────────────────────────────────────────────────────┐
│          GraphRAG Indexing Pipeline                        │
└───────────────────────────────────────────────────────────┘
                           ↓
          ┌────────────────────────────────┐
          │   DarkModeOrchestrator         │
          └────────────────────────────────┘
                    ↓          ↓
        ┌───────────┘          └───────────┐
        ↓ PRIMARY                   SHADOW ↓
┌──────────────────┐          ┌──────────────────┐
│ NetworkX         │          │ Neo4j            │
│ + LanceDB        │          │ (parallel exec)  │
└──────────────────┘          └──────────────────┘
        ↓                              ↓
        │                              │
        ↓                              ↓
┌──────────────────┐          ┌──────────────────┐
│ User Results ✅  │          │ Comparison Logs  │
│ (Production)     │          │ (Validation) 📊  │
└──────────────────┘          └──────────────────┘
```

**Configuration**:
```yaml
storage:
  type: dark_mode
  primary: networkx
  shadow: neo4j
  comparison:
    enabled: true
    log_differences: true
    metrics: [entity_count, community_match, query_f1, latency]
```

**Key Features**:
- Both systems execute identical operations
- NetworkX results returned to user
- Neo4j results logged for comparison
- Neo4j failures don't affect production
- Comparison metrics collected for validation

### Mode 3: Neo4j Only (Target State)

```
┌───────────────────────────────────────┐
│   GraphRAG Indexing Pipeline          │
└───────────────────────────────────────┘
                  ↓
┌───────────────────────────────────────┐
│   Neo4j Unified Storage               │
│   (Graph + Vector index)              │
└───────────────────────────────────────┘
                  ↓
┌───────────────────────────────────────┐
│   User Results ✅                      │
│   (Entities, Communities, Queries)    │
└───────────────────────────────────────┘
```

**Configuration**:
```yaml
storage:
  type: neo4j_only
  neo4j:
    uri: bolt://localhost:7687
    username: neo4j
    password: password
```

---

## Detailed Architecture

### Current Architecture (NetworkX + Parquet)

```
┌─────────────────────────────────────────────────────────────┐
│                    GraphRAG Indexing Pipeline                │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 1-4: Extract entities, relationships, text units      │
│  Output: DataFrames → Parquet files                         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 5: Load Parquet → NetworkX → Calculate Degrees        │
│  Output: Updated DataFrames → Parquet files                 │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 7: Load Parquet → NetworkX → Leiden Clustering        │
│  Output: Communities DataFrame → Parquet files              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 9: Generate Embeddings                                │
│  Output: Embeddings → Separate Vector Store (LanceDB/etc.)  │
└─────────────────────────────────────────────────────────────┘
```

### Proposed Architecture (Neo4j Unified)

```
┌─────────────────────────────────────────────────────────────┐
│                    GraphRAG Indexing Pipeline                │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 1-4: Extract entities, relationships, text units      │
│  Output: DataFrames → Neo4j (batch write) + Optional Parquet│
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 5: Neo4j GDS → Calculate Degrees → Write Back         │
│  Output: Updated Neo4j properties + Optional Parquet        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 7: Neo4j GDS Louvain → Community Detection            │
│  Output: Community nodes + relationships in Neo4j           │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 9: Generate Embeddings → Neo4j Vector Index           │
│  Output: Vector properties in Neo4j + Optional Parquet      │
└─────────────────────────────────────────────────────────────┘
```

**Key Changes**:
1. Write to Neo4j instead of (or in addition to) Parquet
2. Use Neo4j GDS for graph operations
3. Store embeddings in Neo4j Vector Index
4. Optional Parquet output for backward compatibility

---

## Dark Mode Orchestrator Architecture

### Overview

The `DarkModeOrchestrator` is the core component that enables parallel execution of NetworkX and Neo4j operations without affecting production.

### Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                   DarkModeOrchestrator                       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Execution   │  │  Comparison  │  │   Metrics    │      │
│  │  Coordinator │  │  Framework   │  │  Collector   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│         │                  │                  │              │
│         └──────────────────┴──────────────────┘              │
│                           │                                  │
└───────────────────────────┼──────────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            │                               │
            ↓                               ↓
  ┌───────────────────┐          ┌───────────────────┐
  │  Primary Backend  │          │  Shadow Backend   │
  │  (NetworkX)       │          │  (Neo4j)          │
  └───────────────────┘          └───────────────────┘
```

### Execution Coordinator

Responsible for dispatching operations to both backends:

```python
class ExecutionCoordinator:
    """Coordinates parallel execution across primary and shadow backends."""

    def __init__(
        self,
        primary: GraphStorage,  # NetworkX
        shadow: GraphStorage,   # Neo4j
        mode: ExecutionMode
    ):
        self.primary = primary
        self.shadow = shadow
        self.mode = mode

    async def execute_operation(
        self,
        operation: str,
        *args,
        **kwargs
    ) -> tuple[Any, OperationMetrics]:
        """
        Execute operation on primary backend, optionally on shadow.

        Returns:
            (primary_result, metrics)
        """

        # Always execute on primary
        primary_start = time.time()
        try:
            primary_result = await self._execute_on_backend(
                self.primary, operation, *args, **kwargs
            )
            primary_latency = time.time() - primary_start
            primary_error = None
        except Exception as e:
            primary_result = None
            primary_latency = time.time() - primary_start
            primary_error = str(e)
            # Primary failure is fatal - propagate
            raise

        # Execute on shadow only in dark_mode
        shadow_result = None
        shadow_latency = None
        shadow_error = None

        if self.mode == ExecutionMode.DARK_MODE:
            shadow_start = time.time()
            try:
                shadow_result = await self._execute_on_backend(
                    self.shadow, operation, *args, **kwargs
                )
                shadow_latency = time.time() - shadow_start
            except Exception as e:
                # Shadow failure is NOT fatal - just log
                shadow_latency = time.time() - shadow_start
                shadow_error = str(e)
                logger.warning(f"Shadow operation failed: {operation}: {e}")

        # Collect metrics
        metrics = OperationMetrics(
            operation=operation,
            primary_latency=primary_latency,
            shadow_latency=shadow_latency,
            primary_error=primary_error,
            shadow_error=shadow_error,
            timestamp=datetime.now()
        )

        # Compare results if shadow succeeded
        if shadow_result is not None:
            comparison = self._compare_results(
                operation, primary_result, shadow_result
            )
            metrics.comparison = comparison

        return primary_result, metrics

    async def _execute_on_backend(
        self, backend: GraphStorage, operation: str, *args, **kwargs
    ):
        """Execute operation on specific backend."""
        method = getattr(backend, operation)
        return await method(*args, **kwargs)

    def _compare_results(
        self, operation: str, primary_result: Any, shadow_result: Any
    ) -> ComparisonResult:
        """Compare results from primary and shadow backends."""
        # Comparison logic depends on operation type
        # See Comparison Framework section below
        pass
```

### Comparison Framework

Compares results between NetworkX and Neo4j:

```python
class ComparisonFramework:
    """Compares results between primary and shadow backends."""

    def compare_entities(
        self,
        primary_entities: pd.DataFrame,
        shadow_entities: pd.DataFrame
    ) -> EntityComparison:
        """Compare entity DataFrames."""

        # Count comparison
        count_match = len(primary_entities) == len(shadow_entities)

        # ID overlap
        primary_ids = set(primary_entities['id'])
        shadow_ids = set(shadow_entities['id'])

        overlap = primary_ids & shadow_ids
        precision = len(overlap) / len(shadow_ids) if shadow_ids else 0
        recall = len(overlap) / len(primary_ids) if primary_ids else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Missing entities
        missing_in_shadow = primary_ids - shadow_ids
        extra_in_shadow = shadow_ids - primary_ids

        return EntityComparison(
            count_match=count_match,
            primary_count=len(primary_entities),
            shadow_count=len(shadow_entities),
            precision=precision,
            recall=recall,
            f1=f1,
            missing_in_shadow=list(missing_in_shadow)[:10],  # Sample
            extra_in_shadow=list(extra_in_shadow)[:10]
        )

    def compare_communities(
        self,
        primary_communities: pd.DataFrame,
        shadow_communities: pd.DataFrame
    ) -> CommunityComparison:
        """Compare community assignments."""

        # Merge on entity ID
        merged = pd.merge(
            primary_communities[['entity_id', 'community']],
            shadow_communities[['entity_id', 'community']],
            on='entity_id',
            suffixes=('_primary', '_shadow'),
            how='outer'
        )

        # Exact match rate
        exact_matches = (
            merged['community_primary'] == merged['community_shadow']
        ).sum()
        match_rate = exact_matches / len(merged) if len(merged) > 0 else 0

        # NOTE: Louvain is non-deterministic, so < 100% match is expected
        # But should be > 95% for same algorithm

        return CommunityComparison(
            match_rate=match_rate,
            exact_matches=exact_matches,
            total_entities=len(merged),
            primary_levels=primary_communities['level'].nunique(),
            shadow_levels=shadow_communities['level'].nunique()
        )

    def compare_query_results(
        self,
        primary_results: List[Dict],
        shadow_results: List[Dict]
    ) -> QueryComparison:
        """Compare query results."""

        # Extract entity IDs from results
        primary_ids = [r['id'] for r in primary_results]
        shadow_ids = [r['id'] for r in shadow_results]

        # Set overlap (unordered)
        overlap = set(primary_ids) & set(shadow_ids)
        precision = len(overlap) / len(shadow_ids) if shadow_ids else 0
        recall = len(overlap) / len(primary_ids) if primary_ids else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Ranking correlation (ordered)
        # Use rank correlation for entities that appear in both
        ranking_correlation = self._calculate_rank_correlation(
            primary_ids, shadow_ids
        )

        return QueryComparison(
            f1=f1,
            precision=precision,
            recall=recall,
            ranking_correlation=ranking_correlation,
            primary_count=len(primary_results),
            shadow_count=len(shadow_results)
        )

    def _calculate_rank_correlation(
        self, primary_ids: List[str], shadow_ids: List[str]
    ) -> float:
        """Calculate Spearman rank correlation."""
        # Implementation using scipy.stats.spearmanr
        # Only for entities that appear in both lists
        pass
```

### Metrics Collector

Collects and logs comparison metrics:

```python
class MetricsCollector:
    """Collects and logs dark mode comparison metrics."""

    def __init__(self, log_path: str):
        self.log_path = log_path
        self.metrics_buffer = []
        self.flush_interval = 10  # seconds

    async def log_operation(self, metrics: OperationMetrics):
        """Log operation metrics."""
        self.metrics_buffer.append(metrics)

        # Flush to disk periodically
        if len(self.metrics_buffer) >= 100:
            await self._flush_metrics()

    async def _flush_metrics(self):
        """Write metrics buffer to disk."""
        if not self.metrics_buffer:
            return

        # Write to JSONL file
        log_file = Path(self.log_path) / "comparison_metrics.jsonl"
        with open(log_file, 'a') as f:
            for metric in self.metrics_buffer:
                f.write(json.dumps(metric.to_dict()) + '\n')

        # Also update aggregated metrics
        self._update_aggregated_metrics()

        self.metrics_buffer.clear()

    def _update_aggregated_metrics(self):
        """Update aggregated metrics file."""
        # Calculate rolling averages, percentiles, etc.
        # Write to separate aggregated metrics file
        pass

    def generate_report(self) -> DarkModeReport:
        """Generate dark mode validation report."""
        # Read all metrics from log files
        # Calculate summary statistics
        # Check against cutover criteria
        # Return comprehensive report
        pass
```

### Data Models

```python
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

class ExecutionMode(Enum):
    NETWORKX_ONLY = "networkx_only"
    DARK_MODE = "dark_mode"
    NEO4J_ONLY = "neo4j_only"

@dataclass
class OperationMetrics:
    operation: str
    primary_latency: float
    shadow_latency: Optional[float]
    primary_error: Optional[str]
    shadow_error: Optional[str]
    comparison: Optional['ComparisonResult']
    timestamp: datetime

@dataclass
class EntityComparison:
    count_match: bool
    primary_count: int
    shadow_count: int
    precision: float
    recall: float
    f1: float
    missing_in_shadow: List[str]
    extra_in_shadow: List[str]

@dataclass
class CommunityComparison:
    match_rate: float
    exact_matches: int
    total_entities: int
    primary_levels: int
    shadow_levels: int

@dataclass
class QueryComparison:
    f1: float
    precision: float
    recall: float
    ranking_correlation: float
    primary_count: int
    shadow_count: int

@dataclass
class ComparisonResult:
    entities: Optional[EntityComparison] = None
    communities: Optional[CommunityComparison] = None
    queries: Optional[QueryComparison] = None

@dataclass
class DarkModeReport:
    """Summary report for dark mode validation."""
    validation_period: str
    total_operations: int
    operation_breakdown: Dict[str, int]

    # Entity metrics
    entity_match_rate: float  # Should be > 99%
    avg_entity_f1: float

    # Community metrics
    community_match_rate: float  # Should be > 95%

    # Query metrics
    avg_query_f1: float  # Should be > 95%
    avg_ranking_correlation: float  # Should be > 0.90

    # Latency metrics
    avg_latency_ratio: float  # Should be < 2.0
    p95_latency_ratio: float

    # Error metrics
    shadow_error_rate: float  # Should be < 1%
    shadow_error_types: Dict[str, int]

    # Cutover readiness
    meets_cutover_criteria: bool
    blocking_issues: List[str]
```

---

## Neo4j Schema Design

### Node Labels

#### 1. Entity Node

Represents extracted entities (people, organizations, events, etc.)

```cypher
CREATE (e:Entity {
    // Identity
    id: "e_001",                           // Unique ID
    title: "Microsoft",                     // Entity name
    type: "ORGANIZATION",                   // Entity type

    // Content
    description: "Technology company...",   // Merged description

    // Graph Metrics (calculated in Step 5)
    node_degree: 15,                        // Number of relationships
    node_frequency: 23,                     // Occurrence count

    // Community Assignment (calculated in Step 7)
    community: 42,                          // Finest level community
    communities: [42, 15, 3],               // Full hierarchy [fine → coarse]

    // Source References
    text_unit_ids: ["t_001", "t_015"],      // Source text chunks

    // Vector Embedding (generated in Step 9)
    description_embedding: [0.021, -0.045, ...],  // 1536-dim vector

    // Metadata
    source_id: "doc_001"                    // Original document
})
```

**Indexes**:
```cypher
// Primary key
CREATE CONSTRAINT entity_id_unique FOR (e:Entity) REQUIRE e.id IS UNIQUE;

// Lookup indexes
CREATE INDEX entity_title FOR (e:Entity) ON (e.title);
CREATE INDEX entity_type FOR (e:Entity) ON (e.type);

// Vector index
CREATE VECTOR INDEX entity_description_vector
FOR (e:Entity)
ON e.description_embedding
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 1536,
    `vector.similarity_function`: 'cosine'
  }
}
```

#### 2. Relationship (Edge)

Represents relationships between entities

```cypher
CREATE (s:Entity)-[r:RELATED_TO {
    // Identity
    id: "r_001",                            // Unique ID

    // Content
    description: "Co-founded",              // Relationship description

    // Graph Metrics
    weight: 8.0,                            // Relationship strength
    combined_degree: 23,                    // Sum of endpoint degrees

    // Source References
    text_unit_ids: ["t_001", "t_005"],      // Source text chunks

    // Metadata
    source_id: "doc_001"                    // Original document
}]->(t:Entity)
```

**Indexes**:
```cypher
// Relationship property index
CREATE INDEX relationship_weight FOR ()-[r:RELATED_TO]-() ON (r.weight);
```

**Note**: Unlike nodes, relationships in Neo4j cannot have unique constraints, but they can have the `id` property for tracking.

#### 3. Community Node

Represents hierarchical communities from Louvain clustering

```cypher
CREATE (c:Community {
    // Identity
    id: "c_0_42",                           // Format: "c_{level}_{community_id}"
    level: 0,                               // Hierarchy level (0 = finest)
    community_id: 42,                       // ID at this level

    // Content (from Step 8: create_community_reports)
    title: "Technology Companies",          // Generated title
    summary: "This community contains...",  // LLM-generated summary
    full_content: "## Summary\n...",        // Complete report
    findings: [                             // Key findings
        "Microsoft and Apple dominate...",
        "Strong connections to cloud..."
    ],

    // Metrics
    rank: 8.5,                              // Importance score
    rank_explanation: "High centrality...", // Why this rank
    size: 156,                              // Number of entities

    // Vector Embedding
    summary_embedding: [0.034, -0.012, ...], // 1536-dim vector

    // Relationships
    period: "2020-2023",                    // Time period covered

    // Metadata
    source_id: "index_v1"                   // Index version
})
```

**Relationships**:
```cypher
// Entity membership
(entity:Entity)-[:BELONGS_TO {level: 0}]->(community:Community)

// Community hierarchy
(parent:Community)-[:PARENT_OF]->(child:Community)
```

**Indexes**:
```cypher
CREATE CONSTRAINT community_id_unique FOR (c:Community) REQUIRE c.id IS UNIQUE;
CREATE INDEX community_level FOR (c:Community) ON (c.level);
CREATE INDEX community_rank FOR (c:Community) ON (c.rank);

CREATE VECTOR INDEX community_summary_vector
FOR (c:Community)
ON c.summary_embedding
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 1536,
    `vector.similarity_function`: 'cosine'
  }
}
```

#### 4. TextUnit Node

Represents text chunks from documents

```cypher
CREATE (t:TextUnit {
    // Identity
    id: "t_001",                            // Unique ID
    chunk_id: "doc_001_chunk_0",            // Chunk identifier

    // Content
    text: "Microsoft was founded...",       // Chunk text
    n_tokens: 512,                          // Token count

    // Source References
    document_id: "doc_001",                 // Parent document
    entity_ids: ["e_001", "e_042"],         // Entities in chunk
    relationship_ids: ["r_001", "r_015"],   // Relationships in chunk
    covariate_ids: ["cov_001"],             // Claims in chunk (if enabled)

    // Vector Embedding
    text_embedding: [0.015, -0.089, ...],   // 1536-dim vector

    // Metadata
    attributes: {                           // Additional metadata
        "section": "History",
        "page": 1
    }
})
```

**Relationships**:
```cypher
// Entity mentions
(text:TextUnit)-[:MENTIONS]->(entity:Entity)

// Document containment
(doc:Document)-[:CONTAINS]->(text:TextUnit)
```

**Indexes**:
```cypher
CREATE CONSTRAINT text_unit_id_unique FOR (t:TextUnit) REQUIRE t.id IS UNIQUE;
CREATE INDEX text_unit_document FOR (t:TextUnit) ON (t.document_id);

CREATE VECTOR INDEX text_unit_vector
FOR (t:TextUnit)
ON t.text_embedding
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 1536,
    `vector.similarity_function`: 'cosine'
  }
}
```

#### 5. Document Node

Represents source documents

```cypher
CREATE (d:Document {
    // Identity
    id: "doc_001",                          // Unique ID
    title: "Microsoft History",             // Document title

    // Content
    raw_content: "Full document text...",   // Original text

    // Structure
    text_unit_ids: ["t_001", "t_002"],      // Text chunks

    // Metadata
    attributes: {                           // Document metadata
        "type": "article",
        "date": "2023-01-15",
        "author": "John Doe"
    }
})
```

**Indexes**:
```cypher
CREATE CONSTRAINT document_id_unique FOR (d:Document) REQUIRE d.id IS UNIQUE;
CREATE INDEX document_title FOR (d:Document) ON (d.title);
```

#### 6. Covariate Node (Optional)

Represents extracted claims (if covariates enabled)

```cypher
CREATE (cov:Covariate {
    // Identity
    id: "cov_001",                          // Unique ID

    // Content
    subject_id: "e_001",                    // Subject entity
    object_id: "e_042",                     // Object entity
    type: "ACQUISITION",                    // Claim type
    status: "CONFIRMED",                    // Verification status
    description: "Microsoft acquired...",   // Claim description

    // Temporal
    start_date: "2016-12-08",               // Event start
    end_date: "2016-12-08",                 // Event end

    // Source References
    text_unit_ids: ["t_001"],               // Source text chunks
    source_id: "doc_001",                   // Original document

    // Metadata
    attributes: {                           // Additional details
        "amount": "$26.2B",
        "announcement_date": "2016-06-13"
    }
})
```

**Relationships**:
```cypher
// Claim subjects and objects
(cov:Covariate)-[:HAS_SUBJECT]->(subject:Entity)
(cov:Covariate)-[:HAS_OBJECT]->(object:Entity)

// Source references
(text:TextUnit)-[:CONTAINS_CLAIM]->(cov:Covariate)
```

**Indexes**:
```cypher
CREATE CONSTRAINT covariate_id_unique FOR (cov:Covariate) REQUIRE cov.id IS UNIQUE;
CREATE INDEX covariate_type FOR (cov:Covariate) ON (cov.type);
CREATE INDEX covariate_status FOR (cov:Covariate) ON (cov.status);
```

### Complete Schema Visualization

```cypher
// Full schema with all relationships
(Document)-[:CONTAINS]->(TextUnit)
(TextUnit)-[:MENTIONS]->(Entity)
(Entity)-[:RELATED_TO]->(Entity)
(Entity)-[:BELONGS_TO {level}]->(Community)
(Community)-[:PARENT_OF]->(Community)
(TextUnit)-[:CONTAINS_CLAIM]->(Covariate)
(Covariate)-[:HAS_SUBJECT]->(Entity)
(Covariate)-[:HAS_OBJECT]->(Entity)
```

---

## Data Flow: Indexing Pipeline

### Step-by-Step Integration

#### Step 1: Load Input Documents
**Current**: Load documents → DataFrame
**Proposed**: Load documents → DataFrame → Neo4j

```python
# After loading documents
async def write_documents_to_neo4j(
    documents: pd.DataFrame,
    neo4j_driver: neo4j.Driver
):
    with neo4j_driver.session() as session:
        # Batch create Document nodes
        session.execute_write(
            _create_documents,
            documents.to_dict('records')
        )

def _create_documents(tx, documents):
    query = """
    UNWIND $documents AS doc
    CREATE (d:Document {
        id: doc.id,
        title: doc.title,
        raw_content: doc.raw_content,
        attributes: doc.attributes
    })
    """
    tx.run(query, documents=documents)
```

#### Step 2: Create Text Units
**Current**: Chunk documents → DataFrame → Parquet
**Proposed**: Chunk documents → DataFrame → Neo4j + Parquet

```python
async def write_text_units_to_neo4j(
    text_units: pd.DataFrame,
    neo4j_driver: neo4j.Driver
):
    with neo4j_driver.session() as session:
        session.execute_write(
            _create_text_units,
            text_units.to_dict('records')
        )

def _create_text_units(tx, text_units):
    query = """
    UNWIND $units AS unit
    MATCH (d:Document {id: unit.document_id})
    CREATE (t:TextUnit {
        id: unit.id,
        chunk_id: unit.chunk_id,
        text: unit.text,
        n_tokens: unit.n_tokens,
        document_id: unit.document_id,
        attributes: unit.attributes
    })
    CREATE (d)-[:CONTAINS]->(t)
    """
    tx.run(query, units=text_units)
```

#### Step 4: Extract Graph (Entities & Relationships)
**Current**: LLM extraction → DataFrame → Parquet
**Proposed**: LLM extraction → DataFrame → Neo4j (batch) + Parquet

```python
async def write_graph_to_neo4j(
    entities: pd.DataFrame,
    relationships: pd.DataFrame,
    neo4j_driver: neo4j.Driver
):
    with neo4j_driver.session() as session:
        # Batch create entities
        session.execute_write(
            _create_entities,
            entities.to_dict('records')
        )

        # Batch create relationships
        session.execute_write(
            _create_relationships,
            relationships.to_dict('records')
        )

        # Create MENTIONS relationships
        session.execute_write(
            _create_mentions,
            entities.to_dict('records')
        )

def _create_entities(tx, entities):
    query = """
    UNWIND $entities AS entity
    CREATE (e:Entity {
        id: entity.id,
        title: entity.title,
        type: entity.type,
        description: entity.description,
        text_unit_ids: entity.text_unit_ids,
        source_id: entity.source_id
    })
    """
    tx.run(query, entities=entities)

def _create_relationships(tx, relationships):
    query = """
    UNWIND $rels AS rel
    MATCH (s:Entity {id: rel.source})
    MATCH (t:Entity {id: rel.target})
    CREATE (s)-[r:RELATED_TO {
        id: rel.id,
        description: rel.description,
        weight: rel.weight,
        text_unit_ids: rel.text_unit_ids,
        source_id: rel.source_id
    }]->(t)
    """
    tx.run(query, rels=relationships)

def _create_mentions(tx, entities):
    query = """
    UNWIND $entities AS entity
    MATCH (e:Entity {id: entity.id})
    UNWIND entity.text_unit_ids AS text_unit_id
    MATCH (t:TextUnit {id: text_unit_id})
    MERGE (t)-[:MENTIONS]->(e)

    // Also update TextUnit.entity_ids array
    WITH t, collect(e.id) AS entity_ids
    SET t.entity_ids = entity_ids
    """
    tx.run(query, entities=entities)
```

**Optimization**: Use batching for large graphs
```python
# Batch size: 1000 entities per transaction
for batch in chunk_dataframe(entities, batch_size=1000):
    session.execute_write(_create_entities, batch.to_dict('records'))
```

#### Step 5: Finalize Graph (Calculate Degrees)
**Current**: Load Parquet → NetworkX → Calculate degrees → Write Parquet
**Proposed**: Neo4j GDS → Calculate degrees → Update Neo4j + Parquet

```python
async def calculate_degrees_neo4j(neo4j_driver: neo4j.Driver):
    with neo4j_driver.session() as session:
        # Create GDS projection
        session.run("""
            CALL gds.graph.project(
                'degree-calculation',
                'Entity',
                'RELATED_TO',
                {relationshipProperties: ['weight']}
            )
        """)

        # Calculate weighted degree using GDS
        session.run("""
            CALL gds.degree.write('degree-calculation', {
                writeProperty: 'node_degree',
                relationshipWeightProperty: 'weight'
            })
        """)

        # Calculate combined degree for relationships
        session.run("""
            MATCH (s:Entity)-[r:RELATED_TO]->(t:Entity)
            SET r.combined_degree = s.node_degree + t.node_degree
        """)

        # Drop projection
        session.run("CALL gds.graph.drop('degree-calculation')")
```

**Backward Compatibility**: Export to DataFrame if needed
```python
async def export_entities_to_dataframe(neo4j_driver: neo4j.Driver) -> pd.DataFrame:
    with neo4j_driver.session() as session:
        result = session.run("""
            MATCH (e:Entity)
            RETURN e.id AS id,
                   e.title AS title,
                   e.type AS type,
                   e.description AS description,
                   e.node_degree AS node_degree,
                   e.text_unit_ids AS text_unit_ids
        """)
        return pd.DataFrame([record.data() for record in result])
```

#### Step 7: Create Communities (Louvain Clustering)
**Current**: Load Parquet → NetworkX → Leiden → Write Parquet
**Proposed**: Neo4j GDS Louvain → Create Community nodes + relationships

```python
async def run_community_detection_neo4j(neo4j_driver: neo4j.Driver):
    with neo4j_driver.session() as session:
        # Create GDS projection
        session.run("""
            CALL gds.graph.project(
                'community-detection',
                'Entity',
                'RELATED_TO',
                {
                    nodeProperties: ['node_degree'],
                    relationshipProperties: ['weight']
                }
            )
        """)

        # Run Louvain with hierarchy
        result = session.run("""
            CALL gds.louvain.stream('community-detection', {
                relationshipWeightProperty: 'weight',
                maxLevels: 10,
                includeIntermediateCommunities: true,
                seedProperty: 'seed'
            })
            YIELD nodeId, communityId, intermediateCommunityIds
            RETURN
                gds.util.asNode(nodeId).id AS entity_id,
                communityId AS community,
                intermediateCommunityIds AS communities
        """)

        # Store community assignments on entities
        communities_data = [record.data() for record in result]
        session.execute_write(
            _write_community_assignments,
            communities_data
        )

        # Create Community nodes and relationships
        session.execute_write(_create_community_nodes, communities_data)
        session.execute_write(_create_community_relationships, communities_data)

        # Drop projection
        session.run("CALL gds.graph.drop('community-detection')")

def _write_community_assignments(tx, communities_data):
    query = """
    UNWIND $data AS item
    MATCH (e:Entity {id: item.entity_id})
    SET e.community = item.community,
        e.communities = item.communities
    """
    tx.run(query, data=communities_data)

def _create_community_nodes(tx, communities_data):
    # Extract unique communities at each level
    query = """
    UNWIND $data AS item
    UNWIND range(0, size(item.communities) - 1) AS level
    WITH DISTINCT level, item.communities[level] AS community_id
    MERGE (c:Community {
        id: 'c_' + level + '_' + community_id,
        level: level,
        community_id: community_id
    })
    ON CREATE SET c.size = 0
    """
    tx.run(query, data=communities_data)

    # Update community sizes
    query = """
    MATCH (e:Entity)
    UNWIND range(0, size(e.communities) - 1) AS level
    WITH level, e.communities[level] AS community_id, count(e) AS size
    MATCH (c:Community {id: 'c_' + level + '_' + community_id})
    SET c.size = size
    """
    tx.run(query)

def _create_community_relationships(tx, communities_data):
    # Create BELONGS_TO relationships
    query = """
    MATCH (e:Entity)
    UNWIND range(0, size(e.communities) - 1) AS level
    WITH e, level, e.communities[level] AS community_id
    MATCH (c:Community {id: 'c_' + level + '_' + community_id})
    MERGE (e)-[:BELONGS_TO {level: level}]->(c)
    """
    tx.run(query)

    # Create PARENT_OF hierarchy
    query = """
    MATCH (c:Community)
    WHERE c.level < 9  // Assuming max 10 levels
    WITH c
    MATCH (parent:Community {
        level: c.level + 1,
        community_id: c.community_id  // Simplified - needs actual parent ID
    })
    MERGE (parent)-[:PARENT_OF]->(c)
    """
    # Note: This is simplified. Actual implementation needs to track parent IDs from Louvain
    tx.run(query)
```

#### Step 8: Create Community Reports
**Current**: Generate LLM summaries → DataFrame → Parquet
**Proposed**: Generate LLM summaries → Update Community nodes + Parquet

```python
async def write_community_reports_neo4j(
    reports: pd.DataFrame,
    neo4j_driver: neo4j.Driver
):
    with neo4j_driver.session() as session:
        session.execute_write(
            _update_community_reports,
            reports.to_dict('records')
        )

def _update_community_reports(tx, reports):
    query = """
    UNWIND $reports AS report
    MATCH (c:Community {id: report.community_id})
    SET c.title = report.title,
        c.summary = report.summary,
        c.full_content = report.full_content,
        c.findings = report.findings,
        c.rank = report.rank,
        c.rank_explanation = report.rank_explanation
    """
    tx.run(query, reports=reports)
```

#### Step 9: Generate Embeddings
**Current**: Generate embeddings → Write to separate vector store
**Proposed**: Generate embeddings → Write to Neo4j vector properties

```python
async def write_embeddings_neo4j(
    entity_embeddings: pd.DataFrame,
    community_embeddings: pd.DataFrame,
    text_unit_embeddings: pd.DataFrame,
    neo4j_driver: neo4j.Driver
):
    with neo4j_driver.session() as session:
        # Update entity embeddings in batches
        for batch in chunk_dataframe(entity_embeddings, batch_size=1000):
            session.execute_write(
                _update_entity_embeddings,
                batch.to_dict('records')
            )

        # Update community embeddings
        session.execute_write(
            _update_community_embeddings,
            community_embeddings.to_dict('records')
        )

        # Update text unit embeddings
        for batch in chunk_dataframe(text_unit_embeddings, batch_size=1000):
            session.execute_write(
                _update_text_unit_embeddings,
                batch.to_dict('records')
            )

def _update_entity_embeddings(tx, embeddings):
    query = """
    UNWIND $embeddings AS emb
    MATCH (e:Entity {id: emb.id})
    SET e.description_embedding = emb.vector
    """
    tx.run(query, embeddings=embeddings)

def _update_community_embeddings(tx, embeddings):
    query = """
    UNWIND $embeddings AS emb
    MATCH (c:Community {id: emb.id})
    SET c.summary_embedding = emb.vector
    """
    tx.run(query, embeddings=embeddings)

def _update_text_unit_embeddings(tx, embeddings):
    query = """
    UNWIND $embeddings AS emb
    MATCH (t:TextUnit {id: emb.id})
    SET t.text_embedding = emb.vector
    """
    tx.run(query, embeddings=embeddings)
```

### Complete Pipeline Flow

```python
async def run_neo4j_indexing_pipeline(
    config: GraphRagConfig,
    context: PipelineRunContext
):
    """Main indexing pipeline with Neo4j storage."""

    # Initialize Neo4j driver
    neo4j_driver = neo4j.GraphDatabase.driver(
        config.neo4j.uri,
        auth=(config.neo4j.username, config.neo4j.password)
    )

    try:
        # Step 1: Load documents
        documents = await load_input_documents(config, context)
        await write_documents_to_neo4j(documents, neo4j_driver)

        # Step 2: Create text units
        text_units = await create_text_units(documents, config)
        await write_text_units_to_neo4j(text_units, neo4j_driver)

        # Step 3: Create final documents
        final_docs = await create_final_documents(text_units)
        # (Documents already in Neo4j, just update if needed)

        # Step 4: Extract graph
        entities, relationships = await extract_graph(text_units, config)
        await write_graph_to_neo4j(entities, relationships, neo4j_driver)

        # Step 5: Calculate degrees
        await calculate_degrees_neo4j(neo4j_driver)

        # Step 6: Extract covariates (optional)
        if config.covariates.enabled:
            covariates = await extract_covariates(text_units, config)
            await write_covariates_to_neo4j(covariates, neo4j_driver)

        # Step 7: Community detection
        await run_community_detection_neo4j(neo4j_driver)

        # Step 8: Create community reports
        reports = await create_community_reports_from_neo4j(neo4j_driver, config)
        await write_community_reports_neo4j(reports, neo4j_driver)

        # Step 9: Generate embeddings
        entity_emb, community_emb, text_emb = await generate_embeddings_from_neo4j(
            neo4j_driver, config
        )
        await write_embeddings_neo4j(entity_emb, community_emb, text_emb, neo4j_driver)

        # Step 10: Export to Parquet (backward compatibility)
        if config.output.parquet_enabled:
            await export_neo4j_to_parquet(neo4j_driver, context.output_storage)

    finally:
        neo4j_driver.close()
```

---

## Storage Adapter Layer

### Interface Design

To support both Parquet and Neo4j backends, create an abstract storage interface:

```python
# packages/graphrag-storage/graphrag_storage/graph_storage.py

from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd

class GraphStorage(ABC):
    """Abstract interface for graph storage backends."""

    @abstractmethod
    async def write_entities(self, entities: pd.DataFrame) -> None:
        """Write entities to storage."""
        pass

    @abstractmethod
    async def read_entities(self) -> pd.DataFrame:
        """Read entities from storage."""
        pass

    @abstractmethod
    async def write_relationships(self, relationships: pd.DataFrame) -> None:
        """Write relationships to storage."""
        pass

    @abstractmethod
    async def read_relationships(self) -> pd.DataFrame:
        """Read relationships from storage."""
        pass

    @abstractmethod
    async def calculate_degrees(self) -> None:
        """Calculate node degrees and update entities."""
        pass

    @abstractmethod
    async def run_community_detection(
        self,
        max_cluster_size: int = 10,
        use_lcc: bool = True,
        seed: int = 0xDEADBEEF
    ) -> pd.DataFrame:
        """Run community detection algorithm."""
        pass

    @abstractmethod
    async def write_embeddings(
        self,
        entity_embeddings: Optional[pd.DataFrame] = None,
        community_embeddings: Optional[pd.DataFrame] = None,
        text_unit_embeddings: Optional[pd.DataFrame] = None
    ) -> None:
        """Write embeddings to storage."""
        pass
```

### Parquet Implementation (Current)

```python
# packages/graphrag-storage/graphrag_storage/parquet_graph_storage.py

class ParquetGraphStorage(GraphStorage):
    """Parquet-based graph storage (current implementation)."""

    def __init__(self, storage: PipelineStorage):
        self.storage = storage

    async def write_entities(self, entities: pd.DataFrame) -> None:
        await write_table_to_storage(entities, "entities", self.storage)

    async def read_entities(self) -> pd.DataFrame:
        return await load_table_from_storage("entities", self.storage)

    async def write_relationships(self, relationships: pd.DataFrame) -> None:
        await write_table_to_storage(relationships, "relationships", self.storage)

    async def read_relationships(self) -> pd.DataFrame:
        return await load_table_from_storage("relationships", self.storage)

    async def calculate_degrees(self) -> None:
        # Load from Parquet
        entities = await self.read_entities()
        relationships = await self.read_relationships()

        # Create NetworkX graph
        graph = create_graph(entities, relationships)

        # Calculate degrees
        degrees = dict(graph.degree())
        entities["node_degree"] = entities["title"].map(degrees).fillna(0).astype(int)

        # Write back
        await self.write_entities(entities)

    async def run_community_detection(
        self,
        max_cluster_size: int = 10,
        use_lcc: bool = True,
        seed: int = 0xDEADBEEF
    ) -> pd.DataFrame:
        # Load from Parquet
        entities = await self.read_entities()
        relationships = await self.read_relationships()

        # Create NetworkX graph
        graph = create_graph(entities, relationships)

        # Run Leiden clustering
        from graphrag.index.operations.cluster_graph import cluster_graph
        communities = cluster_graph(graph, max_cluster_size, use_lcc, seed)

        # Convert to DataFrame
        return communities_to_dataframe(communities)

    async def write_embeddings(
        self,
        entity_embeddings: Optional[pd.DataFrame] = None,
        community_embeddings: Optional[pd.DataFrame] = None,
        text_unit_embeddings: Optional[pd.DataFrame] = None
    ) -> None:
        # Write to separate vector store (via graphrag-vectors)
        # This is handled by existing vector store logic
        pass
```

### Neo4j Implementation (New)

```python
# packages/graphrag-storage/graphrag_storage/neo4j_graph_storage.py

import neo4j
from typing import Optional
import pandas as pd

class Neo4jGraphStorage(GraphStorage):
    """Neo4j-based graph storage with integrated vectors."""

    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        database: str = "neo4j"
    ):
        self.driver = neo4j.GraphDatabase.driver(uri, auth=(username, password))
        self.database = database

    async def write_entities(self, entities: pd.DataFrame) -> None:
        with self.driver.session(database=self.database) as session:
            for batch in chunk_dataframe(entities, batch_size=1000):
                session.execute_write(
                    _create_entities,
                    batch.to_dict('records')
                )

    async def read_entities(self) -> pd.DataFrame:
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (e:Entity)
                RETURN e.id AS id,
                       e.title AS title,
                       e.type AS type,
                       e.description AS description,
                       e.node_degree AS node_degree,
                       e.community AS community,
                       e.communities AS communities,
                       e.text_unit_ids AS text_unit_ids
            """)
            return pd.DataFrame([record.data() for record in result])

    async def write_relationships(self, relationships: pd.DataFrame) -> None:
        with self.driver.session(database=self.database) as session:
            for batch in chunk_dataframe(relationships, batch_size=1000):
                session.execute_write(
                    _create_relationships,
                    batch.to_dict('records')
                )

    async def read_relationships(self) -> pd.DataFrame:
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (s:Entity)-[r:RELATED_TO]->(t:Entity)
                RETURN r.id AS id,
                       s.title AS source,
                       t.title AS target,
                       r.description AS description,
                       r.weight AS weight,
                       r.combined_degree AS combined_degree,
                       r.text_unit_ids AS text_unit_ids
            """)
            return pd.DataFrame([record.data() for record in result])

    async def calculate_degrees(self) -> None:
        with self.driver.session(database=self.database) as session:
            # Create GDS projection
            session.run("""
                CALL gds.graph.project(
                    'degree-calculation',
                    'Entity',
                    'RELATED_TO',
                    {relationshipProperties: ['weight']}
                )
            """)

            # Calculate degrees
            session.run("""
                CALL gds.degree.write('degree-calculation', {
                    writeProperty: 'node_degree',
                    relationshipWeightProperty: 'weight'
                })
            """)

            # Update relationships
            session.run("""
                MATCH (s:Entity)-[r:RELATED_TO]->(t:Entity)
                SET r.combined_degree = s.node_degree + t.node_degree
            """)

            # Drop projection
            session.run("CALL gds.graph.drop('degree-calculation')")

    async def run_community_detection(
        self,
        max_cluster_size: int = 10,
        use_lcc: bool = True,
        seed: int = 0xDEADBEEF
    ) -> pd.DataFrame:
        with self.driver.session(database=self.database) as session:
            # Create GDS projection
            session.run("""
                CALL gds.graph.project(
                    'community-detection',
                    'Entity',
                    'RELATED_TO',
                    {
                        nodeProperties: ['node_degree'],
                        relationshipProperties: ['weight']
                    }
                )
            """)

            # Run Louvain
            result = session.run("""
                CALL gds.louvain.stream('community-detection', {
                    relationshipWeightProperty: 'weight',
                    maxLevels: 10,
                    includeIntermediateCommunities: true
                })
                YIELD nodeId, communityId, intermediateCommunityIds
                RETURN
                    gds.util.asNode(nodeId).id AS entity_id,
                    gds.util.asNode(nodeId).title AS entity_title,
                    communityId AS community,
                    intermediateCommunityIds AS communities
            """)

            communities_df = pd.DataFrame([record.data() for record in result])

            # Write back to Neo4j
            session.execute_write(
                _write_community_assignments,
                communities_df.to_dict('records')
            )

            # Create Community nodes
            session.execute_write(
                _create_community_nodes,
                communities_df.to_dict('records')
            )

            # Drop projection
            session.run("CALL gds.graph.drop('community-detection')")

            return communities_df

    async def write_embeddings(
        self,
        entity_embeddings: Optional[pd.DataFrame] = None,
        community_embeddings: Optional[pd.DataFrame] = None,
        text_unit_embeddings: Optional[pd.DataFrame] = None
    ) -> None:
        with self.driver.session(database=self.database) as session:
            if entity_embeddings is not None:
                for batch in chunk_dataframe(entity_embeddings, batch_size=1000):
                    session.execute_write(
                        _update_entity_embeddings,
                        batch.to_dict('records')
                    )

            if community_embeddings is not None:
                session.execute_write(
                    _update_community_embeddings,
                    community_embeddings.to_dict('records')
                )

            if text_unit_embeddings is not None:
                for batch in chunk_dataframe(text_unit_embeddings, batch_size=1000):
                    session.execute_write(
                        _update_text_unit_embeddings,
                        batch.to_dict('records')
                    )

    def close(self):
        self.driver.close()
```

### Factory Pattern

```python
# packages/graphrag-storage/graphrag_storage/__init__.py

def create_graph_storage(config: GraphRagConfig) -> GraphStorage:
    """Factory function to create appropriate storage backend."""

    if config.storage.type == "parquet":
        return ParquetGraphStorage(config.storage)

    elif config.storage.type == "neo4j":
        return Neo4jGraphStorage(
            uri=config.neo4j.uri,
            username=config.neo4j.username,
            password=config.neo4j.password,
            database=config.neo4j.database
        )

    elif config.storage.type == "hybrid":
        # Write to both Parquet and Neo4j
        return HybridGraphStorage(
            parquet=ParquetGraphStorage(config.storage),
            neo4j=Neo4jGraphStorage(
                uri=config.neo4j.uri,
                username=config.neo4j.username,
                password=config.neo4j.password
            )
        )

    else:
        raise ValueError(f"Unknown storage type: {config.storage.type}")
```

---

## Query Operation Changes

### Current Query Flow (Parquet + LanceDB)

```python
# Global Search example
async def global_search(query: str, config: GraphRagConfig):
    # 1. Load community reports from Parquet
    communities = pd.read_parquet("communities.parquet")

    # 2. Search via vector store
    vector_store = create_vector_store(config.vector_store)
    results = await vector_store.similarity_search(
        collection="community_reports",
        query_embedding=embed(query),
        limit=10
    )

    # 3. MAP: Generate answers for each community
    # 4. REDUCE: Aggregate answers
    ...
```

### Proposed Query Flow (Neo4j Unified)

```python
# Global Search with Neo4j
async def global_search_neo4j(query: str, neo4j_driver: neo4j.Driver):
    # 1. Embed query
    query_embedding = await embed(query)

    # 2. Vector search for relevant communities (single query)
    with neo4j_driver.session() as session:
        results = session.run("""
            CALL db.index.vector.queryNodes(
                'community_summary_vector',
                10,
                $query_embedding
            )
            YIELD node, score
            RETURN
                node.id AS community_id,
                node.title AS title,
                node.summary AS summary,
                node.full_content AS content,
                node.rank AS rank,
                score
            ORDER BY score DESC
        """, query_embedding=query_embedding)

        communities = [record.data() for record in results]

    # 3. MAP: Generate answers for each community
    # 4. REDUCE: Aggregate answers
    ...
```

### Local Search with Hybrid Query

```python
# Local Search: Find entities + their neighborhoods
async def local_search_neo4j(
    query: str,
    neo4j_driver: neo4j.Driver
):
    query_embedding = await embed(query)

    with neo4j_driver.session() as session:
        # Hybrid query: Vector similarity + graph traversal
        results = session.run("""
            // Find similar entities via vector search
            CALL db.index.vector.queryNodes(
                'entity_description_vector',
                20,
                $query_embedding
            )
            YIELD node AS entity, score

            // Get their neighborhoods (1-2 hops)
            MATCH (entity)-[r1:RELATED_TO]-(neighbor1)
            OPTIONAL MATCH (neighbor1)-[r2:RELATED_TO]-(neighbor2)
            WHERE neighbor2 <> entity

            // Get related text units
            MATCH (t:TextUnit)-[:MENTIONS]->(entity)

            RETURN DISTINCT
                entity.title AS entity,
                entity.description AS description,
                collect(DISTINCT neighbor1.title) AS neighbors_1hop,
                collect(DISTINCT neighbor2.title) AS neighbors_2hop,
                collect(DISTINCT t.text) AS source_texts,
                score
            ORDER BY score DESC
            LIMIT 10
        """, query_embedding=query_embedding)

        return [record.data() for record in results]
```

### New Capability: Hybrid Search

```python
# Find entities similar to query that are connected to specific entity
async def hybrid_search_neo4j(
    query: str,
    anchor_entity: str,
    neo4j_driver: neo4j.Driver
):
    """
    Example: "Find technology companies similar to 'cloud computing'
    that are connected to Microsoft"
    """

    query_embedding = await embed(query)

    with neo4j_driver.session() as session:
        results = session.run("""
            // Find anchor entity
            MATCH (anchor:Entity {title: $anchor_entity})

            // Vector search for similar entities
            CALL db.index.vector.queryNodes(
                'entity_description_vector',
                100,
                $query_embedding
            )
            YIELD node AS candidate, score

            // Filter: must be connected to anchor within 2 hops
            WHERE candidate <> anchor
              AND EXISTS {
                  MATCH (anchor)-[:RELATED_TO*1..2]-(candidate)
              }

            // Get connection path
            MATCH path = shortestPath((anchor)-[:RELATED_TO*]-(candidate))

            RETURN
                candidate.title AS entity,
                candidate.description AS description,
                score AS similarity,
                length(path) AS distance,
                [rel IN relationships(path) | rel.description] AS connection_path
            ORDER BY score DESC
            LIMIT 10
        """,
        anchor_entity=anchor_entity,
        query_embedding=query_embedding)

        return [record.data() for record in results]
```

---

## Configuration

### New Configuration Schema (Dark Mode)

```yaml
# settings.yaml

storage:
  # Execution mode: networkx_only, dark_mode, neo4j_only
  type: dark_mode

  # NetworkX configuration (for networkx_only and dark_mode)
  networkx:
    enabled: true
    cache_dir: "./cache"
    vector_store:
      type: lancedb
      uri: "./output/lancedb"

  # Neo4j configuration (for neo4j_only and dark_mode)
  neo4j:
    enabled: true
    uri: "bolt://localhost:7687"
    username: "neo4j"
    password: "password"
    database: "neo4j"

    # Connection pool settings
    max_connection_pool_size: 50
    connection_acquisition_timeout: 60

    # Batch settings
    batch_size: 1000

    # GDS settings
    gds:
      enabled: true
      projection_prefix: "graphrag_"

    # Vector index settings
    vector_index:
      enabled: true
      dimensions: 1536
      similarity_function: cosine  # Options: cosine, euclidean

  # Dark mode specific configuration
  dark_mode:
    enabled: true  # Only used when type=dark_mode
    primary_backend: networkx  # Always networkx
    shadow_backend: neo4j      # Always neo4j

    # Comparison settings
    comparison:
      enabled: true
      log_path: "./dark_mode_logs"
      log_format: jsonl
      flush_interval_seconds: 10

      # What to compare
      compare_indexing: true
      compare_queries: true

      # Metrics to collect
      metrics:
        - entity_count
        - relationship_count
        - community_match_rate
        - query_f1
        - query_ranking_correlation
        - latency_ratio
        - error_rates

      # Sampling (to reduce overhead)
      sample_rate: 1.0  # 1.0 = 100% of operations

    # Error handling
    error_handling:
      shadow_failure_action: log  # Options: log, alert, fail
      continue_on_shadow_error: true

    # Cutover criteria
    cutover_criteria:
      validation_period_days: 14
      min_operations: 1000
      entity_match_rate_threshold: 0.99
      community_match_rate_threshold: 0.95
      query_f1_threshold: 0.95
      query_ranking_correlation_threshold: 0.90
      latency_ratio_threshold: 2.0
      shadow_error_rate_threshold: 0.01
```

### Configuration Examples

**Mode 1: NetworkX Only (Current)**
```yaml
storage:
  type: networkx_only
  networkx:
    enabled: true
    cache_dir: "./cache"
    vector_store:
      type: lancedb
      uri: "./output/lancedb"
```

**Mode 2: Dark Mode (Validation)**
```yaml
storage:
  type: dark_mode
  networkx:
    enabled: true
    cache_dir: "./cache"
    vector_store:
      type: lancedb
      uri: "./output/lancedb"
  neo4j:
    enabled: true
    uri: "bolt://localhost:7687"
    username: "neo4j"
    password: "password"
  dark_mode:
    enabled: true
    comparison:
      enabled: true
      log_path: "./dark_mode_logs"
```

**Mode 3: Neo4j Only (Target)**
```yaml
storage:
  type: neo4j_only
  neo4j:
    enabled: true
    uri: "bolt://localhost:7687"
    username: "neo4j"
    password: "password"
    database: "neo4j"
```

### Configuration Class

```python
# packages/graphrag/graphrag/config/storage_config.py

from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, Literal

class StorageType(str, Enum):
    PARQUET = "parquet"
    NEO4J = "neo4j"
    HYBRID = "hybrid"

class Neo4jConfig(BaseModel):
    """Neo4j connection configuration."""

    uri: str = Field(default="bolt://localhost:7687")
    username: str = Field(default="neo4j")
    password: str = Field(default="password")
    database: str = Field(default="neo4j")

    # Connection pool
    max_connection_pool_size: int = Field(default=50)
    connection_acquisition_timeout: int = Field(default=60)

    # Batch settings
    batch_size: int = Field(default=1000)

    # GDS settings
    gds_enabled: bool = Field(default=True)
    gds_projection_prefix: str = Field(default="graphrag_")

    # Vector index
    vector_index_enabled: bool = Field(default=True)
    vector_dimensions: int = Field(default=1536)
    vector_similarity_function: Literal["cosine", "euclidean"] = Field(default="cosine")

class StorageConfig(BaseModel):
    """Storage backend configuration."""

    type: StorageType = Field(default=StorageType.PARQUET)

    # Parquet settings
    parquet_base_dir: str = Field(default="./output")

    # Neo4j settings
    neo4j: Optional[Neo4jConfig] = None

    def __init__(self, **data):
        super().__init__(**data)

        # Initialize Neo4j config if needed
        if self.type in [StorageType.NEO4J, StorageType.HYBRID]:
            if self.neo4j is None:
                self.neo4j = Neo4jConfig()
```

---

## Backward Compatibility Strategy (Dark Mode)

### Three-Mode Architecture

Support three execution modes with easy transitions:

```python
class DarkModeGraphStorage(GraphStorage):
    """
    Orchestrates execution across NetworkX and Neo4j backends.

    Modes:
    - networkx_only: Only NetworkX, no Neo4j
    - dark_mode: Both run, NetworkX results returned, Neo4j validated
    - neo4j_only: Only Neo4j
    """

    def __init__(
        self,
        mode: ExecutionMode,
        networkx_backend: Optional[NetworkXGraphStorage] = None,
        neo4j_backend: Optional[Neo4jGraphStorage] = None,
        orchestrator: Optional[DarkModeOrchestrator] = None
    ):
        self.mode = mode
        self.networkx = networkx_backend
        self.neo4j = neo4j_backend
        self.orchestrator = orchestrator

    async def write_entities(self, entities: pd.DataFrame) -> None:
        """Write entities according to current mode."""

        if self.mode == ExecutionMode.NETWORKX_ONLY:
            # Only NetworkX
            await self.networkx.write_entities(entities)

        elif self.mode == ExecutionMode.DARK_MODE:
            # Both backends, orchestrator coordinates
            result, metrics = await self.orchestrator.execute_operation(
                "write_entities",
                entities
            )
            # Return NetworkX result, log comparison

        elif self.mode == ExecutionMode.NEO4J_ONLY:
            # Only Neo4j
            await self.neo4j.write_entities(entities)

    async def read_entities(self) -> pd.DataFrame:
        """Read entities according to current mode."""

        if self.mode == ExecutionMode.NETWORKX_ONLY:
            return await self.networkx.read_entities()

        elif self.mode == ExecutionMode.DARK_MODE:
            # Read from primary (NetworkX) in dark mode
            return await self.networkx.read_entities()

        elif self.mode == ExecutionMode.NEO4J_ONLY:
            return await self.neo4j.read_entities()

    # ... similar for all methods
```

### Migration Path (Safe Transitions)

**Phase 1: NetworkX Only (Current)**
```yaml
storage:
  type: networkx_only
```
- No changes to current behavior
- No Neo4j required

**Phase 2: Dark Mode Validation (2-4 weeks)**
```yaml
storage:
  type: dark_mode
  networkx:
    enabled: true
  neo4j:
    enabled: true
  dark_mode:
    comparison:
      enabled: true
```
- Both systems run in parallel
- NetworkX results returned to users (zero production impact)
- Neo4j results compared and logged
- Build confidence with real traffic
- Collect metrics for go/no-go decision

**Phase 3: Neo4j Only (After Validation)**
```yaml
storage:
  type: neo4j_only
  neo4j:
    enabled: true
```
- Switch to Neo4j only when metrics pass
- NetworkX no longer needed
- Can revert instantly if issues found

**Emergency Rollback (Any Time)**
```yaml
storage:
  type: networkx_only  # Instant revert
```
- One line change
- No data loss (NetworkX maintained in parallel during dark mode)
- Can rollback from dark_mode or neo4j_only

### Export Utility

Provide tool to export Neo4j data to Parquet:

```python
# graphrag/cli/export.py

async def export_neo4j_to_parquet(
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    output_dir: str
):
    """Export Neo4j graph to Parquet files."""

    driver = neo4j.GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    try:
        # Export entities
        entities_df = await export_entities(driver)
        entities_df.to_parquet(f"{output_dir}/entities.parquet")

        # Export relationships
        relationships_df = await export_relationships(driver)
        relationships_df.to_parquet(f"{output_dir}/relationships.parquet")

        # Export communities
        communities_df = await export_communities(driver)
        communities_df.to_parquet(f"{output_dir}/communities.parquet")

        # Export text units
        text_units_df = await export_text_units(driver)
        text_units_df.to_parquet(f"{output_dir}/text_units.parquet")

        print(f"✅ Exported Neo4j data to {output_dir}")

    finally:
        driver.close()

# CLI command
# graphrag export-neo4j --uri bolt://localhost:7687 --output ./output
```

---

## Summary: Architecture Design (Dark Mode)

### Key Design Decisions

1. **Three Execution Modes**: networkx_only, dark_mode, neo4j_only with easy transitions
2. **Dark Mode Orchestrator**: Coordinates parallel execution, comparison, metrics collection
3. **Zero Production Risk**: NetworkX remains authoritative during validation
4. **Comprehensive Comparison**: Entity counts, community matches, query F1, latency ratios
5. **Schema**: Neo4j property graph with Entity, Community, TextUnit, Document, and Covariate nodes
6. **Integration**: Abstract storage interface with NetworkX and Neo4j implementations
7. **Data Flow**: Batch writes to backends, use GDS for graph operations in Neo4j
8. **Query Changes**: Use Neo4j vector index and Cypher for unified graph+vector queries

### Changes to Codebase (Updated for Dark Mode)

| Component | Changes Required | Complexity |
|-----------|------------------|------------|
| **Storage Interface** | Add abstract `GraphStorage` class with mode support | Medium |
| **NetworkX Adapter** | Refactor existing code into adapter | Medium |
| **Neo4j Adapter** | Implement `Neo4jGraphStorage` | High |
| **Dark Mode Orchestrator** | NEW: Parallel execution coordinator | High |
| **Comparison Framework** | NEW: Result comparison logic | High |
| **Metrics Collector** | NEW: Logging and reporting | Medium |
| **Indexing Workflows** | Use `DarkModeGraphStorage` abstraction | Medium |
| **Query Operations** | Add Neo4j query methods + dark mode support | High |
| **Configuration** | Add three-mode config schema | Medium |
| **CLI** | Add export/import/dark-mode-report commands | Medium |
| **Tests** | Add Neo4j integration tests + dark mode tests | Very High |
| **Documentation** | Update guides and examples for dark mode | High |

### Implementation Complexity (Updated)

- **Total Estimated Effort**: 8-10 weeks for Phases 2-3 (Core Integration + Dark Mode Framework)
  - Core Neo4j Implementation: 6 weeks
  - Dark Mode Infrastructure: 4 weeks
- **High-Risk Areas**: Dark mode orchestration, comparison accuracy, performance overhead
- **Medium-Risk Areas**: Community detection comparison (Louvain variance), embedding migration
- **Dependencies**: Neo4j 5.17+, GDS 2.6+, Python neo4j driver

### Dark Mode Advantages

1. **Risk Mitigation**: Validates with 100% of real traffic before cutover
2. **Confidence Building**: Weeks of comparison data builds team confidence
3. **Early Issue Detection**: Find discrepancies before they affect users
4. **Performance Validation**: Measure actual latency in production environment
5. **Easy Rollback**: Instant revert with zero data loss

### Dark Mode Costs

1. **Development**: +20% implementation time (4 extra weeks)
2. **Runtime**: 2x compute during dark mode period (2-4 weeks)
3. **Storage**: Temporary duplication during validation
4. **Complexity**: More code to maintain (orchestrator, comparison)

**Trade-off**: +20% development cost → 80% risk reduction

### Next Steps

With architecture design complete, we can now:

1. ✅ Defined Neo4j schema for GraphRAG
2. ✅ Designed storage abstraction layer with dark mode support
3. ✅ Designed dark mode orchestrator architecture
4. ✅ Designed comparison framework and metrics collection
5. ✅ Mapped indexing pipeline to Neo4j operations
6. ✅ Planned safe migration path via dark mode
7. ⏳ Create proof-of-concept implementation
8. ⏳ Run performance benchmarks
9. ⏳ Implement dark mode framework
10. ⏳ Finalize benefits/trade-offs analysis

---

**Status**: ✅ Complete (Replanned for Dark Mode)
**Next Document**: `04_performance_benchmarks.md` - Performance comparison with real data (POC required)
**Date Updated**: 2026-01-31
