# GraphRAG Neo4j Migration Assessment Plan

**Date**: 2026-01-29
**Objective**: Assess feasibility and approach for migrating GraphRAG from NetworkX to Neo4j as primary graph storage and vector store
**Status**: Planning Phase

---

## Executive Summary

This assessment evaluates replacing GraphRAG's current in-memory NetworkX graph processing with Neo4j as a persistent graph database and integrated vector store. The goal is to understand benefits, challenges, and implementation strategy.

---

## Current Architecture (Baseline)

### How GraphRAG Currently Uses NetworkX

#### 1. **Community Detection** (Step 7 - create_communities)
**Location**: `packages/graphrag/graphrag/index/operations/cluster_graph.py`

**Current Flow**:
```python
# Load entities and relationships from Parquet
entities_df = pd.read_parquet("entities.parquet")
relationships_df = pd.read_parquet("relationships.parquet")

# Build in-memory NetworkX graph
G = nx.Graph()
for entity in entities_df:
    G.add_node(entity["title"], **entity.to_dict())
for rel in relationships_df:
    G.add_edge(rel["source"], rel["target"], weight=rel["weight"])

# Run Leiden algorithm (via leidenalg library)
communities = leiden_clustering(G)
```

**NetworkX Operations**:
- `nx.Graph()` - Create undirected graph
- `G.add_node()` - Add entities as nodes
- `G.add_edge()` - Add relationships as edges
- `G.degree()` - Calculate node degrees
- Leiden algorithm via `igraph`/`leidenalg` (converts NetworkX → igraph)

#### 2. **Graph Metrics Calculation** (Step 5 - finalize_graph)
**Location**: `packages/graphrag/graphrag/index/workflows/finalize_graph.py`

**Current Flow**:
```python
# Calculate node degrees
degrees = dict(G.degree())
entities_df["node_degree"] = entities_df["title"].map(degrees)

# Calculate combined degrees for relationships
relationship["combined_degree"] = (
    source_degree + target_degree
)
```

**NetworkX Operations**:
- `G.degree()` - Node connectivity count
- Graph structure validation

#### 3. **Vector Storage**
**Current Implementation**: Uses `graphrag-vectors` package

**Supported Vector Stores**:
- LanceDB (default, local file-based)
- Qdrant
- OpenSearch
- Azure AI Search
- Pinecone
- Milvus

**Current Flow**:
```python
# Separate from graph storage
vector_store = create_vector_store(config.vector_store)
vector_store.add_documents(
    collection="entity_description",
    ids=entity_ids,
    embeddings=embeddings
)
```

**Storage Separation**:
- ✅ Graph data: Parquet files
- ✅ Vector embeddings: Separate vector store
- ❌ No integration between graph and vectors

---

## Proposed Architecture (Neo4j-Based)

### Vision: Unified Graph + Vector Storage in Neo4j

#### 1. **Graph Storage in Neo4j**
Replace Parquet files + NetworkX with Neo4j native graph:

```cypher
// Entities as nodes
CREATE (e:Entity {
    id: "e_001",
    title: "Microsoft",
    type: "ORGANIZATION",
    description: "Technology company...",
    node_degree: 15,
    node_frequency: 23
})

// Relationships as edges
CREATE (source)-[:RELATED_TO {
    weight: 8.0,
    description: "Co-founded...",
    combined_degree: 23
}]->(target)
```

#### 2. **Vector Embeddings in Neo4j**
Use Neo4j Vector Index (available in Neo4j 5.11+):

```cypher
// Create vector index on entity descriptions
CREATE VECTOR INDEX entity_description_vector
FOR (e:Entity)
ON e.description_embedding
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 1536,
    `vector.similarity_function`: 'cosine'
  }
}

// Store embedding with entity
CREATE (e:Entity {
    id: "e_001",
    title: "Microsoft",
    description_embedding: [0.021, -0.045, ...]  // 1536-dim vector
})
```

#### 3. **Community Detection in Neo4j**
Use Neo4j Graph Data Science (GDS) library:

```cypher
// Create GDS projection
CALL gds.graph.project(
    'graphrag-graph',
    'Entity',
    'RELATED_TO',
    {relationshipProperties: ['weight']}
)

// Run Louvain (similar to Leiden)
CALL gds.louvain.write('graphrag-graph', {
    writeProperty: 'community',
    relationshipWeightProperty: 'weight',
    includeIntermediateCommunities: true
})
```

---

## Assessment Areas

### 1. Technical Feasibility

**Questions to Answer**:
- ✅ Can Neo4j replace all NetworkX operations?
- ✅ Does Neo4j support required graph algorithms?
- ✅ Can Neo4j vector index handle GraphRAG embedding volumes?
- ✅ What's the performance comparison (NetworkX vs Neo4j)?
- ✅ Are there feature gaps (Leiden vs Louvain)?

**Assessment Tasks**:
1. Map all NetworkX operations to Neo4j equivalents
2. Verify Neo4j GDS algorithm availability
3. Test Neo4j vector index with sample embeddings
4. Benchmark performance on sample graphs
5. Identify missing features or limitations

### 2. Architecture Analysis

**Questions to Answer**:
- How would the indexing pipeline change?
- Where does Neo4j fit in the workflow?
- How to handle streaming writes during indexing?
- What happens to Parquet outputs (backward compatibility)?
- How to integrate with existing `graphrag-vectors` abstraction?

**Assessment Tasks**:
1. Design revised pipeline architecture
2. Define Neo4j schema for GraphRAG data model
3. Plan data flow (LLM → Neo4j)
4. Design adapter layer for `graphrag-storage` interface
5. Plan backward compatibility strategy

### 3. Performance Implications

**Questions to Answer**:
- Is Neo4j faster or slower than NetworkX for community detection?
- What's the overhead of network I/O vs in-memory?
- How does vector search performance compare to LanceDB?
- Can Neo4j handle large graphs (1M+ nodes)?
- What are memory requirements?

**Assessment Tasks**:
1. Benchmark community detection (NetworkX vs Neo4j GDS)
2. Benchmark vector search (LanceDB vs Neo4j Vector Index)
3. Test scalability with increasing graph sizes
4. Measure indexing throughput
5. Profile memory usage

### 4. Benefits Analysis

**Potential Benefits**:
- ✅ **Unified storage**: Graph + vectors in one place
- ✅ **Persistent storage**: No need to reload graph into memory
- ✅ **ACID transactions**: Consistency guarantees
- ✅ **Concurrent access**: Multiple readers/writers
- ✅ **Rich query language**: Cypher for complex patterns
- ✅ **Built-in visualization**: Neo4j Browser
- ✅ **Production-ready**: Enterprise features (clustering, backup, monitoring)
- ✅ **Hybrid search**: Combine vector similarity + graph traversal in single query

**Assessment Tasks**:
1. Quantify each benefit with concrete examples
2. Identify new capabilities enabled by Neo4j
3. Evaluate operational improvements
4. Assess developer experience improvements

### 5. Trade-offs and Challenges

**Potential Challenges**:
- ❌ **External dependency**: Requires Neo4j installation/hosting
- ❌ **Network overhead**: I/O latency vs in-memory
- ❌ **Algorithm differences**: Leiden (NetworkX) vs Louvain (Neo4j)
- ❌ **Learning curve**: Cypher query language
- ❌ **Backward compatibility**: Existing Parquet-based workflows
- ❌ **Cost**: Neo4j Enterprise licensing for production
- ❌ **Migration effort**: Code changes across multiple packages

**Assessment Tasks**:
1. Quantify performance overhead
2. Evaluate algorithm equivalence
3. Estimate development effort
4. Calculate infrastructure costs
5. Design mitigation strategies

### 6. Implementation Strategy

**Phases**:
1. **Phase 1: Proof of Concept** (2-3 weeks)
   - Implement basic Neo4j adapter
   - Test community detection with GDS
   - Validate vector index functionality
   - Benchmark on sample data

2. **Phase 2: Core Integration** (4-6 weeks)
   - Implement full Neo4j storage backend
   - Integrate with indexing pipeline
   - Support both Parquet and Neo4j output
   - Add configuration options

3. **Phase 3: Production Readiness** (4-6 weeks)
   - Performance optimization
   - Error handling and recovery
   - Documentation and examples
   - Migration tools for existing indexes

4. **Phase 4: Advanced Features** (ongoing)
   - Hybrid queries (graph + vector)
   - Real-time updates
   - Distributed deployment
   - Monitoring and observability

**Assessment Tasks**:
1. Define detailed implementation plan
2. Identify required code changes
3. Estimate development time
4. Plan testing strategy
5. Design rollout approach

### 7. Use Case Validation

**Target Use Cases**:
1. **Real-time updates**: Add new entities without full re-index
2. **Hybrid search**: "Find entities similar to X that are connected to Y"
3. **Graph analytics**: Run PageRank, betweenness centrality on live data
4. **Multi-user access**: Concurrent indexing and querying
5. **Large-scale graphs**: Handle 10M+ nodes efficiently

**Assessment Tasks**:
1. Design query patterns for each use case
2. Implement prototypes
3. Measure performance
4. Validate functionality
5. Document best practices

---

## Assessment Deliverables

### Documents to Create

1. **Current Architecture Analysis** (`01_current_architecture.md`)
   - Detailed NetworkX usage map
   - All graph operations inventory
   - Data flow diagrams
   - Performance baselines

2. **Neo4j Capabilities Assessment** (`02_neo4j_capabilities.md`)
   - Feature comparison matrix
   - Algorithm equivalence analysis
   - Vector index capabilities
   - Limitations and gaps

3. **Architecture Design** (`03_architecture_design.md`)
   - Proposed Neo4j schema
   - Integration points
   - Data flow redesign
   - API changes

4. **Performance Benchmarks** (`04_performance_benchmarks.md`)
   - Community detection comparison
   - Vector search comparison
   - Scalability analysis
   - Memory profiling

5. **Benefits and Trade-offs** (`05_benefits_tradeoffs.md`)
   - Quantified benefits
   - Risk analysis
   - Cost analysis
   - Decision matrix

6. **Implementation Plan** (`06_implementation_plan.md`)
   - Phase breakdown
   - Task list with estimates
   - Resource requirements
   - Timeline

7. **Migration Strategy** (`07_migration_strategy.md`)
   - Backward compatibility plan
   - Data migration tools
   - Rollout strategy
   - Rollback procedures

8. **Proof of Concept Code** (`poc/`)
   - Neo4j adapter implementation
   - Sample scripts
   - Benchmark code
   - Test cases

---

## Success Criteria

### Must Have
- ✅ Feature parity with NetworkX for all graph operations
- ✅ Vector index supports all embedding types (text units, entities, communities)
- ✅ Performance within 2x of NetworkX for community detection
- ✅ Backward compatibility with Parquet outputs (dual-mode)
- ✅ Clear migration path for existing users

### Should Have
- ✅ Performance equivalent or better than LanceDB for vector search
- ✅ Support for incremental updates (add entities without full rebuild)
- ✅ Hybrid query capabilities (graph + vector in single query)
- ✅ Production deployment guide (Docker, Kubernetes)

### Nice to Have
- ✅ Performance better than NetworkX (leverage GDS optimizations)
- ✅ Real-time streaming updates
- ✅ Distributed deployment support
- ✅ Advanced analytics (PageRank, centrality, etc.)

---

## Timeline

### Week 1: Research & Analysis
- Study current NetworkX usage
- Research Neo4j capabilities
- Create architecture diagrams
- Define detailed assessment plan

**Deliverables**:
- `01_current_architecture.md`
- `02_neo4j_capabilities.md`

### Week 2: Design & Prototyping
- Design Neo4j schema
- Implement basic POC
- Test community detection
- Test vector indexing

**Deliverables**:
- `03_architecture_design.md`
- `poc/basic_neo4j_adapter.py`

### Week 3: Performance Testing
- Run benchmarks
- Analyze results
- Identify bottlenecks
- Optimize queries

**Deliverables**:
- `04_performance_benchmarks.md`
- Benchmark scripts and data

### Week 4: Analysis & Decision
- Complete benefits/trade-offs analysis
- Create implementation plan
- Design migration strategy
- Present findings

**Deliverables**:
- `05_benefits_tradeoffs.md`
- `06_implementation_plan.md`
- `07_migration_strategy.md`
- Final recommendation presentation

---

## Open Questions

### Technical
1. How to handle the NetworkX → igraph conversion for Leiden?
   - Does Neo4j GDS Louvain produce equivalent results?
   - Can we use hierarchical Louvain for same community structure?

2. What's the optimal Neo4j schema for GraphRAG?
   - Single Entity label or typed labels (Person, Organization, etc.)?
   - Property graph vs separate embedding nodes?

3. How to handle large embeddings (1536+ dimensions)?
   - Store in Neo4j properties or external blob storage?
   - Performance impact of large property sizes?

4. What about other vector stores (Qdrant, Pinecone)?
   - Do we still support them or only Neo4j vectors?
   - How to maintain `graphrag-vectors` abstraction?

### Operational
5. How to deploy Neo4j for GraphRAG?
   - Docker Compose sufficient?
   - Need Enterprise edition for production?
   - Cloud-managed vs self-hosted?

6. What about existing users with Parquet-based workflows?
   - Dual-mode support (Parquet + Neo4j)?
   - Migration tools?
   - Deprecation timeline?

7. How to handle backup and recovery?
   - Neo4j native backup?
   - Export to Parquet for portability?

### Strategic
8. Is this the right direction for GraphRAG?
   - Does it align with project goals?
   - Is the complexity justified?
   - What about other graph databases (Amazon Neptune, TigerGraph)?

---

## Risk Assessment

### High Risk
- **Performance Degradation**: Neo4j slower than NetworkX
  - **Mitigation**: Thorough benchmarking, query optimization
- **Algorithm Mismatch**: Louvain ≠ Leiden, different community structures
  - **Mitigation**: Test on real data, validate quality
- **Backward Incompatibility**: Breaking changes for existing users
  - **Mitigation**: Dual-mode support, gradual migration

### Medium Risk
- **Operational Complexity**: Neo4j adds deployment complexity
  - **Mitigation**: Docker Compose templates, documentation
- **Cost**: Neo4j Enterprise licensing
  - **Mitigation**: Start with Community Edition, evaluate ROI
- **Learning Curve**: Team needs to learn Cypher, Neo4j operations
  - **Mitigation**: Training, documentation, examples

### Low Risk
- **Feature Gaps**: Missing NetworkX features
  - **Mitigation**: Neo4j GDS is mature, covers most needs
- **Vendor Lock-in**: Dependency on Neo4j
  - **Mitigation**: Abstract behind `graphrag-storage` interface

---

## Next Steps

### Immediate Actions (This Week)
1. ✅ Create assessment plan (this document)
2. ⏳ Map all NetworkX operations used in GraphRAG
3. ⏳ Research Neo4j Vector Index capabilities (5.11+)
4. ⏳ Research Neo4j GDS algorithms (Louvain vs Leiden)
5. ⏳ Set up test Neo4j instance with GDS plugin

### Week 2 Actions
1. ⏳ Design Neo4j schema for GraphRAG entities/relationships
2. ⏳ Implement basic Neo4j adapter POC
3. ⏳ Test community detection with sample data
4. ⏳ Test vector indexing with sample embeddings
5. ⏳ Document findings

### Week 3 Actions
1. ⏳ Run comprehensive benchmarks
2. ⏳ Analyze performance data
3. ⏳ Test scalability (10K, 100K, 1M nodes)
4. ⏳ Document performance results

### Week 4 Actions
1. ⏳ Complete benefit/trade-off analysis
2. ⏳ Create detailed implementation plan
3. ⏳ Design migration strategy
4. ⏳ Present findings and recommendation

---

## Team & Resources

### Required Expertise
- GraphRAG architecture (current codebase)
- Neo4j and Cypher
- Graph algorithms (community detection, centrality)
- Vector databases and similarity search
- Python graph libraries (NetworkX, igraph)
- Performance benchmarking

### Infrastructure Needs
- Neo4j instance (Community or Enterprise)
- Test data sets (small, medium, large graphs)
- Benchmark server (consistent environment)
- Sample GraphRAG indexes for testing

---

## Conclusion

This assessment will provide a comprehensive evaluation of migrating GraphRAG from NetworkX to Neo4j. The goal is to make an **informed, data-driven decision** on whether this migration is beneficial, feasible, and worth the investment.

**Expected Outcome**: A clear recommendation (GO/NO-GO) with detailed justification, implementation plan, and risk mitigation strategies.

---

## References

### Neo4j Documentation
- [Neo4j Vector Index](https://neo4j.com/docs/cypher-manual/current/indexes-for-vector-search/)
- [Neo4j Graph Data Science](https://neo4j.com/docs/graph-data-science/current/)
- [Neo4j Python Driver](https://neo4j.com/docs/python-manual/current/)

### GraphRAG Codebase
- `packages/graphrag/graphrag/index/operations/cluster_graph.py`
- `packages/graphrag/graphrag/index/operations/create_graph.py`
- `packages/graphrag/graphrag/index/workflows/finalize_graph.py`
- `packages/graphrag-vectors/` - Current vector store abstraction

### Algorithms
- [Leiden Algorithm Paper](https://www.nature.com/articles/s41598-019-41695-z)
- [Louvain Algorithm](https://en.wikipedia.org/wiki/Louvain_method)
- [Neo4j GDS Louvain](https://neo4j.com/docs/graph-data-science/current/algorithms/louvain/)

---

**Status**: ✅ Planning Complete - Ready to begin assessment
**Next Update**: Week 1 completion - Current Architecture Analysis
