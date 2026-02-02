# Neo4j Claims/Covariates Support Assessment

**Date:** 2026-02-01
**Current Status:** Claims extraction is enabled and working, but Neo4j backend does NOT support loading covariates

---

## Executive Summary

✅ **Claims Extraction:** WORKING - 194 covariates extracted from the Christmas Carol document
❌ **Neo4j Integration:** NOT IMPLEMENTED - No covariate loading capability in Neo4jBackend
⚠️ **Impact:** Claims data exists in parquet files but is not loaded into Neo4j graph

---

## Current Claims/Covariates Data

### Data Statistics
- **Total covariates extracted:** 194
- **Covariates with relationships (subject→object):** 182 (94%)
- **Covariates as attributes (no object):** 12 (6%)

### Covariate Schema (from output/covariates.parquet)
```
Columns:
- id                 : UUID for covariate
- human_readable_id  : Sequential integer ID
- covariate_type     : Always "claim"
- type               : Claim type (AUTHORSHIP, DEATH, EMPLOYMENT RELATIONSHIP, etc.)
- description        : Full text description of the claim
- subject_id         : Entity the claim is about (matches entity title)
- object_id          : Related entity (for relationships), or None
- status             : Claim status (optional)
- start_date         : Temporal start (optional)
- end_date           : Temporal end (optional)
- source_text        : Text snippet where claim was extracted
- text_unit_id       : ID of text unit containing this claim
```

### Top Claim Types
1. DEATH (7 claims)
2. EMPLOYMENT RELATIONSHIP (6 claims)
3. BUSINESS PARTNERSHIP (5 claims)
4. SUPERNATURAL VISITATION (5 claims)
5. THEFT FROM DECEASED (4 claims)
6. FAMILY RELATIONSHIP (4 claims)
7. SOCIAL ISOLATION (3 claims)
8. PERSONAL TRANSFORMATION (3 claims)
9. SUPERNATURAL ENCOUNTER (3 claims)
10. MEDICAL CONDITION (3 claims)

### Example Covariate
```json
{
  "id": "uuid-here",
  "human_readable_id": 0,
  "covariate_type": "claim",
  "type": "AUTHORSHIP",
  "description": "Charles Dickens is the author of 'A Christmas Carol'...",
  "subject_id": "CHARLES DICKENS",
  "object_id": null,
  "status": null,
  "start_date": null,
  "end_date": null,
  "source_text": "...preface signed by C.D. dated December 1843...",
  "text_unit_id": "d4eaebda..."
}
```

---

## Neo4j Backend Current State

### What's Implemented
- ✅ Entity nodes with multi-type labels (`:Entity:Person`, `:Entity:Organization`, etc.)
- ✅ RELATED_TO relationships between entities
- ✅ TextUnit nodes with MENTIONS relationships to entities
- ✅ Community nodes with CONTAINS and PARENT_OF relationships
- ✅ Document nodes with HAS_TEXT_UNIT relationships

### What's NOT Implemented
- ❌ Claim/Covariate nodes
- ❌ Relationships based on claims
- ❌ Claim metadata (type, temporal info, source)
- ❌ Links between claims and entities
- ❌ Links between claims and text units

---

## Proposed Neo4j Schema for Claims

### Option 1: Claims as Separate Nodes (RECOMMENDED)

**Rationale:** Preserves all claim metadata, enables claim-based queries, maintains provenance

```cypher
# Node Types
(:Claim {
  id: string,
  human_readable_id: int,
  covariate_type: string,
  type: string,              # AUTHORSHIP, DEATH, etc.
  description: string,       # Full claim text
  status: string,           # Optional
  start_date: string,       # Optional temporal
  end_date: string,         # Optional temporal
  source_text: string       # Text snippet
})

# Relationships
(:Claim)-[:ABOUT]->(:Entity)           # subject_id
(:Claim)-[:INVOLVES]->(:Entity)        # object_id (if present)
(:Claim)-[:EXTRACTED_FROM]->(:TextUnit)

# Example queries enabled:
MATCH (c:Claim)-[:ABOUT]->(e:Entity {title: "CHARLES DICKENS"})
RETURN c.type, c.description

MATCH (c:Claim {type: "DEATH"})-[:ABOUT]->(subject),
      (c)-[:INVOLVES]->(object)
RETURN subject.title, object.title, c.description
```

**Advantages:**
- ✅ Full claim metadata preserved
- ✅ Enables claim-based queries and analytics
- ✅ Clear provenance (which text unit, which entities)
- ✅ Temporal queries on claims (start_date, end_date)
- ✅ Can aggregate claims by type, status, etc.

**Disadvantages:**
- ❌ More nodes and relationships (adds ~194 nodes, ~400+ relationships for this dataset)
- ❌ Slightly more complex query patterns

---

### Option 2: Claims as Relationship Properties (Alternative)

**Rationale:** Simpler schema, claims enhance existing entity relationships

```cypher
# Enhance existing RELATED_TO relationships with claim properties
(:Entity)-[:RELATED_TO {
  weight: float,
  description: string,
  claims: [                   # Array of claim objects
    {
      id: string,
      type: string,
      description: string,
      source_text: string,
      text_unit_id: string
    }
  ]
}]->(:Entity)
```

**Advantages:**
- ✅ Simpler schema (no new node type)
- ✅ Claims directly annotate entity relationships

**Disadvantages:**
- ❌ Loses claims without object_id (12 attribute claims)
- ❌ Cannot query claims independently
- ❌ Temporal information harder to query
- ❌ No direct link between claim and text unit nodes
- ❌ Duplicate claim data if multiple entities involved

---

## Recommendation: OPTION 1 (Separate Claim Nodes)

Option 1 is strongly recommended because:

1. **Data Fidelity:** Preserves all 194 claims including 12 attribute claims (no object_id)
2. **Query Flexibility:** Enables rich claim-based queries (e.g., "all death claims", "claims from 1840s")
3. **Provenance:** Clear links to source text units and involved entities
4. **Consistency:** Matches the existing pattern (TextUnit, Community are separate nodes)
5. **Future-Proof:** Supports claim evolution, versioning, confidence scores

---

## Implementation Plan

### Phase 1: Add Claim Loading to Neo4jBackend

**File:** `packages/graphrag/graphrag/index/graph/neo4j_backend.py`

Add new method:
```python
def load_claims(
    self,
    covariates: pd.DataFrame,
) -> None:
    """Load claims/covariates into Neo4j.

    Creates Claim nodes and relationships:
    - ABOUT: Claim -> Entity (subject)
    - INVOLVES: Claim -> Entity (object, if present)
    - EXTRACTED_FROM: Claim -> TextUnit
    """
```

**Implementation steps:**
1. Clear existing Claim nodes
2. Create Claim nodes in batches (convert dates to strings, handle nulls)
3. Create ABOUT relationships (subject_id -> entity)
4. Create INVOLVES relationships (object_id -> entity, if not null)
5. Create EXTRACTED_FROM relationships (text_unit_id -> TextUnit)

**Estimated complexity:** ~150 lines of code

---

### Phase 2: Update Workflow Integration

**File:** `packages/graphrag/graphrag/index/workflows/load_neo4j_graph.py`

Add claim loading after communities:
```python
# Load claims
logger.info("Loading claims")
covariates = await load_table_from_storage("covariates", context.output_storage)
if covariates is not None and not covariates.empty:
    logger.info("Loading %d claims into Neo4j", len(covariates))
    backend.load_claims(covariates)
else:
    logger.warning("No covariates found in storage")
```

**Estimated complexity:** ~15 lines of code

---

### Phase 3: Testing and Verification

**Test queries:**
```cypher
// Count claims by type
MATCH (c:Claim)
RETURN c.type AS claim_type, count(*) AS count
ORDER BY count DESC

// Find all claims about a specific entity
MATCH (c:Claim)-[:ABOUT]->(e:Entity {title: "SCROOGE"})
RETURN c.type, c.description

// Find relationship claims (subject->object)
MATCH (c:Claim)-[:ABOUT]->(subject:Entity),
      (c)-[:INVOLVES]->(object:Entity)
RETURN subject.title, c.type, object.title, c.description
LIMIT 10

// Trace claim provenance
MATCH (c:Claim)-[:EXTRACTED_FROM]->(t:TextUnit)<-[:HAS_TEXT_UNIT]-(d:Document)
RETURN d.title, t.id, c.type, c.description
LIMIT 5

// Temporal claims
MATCH (c:Claim)
WHERE c.start_date IS NOT NULL
RETURN c.type, c.description, c.start_date, c.end_date
```

---

## Resource Requirements

### Storage Impact
- **Claim nodes:** ~194 nodes (for current dataset)
- **Relationships:** ~400 relationships (1 ABOUT per claim, 1 INVOLVES if object_id, 1 EXTRACTED_FROM per claim)
- **Estimated size increase:** ~10-15% of current graph size

### Performance Impact
- **Load time:** +2-5 seconds to indexing pipeline
- **Query performance:** Minimal impact, claims are optional in most queries

---

## Benefits of Adding Claims Support

1. **Richer Knowledge Graph:** Claims add factual assertions with provenance
2. **Enhanced Query Capabilities:**
   - "What do we know about this entity?" (all claims about it)
   - "What events happened in 1843?" (temporal claims)
   - "Show me all family relationships" (claims by type)
3. **Improved RAG Quality:** Claims provide structured facts for retrieval
4. **Better Explainability:** Claims link answers to source text
5. **Competitive Feature:** Many graph RAG systems don't have structured claims

---

## Risk Assessment

### Low Risk ✅
- Claims are optional - can be disabled via `extract_claims.enabled: false`
- No breaking changes to existing schema
- Claims load independently from other data
- Can be added/removed without affecting other nodes

### Medium Risk ⚠️
- Increased graph size may impact memory usage for very large datasets
- Mitigation: Use batch processing, add claim type filtering config

---

## Alternatives Considered

### Do Nothing
- ❌ Wastes valuable extracted data (194 claims already generated)
- ❌ Missing query capabilities
- ❌ Incomplete Neo4j representation of GraphRAG knowledge

### Load Claims Later (Manual Script)
- ⚠️ Creates inconsistency (some runs have claims, others don't)
- ⚠️ User has to remember to run manual script
- ✅ Lower initial implementation effort

### Load to Separate Database
- ❌ Breaks unified query capability
- ❌ Requires multiple database connections
- ❌ More complex application architecture

---

## Conclusion

**ASSESSMENT: Claims/Covariates are NOT currently supported in Neo4j backend**

**RECOMMENDATION: Implement Option 1 (Separate Claim Nodes)**

**PRIORITY: MEDIUM-HIGH**
- Claims data is already being extracted (194 claims)
- Implementation is straightforward (~165 lines of code)
- Significant value for advanced queries and RAG quality
- Low risk, optional feature

**ESTIMATED EFFORT:** 3-4 hours
- Implementation: 2-3 hours
- Testing: 1 hour
- Documentation: 30 minutes

**NEXT STEPS:**
1. Implement `load_claims()` method in Neo4jBackend
2. Integrate into `load_neo4j_graph` workflow
3. Test with sample queries
4. Document claim schema in README
5. (Optional) Add claim filtering config to limit by type/date

---

## Appendix: Full Graph Schema with Claims

```
# Nodes (7 types):
(:Entity:Person)         - 109 nodes
(:Entity:Organization)   - 15 nodes
(:Entity:Geo)           - 25 nodes
(:Entity:Event)         - 10 nodes
(:TextUnit)             - 42 nodes
(:Community)            - 44 nodes
(:Document)             - 1 node
(:Claim)                - 194 nodes (NEW)

# Relationships (8 types):
RELATED_TO              - Entity to Entity (352)
MENTIONS                - TextUnit to Entity (336)
CONTAINS                - Community to Entity (293)
HAS_TEXT_UNIT          - Document to TextUnit (42)
PARENT_OF              - Community to Community (31)
ABOUT                  - Claim to Entity (194, NEW)
INVOLVES               - Claim to Entity (182, NEW)
EXTRACTED_FROM         - Claim to TextUnit (194, NEW)

# Total with Claims:
Nodes: 440 (246 + 194)
Relationships: 1,624 (1,054 + 570)
```
