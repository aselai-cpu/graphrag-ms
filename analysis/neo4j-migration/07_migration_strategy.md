# Migration Strategy
## Dark Mode Enabled Migration

**Document**: 07 - Migration Strategy
**Date**: 2026-01-31
**Status**: Replanned for Dark Mode

---

## Purpose

This document provides a comprehensive migration strategy for transitioning existing GraphRAG users from NetworkX + LanceDB to Neo4j using a **dark mode validation approach**. It includes step-by-step guides, dark mode configuration, migration tools, rollback procedures, and support resources.

---

## Overview

### Migration Philosophy

**Principles**:
1. **Zero Forced Migration**: Users can continue using NetworkX indefinitely
2. **Dark Mode Validation**: Test Neo4j in parallel without production risk
3. **Data Safety**: No data loss at any point
4. **Easy Rollback**: Can revert at any time with one config change
5. **Clear Communication**: Users understand benefits and trade-offs
6. **Risk-Free Cutover**: Full validation before production impact

### Migration Paths (Dark Mode Strategy)

```
Current State (NetworkX + LanceDB)
        ↓
    [Choose Path]
        ↓
   ┌────┴────────────┐
   │                 │
Path A           Path B
(Stay)        (Validate & Migrate)
   │                 │
   │           Enable Dark Mode
   │                 ↓
   │           Run in Parallel (NetworkX + Neo4j)
   │                 ↓
   │           Collect Metrics (2-4 weeks)
   │                 ↓
   │           Review Comparison Report
   │                 ↓
   │           ┌─────┴─────┐
   │           │           │
   │        Metrics    Metrics
   │         Pass       Fail
   │           │           │
   │      Switch to    Fix Issues
   │      Neo4j Only    or Stay
   │           ↓           │
   └────→ [End State] ←────┘
           (User Choice)
```

**Key Insight**: Dark mode eliminates "hope and pray" migrations. Users validate with 100% of their real data before cutover.

---

## User Segments

### Segment 1: Simple Use Cases

**Profile**:
- Small datasets (< 100 documents)
- Single-user, local development
- Infrequent re-indexing
- No real-time requirements

**Recommendation**: ✅ **Stay on NetworkX**

**Configuration**:
```yaml
storage:
  type: networkx_only
```

**Reasoning**:
- NetworkX + LanceDB is simpler (no database setup)
- Performance difference negligible for small datasets
- File-based storage is portable

**Messaging**:
> "For small, local projects, NetworkX + LanceDB remains the simplest option. Neo4j is available when you need its advanced features."

---

### Segment 2: Growing Projects

**Profile**:
- Medium datasets (100-1000 documents)
- Team collaboration
- Regular re-indexing
- Experimenting with features

**Recommendation**: ⚠️ **Try Dark Mode**

**Configuration**:
```yaml
storage:
  type: dark_mode  # Run both, validate Neo4j
  networkx:
    enabled: true
  neo4j:
    enabled: true
    uri: bolt://localhost:7687
  dark_mode:
    comparison:
      enabled: true
```

**Reasoning**:
- Validate Neo4j with your actual data
- See performance benefits (6x faster community detection)
- Test hybrid queries
- Keep NetworkX as safety net (zero risk)
- Generate comparison report to inform decision

**Messaging**:
> "Try Neo4j in dark mode to validate it works with your data. NetworkX continues serving requests while Neo4j runs in parallel. Review the comparison report to decide if migration is worth it."

**Duration**: Run dark mode for 1-2 weeks, then decide:
- ✅ Switch to `neo4j_only` if metrics pass
- ⏸️ Stay in `dark_mode` longer if needed
- 🔄 Return to `networkx_only` if not worth it

---

### Segment 3: Production Deployments

**Profile**:
- Large datasets (1000+ documents)
- Multi-user access
- Real-time or incremental updates
- Production services

**Recommendation**: ✅ **Migrate to Neo4j via Dark Mode**

**Migration Steps**:
1. **Week 1-2**: Enable dark mode in staging
2. **Week 2-3**: Enable dark mode in production
3. **Week 3-4**: Collect metrics, review comparison report
4. **Week 5**: Cutover to `neo4j_only` (if metrics pass)

**Configuration (Final State)**:
```yaml
storage:
  type: neo4j_only
  neo4j:
    uri: bolt://production-neo4j:7687
    username: neo4j
    password: ${NEO4J_PASSWORD}
```

**Reasoning**:
- 6x faster community detection (critical at scale)
- Concurrent access required for multi-user
- Incremental updates valuable for real-time
- Production features (ACID, backup, monitoring) needed
- **Dark mode ensures safe migration** (validated before cutover)

**Messaging**:
> "Neo4j enables production-ready deployments at scale. Dark mode validation ensures your specific workload performs well before cutover. Zero risk."

**Risk Mitigation**:
- Dark mode validation period: 2-4 weeks
- Comparison metrics: entity match > 99%, query F1 > 95%
- Instant rollback to `networkx_only` if issues found
- No data loss (both systems write data during dark mode)

---

## Migration Timeline (User Perspective)

### Version Roadmap (Dark Mode Strategy)

#### v3.1.0 (Beta) - Month 0 (Week 23)
**Status**: Neo4j + Dark Mode available as opt-in

**Default Configuration**:
```yaml
storage:
  type: networkx_only  # Default (unchanged)
```

**New Modes Available**:
- `networkx_only`: Current default
- `dark_mode`: Parallel validation (NEW) 🆕
- `neo4j_only`: Neo4j only (for early adopters)

**User Action**: None required (backward compatible)

**Who Should Try Dark Mode**:
- Early adopters wanting to validate Neo4j
- Users with performance issues
- Users needing hybrid queries
- Anyone curious about Neo4j

**Messaging**:
> "Try dark mode risk-free! NetworkX continues serving requests while Neo4j runs in parallel. Get a comparison report to see if Neo4j is right for you."

---

#### v3.1.x (Stable) - Month 3-5 (Weeks 35-43)
**Status**: Neo4j production-ready, Dark mode validated by core team

**Default Configuration**:
```yaml
storage:
  type: networkx_only  # Still default (stable)
```

**User Action**: None required

**Who Should Migrate via Dark Mode**:
- Production deployments
- Large datasets
- Real-time requirements
- Teams wanting risk-free migration

**Recommended Migration Process**:
1. Enable `dark_mode` in staging (1-2 weeks)
2. Enable `dark_mode` in production (2-4 weeks)
3. Review comparison report
4. Switch to `neo4j_only` (if validated)

---

#### v3.2.0 - Month 6-8 (Weeks 44-60)
**Status**: Neo4j recommended default for new projects

**Default Configuration (New Projects)**:
```yaml
storage:
  type: neo4j_only  # New default for new projects
```

**Existing Projects**: Unchanged (`networkx_only`)

**User Action**:
- **New projects**: Follow Neo4j setup guide
- **Existing projects**: Consider dark mode migration

**Who Should Migrate**:
- All new projects (unless simple use case)
- Existing projects with scale/performance needs
- Teams wanting modern infrastructure

**Dark Mode Success Stories**:
- Blog posts showcasing dark mode migrations
- Case studies with metrics
- User testimonials

---

#### v3.3.0 - Month 12+ (Week 88+)
**Status**: NetworkX deprecated (optional)

**Warning Shown** (if using `networkx_only`):
```
⚠️  NetworkX storage is deprecated and will be removed in v4.0.0
   Migrate to Neo4j using dark mode: docs/neo4j/dark_mode_migration.md
   Dark mode enables risk-free validation before cutover.
   To suppress this warning: --no-deprecation-warnings
```

**User Action**:
- **Option A**: Migrate to Neo4j via dark mode (recommended)
- **Option B**: Accept staying on v3.x long-term
- **Option C**: Evaluate if NetworkX meets your needs indefinitely

**Support**:
- NetworkX remains fully supported in v3.x
- Security updates continue
- Bug fixes continue

---

#### v4.0.0 - Month 18+ (Optional, Conditional)
**Status**: NetworkX removed (only if conditions met)

**Removal Only if ALL True**:
- > 90% of active users migrated to Neo4j
- Strong business case for removal
- Significant maintenance burden
- Alternative: Keep NetworkX indefinitely

**User Action** (if removal happens):
- Must migrate to Neo4j
- Dark mode migration path still available in v3.x
- Migration support provided

**Note**: This removal is **optional** and may never happen if NetworkX users remain significant.
- Or stay on v3.x LTS (long-term support)

---

## Step-by-Step Migration Guide

### Prerequisites

**Required**:
- GraphRAG v3.1.0 or later
- Existing Parquet-based index

**Neo4j Installation** (choose one):
1. **Docker** (recommended for development)
2. **Neo4j Desktop** (GUI, good for exploration)
3. **Neo4j Aura** (cloud, good for production)
4. **Native Installation** (Linux/Mac/Windows)

---

### Step 1: Install Neo4j

#### Option A: Docker (Recommended)

```bash
# Create docker-compose.yml
cat > docker-compose.yml <<EOF
version: '3.8'
services:
  neo4j:
    image: neo4j:5.17.0
    ports:
      - "7474:7474"  # HTTP
      - "7687:7687"  # Bolt
    environment:
      NEO4J_AUTH: neo4j/your-secure-password
      NEO4J_PLUGINS: '["graph-data-science"]'
      NEO4J_dbms_memory_heap_initial__size: 2G
      NEO4J_dbms_memory_heap_max__size: 4G
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs

volumes:
  neo4j_data:
  neo4j_logs:
EOF

# Start Neo4j
docker-compose up -d

# Verify
docker-compose logs neo4j | grep "Started"
# Should see: "Started."
```

#### Option B: Neo4j Aura (Cloud)

```bash
# 1. Sign up at https://neo4j.com/cloud/aura/
# 2. Create a free instance
# 3. Download credentials file
# 4. Note: uri, username, password
```

#### Option C: Neo4j Desktop

```bash
# 1. Download from https://neo4j.com/download/
# 2. Install application
# 3. Create new database
# 4. Install Graph Data Science plugin
# 5. Start database
```

**Verify Installation**:
```bash
# Install Neo4j Python driver
pip install neo4j

# Test connection
python -c "
from neo4j import GraphDatabase
driver = GraphDatabase.driver(
    'bolt://localhost:7687',
    auth=('neo4j', 'your-secure-password')
)
with driver.session() as session:
    result = session.run('RETURN 1')
    print('✅ Connected!')
driver.close()
"
```

---

### Step 2: Update GraphRAG Configuration for Dark Mode

#### Update settings.yaml

```yaml
# Before (NetworkX only)
storage:
  type: networkx_only
  networkx:
    cache_dir: ./cache
    vector_store:
      type: lancedb
      uri: ./output/lancedb

# After (Dark mode - RECOMMENDED for migration)
storage:
  type: dark_mode  # Run both in parallel, validate Neo4j

  # NetworkX config (continues serving production)
  networkx:
    enabled: true
    cache_dir: ./cache
    vector_store:
      type: lancedb
      uri: ./output/lancedb

  # Neo4j config (shadow execution)
  neo4j:
    enabled: true
    uri: "bolt://localhost:7687"
    username: "neo4j"
    password: "${NEO4J_PASSWORD}"  # Use env var for security
    database: "neo4j"

    # Optional: tune performance
    batch_size: 1000
    max_connection_pool_size: 50

    # GDS settings
    gds:
      enabled: true
      max_cluster_size: 10

    # Vector index settings
    vector_index:
      enabled: true
      dimensions: 1536  # Or 384 for SentenceTransformer
      similarity_function: cosine

  # Dark mode specific settings
  dark_mode:
    enabled: true
    primary_backend: networkx  # NetworkX is production
    shadow_backend: neo4j      # Neo4j is shadow

    # Comparison settings
    comparison:
      enabled: true
      log_path: ./dark_mode_logs  # Where to save comparison data
      log_format: jsonl

      # Metrics to collect
      metrics:
        - entity_count
        - relationship_count
        - community_match_rate
        - query_f1
        - query_ranking_correlation
        - latency_ratio
        - error_rates

      # Sample rate (1.0 = 100% of operations)
      sample_rate: 1.0

    # Error handling
    error_handling:
      shadow_failure_action: log  # Don't fail on Neo4j errors
      continue_on_shadow_error: true

    # Cutover criteria (when it's safe to switch)
    cutover_criteria:
      entity_match_rate_threshold: 0.99      # 99% match
      community_match_rate_threshold: 0.95   # 95% match
      query_f1_threshold: 0.95               # 95% F1
      latency_ratio_threshold: 2.0           # < 2x NetworkX
      shadow_error_rate_threshold: 0.01      # < 1% errors
```

**What Dark Mode Does**:
- NetworkX handles all user requests (production)
- Neo4j runs identical operations in parallel (shadow)
- Comparison framework logs differences
- Neo4j failures don't affect users
- You get a detailed comparison report

#### Set Environment Variables

```bash
# Add to .env file
echo "NEO4J_PASSWORD=your-secure-password" >> .env

# Or export directly
export NEO4J_PASSWORD=your-secure-password
```

---

### Step 3: Import Existing Data to Neo4j

#### Option A: Re-index (Recommended)

**Why**: Ensures consistency, updates embeddings, validates workflow

```bash
# Run indexing with hybrid mode
graphrag index --config settings.yaml

# This will:
# 1. Re-extract entities and relationships
# 2. Write to both Parquet and Neo4j
# 3. Run community detection in Neo4j
# 4. Store embeddings in Neo4j vector index
```

**Estimated Time**:
- Small dataset (100 docs): 10-30 minutes
- Medium dataset (1000 docs): 1-3 hours
- Large dataset (10K+ docs): 5-10 hours

#### Option B: Import from Parquet (Faster)

**Why**: Faster, preserves existing communities, no LLM calls needed

```bash
# Import existing Parquet files to Neo4j
graphrag import-to-neo4j \
  --input ./output \
  --neo4j-uri bolt://localhost:7687 \
  --neo4j-user neo4j \
  --neo4j-password your-secure-password

# This will:
# 1. Read entities.parquet, relationships.parquet, etc.
# 2. Create Neo4j nodes and relationships
# 3. Import embeddings to vector indexes
# 4. Verify data integrity
```

**Estimated Time**:
- Small dataset: 1-5 minutes
- Medium dataset: 5-20 minutes
- Large dataset: 30-120 minutes

**Note**: Communities may need to be recalculated if using Louvain instead of Leiden

---

### Step 4: Validate Migration

#### Verify Data Integrity

```bash
# Run validation tool
graphrag validate-neo4j \
  --neo4j-uri bolt://localhost:7687 \
  --neo4j-user neo4j \
  --neo4j-password your-secure-password

# Checks:
# ✓ All entities present
# ✓ All relationships present
# ✓ All communities present
# ✓ Vector indexes created
# ✓ Embeddings stored
# ✓ No orphaned nodes
# ✓ Schema correct
```

#### Compare Results

```bash
# Export Neo4j data to Parquet for comparison
graphrag export-from-neo4j \
  --output ./output-neo4j \
  --neo4j-uri bolt://localhost:7687

# Compare counts
python -c "
import pandas as pd
parquet_entities = pd.read_parquet('./output/entities.parquet')
neo4j_entities = pd.read_parquet('./output-neo4j/entities.parquet')

print(f'Parquet: {len(parquet_entities)} entities')
print(f'Neo4j:   {len(neo4j_entities)} entities')

# Should be identical (or very close)
assert abs(len(parquet_entities) - len(neo4j_entities)) < 10
print('✅ Entity counts match!')
"
```

#### Test Queries

```bash
# Run test queries
graphrag query \
  --method global \
  --query "What are the main themes?" \
  --config settings.yaml

# Try hybrid query (Neo4j only)
graphrag query \
  --method hybrid \
  --query "Find entities about 'technology' connected to 'Microsoft'" \
  --config settings.yaml
```

---

### Step 4: Run Dark Mode and Collect Metrics

With dark mode enabled, run your normal operations:

```bash
# Re-index your data (both NetworkX and Neo4j will process)
graphrag index --config settings.yaml

# Run queries (both backends execute, NetworkX results returned)
graphrag query --method local "your query"
graphrag query --method global "your query"

# Run for 1-2 weeks to collect sufficient data
# - Indexing operations: both backends process
# - Query operations: both backends execute
# - NetworkX results served to users (zero risk)
# - Neo4j results logged for comparison
```

**What's Happening**:
```
User Request → DarkModeOrchestrator
                   ↓         ↓
              NetworkX    Neo4j (shadow)
                   ↓         ↓
              User Gets   Logged for
              Result ✅   Comparison 📊
```

**Monitoring**:
```bash
# Check dark mode logs
tail -f logs/graphrag.log | grep "dark_mode"

# Check Neo4j is running (should see INFO logs, not ERROR)
tail -f logs/graphrag.log | grep "neo4j"

# Watch comparison metrics accumulate
ls -lh dark_mode_logs/
# Should see: comparison_metrics.jsonl growing
```

**Duration**:
- **Minimum**: 1 week, 100+ operations
- **Recommended**: 2-4 weeks, 1000+ operations
- **For production**: Run until confident

---

### Step 5: Review Dark Mode Comparison Report

After collecting data, generate the comparison report:

```bash
# Generate comprehensive report
graphrag dark-mode-report \
  --log-path ./dark_mode_logs \
  --output dark_mode_report.html

# Open report in browser
open dark_mode_report.html  # Mac
xdg-open dark_mode_report.html  # Linux
```

**Report Sections**:

1. **Validation Summary**
   - Period: Feb 1 - Feb 28, 2026
   - Total operations: 2,847
   - Breakdown: 12 indexing, 2,835 queries

2. **Comparison Metrics**
   ```
   Metric                      | Threshold | Actual | Pass
   ----------------------------|-----------|--------|------
   Entity match rate           | > 99%     | 99.8%  | ✅
   Relationship match rate     | > 99%     | 99.7%  | ✅
   Community match rate        | > 95%     | 96.3%  | ✅
   Avg query F1 score          | > 95%     | 97.2%  | ✅
   Avg ranking correlation     | > 0.90    | 0.94   | ✅
   P95 latency ratio           | < 2.0x    | 1.3x   | ✅
   Neo4j error rate            | < 1%      | 0.2%   | ✅
   ```

3. **Performance Analysis**
   - Indexing: Neo4j 6.1x faster (5.2s vs 31.8s)
   - Queries: Neo4j 1.3x faster (15ms vs 19ms p95)
   - Memory: Neo4j uses 1.5x more (acceptable)

4. **Issue Log**
   - 5 Neo4j timeouts (0.2% of operations)
   - 3 community assignment differences (0.1%)
   - All within acceptable thresholds

5. **Cutover Recommendation**
   ```
   ✅ RECOMMENDATION: GO for cutover

   All metrics pass cutover criteria.
   Neo4j is ready for production use.
   ```

**Decision Matrix**:

| Metrics Pass | Recommendation | Action |
|--------------|----------------|--------|
| ✅ All pass | **GO** | Proceed to Step 6 (cutover) |
| ⚠️ Some fail | **INVESTIGATE** | Extend dark mode, fix issues |
| ❌ Many fail | **NO-GO** | Stay on NetworkX, reconsider |

---

### Step 6: Cutover to Neo4j Only (If Metrics Pass)

Once the report shows all metrics pass, switch to Neo4j-only mode:

**Configuration Change**:
```yaml
# settings.yaml
storage:
  type: neo4j_only  # Neo4j only (validated via dark mode)

  neo4j:
    uri: "bolt://localhost:7687"
    username: "neo4j"
    password: "${NEO4J_PASSWORD}"
    database: "neo4j"
```

**Deployment**:
```bash
# 1. Backup current state
cp settings.yaml settings.yaml.backup

# 2. Update configuration to neo4j_only
# (edit settings.yaml as shown above)

# 3. Restart GraphRAG
# (depends on your deployment: restart service, re-run, etc.)

# 4. Verify cutover
graphrag query --method local "test query"
# Should use Neo4j, no dark mode logging

# 5. Monitor closely for 24-48 hours
tail -f logs/graphrag.log | grep "ERROR"
```

**Rollback Procedure** (if issues arise):
```yaml
# Instant rollback - one line change
storage:
  type: networkx_only  # Revert to NetworkX

# Or go back to dark mode for more validation
storage:
  type: dark_mode
```

**Benefits of Cutover**:
- ✅ 6x faster community detection (validated in dark mode)
- ✅ Faster queries (validated in dark mode)
- ✅ Hybrid query capabilities
- ✅ Production features (ACID, backup)
- ✅ No more dark mode overhead (single backend)

**Monitoring Post-Cutover**:
- Watch error rates (should be < 0.1%)
- Monitor latency (should match dark mode predictions)
- Track user feedback
- Ready to rollback if needed

**Success!** 🎉 You've migrated to Neo4j with zero risk using dark mode.

---

## Migration Tools

### Import Tool

**Command**:
```bash
graphrag import-to-neo4j [OPTIONS]
```

**Options**:
```
--input PATH              Path to Parquet files directory (required)
--neo4j-uri TEXT          Neo4j connection URI (required)
--neo4j-user TEXT         Neo4j username (default: neo4j)
--neo4j-password TEXT     Neo4j password (required)
--neo4j-database TEXT     Neo4j database name (default: neo4j)
--batch-size INTEGER      Batch size for imports (default: 1000)
--skip-embeddings         Skip importing embeddings
--skip-communities        Skip importing communities
--skip-covariates         Skip importing covariates
--checkpoint-file PATH    Resume from checkpoint on failure
--verbose                 Verbose logging
```

**Example**:
```bash
graphrag import-to-neo4j \
  --input ./output \
  --neo4j-uri bolt://localhost:7687 \
  --neo4j-password $NEO4J_PASSWORD \
  --batch-size 2000 \
  --verbose
```

**Output**:
```
Importing GraphRAG data to Neo4j...
✓ Connected to Neo4j
✓ Validated schema
→ Importing entities...
  [████████████████████] 1247/1247 (100%)
✓ Imported 1247 entities
→ Importing relationships...
  [████████████████████] 3891/3891 (100%)
✓ Imported 3891 relationships
→ Importing communities...
  [████████████████████] 156/156 (100%)
✓ Imported 156 communities
→ Importing embeddings...
  [████████████████████] 1247/1247 (100%)
✓ Imported 1247 embeddings
→ Creating vector indexes...
✓ Created entity_description_vector
✓ Created community_summary_vector
✓ Created text_unit_vector
→ Validating import...
✓ All data validated

Import complete! ✅
- Entities: 1247
- Relationships: 3891
- Communities: 156
- Text Units: 542
- Documents: 15
- Embeddings: 1247
- Time: 3m 42s
```

---

### Export Tool

**Command**:
```bash
graphrag export-from-neo4j [OPTIONS]
```

**Options**:
```
--output PATH             Output directory for Parquet files (required)
--neo4j-uri TEXT          Neo4j connection URI (required)
--neo4j-user TEXT         Neo4j username (default: neo4j)
--neo4j-password TEXT     Neo4j password (required)
--neo4j-database TEXT     Neo4j database name (default: neo4j)
--include-embeddings      Include embeddings in export
--format TEXT             Output format: parquet, csv (default: parquet)
--verbose                 Verbose logging
```

**Example**:
```bash
graphrag export-from-neo4j \
  --output ./backup-parquet \
  --neo4j-uri bolt://localhost:7687 \
  --neo4j-password $NEO4J_PASSWORD \
  --include-embeddings \
  --verbose
```

**Output**:
```
Exporting Neo4j data to Parquet...
✓ Connected to Neo4j
→ Exporting entities...
✓ Exported 1247 entities → entities.parquet
→ Exporting relationships...
✓ Exported 3891 relationships → relationships.parquet
→ Exporting communities...
✓ Exported 156 communities → communities.parquet
→ Exporting text units...
✓ Exported 542 text units → text_units.parquet
→ Exporting documents...
✓ Exported 15 documents → documents.parquet
→ Exporting embeddings...
✓ Exported 1247 embeddings → (embedded in files)

Export complete! ✅
Output directory: ./backup-parquet
Time: 1m 23s
```

---

### Validation Tool

**Command**:
```bash
graphrag validate-neo4j [OPTIONS]
```

**Options**:
```
--neo4j-uri TEXT          Neo4j connection URI (required)
--neo4j-user TEXT         Neo4j username (default: neo4j)
--neo4j-password TEXT     Neo4j password (required)
--neo4j-database TEXT     Neo4j database name (default: neo4j)
--check-schema            Validate schema (default: true)
--check-indexes           Validate indexes (default: true)
--check-data              Validate data integrity (default: true)
--check-embeddings        Validate embeddings (default: true)
--verbose                 Verbose logging
```

**Example**:
```bash
graphrag validate-neo4j \
  --neo4j-uri bolt://localhost:7687 \
  --neo4j-password $NEO4J_PASSWORD \
  --verbose
```

**Output**:
```
Validating Neo4j graph...
✓ Connected to Neo4j

Schema Validation:
✓ Entity nodes exist
✓ Community nodes exist
✓ TextUnit nodes exist
✓ Document nodes exist
✓ RELATED_TO relationships exist
✓ BELONGS_TO relationships exist
✓ MENTIONS relationships exist
✓ CONTAINS relationships exist

Index Validation:
✓ entity_id_unique constraint exists
✓ community_id_unique constraint exists
✓ entity_description_vector index exists (1536 dims, cosine)
✓ community_summary_vector index exists (1536 dims, cosine)
✓ text_unit_vector index exists (1536 dims, cosine)

Data Integrity:
✓ No orphaned entities
✓ No orphaned communities
✓ All relationships have valid endpoints
✓ All BELONGS_TO relationships valid
✓ Community hierarchy valid

Embedding Validation:
✓ 1247/1247 entities have embeddings (100%)
✓ 156/156 communities have embeddings (100%)
✓ 542/542 text units have embeddings (100%)

Validation complete! ✅
Status: Healthy
```

---

## Rollback Procedures

### Scenario 1: Testing Neo4j, Want to Go Back

**Situation**: Tried hybrid mode, prefer Parquet

**Steps**:
```yaml
# 1. Update settings.yaml
storage:
  type: parquet  # Back to Parquet only
  base_dir: ./output
```

```bash
# 2. Verify Parquet files intact
ls -lh ./output/
# Should see: entities.parquet, relationships.parquet, etc.

# 3. Run query to test
graphrag query --method global --query "Test query"
# Should work with Parquet files

# 4. (Optional) Stop Neo4j
docker-compose down
```

**Data Loss**: None (Parquet files preserved in hybrid mode)

---

### Scenario 2: Neo4j Data Corrupted

**Situation**: Neo4j data corrupted, need to restore

**Steps**:
```bash
# 1. Stop Neo4j
docker-compose down

# 2. Restore from backup (if you have one)
docker-compose exec neo4j neo4j-admin restore \
  --from=/backups/2024-01-15 \
  --database=neo4j

# 3. Or re-import from Parquet
# (Neo4j will be empty after restart)
docker-compose up -d
graphrag import-to-neo4j \
  --input ./output \
  --neo4j-uri bolt://localhost:7687

# 4. Or switch back to Parquet-only
# (Update settings.yaml as in Scenario 1)
```

**Data Loss**: None if Parquet backup exists

---

### Scenario 3: Neo4j Performance Unacceptable

**Situation**: Neo4j too slow for your use case

**Steps**:
```bash
# 1. Try optimization first
# - Increase Neo4j memory (docker-compose.yml)
# - Tune batch sizes (settings.yaml)
# - Profile slow queries

# 2. If still slow, switch back to Parquet
# (Update settings.yaml as in Scenario 1)

# 3. Export Neo4j data for future reference
graphrag export-from-neo4j --output ./neo4j-backup
```

**Data Loss**: None

---

## Troubleshooting

### Common Issues

#### Issue 1: Neo4j Connection Refused

**Symptoms**:
```
Error: Could not connect to Neo4j at bolt://localhost:7687
```

**Solutions**:
```bash
# 1. Check Neo4j is running
docker-compose ps neo4j
# Should show "Up"

# 2. Check logs
docker-compose logs neo4j | tail -50

# 3. Verify port not blocked
telnet localhost 7687

# 4. Check credentials
# Wrong password will show: "Authentication failed"
```

---

#### Issue 2: Out of Memory

**Symptoms**:
```
Neo4jError: Java heap space
```

**Solutions**:
```yaml
# Increase heap size in docker-compose.yml
environment:
  NEO4J_dbms_memory_heap_initial__size: 4G  # Was 2G
  NEO4J_dbms_memory_heap_max__size: 8G      # Was 4G
```

```bash
# Restart Neo4j
docker-compose down
docker-compose up -d
```

---

#### Issue 3: GDS Plugin Not Loaded

**Symptoms**:
```
There is no procedure with the name `gds.louvain.stream`
```

**Solutions**:
```yaml
# Verify GDS in docker-compose.yml
environment:
  NEO4J_PLUGINS: '["graph-data-science"]'  # Must be array format
```

```bash
# Check Neo4j logs for GDS
docker-compose logs neo4j | grep -i gds
# Should see: "Loaded Graph Data Science"

# Restart if needed
docker-compose restart neo4j
```

---

#### Issue 4: Vector Index Not Working

**Symptoms**:
```
There is no such index: entity_description_vector
```

**Solutions**:
```cypher
// Check if index exists
SHOW INDEXES
// Look for: entity_description_vector

// If missing, create manually
CREATE VECTOR INDEX entity_description_vector
FOR (e:Entity)
ON e.description_embedding
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 1536,
    `vector.similarity_function`: 'cosine'
  }
}

// Wait for index to come online
SHOW INDEXES
// State should be: ONLINE
```

---

#### Issue 5: Slow Import

**Symptoms**:
```
Import taking hours for medium dataset
```

**Solutions**:
```bash
# 1. Increase batch size
graphrag import-to-neo4j \
  --batch-size 5000  # Default is 1000

# 2. Skip embeddings (import later)
graphrag import-to-neo4j \
  --skip-embeddings

# 3. Disable GDS during import
# (in settings.yaml, set gds.enabled: false)

# 4. Use faster disk (SSD)
# 5. Allocate more memory to Neo4j
```

---

#### Issue 6: Community Results Different

**Symptoms**:
```
Neo4j communities different from Parquet/NetworkX
```

**Explanation**:
- Neo4j uses Louvain algorithm
- NetworkX uses Leiden algorithm
- Results will differ slightly (1-5% modularity)

**Solutions**:
```bash
# 1. Accept difference (recommended)
# Both are valid community structures

# 2. Compare community sizes
python -c "
import pandas as pd
parquet = pd.read_parquet('./output/communities.parquet')
neo4j = pd.read_parquet('./output-neo4j/communities.parquet')
print('Parquet communities:', parquet['community'].nunique())
print('Neo4j communities:', neo4j['community'].nunique())
# Should be similar numbers
"

# 3. Validate community quality
# - Are entities in similar communities?
# - Do community reports make sense?
# - Query results still relevant?
```

---

## Support Resources

### Documentation

- **Main Guide**: `docs/neo4j/user_guide.md`
- **Setup Guide**: `docs/neo4j/setup.md`
- **Migration Guide**: `docs/neo4j/migration.md` (this document)
- **Troubleshooting**: `docs/neo4j/troubleshooting.md`
- **API Reference**: `docs/neo4j/api.md`

### Examples

- **Basic Indexing**: `examples/neo4j/01_basic_indexing.py`
- **Hybrid Queries**: `examples/neo4j/02_hybrid_queries.py`
- **Migration**: `examples/neo4j/03_migration.py`
- **Production Setup**: `examples/neo4j/04_production.py`

### Community Support

- **GitHub Discussions**: https://github.com/microsoft/graphrag/discussions
- **Discord**: #neo4j-migration channel
- **Stack Overflow**: Tag `graphrag-neo4j`

### Professional Support

- **Neo4j Support**: https://neo4j.com/support/
- **GraphRAG Support**: support@graphrag.io (if available)

---

## FAQs

### Q1: Do I have to migrate?

**A**: No. Parquet storage will remain supported indefinitely (or at least until v4.0.0, with warnings in v3.3.0).

---

### Q2: Can I migrate back to Parquet after switching to Neo4j?

**A**: Yes, use the export tool:
```bash
graphrag export-from-neo4j --output ./output-parquet
```

---

### Q3: Will my existing queries still work?

**A**: Yes. Query interface remains the same. Neo4j is used internally for storage and retrieval.

---

### Q4: What happens to my Parquet files in hybrid mode?

**A**: They continue to be updated. You have two copies of your data (Neo4j + Parquet).

---

### Q5: How much does Neo4j cost?

**A**:
- **Community Edition**: Free (local/self-hosted)
- **Neo4j Aura** (cloud): ~$65-200/month
- **Enterprise**: Contact Neo4j sales

Most users can use free Community Edition.

---

### Q6: What if I encounter a critical bug in Neo4j integration?

**A**: Switch back to Parquet immediately:
```yaml
storage:
  type: parquet
```

Your Parquet files are intact (if you were in hybrid mode) or can be restored (if you exported).

---

### Q7: Will community detection results be identical?

**A**: No. Neo4j uses Louvain (slightly different from Leiden). Results will be similar but not identical. Quality difference is small (1-5%).

---

### Q8: Can I use Neo4j Enterprise features?

**A**: Yes, if you have a Neo4j Enterprise license. GraphRAG works with both Community and Enterprise editions.

---

### Q9: How do I backup my Neo4j data?

**A**:
```bash
# Neo4j native backup
docker-compose exec neo4j neo4j-admin backup \
  --backup-dir=/backups/$(date +%Y%m%d)

# Or export to Parquet
graphrag export-from-neo4j --output ./backup
```

---

### Q10: What if Neo4j service is down?

**A**: Queries will fail. Consider:
- Running Neo4j with high availability (Enterprise)
- Keeping Parquet export as backup
- Using hybrid mode for safety

---

## Best Practices

### For Development

1. **Use Docker**: Easiest Neo4j setup
2. **Use Hybrid Mode**: Keep Parquet backup
3. **Small Test Dataset**: Validate migration quickly
4. **Version Control**: Track settings.yaml changes

### For Production

1. **Use Neo4j Aura or Self-Hosted**: Reliable infrastructure
2. **Enable Backups**: Regular automated backups
3. **Monitor Performance**: Set up metrics/alerts
4. **Document Configuration**: Keep deployment docs
5. **Test Rollback**: Practice rollback procedure

### For Migration

1. **Test First**: Use test environment
2. **Hybrid Mode**: Transition period with both systems
3. **Validate Results**: Compare Parquet vs Neo4j
4. **Export Backup**: Keep Parquet export before switching
5. **Gradual Rollout**: Start with non-critical projects

---

## Summary

### Migration Checklist

**Before Migration**:
- [ ] Understand benefits and trade-offs
- [ ] Choose appropriate user segment path
- [ ] Set up Neo4j (Docker/Aura/Desktop)
- [ ] Test Neo4j connection
- [ ] Backup existing Parquet files

**During Migration**:
- [ ] Update settings.yaml to hybrid mode
- [ ] Import or re-index data
- [ ] Validate data integrity
- [ ] Test queries
- [ ] Compare results with Parquet

**After Migration**:
- [ ] Document configuration
- [ ] Set up backups
- [ ] Monitor performance
- [ ] Train team on Neo4j queries
- [ ] Plan for full switchover (optional)

### Key Takeaways

1. ✅ **Migration is optional** - Parquet remains supported
2. ✅ **Hybrid mode is safe** - Keeps both systems running
3. ✅ **Rollback is easy** - Can revert at any time
4. ✅ **Tools provided** - Import, export, validation
5. ✅ **Support available** - Documentation and community

### When to Migrate

**Migrate Now** if:
- Large dataset (1000+ documents)
- Need incremental updates
- Need concurrent access
- Need hybrid queries
- Production deployment

**Wait** if:
- Small dataset (< 100 documents)
- Single-user, local use
- Happy with current performance
- Want to avoid complexity

**Try Hybrid** if:
- Unsure about benefits
- Want to evaluate performance
- Need safety net
- Gradual transition preferred

---

**Status**: ✅ Complete
**Assessment Complete**: All 8 documents finished
**Next Step**: Stakeholder review and approval
