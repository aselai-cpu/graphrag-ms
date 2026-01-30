# GraphRAG Index Operation - Comprehensive Analysis

## Overview

The GraphRAG indexing system transforms raw documents into a multi-layered knowledge graph with entities, relationships, communities, and embeddings. The system supports two indexing strategies: **Standard** (LLM-based, high quality) and **Fast** (NLP+LLM hybrid, faster).

---

## 1. Entry Point & CLI

### CLI Command
```bash
graphrag index [OPTIONS]
```

### Implementation
- **CLI Entry**: `packages/graphrag/graphrag/cli/main.py:138-193`
- **Command Handler**: `packages/graphrag/graphrag/cli/index.py:44-62`
- **API Entry**: `packages/graphrag/graphrag/api/index.py:29-93`

### Key Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--root, -r` | `.` | Project root directory |
| `--method, -m` | `standard` | Indexing method (standard/fast) |
| `--verbose, -v` | `false` | Enable verbose logging |
| `--dry-run` | `false` | Validate configuration without executing |
| `--cache/--no-cache` | `true` | Enable/disable LLM response cache |
| `--skip-validation` | `false` | Skip preflight validation |

### Execution Flow
1. Load configuration from root directory
2. Initialize loggers and signal handlers
3. Call `api.build_index()` asynchronously
4. Create pipeline via `PipelineFactory.create_pipeline()`
5. Execute pipeline via `run_pipeline()` generator
6. Track stats and handle errors
7. Exit with error code if any workflow failed

---

## 2. Pipeline Architecture

### Pipeline Factory
**Location**: `packages/graphrag/graphrag/index/workflows/factory.py:17-98`

The factory creates different pipelines based on the indexing method:

### Standard Indexing Pipeline
```
load_input_documents
    ↓
create_base_text_units
    ↓
create_final_documents
    ↓
extract_graph (LLM-based entity/relationship extraction)
    ↓
finalize_graph
    ↓
extract_covariates (optional claims extraction)
    ↓
create_communities (Leiden clustering)
    ↓
create_final_text_units
    ↓
create_community_reports (LLM-summarized reports)
    ↓
generate_text_embeddings
```

**Lines**: `factory.py:52-62, 85-86`

### Fast Indexing Pipeline
```
load_input_documents
    ↓
create_base_text_units
    ↓
create_final_documents
    ↓
extract_graph_nlp (NLP-based noun phrase extraction)
    ↓
prune_graph (edge weight pruning)
    ↓
finalize_graph
    ↓
create_communities
    ↓
create_final_text_units
    ↓
create_community_reports_text (text-based reports)
    ↓
generate_text_embeddings
```

**Lines**: `factory.py:63-73, 87-89`

### Pipeline Execution
**Location**: `packages/graphrag/graphrag/index/run/run_pipeline.py:29-99`

**Key Steps**:
1. Create input, output, and cache storages
2. Load previous state from `context.json`
3. For update runs: prepare delta storage, backup previous index
4. Execute each workflow sequentially
5. Yield results and update stats after each workflow
6. Dump stats and context to storage

### Workflow Registry
**Location**: `packages/graphrag/graphrag/index/workflows/__init__.py:77-100`

**All 22 Registered Workflows**:
1. `load_input_documents`
2. `load_update_documents`
3. `create_base_text_units`
4. `create_communities`
5. `create_community_reports_text`
6. `create_community_reports`
7. `extract_covariates`
8. `create_final_documents`
9. `create_final_text_units`
10. `extract_graph_nlp`
11. `extract_graph`
12. `finalize_graph`
13. `generate_text_embeddings`
14. `prune_graph`
15. `update_final_documents`
16. `update_text_embeddings`
17. `update_community_reports`
18. `update_entities_relationships`
19. `update_communities`
20. `update_covariates`
21. `update_text_units`
22. `update_clean_state`

---

## 3. Index Types & Data Structures

GraphRAG creates **7 primary indexes** stored as Parquet files:

### 3.1 Text Units Index (`text_units.parquet`)
**Schema**: `packages/graphrag/graphrag/data_model/schemas.py:140-149`

| Column | Type | Description |
|--------|------|-------------|
| `id` | string | Unique text unit identifier (UUID) |
| `human_readable_id` | int | Sequential index |
| `text` | string | The actual text content |
| `n_tokens` | int | Token count |
| `document_id` | string | Parent document reference |
| `entity_ids` | list[string] | Entities mentioned in this text unit |
| `relationship_ids` | list[string] | Relationships in this text unit |
| `covariate_ids` | list[string] | Claims/covariates in this text unit |

**Created By**: `create_base_text_units` workflow
**Finalized By**: `create_final_text_units` workflow

### 3.2 Entities Index (`entities.parquet`)
**Schema**: `schemas.py:70-79`
**Data Model**: `packages/graphrag/graphrag/data_model/entity.py`

| Column | Type | Description |
|--------|------|-------------|
| `id` | string | Unique entity ID (UUID) |
| `human_readable_id` | int | Short sequential ID |
| `title` | string | Entity name |
| `type` | string | Entity type (organization, person, geo, event, etc.) |
| `description` | string | Entity description (LLM-generated) |
| `text_unit_ids` | list[string] | Text units where entity appears |
| `node_frequency` | int | Frequency count |
| `node_degree` | int | Graph degree (number of relationships) |

**Created By**: `extract_graph` or `extract_graph_nlp` workflows
**Updated By**: `finalize_graph` workflow (degree calculations)

### 3.3 Relationships Index (`relationships.parquet`)
**Schema**: `schemas.py:81-90`
**Data Model**: `packages/graphrag/graphrag/data_model/relationship.py`

| Column | Type | Description |
|--------|------|-------------|
| `id` | string | Unique relationship ID |
| `human_readable_id` | int | Short sequential ID |
| `source` | string | Source entity name |
| `target` | string | Target entity name |
| `description` | string | Relationship description (LLM-generated) |
| `weight` | float | Edge weight (co-occurrence count) |
| `combined_degree` | int | Degree of source + target |
| `text_unit_ids` | list[string] | Text units containing this relationship |

**Created By**: `extract_graph` or `extract_graph_nlp` workflows

### 3.4 Communities Index (`communities.parquet`)
**Schema**: `schemas.py:92-105`
**Data Model**: `packages/graphrag/graphrag/data_model/community.py`

| Column | Type | Description |
|--------|------|-------------|
| `id` | string | Unique community ID |
| `human_readable_id` | int | Community number |
| `community` | string | Community ID |
| `level` | int | Hierarchical level (0 = leaf, higher = parent) |
| `parent` | string | Parent community ID |
| `children` | list[string] | Child community IDs (hierarchical) |
| `title` | string | Community title |
| `entity_ids` | list[string] | List of entities in community |
| `relationship_ids` | list[string] | List of relationships within community |
| `text_unit_ids` | list[string] | Text units in community |
| `period` | string | Creation date (for tracking updates) |
| `size` | int | Community size (number of entities) |

**Created By**: `create_communities` workflow
**Algorithm**: Hierarchical Leiden clustering (`cluster_graph.py:19-53`)

### 3.5 Community Reports Index (`community_reports.parquet`)
**Schema**: `schemas.py:107-123`
**Data Model**: `packages/graphrag/graphrag/data_model/community_report.py`

| Column | Type | Description |
|--------|------|-------------|
| `id`, `human_readable_id` | string, int | Identifiers |
| `community`, `level` | string, int | Community reference |
| `parent`, `children` | string, list | Hierarchy |
| `title` | string | Community title |
| `summary` | string | LLM-generated summary |
| `full_content` | string | Complete report with findings |
| `rank` | int | Importance rating (0-9) |
| `rating_explanation` | string | Why this rank |
| `findings` | list[dict] | Key findings/insights (structured JSON) |
| `full_content_json` | string | Full report as JSON |
| `period` | string | Creation date |
| `size` | int | Community size |

**Created By**: `create_community_reports` workflow
**Uses**: LLM to summarize community context and relationships

### 3.6 Covariates Index (`covariates.parquet`) - Optional
**Schema**: `schemas.py:125-138`
**Enabled**: Only if `extract_claims.enabled = true`

| Column | Type | Description |
|--------|------|-------------|
| `id`, `human_readable_id` | string, int | IDs |
| `covariate_type` | string | Type of claim |
| `type` | string | Specific type |
| `description` | string | Claim description |
| `subject_id` | string | Subject entity |
| `object_id` | string | Object entity |
| `status` | string | Claim status |
| `start_date`, `end_date` | string | Temporal bounds |
| `source_text` | string | Original text |
| `text_unit_id` | string | Source text unit |

**Created By**: `extract_covariates` workflow (if enabled)

### 3.7 Embedding Indexes (Vector Stores)
**Configuration**: `packages/graphrag/graphrag/config/embeddings.py:5-19`

**Three Types**:
1. **Text Unit Embeddings** (`text_unit_text`) - Embeddings of text unit content
2. **Entity Description Embeddings** (`entity_description`) - "title:description" for entities
3. **Community Report Embeddings** (`community_full_content`) - Full content of community reports

**Storage**: Vector store (configurable: Lancedb, Qdrant, OpenSearch, Azure AI Search, etc.)
**Created By**: `generate_text_embeddings` workflow

---

## 4. Key Components & Operations

### 4.1 Entity & Relationship Extraction

#### Standard Method (LLM-based)
**File**: `packages/graphrag/graphrag/index/operations/extract_graph/graph_extractor.py:38-184`

**Process**:
1. Extract entities and relationships via LLM completion
2. Multi-turn extraction with "gleanings" for comprehensive results
3. Summarize descriptions with separate LLM call
4. Return DataFrames of entities and relationships

**Configuration**: `extract_graph` config
**Model**: Configurable completion model (e.g., GPT-4)

#### Fast Method (NLP-based)
**File**: `packages/graphrag/graphrag/index/operations/extract_graph_nlp/extract_graph_nlp.py`

**Process**:
1. Use NLP text analyzer (regex, syntactic parser, or CFG)
2. Extract noun phrases as entities
3. Create relationships based on co-occurrence
4. Normalize edge weights

**Configuration**: `extract_graph_nlp` config
**No LLM Required**: Pure NLP extraction

### 4.2 Community Detection
**File**: `packages/graphrag/graphrag/index/operations/cluster_graph.py:19-53`

**Algorithm**: Hierarchical Leiden clustering (via `leidenalg` library)
**Optional**: Use largest connected component (LCC) for better clustering
**Configuration**: `cluster_graph` config

**Process**:
1. Convert entity/relationship DataFrames to NetworkX graph
2. Optionally extract largest connected component
3. Apply Leiden algorithm with hierarchical levels
4. Return community assignments with hierarchy

### 4.3 Text Embedding
**File**: `packages/graphrag/graphrag/index/operations/embed_text/embed_text.py:23-89`

**Process**:
1. Batch texts into chunks (configurable batch size)
2. Generate embeddings via LLM embedding model
3. Create vector store and load documents
4. Return ID-to-embedding mappings

**Configuration**: `embed_text` config
**Model**: Configurable embedding model (e.g., text-embedding-3-small)

### 4.4 Graph Creation
**File**: `packages/graphrag/graphrag/index/operations/create_graph.py:10-23`

**Uses**: NetworkX for graph representation
**Creates**: Undirected graph from entity/relationship DataFrames

---

## 5. Storage & Persistence

### Storage Tiers

#### 1. Input Storage
- **Config**: `GraphRagConfig.input_storage`
- **Default Type**: File-based
- **Default Location**: `input/` directory
- **Content**: Source documents to be indexed

#### 2. Output Storage
- **Config**: `GraphRagConfig.output_storage`
- **Default Type**: File-based
- **Default Location**: `output/` directory
- **Content**:
  - Parquet files for all indexes (entities, relationships, communities, etc.)
  - `context.json`: Workflow state
  - `stats.json`: Pipeline statistics
  - Optional GraphML snapshots

#### 3. Update Output Storage
- **Config**: `GraphRagConfig.update_output_storage`
- **Default Location**: `update_output/` directory
- **Used For**: Incremental index updates
- **Structure**: `{timestamp}/delta/` (new data) and `{timestamp}/previous/` (backup)

#### 4. Cache Storage
- **Config**: `GraphRagConfig.cache`
- **Default Type**: NoOp or File-based
- **Location**: `cache/` directory (configurable)
- **Purpose**: Cache LLM responses to avoid re-computation

#### 5. Vector Store
- **Config**: `GraphRagConfig.vector_store`
- **Supported Types** (from graphrag_vectors):
  - Lancedb (default, local)
  - Qdrant
  - OpenSearch
  - Azure AI Search
  - Pinecone
  - Milvus
- **Storage**: Embeddings indexed by embedding type

### File Format
- **Primary Format**: Parquet (columnar, compressed)
- **Helper Format**: JSON (for context.json, stats.json)
- **Optional**: GraphML for graph snapshots

### Storage API
**Interface**: `Storage` from graphrag_storage package

**Operations**:
- `get(key)`: Retrieve file
- `set(key, value)`: Store file
- `find(pattern)`: Search for files
- `child(name)`: Create child storage

---

## 6. LLM Usage During Indexing

### 6.1 Entity & Relationship Extraction
**Workflow**: `extract_graph`
**Model Config**: `config.extract_graph.completion_model_id`
**Prompt**: `config.extract_graph.resolved_prompts().extraction_prompt`

**Process** (`extract_graph.py:96-110`):
1. Load text units from storage
2. Create LLM completion instance with model config and cache
3. Call `extract_graph()` operation on all text units
4. LLM extracts entities and relationships with specified types
5. Max gleanings: Optional multi-turn extraction for thoroughness
6. Store results: entities and relationships DataFrames

### 6.2 Description Summarization
**Workflow**: `extract_graph`
**Model Config**: `config.summarize_descriptions.completion_model_id`

**Process** (`extract_graph.py:156-184`):
1. After initial extraction, summarize entity descriptions
2. Summarize relationship descriptions
3. Merge summaries back into extracted data

### 6.3 Community Summarization
**Workflow**: `create_community_reports`
**Model Config**: `config.community_reports.completion_model_id`
**Prompt**: `config.community_reports.resolved_prompts().graph_prompt`

**Input Context**:
- Community node details (entities with degrees)
- Edge details (relationships with degrees)
- Claim details (if enabled)

**Process** (`create_community_reports.py:94-142`):
1. Build local context for each community
2. Call LLM with context to generate report
3. Report includes: summary, findings, rating (0-9)
4. Finalize and store reports

### 6.4 Text Embeddings
**Workflow**: `generate_text_embeddings`
**Model Config**: `config.embed_text.embedding_model_id`

**Embeddings to Generate** (`embeddings.py`):
1. `text_unit_text`: Text of each text unit
2. `entity_description`: "title:description" for entities
3. `community_full_content`: Full content of community reports

**Process** (`generate_text_embeddings.py:97-152`):
1. Batch texts by token count
2. Call embedding model
3. Store in vector store with IDs
4. Return embedding mappings

### 6.5 Claim Extraction (Optional)
**Workflow**: `extract_covariates`
**Model Config**: `config.extract_claims.completion_model_id`
**Only if**: `config.extract_claims.enabled = true`
**Extracts**: Specific claims/facts as a secondary index

---

## 7. Configuration Options

### Global Settings
**File**: `packages/graphrag/graphrag/config/models/graph_rag_config.py`

| Setting | Default | Description |
|---------|---------|-------------|
| `completion_models` | - | Dict of available chat models |
| `embedding_models` | - | Dict of available embedding models |
| `concurrent_requests` | - | Parallel LLM requests |
| `async_mode` | AsyncIO | AsyncIO or Threaded |
| `input_storage` | `input/` | Input storage location |
| `chunk_size` | 1200 | Text chunk size (tokens) |
| `chunk_overlap` | 100 | Text chunk overlap (tokens) |
| `output_storage` | `output/` | Output location for indexes |
| `cache` | NoOp | Cache type and location |
| `vector_store` | Lancedb | Vector store type |

### Workflow-Specific Configs

#### Extract Graph Config
**File**: `packages/graphrag/graphrag/config/models/extract_graph_config.py`

- `completion_model_id`: Model to use for extraction
- `entity_types`: Entity types to extract (list)
- `max_gleanings`: Number of multi-turn extractions
- `model_instance_name`: For cache partitioning

#### Cluster Graph Config
**File**: `packages/graphrag/graphrag/config/models/cluster_graph_config.py`

- `max_cluster_size`: 10 (default)
- `use_lcc`: true (use largest connected component)
- `seed`: 0xDEADBEEF

#### Community Reports Config
**File**: `packages/graphrag/graphrag/config/models/community_reports_config.py`

- `completion_model_id`: Model for report generation
- `max_report_length`: 2000 (default)
- `max_input_length`: 8000 (default)

#### Embed Text Config
**File**: `packages/graphrag/graphrag/config/models/embed_text_config.py`

- `embedding_model_id`: Model for embeddings
- `batch_size`: 16 (default)
- `batch_max_tokens`: 8191 (default)
- `embeddings_to_generate`: List of embedding types

#### Extract Claims Config
**File**: `packages/graphrag/graphrag/config/models/extract_claims_config.py`

- `enabled`: false (default, must be enabled)
- `completion_model_id`: Model for claim extraction
- `max_gleanings`: 1 (default)

---

## 8. Data Flow Summary

```
Input Documents
       ↓
[load_input_documents] → documents.parquet
       ↓
[create_base_text_units] → chunked text units
       ↓
[create_final_documents] → documents with metadata
       ↓
[extract_graph OR extract_graph_nlp] → raw entities & relationships
       ↓
[finalize_graph] → entities.parquet, relationships.parquet (with degrees)
       ↓
[extract_covariates] → covariates.parquet (optional)
       ↓
[create_communities] → communities.parquet (hierarchical clusters)
       ↓
[create_final_text_units] → text_units.parquet (linked to entities/relationships)
       ↓
[create_community_reports] → community_reports.parquet (LLM summaries)
       ↓
[generate_text_embeddings] → vector store embeddings
       ↓
Index Complete
```

---

## 9. Key Insights

1. **Two-Tier Strategy**: Standard (LLM-based, accurate) vs Fast (NLP+LLM, faster)
2. **Hierarchical Communities**: Leiden clustering creates multi-level community structure
3. **Multiple Embeddings**: Text units, entity descriptions, and community reports all embedded
4. **Incremental Updates**: Separate update workflows support index refreshes
5. **Flexible Storage**: Supports multiple storage backends (file, blob, cosmos) and vector stores
6. **LLM Caching**: Cache layer prevents redundant LLM calls
7. **Modular Workflows**: 22 registered workflows allow mix-and-match pipelines
8. **Rich Metadata**: All indexes include human-readable IDs and extensive cross-references

---

## References

### Key Files
- CLI: `packages/graphrag/graphrag/cli/main.py`, `cli/index.py`
- API: `packages/graphrag/graphrag/api/index.py`
- Pipeline: `packages/graphrag/graphrag/index/workflows/factory.py`, `index/run/run_pipeline.py`
- Workflows: `packages/graphrag/graphrag/index/workflows/` (all workflow modules)
- Data Models: `packages/graphrag/graphrag/data_model/` (entity, relationship, community, etc.)
- Schemas: `packages/graphrag/graphrag/data_model/schemas.py`
- Config: `packages/graphrag/graphrag/config/models/` (all config modules)

### External Dependencies
- **LLM**: Via `graphrag-llm` package (litellm wrapper)
- **Vectors**: Via `graphrag-vectors` package (multiple vector stores)
- **Storage**: Via `graphrag-storage` package (file, blob, cosmos)
- **Clustering**: `leidenalg` library for community detection
- **Graph**: NetworkX for graph operations
- **Data**: Pandas for DataFrames, PyArrow for Parquet
