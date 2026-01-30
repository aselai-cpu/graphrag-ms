# GraphRAG Standard Indexing Pipeline - Step-by-Step Guide

This document provides a detailed walkthrough of each workflow in the Standard Indexing Pipeline, including what happens at each step, the data transformations, and code examples.

---

## Pipeline Overview

```
Input Documents (raw text files)
    ↓
1. load_input_documents
    ↓
2. create_base_text_units
    ↓
3. create_final_documents
    ↓
4. extract_graph
    ↓
5. finalize_graph
    ↓
6. extract_covariates (optional)
    ↓
7. create_communities
    ↓
8. create_final_text_units
    ↓
9. create_community_reports
    ↓
10. generate_text_embeddings
    ↓
Output: Complete Knowledge Graph
```

---

## Step 1: load_input_documents

### Purpose
Read all documents from the input storage and create the initial documents DataFrame.

### Location
`packages/graphrag/graphrag/index/workflows/load_input_documents.py`

### What Happens

1. **Read Input Files**: Scans input storage for documents matching configured patterns
2. **Extract Metadata**: Captures file paths, titles, and attributes
3. **Store Raw Content**: Preserves original document text
4. **Assign IDs**: Generates unique document IDs

### Input
- Raw document files from `input_storage` (configured directory)
- Supported formats: `.txt`, `.md`, `.pdf`, `.docx`, etc. (via markitdown)

### Code Example

```python
# From load_input_documents.py
async def run(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """Load documents from input storage."""

    # Get input storage
    input_storage = context.input_storage

    # Read documents using configured loader
    documents = []
    async for doc_path in input_storage.find(config.input.file_pattern):
        content = await input_storage.get(doc_path)

        documents.append({
            "id": generate_document_id(doc_path),
            "title": extract_title(doc_path),
            "raw_content": content.decode("utf-8"),
            "attributes": {}
        })

    # Create DataFrame
    df = pd.DataFrame(documents)

    # Write to output storage
    await write_table_to_storage(df, "documents", context.output_storage)

    return WorkflowFunctionOutput(
        workflow="load_input_documents",
        output=df
    )
```

### Output DataFrame Schema

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `id` | string | Unique document ID | `"doc_001"` |
| `title` | string | Document title | `"Annual Report 2023"` |
| `raw_content` | string | Full document text | `"This is the content..."` |
| `attributes` | dict | Metadata attributes | `{"year": 2023}` |

### Output File
- `documents.parquet` (initial version, will be enriched later)

### Example Output
```python
# documents DataFrame
   id        title                    raw_content                           attributes
0  doc_001   Annual Report 2023       "Microsoft Corporation Annual..."     {}
1  doc_002   Q1 Earnings              "Q1 2023 Financial Results..."        {}
2  doc_003   Product Announcement     "Introducing new features..."         {}
```

---

## Step 2: create_base_text_units

### Purpose
Chunk documents into smaller text units (chunks) for processing and embedding.

### Location
`packages/graphrag/graphrag/index/workflows/create_base_text_units.py`

### What Happens

1. **Load Documents**: Reads documents from previous step
2. **Initialize Chunker**: Creates chunking strategy based on config
3. **Chunk Documents**: Splits each document into overlapping chunks
4. **Count Tokens**: Calculates token count for each chunk
5. **Assign IDs**: Generates unique text unit IDs

### Chunking Strategy

**Default Configuration**:
- **Chunk Size**: 1200 tokens
- **Overlap**: 100 tokens
- **Encoding**: tiktoken (cl100k_base for GPT-4)

**Why Overlap?**: Ensures entities/relationships spanning chunk boundaries aren't missed.

### Code Example

```python
# From create_base_text_units.py
from graphrag_chunking import create_text_chunker

async def run(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """Create text units by chunking documents."""

    # Load documents
    documents = await load_table_from_storage("documents", context.output_storage)

    # Create chunker
    chunker = create_text_chunker(
        strategy=config.chunking.strategy,  # e.g., "tokens"
        chunk_size=config.chunking.chunk_size,  # 1200
        chunk_overlap=config.chunking.chunk_overlap,  # 100
        encoding_model=config.chunking.encoding_model  # "cl100k_base"
    )

    # Chunk all documents
    text_units = []
    for _, doc in documents.iterrows():
        chunks = chunker.chunk(doc["raw_content"])

        for i, chunk in enumerate(chunks):
            text_units.append({
                "id": f"{doc['id']}_chunk_{i}",
                "text": chunk.text,
                "n_tokens": chunk.n_tokens,
                "document_id": doc["id"],
                "chunk_index": i
            })

    df = pd.DataFrame(text_units)
    await write_table_to_storage(df, "text_units", context.output_storage)

    return WorkflowFunctionOutput(
        workflow="create_base_text_units",
        output=df
    )
```

### Chunking Visualization

```
Document (3000 tokens):
[-----------------------------Original Document-----------------------------]

Chunked into text units (chunk_size=1200, overlap=100):
[--------Chunk 0 (1200)--------]
                        [--------Chunk 1 (1200)--------]
                                                [--------Chunk 2 (1200)--------]
                         ↑ 100 token overlap ↑
```

### Output DataFrame Schema

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `id` | string | Unique text unit ID | `"doc_001_chunk_0"` |
| `text` | string | Chunk content | `"Microsoft was founded..."` |
| `n_tokens` | int | Token count | `1150` |
| `document_id` | string | Parent document ID | `"doc_001"` |
| `chunk_index` | int | Position in document | `0` |

### Output File
- `text_units.parquet` (base version, will be enriched later)

### Example Output
```python
# text_units DataFrame
   id                 text                                    n_tokens  document_id    chunk_index
0  doc_001_chunk_0    "Microsoft Corporation is a tech..."    1150      doc_001        0
1  doc_001_chunk_1    "...company. The company develops..."   1180      doc_001        1
2  doc_002_chunk_0    "Q1 revenues increased by 15%..."       980       doc_002        0
```

---

## Step 3: create_final_documents

### Purpose
Enrich documents DataFrame with aggregated metadata from text units.

### Location
`packages/graphrag/graphrag/index/workflows/create_final_documents.py`

### What Happens

1. **Load Documents and Text Units**: Reads both DataFrames
2. **Aggregate Statistics**: Calculates total tokens, chunk count per document
3. **Add Attributes**: Enriches document metadata
4. **Assign Human-Readable IDs**: Sequential numbering for readability

### Code Example

```python
# From create_final_documents.py
async def run(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """Finalize documents with metadata."""

    documents = await load_table_from_storage("documents", context.output_storage)
    text_units = await load_table_from_storage("text_units", context.output_storage)

    # Aggregate text unit stats per document
    text_unit_stats = text_units.groupby("document_id").agg({
        "n_tokens": "sum",
        "id": "count"
    }).reset_index()
    text_unit_stats.columns = ["id", "total_tokens", "text_unit_count"]

    # Merge with documents
    documents = documents.merge(text_unit_stats, on="id", how="left")

    # Add human-readable IDs
    documents["human_readable_id"] = range(len(documents))

    # Add text_unit_ids list
    text_unit_map = text_units.groupby("document_id")["id"].apply(list).reset_index()
    text_unit_map.columns = ["id", "text_unit_ids"]
    documents = documents.merge(text_unit_map, on="id", how="left")

    await write_table_to_storage(documents, "documents", context.output_storage)

    return WorkflowFunctionOutput(
        workflow="create_final_documents",
        output=documents
    )
```

### Output DataFrame Schema

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `id` | string | Document ID | `"doc_001"` |
| `human_readable_id` | int | Sequential ID | `0` |
| `title` | string | Document title | `"Annual Report"` |
| `raw_content` | string | Full text | `"Microsoft Corp..."` |
| `total_tokens` | int | Total tokens | `3500` |
| `text_unit_count` | int | Number of chunks | `3` |
| `text_unit_ids` | list[string] | Chunk IDs | `["doc_001_chunk_0", ...]` |
| `attributes` | dict | Metadata | `{}` |

### Output File
- `documents.parquet` (final version with enriched metadata)

---

## Step 4: extract_graph

### Purpose
Extract entities and relationships from text units using LLM-based prompting.

### Location
`packages/graphrag/graphrag/index/workflows/extract_graph.py`

### 📊 Detailed Activity Diagram
**See**: `analysis/index/extract_graph_activity.puml` - Detailed PlantUML activity diagram showing the complete extraction process, including how entities are linked to text units (chunks).

This diagram visualizes:
- LLM extraction workflow for each text unit
- Multi-turn gleaning process
- **Entity-to-text-unit linking mechanism** (how `text_unit_ids` are tracked)
- Duplicate entity merging across text units
- Relationship weight calculation based on co-occurrence

### What Happens

1. **Load Text Units**: Reads all text chunks
2. **Initialize LLM**: Creates completion model with configured settings
3. **Extract Entities & Relationships**: For each text unit:
   - Sends text to LLM with extraction prompt
   - Parses structured response (entities, relationships)
   - Optionally runs "gleanings" (additional extraction passes)
4. **Summarize Descriptions**: Calls LLM to condense verbose descriptions
5. **Merge and Deduplicate**: Combines entities/relationships across chunks
6. **Calculate Weights**: Counts co-occurrences for relationship weights

### Extraction Prompt Example

```
-Goal-
Given a text document, identify all entities and their relationships.

-Steps-
1. Identify all entities mentioned in the text
2. For each entity, extract: name, type, description
3. Identify relationships between entities
4. For each relationship, extract: source, target, description

-Entity Types-
- PERSON: People, including fictional
- ORGANIZATION: Companies, agencies, institutions
- GEO: Countries, cities, locations
- EVENT: Named events, incidents

-Output Format-
Return JSON:
{
  "entities": [
    {"name": "...", "type": "...", "description": "..."},
    ...
  ],
  "relationships": [
    {"source": "...", "target": "...", "description": "..."},
    ...
  ]
}

-Text-
{text_unit}
```

### Code Example

```python
# From extract_graph.py and graph_extractor.py
from graphrag_llm.completion import create_completion
from graphrag.index.operations.extract_graph import GraphExtractor

async def run(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """Extract entities and relationships using LLM."""

    # Load text units
    text_units = await load_table_from_storage("text_units", context.output_storage)

    # Get LLM completion model
    model_config = config.get_completion_model_config(
        config.extract_graph.completion_model_id
    )
    llm = create_completion(model_config)

    # Create graph extractor
    extractor = GraphExtractor(
        llm=llm,
        prompt=config.extract_graph.resolved_prompts().extraction_prompt,
        entity_types=config.extract_graph.entity_types,
        max_gleanings=config.extract_graph.max_gleanings
    )

    # Extract from all text units
    all_entities = []
    all_relationships = []

    for _, unit in text_units.iterrows():
        # Extract entities and relationships
        result = await extractor.extract(unit["text"])

        # Add source text unit ID
        for entity in result.entities:
            entity["text_unit_ids"] = [unit["id"]]
        for rel in result.relationships:
            rel["text_unit_ids"] = [unit["id"]]

        all_entities.extend(result.entities)
        all_relationships.extend(result.relationships)

    # Merge duplicates (same name → same entity)
    entities_df = merge_entities(all_entities)
    relationships_df = merge_relationships(all_relationships)

    # Summarize descriptions (if too long)
    entities_df = await summarize_descriptions(entities_df, llm)
    relationships_df = await summarize_descriptions(relationships_df, llm)

    # Save to storage
    await write_table_to_storage(entities_df, "entities", context.output_storage)
    await write_table_to_storage(relationships_df, "relationships", context.output_storage)

    return WorkflowFunctionOutput(
        workflow="extract_graph",
        output={"entities": entities_df, "relationships": relationships_df}
    )
```

### Multi-Turn Extraction (Gleanings)

**Purpose**: Extract entities/relationships that might be missed in first pass.

```python
# Gleaning process
initial_extraction = await llm.extract(text)

for gleaning_round in range(max_gleanings):
    # Ask LLM if anything was missed
    prompt = f"""
    Previously extracted:
    {initial_extraction}

    Review the text again. Did we miss any entities or relationships?
    """

    additional = await llm.extract(text, prompt)
    initial_extraction.merge(additional)
```

### Entity Extraction Example

**Input Text Unit**:
```
Microsoft was founded by Bill Gates and Paul Allen in 1975.
The company is headquartered in Redmond, Washington.
```

**LLM Extraction Output**:
```json
{
  "entities": [
    {
      "name": "Microsoft",
      "type": "ORGANIZATION",
      "description": "Technology company founded in 1975"
    },
    {
      "name": "Bill Gates",
      "type": "PERSON",
      "description": "Co-founder of Microsoft"
    },
    {
      "name": "Paul Allen",
      "type": "PERSON",
      "description": "Co-founder of Microsoft"
    },
    {
      "name": "Redmond",
      "type": "GEO",
      "description": "City in Washington, headquarters of Microsoft"
    }
  ],
  "relationships": [
    {
      "source": "Bill Gates",
      "target": "Microsoft",
      "description": "Co-founded the company in 1975"
    },
    {
      "source": "Paul Allen",
      "target": "Microsoft",
      "description": "Co-founded the company in 1975"
    },
    {
      "source": "Microsoft",
      "target": "Redmond",
      "description": "Headquartered in the city"
    }
  ]
}
```

### Output DataFrame Schemas

**Entities**:
| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `id` | string | UUID | `"e_001"` |
| `title` | string | Entity name | `"Microsoft"` |
| `type` | string | Entity type | `"ORGANIZATION"` |
| `description` | string | Description | `"Tech company founded in 1975"` |
| `text_unit_ids` | list[string] | Source chunks | `["doc_001_chunk_0"]` |

**Relationships**:
| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `id` | string | UUID | `"r_001"` |
| `source` | string | Source entity name | `"Bill Gates"` |
| `target` | string | Target entity name | `"Microsoft"` |
| `description` | string | Relationship description | `"Co-founded the company"` |
| `weight` | float | Co-occurrence count | `1.0` |
| `text_unit_ids` | list[string] | Source chunks | `["doc_001_chunk_0"]` |

### Output Files
- `entities.parquet` (raw, before degree calculation)
- `relationships.parquet` (raw, before degree calculation)

---

## Step 5: finalize_graph

### Purpose
Calculate graph metrics (node degrees, relationship weights) and create graph structure.

### Location
`packages/graphrag/graphrag/index/workflows/finalize_graph.py`

### 📊 Detailed Activity Diagram
**See**: `analysis/index/finalize_graph_activity.puml` - Detailed PlantUML activity diagram showing graph metric calculations and link verification.

This diagram visualizes:
- NetworkX graph construction from entities and relationships
- Node degree and node frequency calculations
- **Verification of entity-text unit bidirectional links**
- Combined degree calculation for relationships
- How graph metrics indicate entity and relationship importance

### What Happens

1. **Load Entities & Relationships**: Reads extraction results
2. **Build NetworkX Graph**: Creates graph structure
3. **Calculate Node Degrees**: Counts connections per entity
4. **Calculate Combined Degrees**: Sum of source and target degrees for relationships
5. **Assign Human-Readable IDs**: Sequential numbering
6. **Update DataFrames**: Add graph metrics to entities/relationships

### Code Example

```python
# From finalize_graph.py
import networkx as nx
from graphrag.index.operations.create_graph import create_graph

async def run(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """Finalize graph with metrics."""

    entities = await load_table_from_storage("entities", context.output_storage)
    relationships = await load_table_from_storage("relationships", context.output_storage)

    # Create NetworkX graph
    graph = create_graph(entities, relationships)

    # Calculate node degrees
    degrees = dict(graph.degree())
    entities["node_degree"] = entities["title"].map(degrees).fillna(0).astype(int)

    # Calculate node frequency (how many text units mention entity)
    entities["node_frequency"] = entities["text_unit_ids"].apply(len)

    # Add human-readable IDs
    entities["human_readable_id"] = range(len(entities))
    relationships["human_readable_id"] = range(len(relationships))

    # Calculate combined degree for relationships
    entity_degree_map = entities.set_index("title")["node_degree"].to_dict()
    relationships["combined_degree"] = (
        relationships["source"].map(entity_degree_map).fillna(0) +
        relationships["target"].map(entity_degree_map).fillna(0)
    ).astype(int)

    # Update weight (count co-occurrences)
    relationship_counts = relationships.groupby(["source", "target"]).size()
    relationships["weight"] = relationships.apply(
        lambda r: relationship_counts.get((r["source"], r["target"]), 1.0),
        axis=1
    )

    await write_table_to_storage(entities, "entities", context.output_storage)
    await write_table_to_storage(relationships, "relationships", context.output_storage)

    return WorkflowFunctionOutput(
        workflow="finalize_graph",
        output={"entities": entities, "relationships": relationships, "graph": graph}
    )
```

### Graph Metrics Visualization

```
Entity: "Microsoft"
- node_degree: 15 (connected to 15 other entities)
- node_frequency: 23 (mentioned in 23 text units)

Relationship: "Bill Gates" → "Microsoft"
- weight: 8.0 (appears in 8 text units)
- combined_degree: 20 (Bill Gates degree=5 + Microsoft degree=15)
```

### Example Output

**Entities (after finalization)**:
```python
   id      human_readable_id  title          type            node_degree  node_frequency
0  e_001   0                  Microsoft      ORGANIZATION    15           23
1  e_002   1                  Bill Gates     PERSON          5            12
2  e_003   2                  Paul Allen     PERSON          3            8
```

**Relationships (after finalization)**:
```python
   id      human_readable_id  source        target      weight  combined_degree
0  r_001   0                  Bill Gates    Microsoft   8.0     20
1  r_002   1                  Paul Allen    Microsoft   5.0     18
```

---

## Step 6: extract_covariates (Optional)

### Purpose
Extract claims, facts, or temporal information from text units.

### Location
`packages/graphrag/graphrag/index/workflows/extract_covariates.py`

### Enabled When
`config.extract_claims.enabled = true` (default: `false`)

### 📖 Why Extract Claims?
**See**: `analysis/index/covariates_rationale.md` - Comprehensive explanation of why GraphRAG extracts claims/covariates in addition to entities and relationships.

This document covers:
- The three knowledge layers (entities, relationships, claims)
- What unique information claims capture that entities/relationships miss
- Real-world use cases where claims are critical (financial analysis, legal, news, etc.)
- Why claims are optional and when to enable/skip them
- Research background and further reading

**Key Insight**: Claims capture **temporal facts** and **events** ("Microsoft reported $50B revenue in Q4 2023") while entities capture **things** ("Microsoft") and relationships capture **connections** ("Microsoft → founded_by → Bill Gates").

### What Happens

1. **Load Text Units & Entities**: Reads extraction results
2. **Initialize LLM**: Creates completion model
3. **Extract Claims**: For each text unit:
   - Sends text with claim extraction prompt
   - Extracts structured claims (subject, object, type, temporal info)
4. **Link to Entities**: Associates claims with related entities
5. **Store Covariates**: Saves to storage

### Claim Extraction Prompt Example

```
-Goal-
Extract factual claims from the text.

-Claim Format-
- Subject: Entity making the claim
- Object: Entity being claimed about
- Type: Type of claim (FACT, OPINION, PREDICTION)
- Description: What is being claimed
- Start Date / End Date: Temporal bounds (if applicable)
- Status: CONFIRMED, DISPUTED, SPECULATIVE

-Example-
Text: "Microsoft reported $50B revenue in Q4 2023"
Claim:
  Subject: Microsoft
  Object: Revenue
  Type: FACT
  Description: Reported $50B revenue in Q4 2023
  Start Date: 2023-10-01
  End Date: 2023-12-31
  Status: CONFIRMED
```

### Code Example

```python
# From extract_covariates.py
async def run(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """Extract claims/covariates from text."""

    if not config.extract_claims.enabled:
        return WorkflowFunctionOutput(
            workflow="extract_covariates",
            output=None
        )

    text_units = await load_table_from_storage("text_units", context.output_storage)
    entities = await load_table_from_storage("entities", context.output_storage)

    # Get LLM
    model_config = config.get_completion_model_config(
        config.extract_claims.completion_model_id
    )
    llm = create_completion(model_config)

    # Extract claims from each text unit
    all_covariates = []

    for _, unit in text_units.iterrows():
        result = await llm.completion(
            messages=build_claim_extraction_prompt(unit["text"])
        )

        claims = parse_claims(result.content)

        for claim in claims:
            all_covariates.append({
                "id": generate_id(),
                "covariate_type": claim["type"],
                "type": claim["type"],
                "description": claim["description"],
                "subject_id": find_entity_id(claim["subject"], entities),
                "object_id": find_entity_id(claim["object"], entities),
                "status": claim["status"],
                "start_date": claim.get("start_date"),
                "end_date": claim.get("end_date"),
                "source_text": unit["text"],
                "text_unit_id": unit["id"]
            })

    covariates_df = pd.DataFrame(all_covariates)
    covariates_df["human_readable_id"] = range(len(covariates_df))

    await write_table_to_storage(covariates_df, "covariates", context.output_storage)

    return WorkflowFunctionOutput(
        workflow="extract_covariates",
        output=covariates_df
    )
```

### Output DataFrame Schema

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `id` | string | UUID | `"c_001"` |
| `human_readable_id` | int | Sequential ID | `0` |
| `covariate_type` | string | Claim type | `"FACT"` |
| `type` | string | Specific type | `"FINANCIAL"` |
| `description` | string | Claim text | `"Reported $50B revenue"` |
| `subject_id` | string | Subject entity ID | `"e_001"` |
| `object_id` | string | Object entity ID | `"e_042"` |
| `status` | string | Verification status | `"CONFIRMED"` |
| `start_date` | string | Start date | `"2023-10-01"` |
| `end_date` | string | End date | `"2023-12-31"` |
| `source_text` | string | Original text | `"Microsoft reported..."` |
| `text_unit_id` | string | Source chunk ID | `"doc_001_chunk_5"` |

### Output File
- `covariates.parquet` (only if enabled)

---

## Step 7: create_communities

### Purpose
Detect hierarchical communities in the entity graph using Leiden clustering.

### Location
`packages/graphrag/graphrag/index/workflows/create_communities.py`

### What Happens

1. **Load Graph**: Reads entities and relationships
2. **Build NetworkX Graph**: Creates undirected graph structure
3. **Run Leiden Algorithm**: Hierarchical community detection
   - Level 0: Finest granularity (smallest communities)
   - Level 1+: Progressively larger parent communities
4. **Assign Community IDs**: Labels each entity with community at each level
5. **Build Hierarchy**: Establishes parent-child relationships
6. **Store Communities**: Saves community structure

### Leiden Clustering Algorithm

**Leiden** is an improved version of the Louvain algorithm for community detection:
- **Objective**: Maximize modularity (dense connections within communities, sparse between)
- **Hierarchical**: Creates multi-level community structure
- **Deterministic**: With fixed seed, produces same results

### Code Example

```python
# From create_communities.py
import networkx as nx
from graphrag.index.operations.cluster_graph import cluster_graph

async def run(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """Create hierarchical communities using Leiden clustering."""

    entities = await load_table_from_storage("entities", context.output_storage)
    relationships = await load_table_from_storage("relationships", context.output_storage)

    # Create graph
    graph = create_graph(entities, relationships)

    # Run hierarchical Leiden clustering
    community_hierarchy = cluster_graph(
        graph=graph,
        max_cluster_size=config.cluster_graph.max_cluster_size,  # 10
        use_lcc=config.cluster_graph.use_lcc,  # True (largest connected component)
        seed=config.cluster_graph.seed  # 0xDEADBEEF
    )

    # community_hierarchy structure:
    # {
    #   "entity_name": {
    #     "level_0": "community_0",
    #     "level_1": "community_5",
    #     "level_2": "community_15"
    #   },
    #   ...
    # }

    # Build communities DataFrame
    communities = []

    # Get unique communities at each level
    for level in range(max_level + 1):
        level_communities = defaultdict(lambda: {
            "entity_ids": [],
            "relationship_ids": [],
            "text_unit_ids": set()
        })

        # Aggregate entities per community
        for entity_name, levels in community_hierarchy.items():
            community_id = levels.get(f"level_{level}")
            if community_id:
                entity_row = entities[entities["title"] == entity_name].iloc[0]
                level_communities[community_id]["entity_ids"].append(entity_row["id"])
                level_communities[community_id]["text_unit_ids"].update(
                    entity_row["text_unit_ids"]
                )

        # Add relationships within community
        for community_id, data in level_communities.items():
            entity_names = entities[entities["id"].isin(data["entity_ids"])]["title"].tolist()

            # Find relationships where both source and target are in community
            community_rels = relationships[
                relationships["source"].isin(entity_names) &
                relationships["target"].isin(entity_names)
            ]
            data["relationship_ids"] = community_rels["id"].tolist()

        # Create community records
        for community_id, data in level_communities.items():
            # Determine parent community (from level+1)
            parent_id = None
            if level < max_level:
                # Find parent by looking at entities' level+1 assignment
                sample_entity = entities[entities["id"] == data["entity_ids"][0]].iloc[0]
                entity_name = sample_entity["title"]
                parent_id = community_hierarchy[entity_name].get(f"level_{level+1}")

            communities.append({
                "id": generate_community_id(community_id, level),
                "community": community_id,
                "level": level,
                "parent": parent_id,
                "entity_ids": data["entity_ids"],
                "relationship_ids": data["relationship_ids"],
                "text_unit_ids": list(data["text_unit_ids"]),
                "title": f"Community {community_id}",
                "size": len(data["entity_ids"]),
                "period": current_date()
            })

    communities_df = pd.DataFrame(communities)
    communities_df["human_readable_id"] = range(len(communities_df))

    # Add children relationships
    parent_to_children = defaultdict(list)
    for _, comm in communities_df.iterrows():
        if comm["parent"]:
            parent_to_children[comm["parent"]].append(comm["community"])

    communities_df["children"] = communities_df["community"].map(
        lambda c: parent_to_children.get(c, [])
    )

    await write_table_to_storage(communities_df, "communities", context.output_storage)

    return WorkflowFunctionOutput(
        workflow="create_communities",
        output=communities_df
    )
```

### Community Hierarchy Visualization

```
Level 2 (Highest - Most General)
    Community 15
    ├── Contains 50 entities
    └── Topic: "Technology Industry"

Level 1 (Mid-level)
    Community 5                    Community 6
    ├── Parent: Community 15       ├── Parent: Community 15
    ├── Contains 25 entities       ├── Contains 25 entities
    └── Topic: "Microsoft"         └── Topic: "Competitors"

Level 0 (Lowest - Most Specific)
    Comm 0          Comm 1         Comm 2          Comm 3
    ├── Parent: 5   ├── Parent: 5  ├── Parent: 6   ├── Parent: 6
    ├── 8 entities  ├── 10 ents    ├── 12 ents     ├── 15 ents
    └── "Products"  └── "Founders" └── "Google"    └── "Apple"
```

### Output DataFrame Schema

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `id` | string | UUID | `"comm_001"` |
| `human_readable_id` | int | Sequential ID | `0` |
| `community` | string | Community ID | `"community_5"` |
| `level` | int | Hierarchy level | `1` |
| `parent` | string | Parent community ID | `"community_15"` |
| `children` | list[string] | Child community IDs | `["community_0", "community_1"]` |
| `title` | string | Community name | `"Community 5"` |
| `entity_ids` | list[string] | Entities in community | `["e_001", "e_002", ...]` |
| `relationship_ids` | list[string] | Relationships within | `["r_001", "r_002", ...]` |
| `text_unit_ids` | list[string] | Source text units | `["doc_001_chunk_0", ...]` |
| `size` | int | Number of entities | `25` |
| `period` | string | Creation date | `"2024-01-15"` |

### Output File
- `communities.parquet`

---

## Step 8: create_final_text_units

### Purpose
Enrich text units with entity, relationship, and community linkages.

### Location
`packages/graphrag/graphrag/index/workflows/create_final_text_units.py`

### What Happens

1. **Load All Data**: Reads text units, entities, relationships, communities, covariates
2. **Build Reverse Mappings**: Create lookups from text unit ID to entities/relationships
3. **Enrich Text Units**: Add entity_ids, relationship_ids, covariate_ids to each chunk
4. **Assign Human-Readable IDs**: Sequential numbering
5. **Store Final Version**: Overwrites text_units.parquet

### Code Example

```python
# From create_final_text_units.py
async def run(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """Finalize text units with all linkages."""

    text_units = await load_table_from_storage("text_units", context.output_storage)
    entities = await load_table_from_storage("entities", context.output_storage)
    relationships = await load_table_from_storage("relationships", context.output_storage)

    # Optional covariates
    try:
        covariates = await load_table_from_storage("covariates", context.output_storage)
    except:
        covariates = None

    # Build reverse mappings: text_unit_id -> entity_ids
    text_unit_to_entities = defaultdict(list)
    for _, entity in entities.iterrows():
        for text_unit_id in entity["text_unit_ids"]:
            text_unit_to_entities[text_unit_id].append(entity["id"])

    # Build reverse mappings: text_unit_id -> relationship_ids
    text_unit_to_relationships = defaultdict(list)
    for _, rel in relationships.iterrows():
        for text_unit_id in rel["text_unit_ids"]:
            text_unit_to_relationships[text_unit_id].append(rel["id"])

    # Build reverse mappings: text_unit_id -> covariate_ids (if enabled)
    text_unit_to_covariates = defaultdict(list)
    if covariates is not None:
        for _, cov in covariates.iterrows():
            text_unit_to_covariates[cov["text_unit_id"]].append(cov["id"])

    # Enrich text units
    text_units["entity_ids"] = text_units["id"].map(
        lambda tid: text_unit_to_entities.get(tid, [])
    )
    text_units["relationship_ids"] = text_units["id"].map(
        lambda tid: text_unit_to_relationships.get(tid, [])
    )
    text_units["covariate_ids"] = text_units["id"].map(
        lambda tid: text_unit_to_covariates.get(tid, [])
    )

    # Add human-readable IDs
    text_units["human_readable_id"] = range(len(text_units))

    await write_table_to_storage(text_units, "text_units", context.output_storage)

    return WorkflowFunctionOutput(
        workflow="create_final_text_units",
        output=text_units
    )
```

### Final Text Units Schema

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `id` | string | Text unit ID | `"doc_001_chunk_0"` |
| `human_readable_id` | int | Sequential ID | `0` |
| `text` | string | Chunk content | `"Microsoft was founded..."` |
| `n_tokens` | int | Token count | `1150` |
| `document_id` | string | Parent document | `"doc_001"` |
| `entity_ids` | list[string] | Entities mentioned | `["e_001", "e_002"]` |
| `relationship_ids` | list[string] | Relationships present | `["r_001", "r_002"]` |
| `covariate_ids` | list[string] | Claims/facts | `["c_001"]` |

### Example Output

```python
# text_units DataFrame (final version)
   id                  human_readable_id  text                           entity_ids            relationship_ids
0  doc_001_chunk_0     0                  "Microsoft was founded..."     [e_001, e_002, e_003] [r_001, r_002]
1  doc_001_chunk_1     1                  "The company develops..."      [e_001, e_015]        [r_010, r_011]
```

### Output File
- `text_units.parquet` (final version with all linkages)

---

## Step 9: create_community_reports

### Purpose
Generate LLM-based natural language summaries for each community.

### Location
`packages/graphrag/graphrag/index/workflows/create_community_reports.py`

### What Happens

1. **Load Communities**: Reads community structure
2. **For Each Community**:
   - Build local context (entities, relationships, claims in community)
   - Format context as structured data (CSV/JSON)
   - Send to LLM with report generation prompt
   - Parse structured response (summary, findings, rating)
3. **Store Reports**: Saves community_reports.parquet

### Report Generation Prompt Example

```
-Role-
You are an AI assistant analyzing a community of entities in a knowledge graph.

-Goal-
Generate a comprehensive report about this community.

-Community Data-
Entities (id, name, type, description, degree):
{entity_table_csv}

Relationships (source, target, description, weight):
{relationship_table_csv}

Claims (subject, object, type, description):
{claims_table_csv}

-Report Structure-
1. Title: Short descriptive title (5-10 words)
2. Summary: 2-3 sentence overview
3. Rating: Importance rating 0-9 (9 = most important)
4. Rating Explanation: Why this rating?
5. Findings: List of key insights (3-5 bullet points)

-Output Format-
Return JSON:
{
  "title": "...",
  "summary": "...",
  "rating": 7,
  "rating_explanation": "...",
  "findings": [
    {"summary": "...", "explanation": "..."},
    ...
  ]
}
```

### Code Example

```python
# From create_community_reports.py
from graphrag.index.operations.community_reports import CommunityReportsExtractor

async def run(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """Generate community reports using LLM."""

    communities = await load_table_from_storage("communities", context.output_storage)
    entities = await load_table_from_storage("entities", context.output_storage)
    relationships = await load_table_from_storage("relationships", context.output_storage)

    # Optional covariates
    try:
        covariates = await load_table_from_storage("covariates", context.output_storage)
    except:
        covariates = None

    # Get LLM
    model_config = config.get_completion_model_config(
        config.community_reports.completion_model_id
    )
    llm = create_completion(model_config)

    # Generate report for each community
    reports = []

    for _, community in communities.iterrows():
        # Build local context
        community_entities = entities[
            entities["id"].isin(community["entity_ids"])
        ]
        community_relationships = relationships[
            relationships["id"].isin(community["relationship_ids"])
        ]

        if covariates is not None:
            community_covariates = covariates[
                covariates["subject_id"].isin(community["entity_ids"]) |
                covariates["object_id"].isin(community["entity_ids"])
            ]
        else:
            community_covariates = None

        # Format as CSV tables
        entity_context = build_entity_context(community_entities)
        relationship_context = build_relationship_context(community_relationships)
        claims_context = build_claims_context(community_covariates) if covariates else ""

        # Generate report
        prompt = build_community_report_prompt(
            entity_context,
            relationship_context,
            claims_context
        )

        result = await llm.completion(messages=prompt)
        report_data = parse_community_report(result.content)

        # Store report
        reports.append({
            "id": community["id"],
            "community": community["community"],
            "level": community["level"],
            "parent": community["parent"],
            "children": community["children"],
            "title": report_data["title"],
            "summary": report_data["summary"],
            "full_content": json.dumps(report_data["findings"]),
            "rank": report_data["rating"],
            "rating_explanation": report_data["rating_explanation"],
            "findings": report_data["findings"],
            "full_content_json": json.dumps(report_data),
            "size": community["size"],
            "period": community["period"]
        })

    reports_df = pd.DataFrame(reports)
    reports_df["human_readable_id"] = range(len(reports_df))

    await write_table_to_storage(reports_df, "community_reports", context.output_storage)

    return WorkflowFunctionOutput(
        workflow="create_community_reports",
        output=reports_df
    )
```

### Context Building Example

**Input Community**:
- Level: 1
- Size: 25 entities
- Topic: "Microsoft"

**Entity Context (CSV format)**:
```csv
id,name,type,description,degree
e_001,Microsoft,ORGANIZATION,Technology company founded in 1975,15
e_002,Bill Gates,PERSON,Co-founder and former CEO,8
e_003,Satya Nadella,PERSON,Current CEO since 2014,6
...
```

**Relationship Context (CSV format)**:
```csv
source,target,description,weight
Bill Gates,Microsoft,Co-founded in 1975,8
Satya Nadella,Microsoft,Appointed CEO in 2014,5
Microsoft,Windows,Developed operating system,12
...
```

**LLM Generated Report**:
```json
{
  "title": "Microsoft Corporation Leadership and Products",
  "summary": "This community focuses on Microsoft, a major technology company, its leadership including founders Bill Gates and Paul Allen and current CEO Satya Nadella, and its primary products like Windows and Office.",
  "rating": 8,
  "rating_explanation": "High importance due to Microsoft's significant role in technology industry and connections to many influential people and products.",
  "findings": [
    {
      "summary": "Microsoft founded in 1975 by Bill Gates and Paul Allen",
      "explanation": "The company has a long history with strong founding leadership that shaped the technology industry."
    },
    {
      "summary": "Leadership transition to Satya Nadella in 2014",
      "explanation": "Satya Nadella became CEO, marking a new era focusing on cloud computing and AI."
    },
    {
      "summary": "Core products include Windows and Office suite",
      "explanation": "These products are fundamental to Microsoft's business and widely used globally."
    }
  ]
}
```

### Output DataFrame Schema

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `id` | string | Community ID | `"comm_001"` |
| `human_readable_id` | int | Sequential ID | `0` |
| `community` | string | Community identifier | `"community_5"` |
| `level` | int | Hierarchy level | `1` |
| `parent` | string | Parent community | `"community_15"` |
| `children` | list[string] | Child communities | `["community_0", ...]` |
| `title` | string | Report title | `"Microsoft Leadership"` |
| `summary` | string | Brief summary | `"This community focuses on..."` |
| `full_content` | string | Complete report | `"[{findings...}]"` |
| `rank` | int | Importance (0-9) | `8` |
| `rating_explanation` | string | Rating justification | `"High importance due to..."` |
| `findings` | list[dict] | Key insights | `[{summary, explanation}, ...]` |
| `full_content_json` | string | Full JSON report | `"{...}"` |
| `size` | int | Number of entities | `25` |
| `period` | string | Creation date | `"2024-01-15"` |

### Output File
- `community_reports.parquet`

---

## Step 10: generate_text_embeddings

### Purpose
Create vector embeddings for semantic search over text units, entities, and community reports.

### Location
`packages/graphrag/graphrag/index/workflows/generate_text_embeddings.py`

### What Happens

1. **Load Data**: Reads text_units, entities, community_reports
2. **For Each Embedding Type**:
   - **text_unit_text**: Embed text unit content
   - **entity_description**: Embed "title: description" for entities
   - **community_full_content**: Embed full community reports
3. **Batch Texts**: Group by token count (configurable batch size)
4. **Generate Embeddings**: Call embedding model for each batch
5. **Store in Vector Store**: Load embeddings with IDs into configured vector store

### Embedding Types

| Type | Content | Purpose | Example |
|------|---------|---------|---------|
| `text_unit_text` | Raw text chunk | Retrieve relevant text for queries | `"Microsoft was founded..."` |
| `entity_description` | "title: description" | Find relevant entities by semantic similarity | `"Microsoft: Technology company..."` |
| `community_full_content` | Full community report | Retrieve relevant communities for queries | `"Microsoft Leadership and Products..."` |

### Code Example

```python
# From generate_text_embeddings.py
from graphrag_llm.embedding import create_embedding
from graphrag_vectors import create_vector_store

async def run(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """Generate embeddings and store in vector store."""

    # Get embedding model
    embedding_config = config.get_embedding_model_config(
        config.embed_text.embedding_model_id
    )
    embedding_model = create_embedding(embedding_config)

    # Get vector store
    vector_store = create_vector_store(config.vector_store)

    # Load data
    text_units = await load_table_from_storage("text_units", context.output_storage)
    entities = await load_table_from_storage("entities", context.output_storage)
    community_reports = await load_table_from_storage(
        "community_reports", context.output_storage
    )

    # Embedding 1: Text Unit Text
    if "text_unit_text" in config.embed_text.embeddings:
        texts = text_units["text"].tolist()
        ids = text_units["id"].tolist()

        embeddings = await generate_embeddings_batched(
            texts=texts,
            embedding_model=embedding_model,
            batch_size=config.embed_text.batch_size,  # 16
            batch_max_tokens=config.embed_text.batch_max_tokens  # 8191
        )

        await vector_store.load_documents(
            collection_name="text_unit_text",
            ids=ids,
            texts=texts,
            embeddings=embeddings
        )

    # Embedding 2: Entity Description
    if "entity_description" in config.embed_text.embeddings:
        # Format: "title: description"
        texts = [
            f"{row['title']}: {row['description']}"
            for _, row in entities.iterrows()
        ]
        ids = entities["id"].tolist()

        embeddings = await generate_embeddings_batched(
            texts=texts,
            embedding_model=embedding_model,
            batch_size=config.embed_text.batch_size,
            batch_max_tokens=config.embed_text.batch_max_tokens
        )

        await vector_store.load_documents(
            collection_name="entity_description",
            ids=ids,
            texts=texts,
            embeddings=embeddings
        )

    # Embedding 3: Community Full Content
    if "community_full_content" in config.embed_text.embeddings:
        texts = community_reports["full_content"].tolist()
        ids = community_reports["id"].tolist()

        embeddings = await generate_embeddings_batched(
            texts=texts,
            embedding_model=embedding_model,
            batch_size=config.embed_text.batch_size,
            batch_max_tokens=config.embed_text.batch_max_tokens
        )

        await vector_store.load_documents(
            collection_name="community_full_content",
            ids=ids,
            texts=texts,
            embeddings=embeddings
        )

    return WorkflowFunctionOutput(
        workflow="generate_text_embeddings",
        output=None  # Embeddings stored in vector store
    )


async def generate_embeddings_batched(
    texts: list[str],
    embedding_model: LLMEmbedding,
    batch_size: int,
    batch_max_tokens: int
) -> list[list[float]]:
    """Generate embeddings in batches respecting token limits."""

    all_embeddings = []
    current_batch = []
    current_batch_tokens = 0

    for text in texts:
        text_tokens = count_tokens(text)

        # Check if adding this text exceeds batch limits
        if (len(current_batch) >= batch_size or
            current_batch_tokens + text_tokens > batch_max_tokens):
            # Process current batch
            batch_embeddings = await embedding_model.embedding(
                input=current_batch
            )
            all_embeddings.extend(batch_embeddings.embeddings)

            # Reset batch
            current_batch = []
            current_batch_tokens = 0

        current_batch.append(text)
        current_batch_tokens += text_tokens

    # Process remaining batch
    if current_batch:
        batch_embeddings = await embedding_model.embedding(
            input=current_batch
        )
        all_embeddings.extend(batch_embeddings.embeddings)

    return all_embeddings
```

### Batching Strategy

**Why Batching?**
- Embedding models have token limits (e.g., 8191 tokens for text-embedding-3-small)
- Batching improves API efficiency (fewer requests)
- Token counting ensures no batch exceeds limits

**Example Batching**:
```
Batch 1 (15 texts, 7800 tokens):
  text_1 (500 tokens) + text_2 (480 tokens) + ... + text_15 (520 tokens)
  → Call embedding_model.embedding(input=[text_1, ..., text_15])
  → Returns 15 embedding vectors

Batch 2 (12 texts, 6500 tokens):
  text_16 (550 tokens) + ... + text_27 (490 tokens)
  → Call embedding_model.embedding(input=[text_16, ..., text_27])
  → Returns 12 embedding vectors
```

### Vector Store Structure

**LanceDB Example**:
```
vector_store/
├── text_unit_text/
│   ├── data.lance          # Binary vector storage
│   └── index.json          # Metadata
├── entity_description/
│   ├── data.lance
│   └── index.json
└── community_full_content/
    ├── data.lance
    └── index.json
```

### Embedding Dimensions

**Common Models**:
- `text-embedding-3-small`: 1536 dimensions
- `text-embedding-3-large`: 3072 dimensions
- `text-embedding-ada-002`: 1536 dimensions

### Output
- **No parquet file**: Embeddings stored directly in vector store
- **Vector store collections**: 3 collections created (text_unit, entity, community)

---

## Complete Pipeline Output

### Final Output Directory Structure

```
output/
├── documents.parquet                # Final documents with metadata
├── text_units.parquet               # Text chunks with all linkages
├── entities.parquet                 # Extracted entities with graph metrics
├── relationships.parquet            # Extracted relationships with weights
├── communities.parquet              # Hierarchical community structure
├── community_reports.parquet        # LLM-generated community summaries
├── covariates.parquet               # Claims/facts (if enabled)
├── context.json                     # Pipeline execution state
└── stats.json                       # Execution statistics

vector_store/                        # Vector embeddings
├── text_unit_text/                  # Text chunk embeddings
├── entity_description/              # Entity embeddings
└── community_full_content/          # Community report embeddings
```

### Pipeline Execution Statistics

**Example `stats.json`**:
```json
{
  "workflow_stats": {
    "load_input_documents": {
      "duration_seconds": 2.5,
      "num_documents": 150
    },
    "create_base_text_units": {
      "duration_seconds": 5.2,
      "num_text_units": 2500
    },
    "extract_graph": {
      "duration_seconds": 450.0,
      "llm_calls": 2600,
      "prompt_tokens": 3500000,
      "output_tokens": 850000,
      "num_entities": 1200,
      "num_relationships": 3500
    },
    "create_communities": {
      "duration_seconds": 15.0,
      "num_communities": 85,
      "num_levels": 3
    },
    "create_community_reports": {
      "duration_seconds": 180.0,
      "llm_calls": 85,
      "prompt_tokens": 250000,
      "output_tokens": 45000
    },
    "generate_text_embeddings": {
      "duration_seconds": 60.0,
      "num_embeddings": 3785,
      "embedding_calls": 250
    }
  },
  "total_duration_seconds": 850.0,
  "total_llm_calls": 2685,
  "total_prompt_tokens": 3750000,
  "total_output_tokens": 895000
}
```

---

## Summary

The Standard Indexing Pipeline transforms raw documents into a rich, queryable knowledge graph through 10 carefully orchestrated workflows:

1. ✅ **Load documents** → Raw content ingestion
2. ✅ **Chunk documents** → Create text units (1200 tokens, 100 overlap)
3. ✅ **Enrich documents** → Add metadata
4. ✅ **Extract graph** → LLM-based entity/relationship extraction
5. ✅ **Finalize graph** → Calculate graph metrics
6. ✅ **Extract claims** → Optional covariate extraction
7. ✅ **Detect communities** → Hierarchical Leiden clustering
8. ✅ **Link text units** → Connect chunks to entities/relationships
9. ✅ **Generate reports** → LLM summaries for communities
10. ✅ **Create embeddings** → Vector indexes for semantic search

**Key Takeaways**:
- **Multi-layered**: Creates both graph structure and text embeddings
- **LLM-powered**: Uses LLMs for extraction, summarization, and reporting
- **Hierarchical**: Communities at multiple levels enable multi-scale reasoning
- **Linked**: All data structures cross-reference each other
- **Production-ready**: Configurable, scalable, and well-tested

This pipeline enables powerful query capabilities through the hierarchical knowledge graph structure!
