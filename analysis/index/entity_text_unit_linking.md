# Entity-Text Unit Linking Mechanism in GraphRAG

## Overview

One of the most important aspects of GraphRAG is the **bidirectional linking between entities and text units (chunks)**. This mechanism enables:
- **Provenance tracking**: Know where each entity was mentioned
- **Citation support**: Show source text for entity information
- **Context retrieval**: Find relevant text chunks for entities during queries
- **Co-occurrence analysis**: Identify entities that appear together

This document explains exactly how these links are created and maintained throughout the indexing pipeline.

---

## Core Concept: Bidirectional Links

GraphRAG maintains two complementary link structures:

### 1. Forward Links: Entity → Text Units
**Created in**: Step 4 (extract_graph)
**Stored in**: `entities.parquet` column `text_unit_ids`

Each entity maintains a list of ALL text unit IDs where it was mentioned:

```python
entity = {
    "id": "e_001",
    "title": "Microsoft",
    "type": "ORGANIZATION",
    "description": "Technology company...",
    "text_unit_ids": [
        "doc_001_chunk_0",  # First document, first chunk
        "doc_001_chunk_2",  # First document, third chunk
        "doc_001_chunk_5",  # First document, sixth chunk
        "doc_002_chunk_1",  # Second document, second chunk
        "doc_003_chunk_0",  # Third document, first chunk
        ...                 # More mentions
    ]  # 23 mentions total
}
```

### 2. Reverse Links: Text Unit → Entities
**Created in**: Step 8 (create_final_text_units)
**Stored in**: `text_units.parquet` column `entity_ids`

Each text unit maintains a list of ALL entity IDs mentioned in it:

```python
text_unit = {
    "id": "doc_001_chunk_0",
    "text": "Microsoft was founded by Bill Gates and Paul Allen in 1975...",
    "n_tokens": 1150,
    "document_id": "doc_001",
    "entity_ids": [
        "e_001",  # Microsoft
        "e_002",  # Bill Gates
        "e_003",  # Paul Allen
        "e_125"   # 1975 (if year treated as entity)
    ]
}
```

---

## Step-by-Step Linking Process

### Phase 1: Initial Link Creation (Step 4 - extract_graph)

#### Process Flow

```
Text Unit: doc_001_chunk_0
Text Content: "Microsoft was founded by Bill Gates and Paul Allen in 1975..."
    ↓
[LLM Extraction]
    ↓
Extracted Entities:
1. Microsoft (ORGANIZATION)
2. Bill Gates (PERSON)
3. Paul Allen (PERSON)
    ↓
[Link Creation]
    ↓
For EACH extracted entity:
    entity["text_unit_ids"] = [current_text_unit_id]
    ↓
Result:
- Microsoft → text_unit_ids: ["doc_001_chunk_0"]
- Bill Gates → text_unit_ids: ["doc_001_chunk_0"]
- Paul Allen → text_unit_ids: ["doc_001_chunk_0"]
```

#### Code Example

```python
# From extract_graph.py - simplified
for text_unit in text_units:
    text_unit_id = text_unit["id"]
    text_content = text_unit["text"]

    # Extract entities from this text unit
    extracted = await llm.extract_entities(text_content)

    for entity in extracted:
        # KEY: Link entity to source text unit
        entity["text_unit_ids"] = [text_unit_id]

        # Store entity
        all_entities.append(entity)
```

### Phase 2: Merging Duplicate Entities (Step 4 - extract_graph)

When the same entity appears in multiple text units, we merge them and **combine their text_unit_ids**:

#### Example Scenario

```
Processing text_units sequentially:

After doc_001_chunk_0:
    Microsoft → text_unit_ids: ["doc_001_chunk_0"]

After doc_001_chunk_2:
    Microsoft → text_unit_ids: ["doc_001_chunk_0", "doc_001_chunk_2"]

After doc_001_chunk_5:
    Microsoft → text_unit_ids: ["doc_001_chunk_0", "doc_001_chunk_2", "doc_001_chunk_5"]

After doc_002_chunk_1:
    Microsoft → text_unit_ids: [
        "doc_001_chunk_0",
        "doc_001_chunk_2",
        "doc_001_chunk_5",
        "doc_002_chunk_1"
    ]

... continues for all text units ...

Final Result:
    Microsoft → text_unit_ids: [23 text unit IDs]
```

#### Merging Algorithm

```python
# Merge duplicate entities by name
entity_groups = defaultdict(list)

# Group all entity instances by normalized name
for entity in all_entities:
    normalized_name = normalize(entity["name"])  # e.g., lowercase, trim
    entity_groups[normalized_name].append(entity)

# Merge each group
merged_entities = []
for name, entity_instances in entity_groups.items():
    # Collect all text_unit_ids from all instances
    all_text_unit_ids = []
    for instance in entity_instances:
        all_text_unit_ids.extend(instance["text_unit_ids"])

    # Remove duplicates
    unique_text_unit_ids = list(set(all_text_unit_ids))

    # Create merged entity
    merged = {
        "id": generate_uuid(),
        "title": name,
        "type": entity_instances[0]["type"],
        "description": merge_descriptions(entity_instances),
        "text_unit_ids": unique_text_unit_ids  # Combined from all instances
    }

    merged_entities.append(merged)
```

### Phase 3: Calculate Node Frequency (Step 5 - finalize_graph)

The **node_frequency** metric is derived from the length of `text_unit_ids`:

```python
# From finalize_graph.py
for entity in entities_df:
    # node_frequency = how many text units mention this entity
    entity["node_frequency"] = len(entity["text_unit_ids"])
```

**Interpretation**:
- High node_frequency → Entity is mentioned frequently → Important entity
- Low node_frequency → Entity is mentioned rarely → Less important entity

**Example**:
```
Microsoft: node_frequency = 23  (mentioned in 23 chunks)
Bill Gates: node_frequency = 12  (mentioned in 12 chunks)
Paul Allen: node_frequency = 8   (mentioned in 8 chunks)
```

### Phase 4: Create Reverse Links (Step 8 - create_final_text_units)

Now we create the **reverse mapping** from text units back to entities:

#### Process Flow

```
Goal: For each text_unit, populate entity_ids list

Step 1: Build Reverse Mapping
    For each entity in entities_df:
        For each text_unit_id in entity["text_unit_ids"]:
            reverse_map[text_unit_id].append(entity["id"])

Step 2: Apply to Text Units
    For each text_unit in text_units_df:
        text_unit["entity_ids"] = reverse_map[text_unit["id"]]
```

#### Code Example

```python
# From create_final_text_units.py
from collections import defaultdict

# Build reverse mapping: text_unit_id → [entity_ids]
text_unit_to_entities = defaultdict(list)

for _, entity in entities_df.iterrows():
    entity_id = entity["id"]
    for text_unit_id in entity["text_unit_ids"]:
        text_unit_to_entities[text_unit_id].append(entity_id)

# Apply to text units
text_units_df["entity_ids"] = text_units_df["id"].map(
    lambda text_unit_id: text_unit_to_entities.get(text_unit_id, [])
)
```

#### Result

```python
# Before Step 8
text_unit = {
    "id": "doc_001_chunk_0",
    "text": "Microsoft was founded by Bill Gates...",
    "entity_ids": None  # Not yet populated
}

# After Step 8
text_unit = {
    "id": "doc_001_chunk_0",
    "text": "Microsoft was founded by Bill Gates...",
    "entity_ids": ["e_001", "e_002", "e_003"]  # Microsoft, Bill Gates, Paul Allen
}
```

---

## Complete Linking Example

Let's trace a complete example through the entire pipeline:

### Input Documents

**Document 1** (`doc_001`):
```
Microsoft Corporation was founded by Bill Gates and Paul Allen in 1975.
The company is headquartered in Redmond, Washington.
Microsoft develops software products including Windows and Office.
```

**Document 2** (`doc_002`):
```
Bill Gates served as CEO of Microsoft until 2000.
The company reported strong quarterly earnings.
```

### Step 2: Text Units Created

After chunking with chunk_size=1200, overlap=100:

```
doc_001_chunk_0: "Microsoft Corporation was founded by Bill Gates and Paul Allen in 1975.
                  The company is headquartered in Redmond, Washington."

doc_001_chunk_1: "The company is headquartered in Redmond, Washington.
                  Microsoft develops software products including Windows and Office."

doc_002_chunk_0: "Bill Gates served as CEO of Microsoft until 2000.
                  The company reported strong quarterly earnings."
```

### Step 4: Entity Extraction and Forward Link Creation

**Processing doc_001_chunk_0**:
```python
# LLM extracts:
entities_from_chunk_0 = [
    {"name": "Microsoft", "type": "ORG", "text_unit_ids": ["doc_001_chunk_0"]},
    {"name": "Bill Gates", "type": "PERSON", "text_unit_ids": ["doc_001_chunk_0"]},
    {"name": "Paul Allen", "type": "PERSON", "text_unit_ids": ["doc_001_chunk_0"]},
    {"name": "Redmond", "type": "GEO", "text_unit_ids": ["doc_001_chunk_0"]},
    {"name": "Washington", "type": "GEO", "text_unit_ids": ["doc_001_chunk_0"]}
]
```

**Processing doc_001_chunk_1**:
```python
# LLM extracts:
entities_from_chunk_1 = [
    {"name": "Redmond", "type": "GEO", "text_unit_ids": ["doc_001_chunk_1"]},  # Duplicate
    {"name": "Washington", "type": "GEO", "text_unit_ids": ["doc_001_chunk_1"]},  # Duplicate
    {"name": "Microsoft", "type": "ORG", "text_unit_ids": ["doc_001_chunk_1"]},  # Duplicate
    {"name": "Windows", "type": "PRODUCT", "text_unit_ids": ["doc_001_chunk_1"]},
    {"name": "Office", "type": "PRODUCT", "text_unit_ids": ["doc_001_chunk_1"]}
]
```

**Processing doc_002_chunk_0**:
```python
# LLM extracts:
entities_from_chunk_2 = [
    {"name": "Bill Gates", "type": "PERSON", "text_unit_ids": ["doc_002_chunk_0"]},  # Duplicate
    {"name": "Microsoft", "type": "ORG", "text_unit_ids": ["doc_002_chunk_0"]}  # Duplicate
]
```

**After Merging Duplicates**:
```python
merged_entities = [
    {
        "id": "e_001",
        "title": "Microsoft",
        "type": "ORGANIZATION",
        "text_unit_ids": ["doc_001_chunk_0", "doc_001_chunk_1", "doc_002_chunk_0"]  # 3 mentions
    },
    {
        "id": "e_002",
        "title": "Bill Gates",
        "type": "PERSON",
        "text_unit_ids": ["doc_001_chunk_0", "doc_002_chunk_0"]  # 2 mentions
    },
    {
        "id": "e_003",
        "title": "Paul Allen",
        "type": "PERSON",
        "text_unit_ids": ["doc_001_chunk_0"]  # 1 mention
    },
    {
        "id": "e_004",
        "title": "Redmond",
        "type": "GEO",
        "text_unit_ids": ["doc_001_chunk_0", "doc_001_chunk_1"]  # 2 mentions
    },
    {
        "id": "e_005",
        "title": "Washington",
        "type": "GEO",
        "text_unit_ids": ["doc_001_chunk_0", "doc_001_chunk_1"]  # 2 mentions
    },
    {
        "id": "e_006",
        "title": "Windows",
        "type": "PRODUCT",
        "text_unit_ids": ["doc_001_chunk_1"]  # 1 mention
    },
    {
        "id": "e_007",
        "title": "Office",
        "type": "PRODUCT",
        "text_unit_ids": ["doc_001_chunk_1"]  # 1 mention
    }
]
```

### Step 5: Calculate Node Frequency

```python
entities_with_frequency = [
    {"id": "e_001", "title": "Microsoft", "node_frequency": 3},      # Most frequent
    {"id": "e_002", "title": "Bill Gates", "node_frequency": 2},
    {"id": "e_004", "title": "Redmond", "node_frequency": 2},
    {"id": "e_005", "title": "Washington", "node_frequency": 2},
    {"id": "e_003", "title": "Paul Allen", "node_frequency": 1},
    {"id": "e_006", "title": "Windows", "node_frequency": 1},
    {"id": "e_007", "title": "Office", "node_frequency": 1}
]
```

### Step 8: Create Reverse Links

**Build Reverse Mapping**:
```python
text_unit_to_entities = {
    "doc_001_chunk_0": ["e_001", "e_002", "e_003", "e_004", "e_005"],  # 5 entities
    "doc_001_chunk_1": ["e_001", "e_004", "e_005", "e_006", "e_007"],  # 5 entities
    "doc_002_chunk_0": ["e_001", "e_002"]                              # 2 entities
}
```

**Apply to Text Units**:
```python
text_units_final = [
    {
        "id": "doc_001_chunk_0",
        "text": "Microsoft Corporation was founded...",
        "entity_ids": ["e_001", "e_002", "e_003", "e_004", "e_005"]
    },
    {
        "id": "doc_001_chunk_1",
        "text": "The company is headquartered...",
        "entity_ids": ["e_001", "e_004", "e_005", "e_006", "e_007"]
    },
    {
        "id": "doc_002_chunk_0",
        "text": "Bill Gates served as CEO...",
        "entity_ids": ["e_001", "e_002"]
    }
]
```

### Final Bidirectional Links

**Forward (Entity → Text Units)**:
```
Microsoft (e_001) → ["doc_001_chunk_0", "doc_001_chunk_1", "doc_002_chunk_0"]
Bill Gates (e_002) → ["doc_001_chunk_0", "doc_002_chunk_0"]
Paul Allen (e_003) → ["doc_001_chunk_0"]
Windows (e_006) → ["doc_001_chunk_1"]
```

**Reverse (Text Unit → Entities)**:
```
doc_001_chunk_0 → ["e_001", "e_002", "e_003", "e_004", "e_005"]
doc_001_chunk_1 → ["e_001", "e_004", "e_005", "e_006", "e_007"]
doc_002_chunk_0 → ["e_001", "e_002"]
```

---

## Use Cases During Query Time

### 1. Citation / Provenance

When displaying entity information, show source text:

```python
# Query: "Tell me about Microsoft"
entity = entities_df[entities_df["title"] == "Microsoft"].iloc[0]

# Retrieve source text units
source_chunks = text_units_df[
    text_units_df["id"].isin(entity["text_unit_ids"])
]

# Display with citations
print(f"Microsoft: {entity['description']}")
print("\nSources:")
for chunk in source_chunks:
    print(f"- {chunk['text'][:100]}...")
```

**Output**:
```
Microsoft: Technology company founded in 1975...

Sources:
- Microsoft Corporation was founded by Bill Gates and Paul Allen in 1975. The company is headquarter...
- The company is headquartered in Redmond, Washington. Microsoft develops software products includi...
- Bill Gates served as CEO of Microsoft until 2000. The company reported strong quarterly earnings...
```

### 2. Context Retrieval for Local Search

Find all text units related to an entity:

```python
# Local search query: "What products does Microsoft make?"

# Step 1: Find Microsoft entity via vector search
entity = find_entity_by_vector_search("Microsoft")

# Step 2: Retrieve all text units mentioning Microsoft
related_chunks = text_units_df[
    text_units_df["id"].isin(entity["text_unit_ids"])
]

# Step 3: Build context for LLM
context = "\n\n".join(related_chunks["text"])

# Step 4: Generate answer
answer = llm.completion(
    f"Context:\n{context}\n\nQuestion: What products does Microsoft make?"
)
```

### 3. Co-occurrence Analysis

Find entities that frequently appear together:

```python
# Find entities that co-occur with "Microsoft"
microsoft_entity = entities_df[entities_df["title"] == "Microsoft"].iloc[0]
microsoft_chunks = set(microsoft_entity["text_unit_ids"])

# For each other entity, count shared text units
co_occurrences = []
for _, entity in entities_df.iterrows():
    if entity["title"] == "Microsoft":
        continue

    entity_chunks = set(entity["text_unit_ids"])
    shared_chunks = microsoft_chunks.intersection(entity_chunks)

    if shared_chunks:
        co_occurrences.append({
            "entity": entity["title"],
            "co_occurrence_count": len(shared_chunks),
            "shared_chunks": list(shared_chunks)
        })

# Sort by co-occurrence
co_occurrences.sort(key=lambda x: x["co_occurrence_count"], reverse=True)
```

**Result**:
```python
[
    {"entity": "Bill Gates", "co_occurrence_count": 2, "shared_chunks": ["doc_001_chunk_0", "doc_002_chunk_0"]},
    {"entity": "Redmond", "co_occurrence_count": 2, "shared_chunks": ["doc_001_chunk_0", "doc_001_chunk_1"]},
    {"entity": "Washington", "co_occurrence_count": 2, "shared_chunks": ["doc_001_chunk_0", "doc_001_chunk_1"]},
    {"entity": "Paul Allen", "co_occurrence_count": 1, "shared_chunks": ["doc_001_chunk_0"]},
    {"entity": "Windows", "co_occurrence_count": 1, "shared_chunks": ["doc_001_chunk_1"]},
    {"entity": "Office", "co_occurrence_count": 1, "shared_chunks": ["doc_001_chunk_1"]}
]
```

### 4. Relationship Weight Calculation

Relationship weights are based on co-occurrence in text units:

```python
relationship = {
    "source": "Bill Gates",
    "target": "Microsoft",
    "text_unit_ids": ["doc_001_chunk_0", "doc_002_chunk_0"]
}

# Weight = number of text units where both entities appear
relationship["weight"] = len(relationship["text_unit_ids"])  # 2.0
```

**Interpretation**: Higher weight means entities are mentioned together more frequently, indicating a stronger relationship.

---

## Performance Considerations

### Memory Usage

Each entity stores a list of text unit IDs:
- **Small dataset** (1000 entities, 10 mentions each): ~10KB per entity → 10MB total
- **Large dataset** (100K entities, 50 mentions each): ~50KB per entity → 5GB total

**Optimization**: Store text_unit_ids as compressed arrays or use database joins instead of in-memory lists.

### Query Performance

**Fast Operations**:
- ✅ Get text units for entity: `O(k)` where k = number of mentions
- ✅ Get entities in text unit: `O(m)` where m = number of entities in chunk

**Slow Operations**:
- ❌ Find all co-occurring entities: `O(n²)` where n = number of entities
- ❌ Build full co-occurrence matrix: Requires nested loops

**Optimization**: Pre-compute co-occurrence matrices during indexing for faster query-time lookups.

---

## Summary

The entity-text unit linking mechanism in GraphRAG:

1. **Created During Extraction** (Step 4): Each entity tracks which text units mention it
2. **Merged Across Chunks**: Duplicate entities combine their text_unit_ids
3. **Measured as Frequency** (Step 5): node_frequency = len(text_unit_ids)
4. **Reversed in Final Step** (Step 8): Text units get entity_ids list

**Benefits**:
- ✅ Full provenance tracking
- ✅ Citation support
- ✅ Context retrieval for queries
- ✅ Co-occurrence analysis
- ✅ Relationship weight calculation

**Key Insight**: This bidirectional linking is what enables GraphRAG to provide **cited, grounded answers** with source text, distinguishing it from traditional RAG systems that lose provenance after retrieval.
