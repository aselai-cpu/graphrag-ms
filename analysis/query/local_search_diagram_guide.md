# Local Search Activity Diagram - Guide

**File**: `local_search_activity.puml`
**Type**: PlantUML Activity Diagram
**Purpose**: Visual representation of GraphRAG Local Search complete flow

---

## Overview

The Local Search activity diagram shows the complete end-to-end flow of a local search query in GraphRAG, from CLI invocation through answer generation and result display.

## Diagram Structure

The diagram is organized into **5 main phases** with **6 swim lanes**:

### Swim Lanes (by color)

1. **CLI Layer** (AntiqueWhite) - User interaction and command handling
2. **Data Loading** (LightBlue) - Reading parquet files and embeddings
3. **Initialization** (LightGreen) - Search engine and context builder setup
4. **Query Execution** (LightYellow) - Main search orchestration
5. **Phase-specific lanes**:
   - Entity Mapping (LightCoral)
   - Graph Traversal (LightSteelBlue)
   - Context Building (Lavender)
   - Answer Generation (LightGoldenRodYellow)
   - Result Assembly (PaleGreen)

### Main Phases

#### Phase 1: Entity Mapping (Vector Search)
**Purpose**: Find entities relevant to the query

**Steps**:
1. Embed query text using embedding model
2. Perform vector similarity search on entity descriptions
3. Retrieve top candidates (oversampled by 2x)
4. Filter and rank by similarity
5. Select top-k entities (default: 10)

**Key Parameters**:
- `top_k_mapped_entities`: Number of seed entities (default: 10)
- `oversample_scaler`: Oversampling factor (default: 2)
- `embedding_vectorstore_key`: Search by ID or TITLE

**Output**: List of seed entities most relevant to query

---

#### Phase 2: Graph Traversal (Neighborhood Expansion)
**Purpose**: Expand seed entities to their local neighborhoods

**Steps**:
1. Start with seed entities from Phase 1
2. For each seed entity:
   - Find all relationships where entity is source or target
   - Extract connected entities (neighbors)
   - Rank neighbors by entity importance (rank attribute)
   - Select top-k neighbors
3. Combine seed entities + neighbors into final entity set

**Key Parameters**:
- `top_k_relationships`: Neighbors per entity (default: 10)

**Output**: Expanded entity set forming local subgraph

**Algorithm**:
```
entity_set = seed_entities
for entity in seed_entities:
    relationships = find_relationships(entity)
    neighbors = extract_neighbors(relationships)
    top_neighbors = rank_and_select(neighbors, k=top_k_relationships)
    entity_set.add(top_neighbors)
return entity_set
```

---

#### Phase 3: Context Building (Multi-Source Assembly)
**Purpose**: Gather rich contextual information from multiple sources

**Parallel Context Retrieval** (5 concurrent sub-processes):

##### 3.1 Entity Context
- Filter entities by entity_set
- Format as CSV: `id,entity,description`
- Order by rank (most important first)
- No token limit (entities are brief)

##### 3.2 Relationship Context
- Get relationships between entities in entity_set
- Apply count limit: `top_k_relationships` (default: 10)
- Format as CSV: `id,source,target,description`
- Order by rank

##### 3.3 Community Context
- Find communities containing entities in entity_set
- Retrieve community reports for those communities
- Apply token budget: `max_context_tokens × community_prop` (default: 6000 tokens)
- Format as CSV: `id,community,summary`
- Order by community size (largest first)

**Token Budget Algorithm**:
```python
current_tokens = 0
selected_reports = []

for report in sorted_reports:
    report_tokens = count_tokens(report.summary)
    if current_tokens + report_tokens <= community_budget:
        selected_reports.append(report)
        current_tokens += report_tokens
    else:
        break  # Budget exceeded

return selected_reports
```

##### 3.4 Text Unit Context
- Filter text units mentioning entities in entity_set
- Apply token budget: `max_context_tokens × text_unit_prop` (default: 6000 tokens)
- Sort by relevance: entity rank (primary), text order (secondary)
- Format as CSV: `id,text`

##### 3.5 Covariate Context (Claims - Optional)
- **What**: Structured factual assertions about entities
- **When Used**: Only if `extract_claims.enabled = true` in settings.yaml (default: false)
- **Process**:
  1. Filter claims where subject_id matches entities in entity_set
  2. Apply remaining token budget (after text units and communities)
  3. Format as CSV with claim attributes
- **Format**: `id|subject|object|type|status|description|start_date|end_date|source_text`
- **Example Claim**:
  ```
  claim_1|COMPANY A|GOVT AGENCY B|REGULATORY_VIOLATION|TRUE|Company A was fined...|2022-01-10|2022-01-10|According to...
  ```
- **Token Priority**: Lowest (uses only remaining budget)

**Context Assembly**:
```
-----Entities-----
[entity_context CSV]

-----Relationships-----
[relationship_context CSV]

-----Reports-----
[community_context CSV]

-----Sources-----
[text_unit_context CSV]

-----Claims-----
[covariate_context CSV]
```

**Key Parameters**:
- `max_context_tokens`: Total limit (default: 12000)
- `text_unit_prop`: Text unit proportion (default: 0.5)
- `community_prop`: Community proportion (default: 0.5)

**Output**: Formatted context string within token budget

---

#### Phase 4: Answer Generation (LLM Call)
**Purpose**: Generate answer using LLM with rich context

**Steps**:
1. Build completion messages:
   - System message: `LOCAL_SEARCH_SYSTEM_PROMPT` + formatted context
   - User message: Original query
2. Call LLM:
   - Streaming mode: Yield tokens as received
   - Non-streaming: Return complete response
3. Track metadata: tokens, time, LLM calls

**Streaming Flow**:
```
if streaming:
    async for chunk in llm.completion_async(messages, stream=True):
        token = chunk.choices[0].delta.content
        yield token  # Real-time display
        full_response += token
else:
    response = await llm.completion_async(messages)
    return response.content
```

**Key Parameters**:
- `completion_model_id`: LLM model (e.g., gpt-4o, claude-3-5-sonnet)
- `temperature`: Randomness (default: 0)
- `max_tokens`: Output limit (from config)

**Output**: Generated answer text

---

#### Phase 5: Result Assembly
**Purpose**: Package results for return to user

**SearchResult Structure**:
```python
@dataclass
class SearchResult:
    response: str                          # Generated answer
    context_data: dict                     # Raw DataFrames
    context_text: str                      # Formatted context
    completion_time: float                 # Duration in seconds
    llm_calls: int                         # Always 1 for local search
    prompt_tokens: int                     # Input tokens
    output_tokens: int                     # Generated tokens
    llm_calls_categories: dict[str, int]   # {"search": 1}
    prompt_tokens_categories: dict[str, int]
    output_tokens_categories: dict[str, int]
```

---

## Key Characteristics

### Single LLM Call
- Unlike global search (MAP + REDUCE with many calls)
- All context in one prompt
- Efficient for targeted queries

### Token Budget Management
- **Total budget**: 12000 tokens (default)
- **Allocation**: 50% text units, 50% communities
- **Enforcement**: Stop adding items when budget exceeded
- **Priority**: Higher-ranked items first

### Multi-Source Context
- **Entities**: Names and descriptions
- **Relationships**: Connections and descriptions
- **Communities**: Hierarchical summaries
- **Text Units**: Original source text
- **Covariates (Claims)**: Structured assertions about entities (optional, disabled by default)

### Graph-Aware
- Uses entity neighborhoods
- Relationship traversal
- Community structure
- Not just vector similarity

---

## Performance Characteristics

### Speed: Medium (~2-5 seconds)
| Operation | Time |
|-----------|------|
| Vector search | 100-200ms |
| Graph traversal | 50-100ms |
| Context building | 100-200ms |
| LLM generation | 1-4 seconds |
| **Total** | **2-5 seconds** |

### Cost: Medium
| Component | Tokens |
|-----------|--------|
| System prompt | ~500 |
| Context | ~8000-11500 |
| Query | ~50-200 |
| **Prompt total** | **~8500-12000** |
| **Output** | **500-2000** |
| **Total** | **~9000-14000** |

**Cost Example (GPT-4o)**:
- Input: 10000 tokens × $2.50/1M = $0.025
- Output: 1000 tokens × $10/1M = $0.01
- **Total per query**: **~$0.035**

### Quality: High for Specific Queries
- Rich contextual information
- Multiple information sources
- Graph-aware relationships
- Focused on relevant entities

---

## Configuration

### Required Settings

```yaml
local_search:
  # Prompts
  prompt: "prompts/local_search_system_prompt.txt"

  # Models
  completion_model_id: default_completion_model
  embedding_model_id: default_embedding_model

  # Token Budget
  max_context_tokens: 12000
  text_unit_prop: 0.5
  community_prop: 0.5

  # Retrieval
  top_k_entities: 10
  top_k_relationships: 10

  # Conversation
  conversation_history_max_turns: 5

  # LLM Parameters
  temperature: 0
  top_p: 1
  n: 1
  max_gen_tokens: null
```

### Data Requirements

**Parquet Files**:
- ✅ `entities.parquet` - Graph nodes with descriptions
- ✅ `relationships.parquet` - Graph edges with descriptions
- ✅ `communities.parquet` - Community structure
- ✅ `community_reports.parquet` - Community summaries
- ✅ `text_units.parquet` - Source text chunks
- ⚠️ `covariates.parquet` - **Optional** claims/facts (requires `extract_claims.enabled = true`)

**Note on Claims**: Claims are **disabled by default**. See `claims_covariates_guide.md` for details on when and how to enable.

**Vector Stores**:
- ✅ `entity_description` embeddings - For entity mapping
- ✅ `text_unit_text` embeddings - For text unit retrieval

---

## Use Cases

### Ideal For:
✅ **Entity-focused questions**
- "How is Company X related to Person Y?"
- "What are the key relationships of Entity Z?"
- "Describe the connections between A and B"

✅ **Specific information retrieval**
- "What does the data say about Topic X?"
- "Find information related to Concept Y"
- "Explain the role of Person A"

✅ **Neighborhood exploration**
- "What entities are connected to X?"
- "Show me the local context around Y"
- "Describe the community containing Z"

### Not Ideal For:
❌ **Broad, abstract questions**
- "What are the main themes?" (use global search)
- "Summarize the entire dataset" (use global search)
- "What are the high-level patterns?" (use global search)

❌ **Multi-faceted complex questions**
- "Compare multiple perspectives on X" (use DRIFT search)
- "Analyze Y from different angles" (use DRIFT search)

❌ **Simple factoid questions**
- "What does the document say about X?" (use basic search)
- "Find mentions of Y" (use basic search)

---

## Comparison with Other Search Methods

| Aspect | Local Search | Global Search | DRIFT Search | Basic Search |
|--------|--------------|---------------|--------------|--------------|
| **Scope** | Entity neighborhoods | Entire graph | Iterative exploration | Text chunks |
| **LLM Calls** | 1 | Many (MAP) + 1 (REDUCE) | Many (primer + local + reduce) | 1 |
| **Speed** | Medium | Slow | Slowest | Fastest |
| **Cost** | Medium | High | Highest | Low |
| **Context Sources** | 5 (multi-source) | 1 (community reports) | Combined | 1 (text units) |
| **Graph Usage** | Entity relationships | Community hierarchy | Both | None |
| **Best For** | Specific entities | Broad questions | Complex analysis | Simple lookup |

---

## Optimization Tips

### 1. Adjust Token Budgets
```yaml
# More text, less community
text_unit_prop: 0.7
community_prop: 0.3

# More community, less text
text_unit_prop: 0.3
community_prop: 0.7

# Increase total budget for longer context
max_context_tokens: 16000
```

### 2. Tune Retrieval Parameters
```yaml
# More entities for broader context
top_k_entities: 15

# More relationships for richer graph
top_k_relationships: 20

# Less aggressive expansion
top_k_entities: 5
top_k_relationships: 5
```

### 3. Model Selection
```yaml
# Cost-optimized (Claude Haiku)
completion_model_id: haiku_completion

# Quality-optimized (Claude 3.5 Sonnet)
completion_model_id: sonnet_completion

# Speed-optimized (GPT-4o)
completion_model_id: gpt4o_completion
```

---

## Troubleshooting

### Problem: Context too large (exceeds token limit)

**Symptoms**: Warning about context truncation

**Solutions**:
1. Reduce `max_context_tokens`
2. Lower `top_k_entities` or `top_k_relationships`
3. Adjust `text_unit_prop` and `community_prop`

---

### Problem: Poor quality answers

**Symptoms**: Answers lack detail or miss key information

**Solutions**:
1. Increase `top_k_entities` for broader coverage
2. Increase `text_unit_prop` for more source text
3. Check if entities are being mapped correctly (vector search quality)
4. Verify entity descriptions are informative

---

### Problem: Slow performance

**Symptoms**: Queries take > 10 seconds

**Solutions**:
1. Use faster LLM model (GPT-4o vs GPT-4-turbo)
2. Reduce context size (`max_context_tokens`)
3. Lower retrieval parameters (`top_k_*`)
4. Check vector store performance

---

### Problem: High cost

**Symptoms**: Expensive LLM bills

**Solutions**:
1. Switch to cheaper model (Claude 3 Haiku, GPT-3.5)
2. Reduce `max_context_tokens` (less input tokens)
3. Use basic search for simple queries
4. Consider local embeddings (SentenceTransformer)

---

## Code References

### Key Files
- **Search Engine**: `packages/graphrag/graphrag/query/structured_search/local_search/search.py:47-144`
- **Context Builder**: `packages/graphrag/graphrag/query/structured_search/local_search/mixed_context.py:52-282`
- **Entity Extraction**: `packages/graphrag/graphrag/query/context_builder/entity_extraction.py:41-126`
- **Relationship Retrieval**: `packages/graphrag/graphrag/query/input/retrieval/relationships.py`
- **Text Unit Retrieval**: `packages/graphrag/graphrag/query/input/retrieval/text_units.py`
- **Configuration**: `packages/graphrag/graphrag/config/models/local_search_config.py:11-49`

### Key Classes
- `LocalSearch`: Main search orchestrator
- `LocalSearchMixedContext`: Context builder
- `Entity`, `Relationship`, `Community`, `TextUnit`, `Covariate`: Data models
- `SearchResult`: Result container

---

## Viewing the Diagram

### Option 1: PlantUML Online
1. Go to https://www.plantuml.com/plantuml/uml/
2. Paste contents of `local_search_activity.puml`
3. Click "Submit"

### Option 2: VS Code Extension
1. Install "PlantUML" extension
2. Open `local_search_activity.puml`
3. Press `Alt+D` to preview

### Option 3: Command Line
```bash
# Install PlantUML
brew install plantuml  # macOS
apt-get install plantuml  # Ubuntu

# Generate PNG
plantuml local_search_activity.puml

# Generate SVG
plantuml -tsvg local_search_activity.puml
```

---

## Related Documentation

- **Query Analysis**: See `query_analysis.md` for comprehensive overview of all search methods
- **Sequence Diagram**: See `query_sequence.puml` for global search sequence diagram
- **GraphRAG Docs**: https://microsoft.github.io/graphrag/

---

**Last Updated**: 2026-01-30
**Diagram Version**: 1.0
**Status**: Complete ✅
