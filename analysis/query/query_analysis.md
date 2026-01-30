# GraphRAG Query Operation - Comprehensive Analysis

## Overview

The GraphRAG query system provides four distinct search methods to retrieve information from the indexed knowledge graph. Each method balances between search scope, computational cost, and answer quality.

---

## 1. Entry Point & CLI

### CLI Command
```bash
graphrag query [QUERY] [OPTIONS]
```

### Implementation
- **CLI Entry**: `packages/graphrag/graphrag/cli/main.py:364-481`
- **Command Handler**: `packages/graphrag/graphrag/cli/query.py`
- **API Entry**: `packages/graphrag/graphrag/api/query.py`

### Key Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `query` | (required) | The query string to execute |
| `--root, -r` | `.` | Project root directory |
| `--method, -m` | `global` | Search method (global/local/drift/basic) |
| `--verbose, -v` | `false` | Enable verbose logging |
| `--data, -d` | `None` | Index output directory (parquet files location) |
| `--community-level` | `2` | Leiden hierarchy level for community selection |
| `--dynamic-community-selection` | `false` | Enable LLM-based dynamic community filtering |
| `--response-type` | `"Multiple Paragraphs"` | Desired response format |
| `--streaming/--no-streaming` | `false` | Stream response token-by-token |

### Execution Flow
1. Parse CLI arguments
2. Load configuration from root directory (`settings.yaml`)
3. Resolve output files from data directory
4. Dispatch to appropriate search method handler
5. Initialize search engine via factory
6. Execute search and stream results
7. Display final answer and statistics

---

## 2. Search Methods

### SearchMethod Enum
**Location**: `packages/graphrag/graphrag/config/enums.py:31-37`

GraphRAG supports **four query methods**:

### 2.1 GLOBAL SEARCH (Default)
**CLI Handler**: `cli/query.py:25-109`
**Search Engine**: `query/structured_search/global_search/search.py`

**Overview**: Hierarchical search across entire knowledge graph using community structure

**Process**:
1. **MAP Phase**: Query all relevant community reports in parallel
   - Each report receives the query independently
   - LLM generates scored key points from each community
   - Returns structured JSON with importance ratings (0-100)
2. **REDUCE Phase**: Aggregate map results into final answer
   - Combines all key points from map phase
   - LLM synthesizes coherent final response

**Data Required**:
- `entities.parquet`
- `communities.parquet`
- `community_reports.parquet`
- `entity_description` embeddings (if dynamic selection enabled)

**Use Case**: Best for broad questions requiring global context (e.g., "What are the main themes in this dataset?")

**Configuration**: `GlobalSearchConfig`

### 2.2 LOCAL SEARCH
**CLI Handler**: `cli/query.py:112-206`
**Search Engine**: `query/structured_search/local_search/search.py`

**Overview**: Focused search on local entity neighborhoods with mixed context

**Process**:
1. **Entity Mapping**: Vector search to find relevant entities
2. **Neighborhood Traversal**: Expand to connected entities via relationships
3. **Context Building**: Gather entities, relationships, communities, text units, claims
4. **Answer Generation**: Single LLM call with rich context

**Data Required**:
- `entities.parquet`
- `relationships.parquet`
- `communities.parquet`
- `community_reports.parquet`
- `text_units.parquet`
- `covariates.parquet` (optional)
- `entity_description` embeddings
- `text_unit_text` embeddings

**Use Case**: Best for specific questions about particular entities or relationships (e.g., "How is Company X related to Person Y?")

**Configuration**: `LocalSearchConfig`

### 2.3 DRIFT SEARCH
**CLI Handler**: `cli/query.py:209-298`
**Search Engine**: `query/structured_search/drift_search/search.py`

**Overview**: Combines global and local search with iterative refinement

**Process**:
1. **PRIMER**: Global map phase generates follow-up questions
2. **FOLLOW-UP ITERATION**: Run local searches for each generated question
3. **REDUCE**: Synthesize all local answers into final response

**Data Required**:
- All data from both global and local search
- `entities.parquet`, `relationships.parquet`, `communities.parquet`
- `community_reports.parquet`, `text_units.parquet`
- Both embedding types

**Use Case**: Best for complex multi-faceted questions requiring both breadth and depth

**Configuration**: `DRIFTSearchConfig`

### 2.4 BASIC SEARCH
**CLI Handler**: `cli/query.py:301-370`
**Search Engine**: `query/structured_search/basic_search/search.py`

**Overview**: Simple vector search over raw text chunks (traditional RAG)

**Process**:
1. **Vector Search**: Find most similar text units to query
2. **Context Assembly**: Collect top-k text units within token budget
3. **Answer Generation**: Single LLM call with text unit context

**Data Required**:
- `text_units.parquet`
- `text_unit_text` embeddings

**Use Case**: Best for simple factoid questions or when you don't need graph structure (e.g., "What does the document say about topic X?")

**Configuration**: `BasicSearchConfig`

---

## 3. Query Processing Pipeline

### High-Level Pipeline Architecture

```
User Query
    ↓
[Configuration Loading] → Load settings.yaml + CLI overrides
    ↓
[Data Loading] → Read parquet files from output/
    ↓
[Embedding Loading] → Load vector stores (entity, text unit)
    ↓
[Search Engine Factory] → Create search engine with context builders
    ↓
[Context Building] → Retrieve relevant context from indexes
    ↓
[LLM Generation] → Generate answer with streaming support
    ↓
[Result Formatting] → Format and display answer
```

### Stage 1: Configuration & Data Loading
**Location**: `cli/query.py` and `api/query.py`

**Key Functions**:
- `load_config(root)`: Load GraphRagConfig from settings.yaml
- `_resolve_output_files(config, data)`: Read parquet files
  ```python
  entities = pd.read_parquet(f"{data_dir}/entities.parquet")
  relationships = pd.read_parquet(f"{data_dir}/relationships.parquet")
  communities = pd.read_parquet(f"{data_dir}/communities.parquet")
  community_reports = pd.read_parquet(f"{data_dir}/community_reports.parquet")
  text_units = pd.read_parquet(f"{data_dir}/text_units.parquet")
  covariates = pd.read_parquet(f"{data_dir}/covariates.parquet")  # optional
  ```

### Stage 2: Search Engine Initialization
**Location**: `query/factory.py:37-276`

**Factory Functions**:
- `get_global_search_engine()` → Returns `GlobalSearch`
- `get_local_search_engine()` → Returns `LocalSearch`
- `get_drift_search_engine()` → Returns `DRIFTSearch`
- `get_basic_search_engine()` → Returns `BasicSearch`

**Common Initialization Pattern**:
```python
1. Create completion model: create_completion(model_config)
2. Create embedding model: create_embedding(embedding_config)
3. Load prompts from config
4. Initialize context builder with data models
5. Configure tokenizer from model
6. Return search engine instance
```

### Stage 3: Context Building
**Location**: `query/context_builder/` and `query/input/retrieval/`

**Context Builder Types**:
- `GlobalCommunityContext`: Selects relevant community reports
- `LocalSearchMixedContext`: Combines entities, relationships, communities, text units, claims
- `DRIFTSearchContextBuilder`: Combines global and local contexts
- `BasicSearchContext`: Vector search over text units

**Context Building Output**:
```python
@dataclass
class ContextBuilderResult:
    context_chunks: str | list[str]  # Formatted text for LLM
    context_records: dict | list[dict]  # Raw records for tracing
    llm_calls: int  # LLM calls during context building
    prompt_tokens: int  # Tokens used in prompts
    output_tokens: int  # Tokens generated
```

### Stage 4: Answer Generation
**Location**: `query/structured_search/<method>/search.py`

**Generation Process**:
1. Format context into system prompt
2. Build completion messages:
   ```python
   messages = CompletionMessagesBuilder()
       .add_system_message(system_prompt)
       .add_user_message(query)
       .build()
   ```
3. Call LLM with streaming or non-streaming mode:
   ```python
   if stream:
       async for chunk in completion.completion_async(messages, stream=True):
           yield chunk.choices[0].delta.content
   else:
       response = await completion.completion_async(messages)
       return response.content
   ```
4. Track token usage and timing

---

## 4. Context Retrieval Mechanisms

### 4.1 Entity Extraction & Mapping
**Location**: `query/context_builder/entity_extraction.py:41-96`

**Function**: `map_query_to_entities()`

**Process**:
1. **Embed Query**: Convert query text to vector
   ```python
   query_embedding = text_embedding_vectorstore.embed_query(query)
   ```
2. **Vector Search**: Find similar entity descriptions
   ```python
   results = vectorstore.similarity_search_by_text(
       text=query,
       k=top_k * oversample_scaler,  # Oversample by 2x
       text_embedder=embedding_fn
   )
   ```
3. **Filter & Rank**: Remove excluded entities, sort by rank
4. **Return Top-K**: Return most relevant entities

**Parameters**:
- `k`: Number of entities to retrieve (default: 10)
- `oversample_scaler`: Oversampling factor (default: 2)
- `embedding_vectorstore_key`: Search by ID or TITLE

### 4.2 Entity Neighborhood Traversal
**Location**: `query/context_builder/entity_extraction.py:99-126`

**Function**: `find_nearest_neighbors_by_entity_rank()`

**Process**:
1. Find relationships where entity is source or target
2. Extract connected entities
3. Rank by entity importance (rank attribute)
4. Return top-k neighbors

### 4.3 Relationship Retrieval
**Location**: `query/input/retrieval/relationships.py`

**Retrieval Strategies**:
- `get_in_network_relationships()`: Relationships between entity and neighbors
- `get_out_network_relationships()`: Relationships originating from entity
- `get_candidate_relationships()`: All relationships connected to entity

**Filtering**:
- By source/target entities
- By graph paths
- By count limits
- By token budgets

### 4.4 Text Unit Retrieval
**Location**: `query/input/retrieval/text_units.py`

**Process**:
1. Filter text units by associated entities
2. Apply token budget constraints
3. Sort by relevance:
   - Entity rank (primary)
   - Text order (secondary)
4. Return top units within budget

### 4.5 Community Report Selection
**Location**: `query/input/retrieval/community_reports.py`

**Selection Strategies**:

#### A. Static Selection (Default)
- Select all reports at specified community level
- No LLM calls required
- Fast but may include irrelevant reports

#### B. Dynamic Selection (Optional)
**Location**: `query/context_builder/dynamic_community_selection.py`

**Process**:
1. Batch community reports
2. For each batch:
   - Send to LLM with relevance rating prompt
   - LLM rates each report 0-10
   - Filter by rating threshold (default: 7)
3. If children are relevant, optionally keep parent communities
4. Return filtered reports

**Configuration**:
- `dynamic_search_threshold`: Rating threshold (default: 7)
- `dynamic_search_keep_parent`: Include parents (default: true)
- `dynamic_search_num_repeats`: Rating repetitions (default: 1)

### 4.6 Covariate (Claim) Retrieval
**Location**: `query/input/retrieval/covariates.py`

**What Are Claims?**
Claims (covariates) are structured factual assertions extracted from documents about specific entities. They provide:
- Verified facts (status: TRUE/FALSE/SUSPECTED)
- Temporal context (start/end dates)
- Evidence traceability (source text)
- Categorization (claim types)

**Retrieval Process**:
1. Filter claims by selected entities (subject_id match)
2. Extract claim attributes (type, status, description, dates, source)
3. Apply remaining token budget (after entities, relationships, communities, text units)
4. Format as CSV: `id|entity|type|status|description|start_date|end_date|source_text`

**Configuration**:
- **Enabled**: `extract_claims.enabled` in settings.yaml (default: false)
- **Description**: What types of claims to extract
- **Usage**: Local Search and DRIFT Search only (not Global or Basic)

**Example Claim**:
```
Subject: COMPANY A
Object: GOVERNMENT AGENCY B
Type: ANTI-COMPETITIVE PRACTICES
Status: TRUE
Dates: 2022-01-10 to 2022-01-10
Description: Company A was fined for bid rigging in public tenders...
Source: "According to an article on 2022/01/10, Company A was fined..."
```

**See**: `claims_covariates_guide.md` for comprehensive documentation

---

## 5. Answer Generation Strategies

### 5.1 Global Search: MAP-REDUCE Pattern

#### MAP Phase
**Location**: `structured_search/global_search/search.py:216-274`

```python
async def _map_response_single_batch(
    context_data: str,  # Community reports as CSV
    query: str,
    max_length: int,
    **llm_kwargs
) -> SearchResult
```

**Process**:
1. Format with `MAP_SYSTEM_PROMPT`
2. Call LLM for each batch in parallel (with semaphore limit)
3. Parse JSON response with importance scores
4. Return structured `SearchResult`

**Prompt**: `prompts/query/global_search_map_system_prompt.py`
**Output Format**:
```json
{
  "points": [
    {
      "description": "Key point text",
      "score": 75  // Importance 0-100
    },
    ...
  ]
}
```

#### REDUCE Phase
**Location**: `structured_search/global_search/search.py:298+`

```python
async def _reduce_response(
    map_responses: list[SearchResult],
    query: str,
    **llm_kwargs
) -> SearchResult
```

**Process**:
1. Aggregate all map responses
2. Format with `REDUCE_SYSTEM_PROMPT`
3. Call LLM to synthesize final answer
4. Return final `SearchResult`

**Prompt**: `prompts/query/global_search_reduce_system_prompt.py`

### 5.2 Local Search: Single-Phase Generation
**Location**: `structured_search/local_search/search.py:56-144`

```python
async def search(
    query: str,
    conversation_history: ConversationHistory | None = None
) -> SearchResult
```

**Process**:
1. Build mixed context:
   - Conversation history (if present)
   - Selected entities (vector search)
   - Related relationships
   - Community reports
   - Text units
   - Covariates/claims
2. Format with `LOCAL_SEARCH_SYSTEM_PROMPT`
3. Add user message with query
4. Stream response from LLM
5. Track tokens and timing
6. Return `SearchResult`

**Context Composition**:
```
# Entities
id,entity,description
...

# Relationships
id,source,target,description
...

# Reports
id,community,summary
...

# Sources
id,text
...

# Claims
id,subject,object,type,description
...
```

**Prompt**: `prompts/query/local_search_system_prompt.py`

### 5.3 DRIFT Search: Iterative Refinement
**Location**: `structured_search/drift_search/search.py:37-180`

**Three-Stage Process**:

#### Stage 1: PRIMER (Global Context)
**Component**: `DRIFTPrimer` (`drift_search/primer.py`)

```python
async def __call__(self, query: str) -> list[str]
```

**Process**:
1. Run global MAP phase to get community context
2. Format MAP results as input
3. Call LLM to generate k follow-up questions
4. Return list of generated questions

**Purpose**: Generate diverse follow-up questions for depth

#### Stage 2: FOLLOW-UP ITERATION
**Process**:
```python
for question in [original_query] + follow_up_questions:
    result = await local_search_engine.search(question)
    local_results.append(result)
```

**Purpose**: Answer each question with focused local search

#### Stage 3: REDUCE
**Process**:
1. Gather all local search results
2. Format with `DRIFT_REDUCE_PROMPT`
3. Call LLM to synthesize final answer
4. Return comprehensive response

**Use Case**: Complex questions requiring exploration

### 5.4 Basic Search: Direct RAG
**Location**: `structured_search/basic_search/search.py:57-144`

```python
async def search(query: str) -> SearchResult
```

**Process**:
1. Vector search on text_unit_text embeddings
2. Retrieve top-k text units
3. Filter by max_context_tokens
4. Format with `BASIC_SEARCH_SYSTEM_PROMPT`
5. Call LLM with text unit context
6. Return response

**Simplest Method**: No graph traversal, no community structure

**Prompt**: `prompts/query/basic_search_system_prompt.py`

---

## 6. Key Components & Classes

### Core Search Classes

| Class | Location | Purpose |
|-------|----------|---------|
| `BaseSearch` | `structured_search/base.py:46-64` | Abstract base for all search engines |
| `GlobalSearch` | `global_search/search.py:55-` | MAP-REDUCE orchestration |
| `LocalSearch` | `local_search/search.py:47-` | Local neighborhood search |
| `DRIFTSearch` | `drift_search/search.py:37-` | Iterative refinement search |
| `BasicSearch` | `basic_search/search.py:47-` | Simple vector RAG |

### Context Builder Classes

| Class | Location | Purpose |
|-------|----------|---------|
| `GlobalCommunityContext` | `global_search/community_context.py:27-` | Community report selection |
| `LocalSearchMixedContext` | `local_search/mixed_context.py:52-` | Multi-source context assembly |
| `DRIFTSearchContextBuilder` | `drift_search/drift_context.py` | Combined global/local context |
| `BasicSearchContext` | `basic_search/basic_context.py:27-` | Text unit vector search |
| `DynamicCommunitySelection` | `context_builder/dynamic_community_selection.py` | LLM-based community filtering |

### Data Model Classes

| Model | Location | Purpose |
|-------|----------|---------|
| `Entity` | `data_model/entity.py` | Graph nodes with descriptions |
| `Relationship` | `data_model/relationship.py` | Graph edges with descriptions |
| `CommunityReport` | `data_model/community_report.py` | Hierarchical community summaries |
| `Community` | `data_model/community.py` | Community structure |
| `TextUnit` | `data_model/text_unit.py` | Source text chunks |
| `Covariate` | `data_model/covariate.py` | Claims/facts |

### Result Classes

| Class | Location | Purpose |
|-------|----------|---------|
| `SearchResult` | `structured_search/base.py:29-44` | Encapsulates search output |
| `ContextBuilderResult` | `context_builder/builders.py` | Context construction output |

**SearchResult Structure**:
```python
@dataclass
class SearchResult:
    response: str | dict | list  # Generated answer
    context_data: str | list | dict  # Raw context records
    context_text: str | list | dict  # Formatted context text
    completion_time: float
    llm_calls: int
    prompt_tokens: int
    output_tokens: int
    llm_calls_categories: dict[str, int]  # Per-phase breakdown
    prompt_tokens_categories: dict[str, int]
    output_tokens_categories: dict[str, int]
```

---

## 7. Configuration Options

### 7.1 Global Search Configuration
**File**: `config/models/global_search_config.py:11-67`

| Parameter | Default | Purpose |
|-----------|---------|---------|
| **Prompts** | | |
| `map_prompt` | defaults | MAP phase system prompt |
| `reduce_prompt` | defaults | REDUCE phase system prompt |
| `knowledge_prompt` | defaults | General knowledge prompt |
| **Models** | | |
| `completion_model_id` | configured | LLM for generation |
| **Context Limits** | | |
| `max_context_tokens` | 8000 | Max tokens for full context |
| `data_max_tokens` | 8000 | Max data tokens in MAP phase |
| **Response Formatting** | | |
| `map_max_length` | 1000 | Max response length in MAP (words) |
| `reduce_max_length` | 2000 | Max response length in REDUCE (words) |
| **Dynamic Selection** | | |
| `dynamic_search_threshold` | 7 | Rating threshold (0-10) |
| `dynamic_search_keep_parent` | true | Include parent communities |
| `dynamic_search_num_repeats` | 1 | Number of rating passes |
| `dynamic_search_use_summary` | true | Use summary vs full content |
| `dynamic_search_max_level` | 10 | Max hierarchy level |
| **Parallelism** | | |
| `concurrent_coroutines` | 32 | Max parallel MAP calls |

### 7.2 Local Search Configuration
**File**: `config/models/local_search_config.py:11-49`

| Parameter | Default | Purpose |
|-----------|---------|---------|
| **Prompts & Models** | | |
| `prompt` | configured | System prompt |
| `completion_model_id` | configured | LLM model |
| `embedding_model_id` | configured | Embedding model |
| **Context Budget** | | |
| `text_unit_prop` | 0.5 | Proportion for text units |
| `community_prop` | 0.5 | Proportion for communities |
| `max_context_tokens` | 12000 | Total context limit |
| **Retrieval** | | |
| `top_k_entities` | 10 | Entities to retrieve |
| `top_k_relationships` | 10 | Relationships to retrieve |
| **Conversation** | | |
| `conversation_history_max_turns` | 5 | Max conversation turns |

### 7.3 DRIFT Search Configuration
**File**: `config/models/drift_search_config.py:11-123`

| Parameter | Default | Purpose |
|-----------|---------|---------|
| **Prompts & Models** | | |
| `prompt` | configured | Local search prompt |
| `reduce_prompt` | configured | Reduce phase prompt |
| `completion_model_id` | configured | LLM model |
| `embedding_model_id` | configured | Embedding model |
| **DRIFT Parameters** | | |
| `n_depth` | 2 | Number of iterations |
| `drift_k_followups` | 3 | Follow-up questions per iteration |
| `primer_folds` | 10 | Data folds for primer |
| **Token Limits** | | |
| `data_max_tokens` | 8000 | Max data tokens |
| `reduce_max_tokens` | 500 | Max reduce tokens |
| **Local Search Settings** | | |
| `local_search_text_unit_prop` | 0.5 | Text unit proportion |
| `local_search_community_prop` | 0.5 | Community proportion |
| `local_search_top_k_mapped_entities` | 10 | Entities to map |
| `local_search_top_k_relationships` | 10 | Relationships |
| `local_search_max_data_tokens` | 8000 | Context limit |

### 7.4 Basic Search Configuration
**File**: `config/models/basic_search_config.py:11-33`

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `prompt` | configured | System prompt |
| `completion_model_id` | configured | LLM model |
| `embedding_model_id` | configured | Embedding model |
| `k` | 10 | Number of text units |
| `max_context_tokens` | 4000 | Context token limit |

---

## 8. Token Budget Management

**Token Counting**:
- Uses model tokenizer: `model.tokenizer.encode(text)`
- Tracks tokens for all context types
- Enforces hard limits to prevent LLM context overflow

**Budget Allocation Pattern** (Local Search):
```python
text_unit_budget = max_context_tokens * text_unit_prop
community_budget = max_context_tokens * community_prop

for text_unit in sorted_text_units:
    tokens = count_tokens(text_unit.text)
    if current_tokens + tokens > text_unit_budget:
        break
    selected_units.append(text_unit)
    current_tokens += tokens
```

---

## 9. Callback System

**Location**: `callbacks/query_callbacks.py`

**QueryCallbacks Interface**:
```python
class QueryCallbacks(BaseLLMCallback):
    def on_context(context: Any) -> None
        # Called when context is constructed

    def on_map_response_start(map_response_contexts: list[str]) -> None
        # Before MAP phase

    def on_map_response_end(map_response_outputs: list[SearchResult]) -> None
        # After MAP phase

    def on_reduce_response_start(reduce_response_context: str) -> None
        # Before REDUCE phase

    def on_reduce_response_end(reduce_response_output: str) -> None
        # After REDUCE phase

    def on_llm_new_token(token) -> None
        # Each streamed token
```

**Usage**: Allows custom tracking, logging, and UI updates

---

## 10. Streaming Implementation

**Async Generator Pattern**:
```python
async def stream_search(query: str) -> AsyncGenerator[str, None]:
    # Build context
    context = await context_builder.build_context(query)

    # Stream LLM response
    async for chunk in llm.completion_async(messages, stream=True):
        token = chunk.choices[0].delta.content
        if token:
            yield token

    # Finalize stats
    result = SearchResult(...)
```

**CLI Streaming** (`cli/query.py`):
```python
if streaming:
    async for chunk in api.global_search_streaming(...):
        full_response += chunk
        print(chunk, end="", flush=True)
    print()  # Newline
```

**Benefits**:
- Real-time UI updates
- Progressive rendering
- Better user experience for long responses

---

## 11. Key Insights

1. **Four Search Strategies**: Each optimized for different query types
2. **Hierarchical Knowledge**: Community structure enables multi-scale reasoning
3. **Hybrid Retrieval**: Combines vector search, graph traversal, and structured indexes
4. **Token-Aware**: All operations respect LLM context limits
5. **Streaming Support**: Real-time response generation
6. **Callback Hooks**: Extensible for monitoring and UI integration
7. **Dynamic Selection**: Optional LLM-based relevance filtering
8. **Conversation History**: Local search supports multi-turn conversations
9. **Parallel Execution**: MAP phase uses asyncio for performance
10. **Comprehensive Tracking**: Full token usage and timing statistics

---

## 12. Comparison of Search Methods

| Aspect | Global | Local | DRIFT | Basic |
|--------|--------|-------|-------|-------|
| **Scope** | Entire graph | Entity neighborhoods | Iterative refinement | Text chunks |
| **LLM Calls** | Many (MAP) + 1 (REDUCE) | 1 | Many (primer + local + reduce) | 1 |
| **Speed** | Slow (parallel MAP) | Medium | Slowest | Fastest |
| **Cost** | High (many tokens) | Medium | Highest | Low |
| **Quality** | Best for broad | Best for specific | Best for complex | Good for simple |
| **Context** | Community reports | Multi-source | Combined | Text units only |
| **Graph Use** | Community hierarchy | Entity relationships | Both | None |

---

## References

### Key Files
- **CLI**: `cli/main.py:364-481`, `cli/query.py`
- **API**: `api/query.py`
- **Factory**: `query/factory.py`
- **Search Engines**: `query/structured_search/<method>/search.py`
- **Context Builders**: `query/context_builder/`, `query/input/retrieval/`
- **Prompts**: `prompts/query/*_system_prompt.py`
- **Config**: `config/models/*_search_config.py`
- **Data Models**: `data_model/*.py`

### External Dependencies
- **LLM**: Via `graphrag-llm` package (completion, embedding)
- **Vectors**: Via `graphrag-vectors` package (similarity search)
- **Storage**: Via `graphrag-storage` package (read parquet files)
- **Data**: Pandas DataFrames, NetworkX graphs
