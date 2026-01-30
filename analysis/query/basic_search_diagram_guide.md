# Basic Search Activity Diagram - Guide

**File**: `basic_search_activity.puml`
**Type**: PlantUML Activity Diagram
**Purpose**: Visual representation of GraphRAG Basic Search complete flow (Traditional RAG)

---

## Overview

The Basic Search activity diagram shows the complete end-to-end flow of a basic search query in GraphRAG, representing traditional Retrieval-Augmented Generation (RAG) without graph structure utilization.

## Diagram Structure

The diagram is organized into **4 main phases** with **5 swim lanes**:

### Swim Lanes (by color)

1. **CLI Layer** (AntiqueWhite) - User interaction and command handling
2. **Data Loading** (LightBlue) - Reading parquet files and embeddings
3. **Initialization** (LightGreen) - Search engine setup
4. **Query Execution** (LightYellow) - Search orchestration
5. **Phase-specific lanes**:
   - Vector Search (LightCoral)
   - Token Budget Filtering (LightSteelBlue)
   - Answer Generation (Lavender)
   - Result Assembly (PaleGreen)

### Main Phases

#### Phase 1: Vector Search

**Purpose**: Find text chunks most similar to the query

**Steps**:
1. Embed query text using embedding model
2. Perform vector similarity search on text_unit_text embeddings
3. Retrieve top-k most similar text units
4. Extract text unit IDs from search results
5. Filter text_units DataFrame to get full content

**Key Parameters**:
- `k`: Number of text units to retrieve (default: 10)
- `embedding_vectorstore_key`: Search key (default: "id")

**Algorithm**:
```python
# 1. Embed query
query_embedding = text_embedder.embedding(input=[query]).first_embedding

# 2. Vector similarity search
related_texts = text_unit_embeddings.similarity_search_by_text(
    text=query,
    text_embedder=lambda t: text_embedder.embedding(input=[t]).first_embedding,
    k=k  # default: 10
)

# 3. Extract IDs
text_unit_ids = {result.document.id for result in related_texts}

# 4. Filter text units
text_units_filtered = [
    {"id": unit.short_id, "text": unit.text}
    for unit in text_units
    if unit.id in text_unit_ids
]

# 5. Create DataFrame
related_text_df = pd.DataFrame(text_units_filtered)
```

**Output**: DataFrame with top-k similar text units
- Columns: `id`, `text`
- Ordered by similarity score (implicit)

**Similarity Metric**: Cosine similarity
- Range: -1 to 1 (typically 0.5-0.95 for relevant results)
- Higher score = more similar

---

#### Phase 2: Token Budget Filtering

**Purpose**: Fit retrieved text units into LLM context window

**Steps**:
1. Initialize token counter
2. Count header tokens ("id|text\n")
3. For each retrieved text unit:
   - Format as CSV row
   - Count tokens
   - If within budget: add to context
   - If exceeds budget: stop
4. Create final filtered DataFrame
5. Convert to CSV format

**Key Parameters**:
- `max_context_tokens`: Token limit (default: 4000)
- `column_delimiter`: CSV separator (default: "|")
- `text_id_col`: ID column name (default: "id")
- `text_col`: Text column name (default: "text")

**Token Budgeting Algorithm**:
```python
current_tokens = 0
text_ids = []

# Count header
header = text_id_col + column_delimiter + text_col + "\n"
current_tokens = len(tokenizer.encode(header))

# Iterate through retrieved text units
for i, row in related_text_df.iterrows():
    # Format as CSV row
    text = row[text_id_col] + column_delimiter + row[text_col] + "\n"
    tokens = len(tokenizer.encode(text))

    # Check budget
    if current_tokens + tokens > max_context_tokens:
        logger.warning(
            f"Reached token limit: {current_tokens + tokens}. "
            f"Reverting to previous context state"
        )
        break

    # Add to context
    current_tokens += tokens
    text_ids.append(i)

# Filter to selected text units
final_text_df = related_text_df[related_text_df.index.isin(text_ids)]
```

**CSV Format**:
```csv
id|text
text_unit_42|This is the first chunk of text that was retrieved from the original document. It contains relevant information about the topic.
text_unit_87|Another text chunk that was deemed similar to the query. This chunk provides additional context and details.
text_unit_103|A third chunk that discusses related concepts and may help answer the user's question.
```

**Output**: CSV-formatted text units within token budget

**Why Lower Budget?**:
- Basic search: 4000 tokens (default)
- Local search: 12000 tokens
- Global search: 8000 tokens per batch

Reasons:
1. Text units can be long (200-1000 tokens each)
2. Simpler queries = shorter context needed
3. Lower cost per query
4. Faster LLM inference

---

#### Phase 3: Answer Generation

**Purpose**: Generate answer using LLM with text unit context

**Steps**:
1. Build system message with BASIC_SEARCH_SYSTEM_PROMPT
2. Include text units context (CSV format)
3. Add response_type parameter
4. Build user message with query
5. Call LLM with streaming enabled
6. Stream tokens back to user
7. Track token usage and timing

**Prompt Structure**:
```python
system_message = BASIC_SEARCH_SYSTEM_PROMPT.format(
    context_data=final_text_csv,  # Text units in CSV format
    response_type=response_type    # e.g., "Multiple Paragraphs"
)

messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": query}
]
```

**Streaming Flow**:
```python
response = ""
async for chunk in llm.completion_async(messages, stream=True):
    token = chunk.choices[0].delta.content or ""
    response += token
    yield token  # Real-time display to user
```

**Key Parameters**:
- `completion_model_id`: LLM model to use
- `response_type`: Desired format (default: "Multiple Paragraphs")
- `temperature`: Randomness (default: 0)
- `max_tokens`: Output limit (from config)

**Token Tracking**:
```python
llm_calls = {"build_context": 0, "response": 1}
prompt_tokens = {"build_context": 0, "response": len(tokenizer.encode(system_message))}
output_tokens = {"build_context": 0, "response": len(tokenizer.encode(response))}
```

**Output**: Generated answer text

---

#### Phase 4: Result Assembly

**Purpose**: Package results for return to user

**SearchResult Structure**:
```python
@dataclass
class SearchResult:
    # Generated answer
    response: str

    # Context used
    context_data: dict                     # {"sources": DataFrame}
    context_text: str                      # CSV-formatted text units

    # Timing
    completion_time: float                 # Duration in seconds

    # Token usage
    llm_calls: int                         # Always 1 for basic search
    prompt_tokens: int                     # Input tokens
    output_tokens: int                     # Generated tokens

    # Per-phase breakdown
    llm_calls_categories: dict[str, int]   # {"build_context": 0, "response": 1}
    prompt_tokens_categories: dict[str, int]
    output_tokens_categories: dict[str, int]
```

**Context Records**:
```python
context_data = {
    "sources": final_text_df  # DataFrame with columns: id, text
}
```

**Example Result**:
```python
SearchResult(
    response="Based on the provided sources, X refers to...",
    context_data={"sources": pd.DataFrame([...])},
    context_text="id|text\nunit_42|...\nunit_87|...",
    completion_time=2.3,
    llm_calls=1,
    prompt_tokens=4523,
    output_tokens=687,
    llm_calls_categories={"build_context": 0, "response": 1},
    prompt_tokens_categories={"build_context": 0, "response": 4523},
    output_tokens_categories={"build_context": 0, "response": 687}
)
```

---

## Key Characteristics

### Traditional RAG Pattern

**What is RAG?**
Retrieval-Augmented Generation combines:
1. **Retrieval**: Find relevant documents/chunks
2. **Augmentation**: Add retrieved content to prompt
3. **Generation**: LLM generates answer with context

**Basic Search Implementation**:
- ✅ Vector search for retrieval
- ✅ Text units as context
- ✅ Single LLM call for generation
- ❌ No graph traversal
- ❌ No entity/relationship awareness
- ❌ No community structure

### Single LLM Call

**Efficiency**:
- Only 1 LLM call per query
- Similar to local search
- Unlike global search (N+1 calls)
- Unlike DRIFT search (many calls)

**Cost**:
- ~4000-5000 prompt tokens
- ~500-1000 output tokens
- Total: ~5000-6000 tokens per query

**Example Cost (GPT-4o)**:
- Input: 4500 tokens × $2.50/1M = $0.011
- Output: 750 tokens × $10/1M = $0.0075
- **Total: ~$0.02 per query**

### No Graph Structure

**Simplicity**:
- ✅ Easiest to implement
- ✅ Minimal data requirements
- ✅ Fastest search method
- ✅ Lowest cost

**Limitations**:
- ❌ No entity relationships
- ❌ No community hierarchy
- ❌ No graph-based reasoning
- ❌ Pure semantic matching only

**When This is OK**:
- Simple factoid questions
- Direct document lookup
- Quote/citation requests
- Cost-sensitive scenarios

**When This is NOT OK**:
- Entity relationship queries
- Broad thematic questions
- Questions requiring connections
- Graph-aware reasoning needed

### Minimal Data Requirements

**Required Files**:
- ✅ `text_units.parquet` - Source text chunks
- ✅ `text_unit_text` embeddings - Vector store

**NOT Required**:
- ❌ `entities.parquet`
- ❌ `relationships.parquet`
- ❌ `communities.parquet`
- ❌ `community_reports.parquet`
- ❌ `entity_description` embeddings

**Storage Footprint**:
- Minimal: Only text units + embeddings
- Typical: 10-50 MB vs 100-500 MB for full graph

---

## Performance Characteristics

### Speed: Fastest (1-3 seconds)

| Operation | Time | Notes |
|-----------|------|-------|
| Vector search | 100-200ms | Fast similarity search |
| Token filtering | 20-50ms | Pure computation |
| LLM generation | 1-2s | Single call |
| **Total** | **1-3 seconds** | Fastest method |

**Factors Affecting Speed**:
- LLM model speed (GPT-4o > GPT-4 Turbo)
- Vector store performance
- Network latency
- Output length

**Speed Comparison**:
| Method | Time | Relative |
|--------|------|----------|
| Basic | 1-3s | 1x (baseline) |
| Local | 2-5s | 1.5-2x slower |
| Global | 5-20s | 5-10x slower |
| DRIFT | 10-30s | 10-15x slower |

### Cost: Lowest

**Token Breakdown**:
```
Prompt Tokens:
- System prompt: ~200 tokens
- Text unit context: ~3000-4000 tokens
- Query: ~50-200 tokens
- Total: ~4000-5000 tokens

Output Tokens:
- Answer: ~500-1000 tokens

Total per query: ~5000-6000 tokens
```

**Cost Examples**:

| Model | Input Price | Output Price | Cost per Query |
|-------|-------------|--------------|----------------|
| GPT-4o | $2.50/1M | $10/1M | $0.018 |
| GPT-4 Turbo | $10/1M | $30/1M | $0.075 |
| Claude 3.5 Sonnet | $3/1M | $15/1M | $0.028 |
| Claude 3 Haiku | $0.25/1M | $1.25/1M | $0.002 |

**Cost Comparison** (GPT-4o):
| Method | Tokens | Cost | Relative |
|--------|--------|------|----------|
| Basic | 5-6K | $0.02 | 1x |
| Local | 12-14K | $0.04 | 2x |
| Global | 100-250K | $0.35 | 17x |
| DRIFT | 200-400K | $0.70 | 35x |

### Quality: Good for Simple Queries

**Strengths**:
- ✅ Direct text retrieval
- ✅ Semantic similarity matching
- ✅ Fast and efficient
- ✅ Simple and predictable
- ✅ Good for factoid questions

**Limitations**:
- ❌ No graph context
- ❌ May miss connections between entities
- ❌ No hierarchical understanding
- ❌ Can't traverse relationships
- ❌ Limited to chunk-level semantics

**Quality by Query Type**:
| Query Type | Basic | Local | Global |
|------------|-------|-------|--------|
| Factoid ("What is X?") | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Entity relationships | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Broad themes | ⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| Direct quotes | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |

---

## Configuration

### Required Settings

```yaml
basic_search:
  # Prompts
  prompt: "prompts/basic_search_system_prompt.txt"

  # Models
  completion_model_id: default_completion_model
  embedding_model_id: default_embedding_model

  # Retrieval
  k: 10                      # Number of text units
  max_context_tokens: 4000   # Token limit

  # Response
  response_type: "Multiple Paragraphs"

  # LLM Parameters
  temperature: 0
  top_p: 1
  n: 1
  max_gen_tokens: null

  # Vector Store
  embedding_vectorstore_key: "id"
```

### Data Requirements

**Parquet Files**:
- ✅ `text_units.parquet` - Source text chunks

**Vector Stores**:
- ✅ `text_unit_text` embeddings - For similarity search

**Minimal Setup**:
- Only 2 files needed
- Fastest to prepare
- Smallest data footprint

---

## Use Cases

### Ideal For:

✅ **Simple factoid questions**
- "What does the document say about X?"
- "Define term Y"
- "What is the value of Z?"

✅ **Direct document lookup**
- "Find mentions of topic A"
- "Quote text about B"
- "What does section C say?"

✅ **Citation and source retrieval**
- "Where does it mention X?"
- "Show me text about Y"
- "Find the original passage"

✅ **Cost-sensitive scenarios**
- Budget-constrained applications
- High-volume queries
- Simple chatbots

✅ **Speed-priority applications**
- Real-time search
- Interactive Q&A
- Quick lookups

### Not Ideal For:

❌ **Entity relationship questions**
- "How is Company X related to Person Y?" (use local search)
- "What connects A and B?" (use local search)

❌ **Broad thematic questions**
- "What are the main themes?" (use global search)
- "Summarize the dataset" (use global search)

❌ **Complex multi-faceted questions**
- "Analyze X from multiple perspectives" (use DRIFT search)
- "Compare different viewpoints" (use DRIFT search)

❌ **Graph-aware queries**
- "Describe the network around X" (use local search)
- "Show entity neighborhoods" (use local search)

---

## Comparison with Other Search Methods

| Aspect | Basic Search | Local Search | Global Search | DRIFT Search |
|--------|--------------|--------------|---------------|--------------|
| **Pattern** | Simple RAG | Graph-aware RAG | MAP-REDUCE | Iterative |
| **LLM Calls** | 1 | 1 | Many (N+1) | Many |
| **Data Source** | Text units only | Multi-source | Community reports | Combined |
| **Graph Usage** | None | Entity relationships | Community hierarchy | Both |
| **Speed** | ⭐⭐⭐⭐⭐ Fastest | ⭐⭐⭐⭐ Fast | ⭐⭐ Slow | ⭐ Slowest |
| **Cost** | ⭐⭐⭐⭐⭐ Lowest | ⭐⭐⭐⭐ Low | ⭐⭐ Medium | ⭐ High |
| **Setup** | ⭐⭐⭐⭐⭐ Minimal | ⭐⭐⭐ Moderate | ⭐⭐⭐ Moderate | ⭐⭐ Complex |
| **Data Required** | 2 files | 8+ files | 3+ files | 8+ files |
| **Quality (simple)** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **Quality (complex)** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## Optimization Tips

### 1. Adjust Retrieval Count

**More Text Units** (Better coverage):
```yaml
k: 20  # Retrieve more chunks
max_context_tokens: 6000  # Increase budget
# Pro: More context, better answers
# Con: Higher cost, slower
```

**Fewer Text Units** (Faster, cheaper):
```yaml
k: 5   # Fewer chunks
max_context_tokens: 2000  # Lower budget
# Pro: Faster, cheaper
# Con: May miss relevant info
```

**Recommendation**: Start with default (k=10), adjust based on results

### 2. Token Budget Tuning

**Larger Context** (More detailed):
```yaml
max_context_tokens: 8000
# Pro: More text units included
# Con: Higher cost, slower generation
```

**Smaller Context** (Faster, cheaper):
```yaml
max_context_tokens: 2000
# Pro: Lower cost, faster
# Con: Less context, potentially lower quality
```

### 3. Model Selection

**Cost-Optimized** (Claude Haiku):
```yaml
completion_model_id: haiku_completion
# Cost: $0.002 per query (97% savings)
# Speed: 3x faster
# Quality: Good for simple questions
```

**Quality-Optimized** (Claude 3.5 Sonnet / GPT-4):
```yaml
completion_model_id: sonnet_completion
# Cost: Higher (~$0.03 per query)
# Speed: Slower
# Quality: Best for nuanced questions
```

**Balanced** (GPT-4o):
```yaml
completion_model_id: gpt4o_completion
# Cost: $0.018 per query
# Speed: Fast
# Quality: High
```

### 4. Embedding Model Choice

**API-based** (OpenAI):
```yaml
embedding_model_id: openai_embedding
# Cost: ~$0.0001 per query
# Quality: High
# Speed: Network latency
```

**Local** (SentenceTransformer):
```yaml
embedding_model_id: sentence_transformer_embedding
model: BAAI/bge-large-en-v1.5
# Cost: Free (runs locally)
# Quality: Very good
# Speed: Fast (no network)
# Privacy: Complete data privacy
```

### 5. Response Format

**Concise Answers**:
```yaml
response_type: "Single Sentence"
# Pro: Faster, cheaper
# Con: Less detail
```

**Detailed Answers**:
```yaml
response_type: "Multiple Paragraphs"
# Pro: More comprehensive
# Con: Higher output tokens
```

---

## Troubleshooting

### Problem: Poor quality answers

**Symptoms**: Answers miss key information, off-topic, or too vague

**Solutions**:
1. **Increase retrieval count**:
   ```yaml
   k: 20  # More text units
   ```

2. **Increase context budget**:
   ```yaml
   max_context_tokens: 6000
   ```

3. **Check text unit size**:
   - Verify text units are appropriate size (200-1000 tokens)
   - Too small: Lack context
   - Too large: Few units fit in budget

4. **Improve embeddings**:
   - Use better embedding model
   - Consider local SentenceTransformer models
   - Check embedding quality

5. **Use alternative search method**:
   - If question involves entities: use local search
   - If question is broad: use global search

---

### Problem: High cost per query

**Symptoms**: Unexpected costs despite using basic search

**Solutions**:
1. **Reduce context size**:
   ```yaml
   max_context_tokens: 2000
   k: 5
   ```

2. **Use cheaper model**:
   ```yaml
   completion_model_id: haiku_completion
   ```

3. **Shorten responses**:
   ```yaml
   response_type: "Single Paragraph"
   max_gen_tokens: 500
   ```

4. **Use local embeddings**:
   - Switch to SentenceTransformer (free)
   - Eliminates embedding API costs

---

### Problem: Slow performance

**Symptoms**: Queries take > 3 seconds

**Solutions**:
1. **Use faster LLM**:
   ```yaml
   completion_model_id: gpt4o_completion
   ```

2. **Reduce context**:
   ```yaml
   k: 5
   max_context_tokens: 2000
   ```

3. **Optimize vector store**:
   - Check vector store performance
   - Consider in-memory vector stores
   - Optimize index settings

4. **Local embeddings**:
   - Eliminate network latency
   - Use SentenceTransformer locally

---

### Problem: Empty or irrelevant results

**Symptoms**: Returns empty or completely off-topic answers

**Solutions**:
1. **Check data coverage**:
   - Verify text units cover query topic
   - May need to reindex

2. **Increase retrieval count**:
   ```yaml
   k: 20  # Cast wider net
   ```

3. **Check embedding quality**:
   - Test vector search directly
   - Verify similarity scores
   - Consider different embedding model

4. **Try alternative search**:
   - Local search for entity-related queries
   - Global search for broad questions

---

### Problem: Out of memory

**Symptoms**: Memory errors during query execution

**Solutions**:
1. **Reduce context size**:
   ```yaml
   max_context_tokens: 2000
   k: 5
   ```

2. **Batch processing**:
   - Process queries one at a time
   - Clear memory between queries

3. **Optimize vector store**:
   - Use memory-efficient vector stores
   - Load embeddings on-demand

---

## Code References

### Key Files

- **Search Engine**: `packages/graphrag/graphrag/query/structured_search/basic_search/search.py:32-183`
- **Context Builder**: `packages/graphrag/graphrag/query/structured_search/basic_search/basic_context.py:27-110`
- **Configuration**: `packages/graphrag/graphrag/config/models/basic_search_config.py:11-33`

### Key Classes

- `BasicSearch`: Main search orchestrator
- `BasicSearchContext`: Context builder
- `TextUnit`: Data model for text chunks
- `SearchResult`: Result container

### Key Functions

- `BasicSearch.search()`: Main entry point
- `BasicSearch.stream_search()`: Streaming variant
- `BasicSearchContext.build_context()`: Context preparation

---

## Viewing the Diagram

### Option 1: PlantUML Online
1. Go to https://www.plantuml.com/plantuml/uml/
2. Paste contents of `basic_search_activity.puml`
3. Click "Submit"

### Option 2: VS Code Extension
1. Install "PlantUML" extension
2. Open `basic_search_activity.puml`
3. Press `Alt+D` to preview

### Option 3: Command Line
```bash
# Install PlantUML
brew install plantuml  # macOS
apt-get install plantuml  # Ubuntu

# Generate PNG
plantuml basic_search_activity.puml

# Generate SVG
plantuml -tsvg basic_search_activity.puml
```

---

## Related Documentation

- **Query Analysis**: See `query_analysis.md` for comprehensive overview of all search methods
- **Local Search Diagram**: See `local_search_activity.puml` for graph-aware comparison
- **Global Search Diagram**: See `global_search_activity.puml` for MAP-REDUCE comparison
- **GraphRAG Docs**: https://microsoft.github.io/graphrag/

---

**Last Updated**: 2026-01-30
**Diagram Version**: 1.0
**Status**: Complete ✅
