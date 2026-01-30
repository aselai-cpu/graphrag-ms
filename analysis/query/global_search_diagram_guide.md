# Global Search Activity Diagram - Guide

**File**: `global_search_activity.puml`
**Type**: PlantUML Activity Diagram
**Purpose**: Visual representation of GraphRAG Global Search complete flow (MAP-REDUCE pattern)

---

## Overview

The Global Search activity diagram shows the complete end-to-end flow of a global search query in GraphRAG, from CLI invocation through MAP-REDUCE processing to final answer generation.

## Diagram Structure

The diagram is organized into **5 main phases** with **6 swim lanes**:

### Swim Lanes (by color)

1. **CLI Layer** (AntiqueWhite) - User interaction and command handling
2. **Data Loading** (LightBlue) - Reading parquet files
3. **Initialization** (LightGreen) - Search engine setup
4. **Query Execution** (LightYellow) - Search orchestration
5. **Phase-specific lanes**:
   - Community Selection (LightCoral)
   - Context Batching (LightSteelBlue)
   - MAP Phase (Lavender)
   - REDUCE Phase (LightGoldenRodYellow)
   - Result Assembly (PaleGreen)

### Main Phases

#### Phase 1: Community Selection

**Purpose**: Select relevant community reports to query

**Two Paths**:

##### A. Static Selection (Default)
**Steps**:
1. Select all community reports at specified level (default: level 2)
2. Optionally filter by minimum rank
3. Fast, deterministic, no LLM calls

**Key Parameters**:
- `community_level`: Hierarchy level to query (default: 2)
- `min_community_rank`: Minimum rank filter (default: 0)

**Output**: List of community reports

**When to Use**:
- Small to medium datasets (< 1000 communities)
- Cost-sensitive scenarios
- Deterministic results required
- Fast execution needed

##### B. Dynamic Selection (Optional)
**Steps**:
1. Start with root communities (level 0)
2. For each community in queue:
   - Call LLM to rate relevance (0-10 scale)
   - If rating ≥ threshold: mark as relevant, add children to queue
   - If rating < threshold: skip
   - Optionally remove parent if child is relevant
3. Continue until queue empty
4. Fallback: If no relevant communities found, try next level

**Key Parameters**:
- `dynamic_search_threshold`: Rating threshold (default: 7)
- `dynamic_search_keep_parent`: Include parent communities (default: true)
- `dynamic_search_num_repeats`: Rating repetitions (default: 1)
- `dynamic_search_use_summary`: Rate summary vs full content (default: true)
- `dynamic_search_max_level`: Maximum hierarchy level (default: 10)
- `concurrent_coroutines`: Parallel rating calls (default: 8)

**Output**: Filtered list of relevant community reports

**When to Use**:
- Large datasets (> 1000 communities)
- Highly focused queries
- Quality over speed priority
- Acceptable to add LLM cost for precision

**Algorithm**:
```python
queue = root_communities  # Level 0
relevant = set()

while queue:
    # Rate all communities in current queue in parallel
    ratings = await asyncio.gather(*[
        rate_community(community, query)
        for community in queue
    ])

    next_queue = []
    for community, rating in zip(queue, ratings):
        if rating >= threshold:
            relevant.add(community)
            # Add children for deeper exploration
            next_queue.extend(community.children)
            # Optionally remove parent
            if not keep_parent:
                relevant.discard(community.parent)

    queue = next_queue

return [reports[c] for c in relevant]
```

**Cost Analysis**:
- Static: 0 LLM calls
- Dynamic: 10-100+ LLM calls (depends on graph size and query)
- Each rating call: ~500-1000 prompt tokens, ~50-100 output tokens

---

#### Phase 2: Context Batching

**Purpose**: Split community reports into batches within token limits

**Steps**:
1. Format community reports as CSV
2. Optionally shuffle reports (for diversity)
3. Calculate tokens for each report
4. Create batches respecting max_context_tokens
5. Each batch becomes one MAP call

**Key Parameters**:
- `max_context_tokens`: Token limit per batch (default: 8000)
- `shuffle_data`: Randomize order (default: true)
- `random_state`: Seed for reproducibility (default: 86)
- `include_community_rank`: Include rank column (default: false)
- `include_community_weight`: Include weight/occurrence (default: true)
- `normalize_community_weight`: Normalize weights (default: true)

**CSV Format**:
```csv
id,community,level,title,summary,weight
1,Community 1,2,Tech Industry Overview,This community covers...,0.85
2,Community 2,2,Healthcare Sector,Analysis of healthcare...,0.72
...
```

**Batching Algorithm**:
```python
batches = []
current_batch = []
current_tokens = 0

for report in community_reports:
    report_csv = format_as_csv_row(report)
    report_tokens = count_tokens(report_csv)

    if current_tokens + report_tokens > max_context_tokens:
        # Save current batch and start new one
        batches.append(current_batch)
        current_batch = [report]
        current_tokens = report_tokens
    else:
        current_batch.append(report)
        current_tokens += report_tokens

# Don't forget last batch
if current_batch:
    batches.append(current_batch)

return batches
```

**Output**: List of CSV-formatted community report batches

**Why Batching?**:
1. **LLM Context Limits**: Cannot fit all communities in one prompt
2. **Parallelization**: Multiple batches = parallel processing
3. **Better Coverage**: Each batch gets full LLM attention
4. **Scalability**: Works with any graph size

---

#### Phase 3: MAP Phase (Parallel)

**Purpose**: Extract key points from each community report batch in parallel

**Steps** (per batch):
1. Acquire semaphore slot (rate limiting)
2. Build MAP prompt with community reports CSV
3. Call LLM with JSON mode enabled
4. Parse JSON response to extract key points
5. Release semaphore
6. Return SearchResult with key points

**Parallel Execution**:
```python
# Execute all MAP calls in parallel
map_responses = await asyncio.gather(*[
    _map_response_single_batch(
        context_data=batch,
        query=query,
        max_length=map_max_length,
    )
    for batch in context_batches
])
```

**Semaphore Control**:
- Limits concurrent MAP calls
- Default: 32 concurrent requests
- Prevents overwhelming LLM API
- Balances speed vs rate limits

**MAP Prompt Structure**:
```
System Message:
    MAP_SYSTEM_PROMPT (formatted with context_data and max_length)
    Community reports CSV: id,community,level,title,summary
    Instructions: Extract key points, score 0-100

User Message:
    {user_query}
```

**Expected JSON Response**:
```json
{
  "points": [
    {
      "description": "The tech industry shows rapid AI adoption...",
      "score": 85
    },
    {
      "description": "Healthcare sector faces regulatory challenges...",
      "score": 72
    },
    {
      "description": "Minor point about market trends",
      "score": 45
    }
  ]
}
```

**Scoring Interpretation**:
- **0**: No relevant information found
- **1-30**: Low importance/relevance
- **31-60**: Medium importance
- **61-80**: High importance
- **81-100**: Critical information

**Error Handling**:
- If JSON parse fails: Return empty points list
- If LLM error: Return score=0 placeholder
- Continue processing other batches

**Key Parameters**:
- `map_max_length`: Max response length in words (default: 1000)
- `concurrent_coroutines`: Parallel limit (default: 32)
- `map_llm_params`: LLM parameters (temperature, etc.)
- `json_mode`: Enable JSON response format (default: true)

**Output**: List[SearchResult], one per batch
- Each contains: key points with scores, tokens used, timing

---

#### Phase 4: REDUCE Phase

**Purpose**: Synthesize all MAP outputs into final coherent answer

**Steps**:
1. **Aggregate Key Points**: Collect all points from all MAP responses
2. **Filter by Score**: Remove points with score ≤ 0
3. **Sort by Importance**: Order by score (descending)
4. **Apply Token Budget**: Select highest-scored points within limit
5. **Format as Analyst Reports**: Structure for REDUCE prompt
6. **Call LLM**: Generate final synthesized answer
7. **Return Result**: Package as SearchResult

**Detailed Process**:

##### Step 1: Aggregate Key Points
```python
key_points = []
for index, map_response in enumerate(map_responses):
    for element in map_response.response:
        key_points.append({
            "analyst": index,  # Which batch/MAP call
            "answer": element["description"],
            "score": element["score"]
        })
```

##### Step 2: Filter and Sort
```python
# Remove irrelevant points
filtered_points = [p for p in key_points if p["score"] > 0]

# Sort by importance (highest first)
sorted_points = sorted(
    filtered_points,
    key=lambda x: x["score"],
    reverse=True
)
```

##### Step 3: Apply Token Budget
```python
data = []
total_tokens = 0

for point in sorted_points:
    # Format as analyst report
    formatted = f"""----Analyst {point['analyst'] + 1}----
Importance Score: {point['score']}
{point['answer']}"""

    point_tokens = count_tokens(formatted)

    # Check budget
    if total_tokens + point_tokens > data_max_tokens:
        break  # Budget exceeded

    data.append(formatted)
    total_tokens += point_tokens

report_data = "\n\n".join(data)
```

##### Step 4: Build REDUCE Prompt
```python
reduce_prompt = REDUCE_SYSTEM_PROMPT.format(
    report_data=report_data,
    response_type=response_type,  # e.g., "Multiple Paragraphs"
    max_length=reduce_max_length  # e.g., 2000 words
)

if allow_general_knowledge:
    reduce_prompt += "\n" + GENERAL_KNOWLEDGE_INSTRUCTION
```

**REDUCE Prompt Structure**:
```
System Message:
    REDUCE_SYSTEM_PROMPT
    Analyst reports (formatted key points)
    Response type: {Multiple Paragraphs, Single Paragraph, etc.}
    Max length: {2000 words}
    [Optional: General knowledge instruction]

User Message:
    {user_query}
```

##### Step 5: Call LLM
```python
# Streaming mode
async for chunk in llm.completion_async(messages, stream=True):
    token = chunk.choices[0].delta.content
    yield token  # Real-time display
    full_response += token
```

**Key Parameters**:
- `data_max_tokens`: Token budget for analyst reports (default: 8000)
- `reduce_max_length`: Max response length in words (default: 2000)
- `response_type`: Desired format (default: "Multiple Paragraphs")
- `allow_general_knowledge`: Allow external knowledge (default: false)
- `reduce_llm_params`: LLM parameters

**Output**: SearchResult with final answer

**No Data Handling**:
If all MAP responses have score 0 (no relevant info found):
- If `allow_general_knowledge=false`: Return `NO_DATA_ANSWER`
  - "I do not have enough information to answer this question."
- If `allow_general_knowledge=true`: Allow LLM to use general knowledge
  - Risk: May increase hallucinations

---

#### Phase 5: Result Assembly

**Purpose**: Package all results for return to user

**GlobalSearchResult Structure**:
```python
@dataclass
class GlobalSearchResult:
    # Final answer
    response: str

    # Context used
    context_data: dict                     # Community reports DataFrames
    context_text: list[str]                # Formatted CSV batches

    # MAP phase outputs
    map_responses: list[SearchResult]      # All MAP results

    # REDUCE phase context
    reduce_context_data: str               # Analyst reports
    reduce_context_text: str               # Formatted reduce context

    # Timing
    completion_time: float                 # Total duration (seconds)

    # Token usage
    llm_calls: int                         # Total LLM calls
    prompt_tokens: int                     # Total input tokens
    output_tokens: int                     # Total output tokens

    # Per-phase breakdown
    llm_calls_categories: dict[str, int]   # {"build_context": X, "map": N, "reduce": 1}
    prompt_tokens_categories: dict[str, int]
    output_tokens_categories: dict[str, int]
```

**Token Accounting**:
```python
llm_calls = {
    "build_context": dynamic_selection_calls,  # 0 for static
    "map": num_batches,
    "reduce": 1
}
prompt_tokens = {
    "build_context": dynamic_selection_prompt_tokens,
    "map": sum(map_prompt_tokens),
    "reduce": reduce_prompt_tokens
}
output_tokens = {
    "build_context": dynamic_selection_output_tokens,
    "map": sum(map_output_tokens),
    "reduce": reduce_output_tokens
}
```

---

## Key Characteristics

### MAP-REDUCE Pattern

**Why MAP-REDUCE?**
1. **Scalability**: Can handle unlimited community reports
2. **Parallelization**: MAP calls run concurrently
3. **Quality**: Each batch gets focused LLM attention
4. **Aggregation**: REDUCE synthesizes multiple perspectives

**Trade-offs**:
- **Pro**: Comprehensive coverage of entire graph
- **Pro**: Scalable to any data size
- **Con**: Many LLM calls = high cost
- **Con**: Slower than local search

### Multiple LLM Calls

**Call Breakdown**:
1. **Build Context** (Optional): 0-100+ calls for dynamic selection
2. **MAP Phase**: N calls (one per batch)
   - Typical: 5-20 batches
   - Depends on: number of communities, token limit, selection method
3. **REDUCE Phase**: 1 call

**Total**: 1 + N (static) or 10-100 + N + 1 (dynamic)

### Community Hierarchy Utilization

**Leiden Hierarchy**:
- Level 0: Root communities (coarsest)
- Level 1-2: Mid-level communities
- Level 3+: Fine-grained communities

**Default Query Level**: 2
- Balances specificity vs coverage
- Adjust based on graph structure

### Token Budget Management

**Batching Budget** (Phase 2):
- Each batch ≤ `max_context_tokens` (8000)
- Ensures MAP prompts fit in LLM context

**REDUCE Budget** (Phase 4):
- Analyst reports ≤ `data_max_tokens` (8000)
- Prioritizes highest-scored key points

---

## Performance Characteristics

### Speed: Slow (5-20 seconds)

| Operation | Time | Notes |
|-----------|------|-------|
| Community selection (static) | 100-500ms | Fast, no LLM |
| Community selection (dynamic) | 2-5s | Many LLM calls |
| Context batching | 50-100ms | Pure computation |
| MAP phase (parallel) | 3-10s | Depends on batches |
| REDUCE phase | 2-5s | Single LLM call |
| **Total (static)** | **5-15s** | Typical case |
| **Total (dynamic)** | **10-20s** | With filtering |

**Factors Affecting Speed**:
- Number of community reports
- Dynamic vs static selection
- LLM model speed (GPT-4 vs GPT-4o vs Claude)
- Network latency
- Concurrent coroutines limit

### Cost: High

**Static Selection Example** (GPT-4o):
```
Assumptions:
- 1000 community reports
- 10 batches (100 reports per batch)
- GPT-4o pricing: $2.50/1M input, $10/1M output

MAP Phase (10 calls):
- Input: 10 × 8000 tokens = 80,000 tokens → $0.20
- Output: 10 × 1500 tokens = 15,000 tokens → $0.15

REDUCE Phase (1 call):
- Input: 8000 tokens → $0.02
- Output: 2000 tokens → $0.02

Total per query: $0.39
```

**Dynamic Selection Example** (GPT-4o):
```
Additional costs:
- 50 rating calls
- Input: 50 × 800 tokens = 40,000 tokens → $0.10
- Output: 50 × 50 tokens = 2,500 tokens → $0.025

Total per query: $0.515
```

**Cost Comparison**:
| Scenario | LLM Calls | Tokens | Cost (GPT-4o) |
|----------|-----------|--------|---------------|
| Small dataset (static) | 6 calls | ~50K | $0.15 |
| Medium dataset (static) | 11 calls | ~100K | $0.35 |
| Large dataset (static) | 21 calls | ~200K | $0.70 |
| Medium dataset (dynamic) | 60 calls | ~150K | $0.50 |

### Quality: Best for Broad Questions

**Strengths**:
- ✅ Comprehensive coverage of entire knowledge graph
- ✅ Multiple perspectives (analyst reports from different batches)
- ✅ Hierarchical understanding (community structure)
- ✅ Balanced aggregation (scored key points)
- ✅ Scales to any dataset size

**Limitations**:
- ❌ May miss fine-grained entity relationships
- ❌ Less effective for specific entity queries
- ❌ Can be too broad for focused questions
- ❌ High cost for simple queries

---

## Configuration

### Required Settings

```yaml
global_search:
  # Prompts
  map_prompt: "prompts/global_search_map_system_prompt.txt"
  reduce_prompt: "prompts/global_search_reduce_system_prompt.txt"
  knowledge_prompt: "prompts/global_search_knowledge_system_prompt.txt"

  # Models
  completion_model_id: default_completion_model

  # Context Limits
  max_context_tokens: 8000     # Per batch
  data_max_tokens: 8000        # For REDUCE

  # Response Formatting
  response_type: "Multiple Paragraphs"
  map_max_length: 1000         # Words
  reduce_max_length: 2000      # Words

  # Dynamic Selection (Optional)
  dynamic_search_threshold: 7
  dynamic_search_keep_parent: true
  dynamic_search_num_repeats: 1
  dynamic_search_use_summary: true
  dynamic_search_max_level: 10

  # Parallelism
  concurrent_coroutines: 32

  # LLM Parameters
  temperature: 0
  top_p: 1
  n: 1
  max_gen_tokens: null

  # Advanced
  allow_general_knowledge: false
```

### Data Requirements

**Parquet Files**:
- ✅ `communities.parquet` - Community structure
- ✅ `community_reports.parquet` - Community summaries
- ⚠️ `entities.parquet` - Optional (for dynamic selection weights)

**Vector Stores**:
- ⚠️ `entity_description` embeddings - Optional (for dynamic selection)

**Minimal Requirements**:
- Only needs community data
- No entity/relationship/text unit data required
- Lighter data footprint than local search

---

## Use Cases

### Ideal For:

✅ **Broad, abstract questions**
- "What are the main themes in this dataset?"
- "Summarize the key findings"
- "What are the overarching patterns?"

✅ **Dataset-wide analysis**
- "Describe the major trends"
- "What are the high-level insights?"
- "Give me an overview of the entire corpus"

✅ **Multi-faceted exploration**
- "Compare different sectors in the data"
- "What are the diverse perspectives on topic X?"
- "Analyze the dataset from multiple angles"

✅ **Hierarchical understanding**
- "What are the top-level categories?"
- "Describe the structure of the knowledge"
- "What are the major clusters?"

### Not Ideal For:

❌ **Specific entity questions**
- "How is Company X related to Person Y?" (use local search)
- "What are the connections of Entity Z?" (use local search)

❌ **Simple factoid questions**
- "What does the document say about X?" (use basic search)
- "Find mentions of Y" (use basic search)

❌ **Fine-grained relationship queries**
- "Trace the path between A and B" (use local search)
- "What's the shortest connection?" (use local search)

❌ **Cost-sensitive scenarios with specific queries**
- Local search is more cost-effective for targeted questions
- Global search best when breadth is essential

---

## Comparison with Other Search Methods

| Aspect | Global Search | Local Search | DRIFT Search | Basic Search |
|--------|---------------|--------------|--------------|--------------|
| **Pattern** | MAP-REDUCE | Single-shot | Iterative | Single-shot |
| **LLM Calls** | Many (N+1) | 1 | Many (primer + local + reduce) | 1 |
| **Data Source** | Community reports | Multi-source | Combined | Text units only |
| **Scope** | Entire graph | Entity neighborhoods | Iterative exploration | Text chunks |
| **Speed** | Slow (5-20s) | Medium (2-5s) | Slowest (10-30s) | Fastest (1-3s) |
| **Cost** | High ($0.15-0.70) | Medium ($0.03-0.05) | Highest ($0.50-1.00) | Low ($0.01-0.02) |
| **Quality (broad)** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ |
| **Quality (specific)** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Parallelization** | ✅ MAP calls | ❌ | ⚠️ Some | ❌ |
| **Graph Usage** | Community hierarchy | Entity relationships | Both | None |
| **Streaming** | ✅ REDUCE only | ✅ | ✅ Final only | ✅ |

---

## Optimization Tips

### 1. Adjust Community Selection

**For Smaller Datasets** (< 500 communities):
```yaml
# Use static selection for speed and cost
# No configuration needed - static is default
```

**For Larger Datasets** (> 1000 communities):
```yaml
# Enable dynamic selection for precision
dynamic_search_threshold: 7
dynamic_search_keep_parent: true
concurrent_coroutines: 16  # Increase for faster rating
```

**For Highly Focused Queries**:
```yaml
# Stricter filtering
dynamic_search_threshold: 8  # Higher threshold
dynamic_search_keep_parent: false  # Only leaves
```

**For Broader Coverage**:
```yaml
# More permissive
dynamic_search_threshold: 5  # Lower threshold
dynamic_search_keep_parent: true  # Include parents
```

### 2. Tune Batch Size

**Smaller Batches** (More MAP calls):
```yaml
max_context_tokens: 4000  # Smaller batches
# Pro: Better parallel utilization
# Con: More LLM calls, higher cost
```

**Larger Batches** (Fewer MAP calls):
```yaml
max_context_tokens: 12000  # Larger batches
# Pro: Fewer LLM calls, lower cost
# Con: Less parallelization, potentially less focused
```

**Recommendation**: Keep default (8000) unless specific needs

### 3. Control Parallelism

**Higher Concurrency** (Faster):
```yaml
concurrent_coroutines: 64  # More parallel
# Pro: Faster MAP phase
# Con: May hit API rate limits
```

**Lower Concurrency** (Rate limit safe):
```yaml
concurrent_coroutines: 16  # Less parallel
# Pro: Avoids rate limits
# Con: Slower MAP phase
```

**For Dynamic Selection**:
```yaml
# In dynamic_community_selection_kwargs
concurrent_coroutines: 8  # Rating calls
```

### 4. Response Length Tuning

**Shorter Responses** (Faster, cheaper):
```yaml
map_max_length: 500         # Shorter MAP outputs
reduce_max_length: 1000     # Shorter final answer
# Pro: Lower output tokens, faster generation
# Con: Less detailed answers
```

**Longer Responses** (More detailed):
```yaml
map_max_length: 2000        # Longer MAP outputs
reduce_max_length: 3000     # Longer final answer
# Pro: More comprehensive answers
# Con: Higher cost, slower
```

### 5. Model Selection

**Cost-Optimized** (Claude Haiku):
```yaml
completion_model_id: haiku_completion
# Cost: ~97% cheaper than GPT-4
# Speed: 3x faster
# Quality: Good for straightforward queries
```

**Quality-Optimized** (Claude 3.5 Sonnet or GPT-4):
```yaml
completion_model_id: sonnet_completion  # or gpt4_completion
# Cost: Highest
# Speed: Slower
# Quality: Best for complex analysis
```

**Speed-Optimized** (GPT-4o):
```yaml
completion_model_id: gpt4o_completion
# Cost: Medium
# Speed: Fast
# Quality: High
# Balanced option
```

### 6. General Knowledge Control

**Strict Data-Only** (Default):
```yaml
allow_general_knowledge: false
# Returns "I don't know" if no relevant data
# Minimizes hallucinations
```

**Allow External Knowledge**:
```yaml
allow_general_knowledge: true
# Uses general knowledge when data insufficient
# Risk: May hallucinate
# Use when: Query may need external context
```

---

## Troubleshooting

### Problem: High cost per query

**Symptoms**: Expensive bills, many LLM calls

**Solutions**:
1. **Disable dynamic selection**:
   ```yaml
   # Remove or set to false
   # Saves 10-100 LLM calls per query
   ```

2. **Increase batch size**:
   ```yaml
   max_context_tokens: 12000  # Fewer batches
   ```

3. **Use cheaper model**:
   ```yaml
   completion_model_id: haiku_completion
   # 97% cost reduction vs GPT-4
   ```

4. **Reduce output length**:
   ```yaml
   map_max_length: 500
   reduce_max_length: 1000
   ```

5. **Consider alternative search method**:
   - Use local search for specific queries
   - Use basic search for simple lookups

---

### Problem: Slow performance

**Symptoms**: Queries take > 20 seconds

**Solutions**:
1. **Increase parallelism**:
   ```yaml
   concurrent_coroutines: 64
   ```

2. **Use faster model**:
   ```yaml
   completion_model_id: gpt4o_completion  # 2-3x faster
   ```

3. **Disable dynamic selection**:
   - Saves 2-5 seconds per query

4. **Reduce batch count**:
   ```yaml
   max_context_tokens: 12000  # Larger batches
   ```

5. **Optimize community level**:
   ```bash
   --community-level 1  # Fewer communities at higher level
   ```

---

### Problem: Poor quality answers

**Symptoms**: Answers are too vague, miss important info, or are off-topic

**Solutions**:
1. **Enable dynamic selection**:
   ```yaml
   # Ensures only relevant communities queried
   dynamic_search_threshold: 7
   ```

2. **Adjust community level**:
   ```bash
   --community-level 3  # More specific communities
   ```

3. **Increase MAP output length**:
   ```yaml
   map_max_length: 2000  # More detailed key points
   ```

4. **Check data quality**:
   - Verify community reports are informative
   - Check indexing quality
   - Ensure communities are well-formed

5. **Use better model**:
   ```yaml
   completion_model_id: sonnet_completion  # Higher quality
   ```

6. **Increase REDUCE context**:
   ```yaml
   data_max_tokens: 12000  # More key points in REDUCE
   ```

---

### Problem: Empty or "I don't know" responses

**Symptoms**: Returns NO_DATA_ANSWER or very short answers

**Solutions**:
1. **Enable general knowledge**:
   ```yaml
   allow_general_knowledge: true
   ```

2. **Lower dynamic selection threshold**:
   ```yaml
   dynamic_search_threshold: 5  # More permissive
   ```

3. **Use different community level**:
   ```bash
   --community-level 2  # Try different level
   ```

4. **Check data coverage**:
   - Verify community reports cover query topic
   - May need to reindex with different settings

5. **Try static selection**:
   - Dynamic selection may be too strict
   - Static ensures all communities at level are queried

---

### Problem: Rate limit errors

**Symptoms**: API rate limit exceeded errors

**Solutions**:
1. **Reduce concurrency**:
   ```yaml
   concurrent_coroutines: 16  # Lower limit
   ```

2. **Add retry logic**:
   - LiteLLM handles this automatically
   - Configure backoff in model config

3. **Batch processing**:
   - Process multiple queries with delays
   - Use queueing system

4. **Upgrade API tier**:
   - OpenAI: Tier 3+ for higher limits
   - Anthropic: Contact support

---

### Problem: JSON parse errors in MAP phase

**Symptoms**: Warnings about JSON parsing failures

**Solutions**:
1. **Verify JSON mode enabled**:
   ```yaml
   # Should be enabled by default
   # Check map_llm_params
   ```

2. **Use newer model**:
   - GPT-4o has better JSON support
   - Claude 3.5 Sonnet has excellent structured output

3. **Check prompts**:
   - Ensure MAP_SYSTEM_PROMPT clearly requests JSON
   - Verify format examples in prompt

4. **Increase temperature slightly**:
   ```yaml
   temperature: 0.1  # May help with formatting
   ```

---

### Problem: Memory issues with large datasets

**Symptoms**: Out of memory errors, slow data loading

**Solutions**:
1. **Optimize data loading**:
   - Load only required columns
   - Filter communities early

2. **Reduce batch size**:
   ```yaml
   max_context_tokens: 4000
   ```

3. **Use dynamic selection**:
   - Reduces communities processed
   - More memory efficient

4. **Process in chunks**:
   - Split large queries into multiple smaller runs

---

## Code References

### Key Files

- **Search Engine**: `packages/graphrag/graphrag/query/structured_search/global_search/search.py:55-522`
- **Community Context Builder**: `packages/graphrag/graphrag/query/structured_search/global_search/community_context.py:27-146`
- **Dynamic Selection**: `packages/graphrag/graphrag/query/context_builder/dynamic_community_selection.py:26-177`
- **Community Context Helper**: `packages/graphrag/graphrag/query/context_builder/community_context.py`
- **Configuration**: `packages/graphrag/graphrag/config/models/global_search_config.py:11-67`

### Key Classes

- `GlobalSearch`: Main search orchestrator
- `GlobalCommunityContext`: Context builder
- `DynamicCommunitySelection`: LLM-based community filtering
- `GlobalSearchResult`: Extended result with MAP responses
- `SearchResult`: Base result container
- `CommunityReport`, `Community`: Data models

### Key Functions

- `GlobalSearch.search()`: Main entry point
- `GlobalSearch._map_response_single_batch()`: Single MAP call
- `GlobalSearch._reduce_response()`: REDUCE aggregation
- `GlobalSearch._stream_reduce_response()`: Streaming REDUCE
- `GlobalCommunityContext.build_context()`: Context preparation
- `DynamicCommunitySelection.select()`: Relevance-based filtering

---

## Viewing the Diagram

### Option 1: PlantUML Online
1. Go to https://www.plantuml.com/plantuml/uml/
2. Paste contents of `global_search_activity.puml`
3. Click "Submit"

### Option 2: VS Code Extension
1. Install "PlantUML" extension
2. Open `global_search_activity.puml`
3. Press `Alt+D` to preview

### Option 3: Command Line
```bash
# Install PlantUML
brew install plantuml  # macOS
apt-get install plantuml  # Ubuntu

# Generate PNG
plantuml global_search_activity.puml

# Generate SVG
plantuml -tsvg global_search_activity.puml
```

---

## Related Documentation

- **Query Analysis**: See `query_analysis.md` for comprehensive overview of all search methods
- **Local Search Diagram**: See `local_search_activity.puml` and guide for comparison
- **GraphRAG Docs**: https://microsoft.github.io/graphrag/

---

**Last Updated**: 2026-01-30
**Diagram Version**: 1.0
**Status**: Complete ✅
