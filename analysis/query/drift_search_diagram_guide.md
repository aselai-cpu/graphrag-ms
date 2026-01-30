# DRIFT Search Activity Diagram - Guide

**File**: `drift_search_activity.puml`
**Type**: PlantUML Activity Diagram
**Purpose**: Visual representation of GraphRAG DRIFT Search complete flow (Iterative Refinement)

---

## Overview

The DRIFT Search activity diagram shows the complete end-to-end flow of a DRIFT (Dynamic Reasoning and Inference with Flexible Traversal) search query in GraphRAG, combining global and local search with iterative query refinement.

## Diagram Structure

The diagram is organized into **4 main stages** with **6 swim lanes**:

### Swim Lanes (by color)

1. **CLI Layer** (AntiqueWhite) - User interaction and command handling
2. **Data Loading** (LightBlue) - Reading parquet files and embeddings
3. **Initialization** (LightGreen) - Search engine setup
4. **Query Execution** (LightYellow) - Search orchestration
5. **Stage-specific lanes**:
   - PRIMER (LightCoral)
   - Iterative Local Search (LightSteelBlue + Lavender)
   - REDUCE Phase (LightGoldenRodYellow)
   - Result Assembly (PaleGreen)

### Main Stages

#### Stage 1: PRIMER (Global Context)

**Purpose**: Generate initial understanding and follow-up questions using global community structure

**Steps**:
1. Build global context (select community reports)
2. Split reports into parallel folds
3. For each fold:
   - Format DRIFT_PRIMER_PROMPT with reports
   - Call LLM with structured response format
   - Parse PrimerResponse (intermediate_answer + score + follow_up_queries)
4. Collect all primer responses
5. Create initial DriftAction with combined answers
6. Add follow-up queries to QueryState queue

**Key Parameters**:
- `primer_folds`: Number of parallel primer calls (default: 10)
- `community_level`: Community hierarchy level (default: 2)
- `dynamic_community_selection`: Enable LLM-based filtering (default: false)

**PrimerResponse Structure**:
```python
class PrimerResponse(BaseModel):
    intermediate_answer: str  # 2000 characters, markdown formatted
    score: int  # 0-100 relevance score
    follow_up_queries: list[str]  # 5+ follow-up questions
```

**Example Primer Output**:
```json
{
  "intermediate_answer": "## Technology Industry Overview\n\nThe technology sector shows...",
  "score": 85,
  "follow_up_queries": [
    "What are the key relationships between major tech companies?",
    "How do regulatory challenges affect the industry?",
    "What role does innovation play in competitive dynamics?",
    "Which entities are most central to the ecosystem?",
    "What are the emerging trends in the sector?"
  ]
}
```

**Algorithm**:
```python
# 1. Build global context
community_reports = await context_builder.build_context(query)

# 2. Split into folds for parallel processing
folds = np.array_split(community_reports, primer_folds)  # default: 10

# 3. Process folds in parallel
tasks = [
    primer.decompose_query(query, fold)
    for fold in folds
]
responses = await asyncio.gather(*tasks)

# 4. Process responses
for response in responses:
    intermediate_answer = response["intermediate_answer"]
    score = response["score"]
    follow_ups = response["follow_up_queries"]
    # Create DriftAction and add to QueryState

# 5. Create initial action
initial_action = DriftAction(
    query=query,
    answer=combined_intermediate_answers,
    score=average_score,
    follow_ups=all_follow_up_queries
)
query_state.add_action(initial_action)
```

**Token Usage**:
- Typical: 10 primer calls
- ~8000 prompt tokens per call
- ~2000 output tokens per call
- Total: ~80K prompt + ~20K output = ~100K tokens

**Output**: Initial DriftAction with primer answer and follow-up queue

---

#### Stage 2: Iterative Local Search

**Purpose**: Refine understanding by iteratively answering follow-up questions using local search

**Main Loop**:
```python
epochs = 0
while epochs < n_depth:  # default: n_depth = 2
    # 1. Rank incomplete actions
    actions = query_state.rank_incomplete_actions()
    if not actions:
        break

    # 2. Select top-k for processing
    actions = actions[:drift_k_followups]  # default: 3

    # 3. Execute local searches in parallel
    results = await asyncio.gather(*[
        action.search(search_engine=local_search, global_query=query)
        for action in actions
    ])

    # 4. Process results
    for action in results:
        query_state.add_action(action)
        query_state.add_all_follow_ups(action, action.follow_ups)

    epochs += 1
```

**Action Ranking**:
Actions are ranked by:
1. **Score** (descending): Higher-scored queries prioritized
2. **Depth** (ascending): Shallower queries preferred
3. **Completeness**: Only incomplete actions (no answer yet)

**Per-Action Local Search**:
Each action executes a full local search:
1. **Entity Mapping**: Vector search on query
2. **Graph Traversal**: Expand to entity neighborhoods
3. **Context Building**: Multi-source assembly
   - Entities, relationships, communities
   - Text units, covariates
4. **LLM Generation**: JSON response with follow-ups

**JSON Response Format**:
```json
{
  "response": "Answer to the follow-up question...",
  "score": 0.85,
  "follow_up_queries": [
    "Additional question 1...",
    "Additional question 2...",
    "..."
  ]
}
```

**Key Parameters**:
- `n_depth`: Number of iterations (default: 2)
- `drift_k_followups`: Actions per iteration (default: 3)
- `local_search_*`: Local search configuration
  - `local_search_text_unit_prop`: 0.5
  - `local_search_community_prop`: 0.5
  - `local_search_top_k_mapped_entities`: 10
  - `local_search_top_k_relationships`: 10
  - `local_search_max_data_tokens`: 8000

**QueryState Management**:
```python
class QueryState:
    def __init__(self):
        self.graph: dict[str, DriftAction] = {}  # Query -> Action mapping
        self.incomplete_actions: list[DriftAction] = []

    def add_action(self, action: DriftAction):
        """Add completed action to state."""
        self.graph[action.query] = action

    def add_all_follow_ups(self, parent: DriftAction, follow_ups: list[str]):
        """Create new actions for follow-up queries."""
        for query in follow_ups:
            if query not in self.graph:
                new_action = DriftAction(query=query)
                self.incomplete_actions.append(new_action)

    def rank_incomplete_actions(self) -> list[DriftAction]:
        """Return sorted list of incomplete actions."""
        return sorted(
            self.incomplete_actions,
            key=lambda a: (a.score if a.score else 0, a.depth),
            reverse=True
        )
```

**Execution Flow Example** (n_depth=2, k=3):
```
PRIMER:
  └─ Initial Action (answer + 50 follow-ups)

Iteration 1:
  ├─ Action 1 (top-scored follow-up) → answer + 5 new follow-ups
  ├─ Action 2 (2nd-scored follow-up) → answer + 5 new follow-ups
  └─ Action 3 (3rd-scored follow-up) → answer + 5 new follow-ups
  (15 new follow-ups added to queue)

Iteration 2:
  ├─ Action 4 (highest-scored from queue) → answer + 5 new follow-ups
  ├─ Action 5 (2nd-scored from queue) → answer + 5 new follow-ups
  └─ Action 6 (3rd-scored from queue) → answer + 5 new follow-ups

Total: 1 primer + 6 local searches = 7 answers
```

**Token Usage** (typical):
- Iteration 1: 3 local searches × ~12K tokens = ~36K
- Iteration 2: 3 local searches × ~12K tokens = ~36K
- Total: ~72K prompt + ~12K output = ~84K tokens

**Output**: QueryState with graph of completed actions and answers

---

#### Stage 3: REDUCE Phase

**Purpose**: Synthesize all intermediate answers into single comprehensive response

**Steps**:
1. Extract all completed actions from QueryState
2. Collect all answers (primer + local search results)
3. Format as context for REDUCE prompt
4. Call LLM to synthesize final answer
5. Track tokens and timing

**Answer Collection**:
```python
# Extract answers from query state graph
answers = []

# Add primer answer
if initial_action.answer:
    answers.append(initial_action.answer)

# Add all local search answers
for action in query_state.graph.values():
    if action.answer and action != initial_action:
        answers.append(action.answer)

# Format for REDUCE
context_data = answers  # List of strings
```

**REDUCE Prompt Structure**:
```python
reduce_prompt = DRIFT_REDUCE_PROMPT.format(
    context_data=answers,  # List of intermediate answers
    response_type=response_type  # e.g., "Multiple Paragraphs"
)

messages = [
    {"role": "system", "content": reduce_prompt},
    {"role": "user", "content": original_query}
]
```

**LLM Call**:
```python
model_params = {
    "temperature": reduce_temperature,  # from config
    "max_completion_tokens": reduce_max_completion_tokens
}

if streaming:
    async for chunk in llm.completion_async(messages, stream=True, **model_params):
        token = chunk.choices[0].delta.content or ""
        yield token
else:
    response = await llm.completion_async(messages, **model_params)
    final_answer = response.content
```

**Key Parameters**:
- `reduce_temperature`: LLM temperature for REDUCE (from config)
- `reduce_max_completion_tokens`: Max output tokens
- `response_type`: Desired format (default: "Multiple Paragraphs")

**Example REDUCE Context**:
```
Answer 1 (Primer): The technology industry encompasses major companies like...

Answer 2 (Local): Microsoft and Google have complex relationships through both competition and collaboration...

Answer 3 (Local): Regulatory challenges in the EU include antitrust investigations focusing on...

Answer 4 (Local): Innovation cycles in cloud computing show rapid evolution with...

Answer 5 (Local): Central entities in the ecosystem include the FAANG companies which...

Answer 6 (Local): Emerging trends indicate a shift toward AI-first architectures with...

Answer 7 (Local): Competitive dynamics reveal interesting patterns where...
```

**Token Usage**:
- ~8000 prompt tokens (REDUCE prompt + all answers)
- ~2000 output tokens (final synthesis)
- Total: ~10K tokens

**Output**: Final synthesized answer combining all perspectives

---

#### Stage 4: Result Assembly

**Purpose**: Package all results for return to user

**SearchResult Structure**:
```python
@dataclass
class SearchResult:
    # Final answer
    response: str

    # Context used
    context_data: dict                     # Graph of all actions and context
    context_text: str                      # Serialized query state

    # Timing
    completion_time: float                 # Total duration (seconds)

    # Token usage
    llm_calls: int                         # Total LLM calls
    prompt_tokens: int                     # Total input tokens
    output_tokens: int                     # Total output tokens

    # Per-phase breakdown
    llm_calls_categories: dict[str, int]
    # {"build_context": X, "primer": 10, "action": 6+, "reduce": 1}
    prompt_tokens_categories: dict[str, int]
    output_tokens_categories: dict[str, int]
```

**Token Accounting Example**:
```python
llm_calls = {
    "build_context": 0,  # Static selection (or 10-100 for dynamic)
    "primer": 10,  # primer_folds
    "action": 6,  # n_depth × drift_k_followups (2 × 3)
    "reduce": 1
}
# Total: 17 LLM calls

prompt_tokens = {
    "build_context": 0,
    "primer": 80000,  # 10 × 8000
    "action": 72000,  # 6 × 12000
    "reduce": 8000
}
# Total: 160K prompt tokens

output_tokens = {
    "build_context": 0,
    "primer": 20000,  # 10 × 2000
    "action": 12000,  # 6 × 2000
    "reduce": 2000
}
# Total: 34K output tokens

# Grand total: ~194K tokens
```

---

## Key Characteristics

### Iterative Refinement

**What is DRIFT?**
Dynamic Reasoning and Inference with Flexible Traversal combines:
1. **Global Primer**: Broad understanding from community reports
2. **Iterative Local Search**: Focused exploration via follow-up questions
3. **Dynamic Query Generation**: LLM generates next questions
4. **State Management**: Tracks exploration graph
5. **Final Synthesis**: Combines all discoveries

**Why Iterative?**
- ✅ Depth + Breadth: Combines global and local strengths
- ✅ Adaptive: Follows most promising leads
- ✅ Comprehensive: Explores multiple angles
- ✅ Quality: Best for complex questions

### QueryState Management

**QueryState Tracks**:
1. **Action Graph**: Query → Answer mappings
2. **Incomplete Queue**: Unanswered follow-ups
3. **Token Usage**: Per-action accounting
4. **Context Data**: All retrieved information

**Action Lifecycle**:
```
1. Created (from primer or previous action)
   ↓
2. Added to incomplete queue
   ↓
3. Ranked by score
   ↓
4. Selected for processing (if top-k)
   ↓
5. Local search executed
   ↓
6. Answer added, follow-ups extracted
   ↓
7. Marked complete, moved to graph
   ↓
8. New follow-ups added to queue
```

### Multiple LLM Calls

**Call Breakdown** (typical):
1. **Build Context** (optional): 0-100 calls
   - Only if dynamic community selection enabled
   - Default: 0 (static selection)

2. **Primer**: 10 calls (default)
   - `primer_folds` parallel calls
   - Each processes subset of community reports

3. **Action** (local searches): 6-20+ calls
   - `n_depth` × `drift_k_followups`
   - Default: 2 × 3 = 6 calls
   - Can be more if deeper exploration needed

4. **REDUCE**: 1 call
   - Synthesizes all answers

**Total**: 17-130+ LLM calls per query

### Combines Global + Local

**Global (Primer)**:
- Uses community reports
- Broad understanding
- Generates follow-ups
- MAP-like parallel processing

**Local (Actions)**:
- Entity-focused search
- Graph traversal
- Multi-source context
- Targeted exploration

**Best of Both Worlds**:
- Global primer provides direction
- Local searches provide depth
- Iterative refinement adapts
- Final reduce maintains coherence

---

## Performance Characteristics

### Speed: Slowest (10-30 seconds)

| Operation | Time | Notes |
|-----------|------|-------|
| Build context | 100-500ms | Static selection |
| Build context (dynamic) | 2-5s | LLM-based filtering |
| Primer (10 parallel) | 3-5s | Parallel fold processing |
| Iteration 1 (3 local) | 4-8s | Parallel local searches |
| Iteration 2 (3 local) | 4-8s | Parallel local searches |
| REDUCE | 2-5s | Single synthesis call |
| **Total (typical)** | **10-25s** | Full pipeline |

**Factors Affecting Speed**:
- Number of iterations (`n_depth`)
- Actions per iteration (`drift_k_followups`)
- Primer folds (`primer_folds`)
- LLM model speed
- Dynamic vs static community selection

### Cost: Highest

**Token Breakdown** (typical, n_depth=2, k=3):
```
Phase            | LLM Calls | Prompt Tokens | Output Tokens | Subtotal
-----------------|-----------|---------------|---------------|----------
Build Context    |  0        |      0        |      0        |      0
Primer           | 10        |  80,000       | 20,000        | 100,000
Action (Local)   |  6        |  72,000       | 12,000        |  84,000
REDUCE           |  1        |   8,000       |  2,000        |  10,000
-----------------|-----------|---------------|---------------|----------
TOTAL            | 17        | 160,000       | 34,000        | 194,000
```

**Cost Examples**:

| Model | Input Price | Output Price | Total Cost per Query |
|-------|-------------|--------------|----------------------|
| GPT-4o | $2.50/1M | $10/1M | $0.74 |
| GPT-4 Turbo | $10/1M | $30/1M | $2.62 |
| Claude 3.5 Sonnet | $3/1M | $15/1M | $0.99 |
| Claude 3 Haiku | $0.25/1M | $1.25/1M | $0.08 |

**Cost with More Depth** (n_depth=3, k=5):
- Actions: 3 × 5 = 15 local searches
- Total calls: ~26
- Total tokens: ~300K
- Cost (GPT-4o): ~$1.10

**Cost Comparison**:
| Method | Calls | Tokens | Cost (GPT-4o) | Relative |
|--------|-------|--------|---------------|----------|
| Basic | 1 | 5-6K | $0.02 | 1x |
| Local | 1 | 12-14K | $0.04 | 2x |
| Global | 11 | 100K | $0.35 | 17x |
| DRIFT | 17+ | 194K+ | $0.74+ | 37x+ |

### Quality: Best for Complex Queries

**Strengths**:
- ✅ Most comprehensive exploration
- ✅ Multiple perspectives captured
- ✅ Iterative depth refinement
- ✅ Adaptive query following
- ✅ Combines global + local strengths
- ✅ Best for multi-faceted questions

**Limitations**:
- ❌ Highest cost per query
- ❌ Slowest execution time
- ❌ Complex to configure
- ❌ Overkill for simple questions

**Quality by Query Type**:
| Query Type | DRIFT | Global | Local | Basic |
|------------|-------|--------|-------|-------|
| Complex multi-faceted | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| Broad themes | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐ |
| Entity relationships | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| Simple factoids | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Multi-angle analysis | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐ |

---

## Configuration

### Required Settings

```yaml
drift_search:
  # Prompts
  prompt: "prompts/drift_search_system_prompt.txt"  # Local search prompt
  reduce_prompt: "prompts/drift_search_reduce_prompt.txt"

  # Models
  completion_model_id: default_completion_model
  embedding_model_id: default_embedding_model

  # DRIFT Parameters
  n_depth: 2                    # Number of iterations
  drift_k_followups: 3          # Actions per iteration
  primer_folds: 10              # Parallel primer calls

  # Token Limits
  data_max_tokens: 8000         # For primer context
  reduce_max_tokens: 500        # REDUCE output limit

  # Local Search Settings
  local_search_text_unit_prop: 0.5
  local_search_community_prop: 0.5
  local_search_top_k_mapped_entities: 10
  local_search_top_k_relationships: 10
  local_search_max_data_tokens: 8000

  # LLM Parameters
  local_search_temperature: 0
  local_search_top_p: 1
  local_search_n: 1
  local_search_llm_max_gen_completion_tokens: null
  reduce_temperature: 0
  reduce_max_completion_tokens: null

  # Response
  response_type: "Multiple Paragraphs"
```

### Data Requirements

**Parquet Files**:
- ✅ `entities.parquet` - For local search
- ✅ `relationships.parquet` - For graph traversal
- ✅ `communities.parquet` - For community structure
- ✅ `community_reports.parquet` - For primer
- ✅ `text_units.parquet` - For local context
- ⚠️ `covariates.parquet` - Optional claims/facts

**Vector Stores**:
- ✅ `entity_description` embeddings - For entity mapping
- ✅ `text_unit_text` embeddings - For text retrieval

**Full Requirements**:
- All data from both global and local search needed
- Most comprehensive data requirements

---

## Use Cases

### Ideal For:

✅ **Complex multi-faceted questions**
- "Analyze the technology industry from multiple perspectives"
- "Compare different viewpoints on climate policy"
- "Explore the multifaceted nature of healthcare reform"

✅ **Deep exploratory analysis**
- "Investigate the relationships between financial institutions"
- "Examine the evolution of AI research from different angles"
- "Analyze geopolitical dynamics comprehensively"

✅ **Questions requiring both breadth and depth**
- "Provide a comprehensive analysis of topic X"
- "Explore all aspects of issue Y"
- "Give me an in-depth understanding of Z"

✅ **Multi-angle comparative analysis**
- "Compare different stakeholder perspectives on X"
- "Analyze Y from business, technical, and social angles"
- "Contrast approaches to problem Z"

### Not Ideal For:

❌ **Simple factoid questions**
- "What is X?" (use basic search)
- "Define term Y" (use basic search)

❌ **Direct entity queries**
- "How is Company A related to Person B?" (use local search)
- "What are the connections of Entity C?" (use local search)

❌ **Single-perspective questions**
- "What are the main themes?" (use global search)
- "Summarize the dataset" (use global search)

❌ **Cost-sensitive scenarios**
- High-volume queries (use local/basic)
- Budget-constrained applications (use cheaper methods)

❌ **Speed-priority applications**
- Real-time search (use basic/local)
- Interactive quick lookups (use basic)

---

## Comparison with Other Search Methods

| Aspect | DRIFT Search | Global Search | Local Search | Basic Search |
|--------|--------------|---------------|--------------|--------------|
| **Pattern** | Iterative refinement | MAP-REDUCE | Graph-aware RAG | Simple RAG |
| **LLM Calls** | Many (17-130+) | Many (N+1) | 1 | 1 |
| **Phases** | 3 (Primer + Iterate + Reduce) | 2 (MAP + REDUCE) | 1 | 1 |
| **Data Source** | Combined | Community reports | Multi-source | Text units only |
| **Graph Usage** | Full (community + entity) | Community hierarchy | Entity relationships | None |
| **Adaptiveness** | ⭐⭐⭐⭐⭐ High | ⭐ Low | ⭐ Low | ⭐ Low |
| **Speed** | ⭐ Slowest (10-30s) | ⭐⭐ Slow (5-20s) | ⭐⭐⭐⭐ Fast (2-5s) | ⭐⭐⭐⭐⭐ Fastest (1-3s) |
| **Cost** | ⭐ Highest ($0.70+) | ⭐⭐ High ($0.35) | ⭐⭐⭐⭐ Low ($0.04) | ⭐⭐⭐⭐⭐ Lowest ($0.02) |
| **Setup** | ⭐⭐ Complex | ⭐⭐⭐ Moderate | ⭐⭐⭐ Moderate | ⭐⭐⭐⭐⭐ Simple |
| **Quality (complex)** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **Quality (simple)** | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Follow-ups** | ✅ Generated | ❌ | ❌ | ❌ |
| **Iteration** | ✅ Multi-depth | ❌ | ❌ | ❌ |
| **Streaming** | ✅ REDUCE only | ✅ REDUCE only | ✅ | ✅ |

---

## Optimization Tips

### 1. Adjust Depth and Breadth

**Shallow + Narrow** (Faster, cheaper):
```yaml
n_depth: 1                # Single iteration
drift_k_followups: 2      # Fewer actions
# Cost: ~30% reduction
# Speed: ~40% faster
# Quality: Moderate reduction
```

**Default** (Balanced):
```yaml
n_depth: 2
drift_k_followups: 3
# Good balance for most queries
```

**Deep + Wide** (Best quality, expensive):
```yaml
n_depth: 3                # More iterations
drift_k_followups: 5      # More actions per iteration
# Cost: ~2x increase
# Speed: ~2x slower
# Quality: Maximum depth
```

### 2. Tune Primer Folds

**Fewer Folds** (Faster, less diverse):
```yaml
primer_folds: 5           # Half the default
# Pro: Faster primer phase
# Con: Less diverse follow-ups
```

**More Folds** (More diverse, slower):
```yaml
primer_folds: 20          # Double default
# Pro: More diverse perspectives
# Con: Higher cost, slower
```

### 3. Model Selection

**Cost-Optimized** (Claude Haiku):
```yaml
completion_model_id: haiku_completion
# Cost: ~$0.08 per query (90% savings)
# Speed: 3x faster
# Quality: Good for straightforward questions
```

**Quality-Optimized** (Claude 3.5 Sonnet / GPT-4):
```yaml
completion_model_id: sonnet_completion
# Cost: ~$0.99 per query
# Speed: Slower
# Quality: Best for complex analysis
```

**Balanced** (GPT-4o):
```yaml
completion_model_id: gpt4o_completion
# Cost: ~$0.74 per query
# Speed: Fast
# Quality: High
```

### 4. Local Search Tuning

**Smaller Context** (Faster local searches):
```yaml
local_search_top_k_mapped_entities: 5
local_search_top_k_relationships: 5
local_search_max_data_tokens: 4000
# Pro: Faster iterations
# Con: Less local context
```

**Larger Context** (Better local quality):
```yaml
local_search_top_k_mapped_entities: 15
local_search_top_k_relationships: 15
local_search_max_data_tokens: 12000
# Pro: Richer local context
# Con: Slower, more expensive
```

### 5. Dynamic Community Selection

**Disable for Speed** (default):
```yaml
# Not configured - uses static selection
# Pro: Fast, deterministic
# Con: May include irrelevant communities
```

**Enable for Quality**:
```yaml
dynamic_community_selection: true
dynamic_search_threshold: 7
# Pro: Only relevant communities
# Con: +10-100 LLM calls, slower
```

---

## Troubleshooting

### Problem: Extremely high cost

**Symptoms**: Bills > $1 per query

**Solutions**:
1. **Reduce depth**:
   ```yaml
   n_depth: 1              # Single iteration
   drift_k_followups: 2    # Fewer actions
   ```

2. **Reduce primer folds**:
   ```yaml
   primer_folds: 5         # Fewer parallel calls
   ```

3. **Use cheaper model**:
   ```yaml
   completion_model_id: haiku_completion
   # 90% cost reduction
   ```

4. **Consider alternative method**:
   - Use global search for broad questions
   - Use local search for entity-focused queries

5. **Optimize local search**:
   ```yaml
   local_search_max_data_tokens: 4000
   local_search_top_k_mapped_entities: 5
   ```

---

### Problem: Very slow performance

**Symptoms**: Queries take > 30 seconds

**Solutions**:
1. **Reduce iterations**:
   ```yaml
   n_depth: 1              # Cut iterations in half
   ```

2. **Reduce actions**:
   ```yaml
   drift_k_followups: 2    # Fewer parallel searches
   ```

3. **Use faster model**:
   ```yaml
   completion_model_id: gpt4o_completion
   # or haiku_completion for maximum speed
   ```

4. **Reduce primer folds**:
   ```yaml
   primer_folds: 5         # Faster primer
   ```

5. **Optimize local search speed**:
   ```yaml
   local_search_max_data_tokens: 4000
   local_search_top_k_mapped_entities: 5
   ```

---

### Problem: Poor quality answers

**Symptoms**: Answers lack depth, miss key perspectives

**Solutions**:
1. **Increase depth**:
   ```yaml
   n_depth: 3              # More iterations
   drift_k_followups: 5    # More actions
   ```

2. **Increase primer folds**:
   ```yaml
   primer_folds: 15        # More diverse starting points
   ```

3. **Enable dynamic selection**:
   ```yaml
   dynamic_community_selection: true
   # Ensures relevant communities
   ```

4. **Use better model**:
   ```yaml
   completion_model_id: sonnet_completion
   # Or opus for maximum quality
   ```

5. **Improve local search quality**:
   ```yaml
   local_search_top_k_mapped_entities: 15
   local_search_max_data_tokens: 12000
   ```

6. **Check data quality**:
   - Verify community reports are informative
   - Ensure entity descriptions are detailed
   - Check text unit quality

---

### Problem: JSON parse errors

**Symptoms**: Errors parsing LLM responses

**Solutions**:
1. **Use newer model**:
   - GPT-4o has excellent JSON support
   - Claude 3.5 Sonnet has structured output
   - Avoid older models

2. **Check prompts**:
   - Ensure LOCAL_SEARCH_SYSTEM_PROMPT requests JSON
   - Verify PrimerResponse schema is clear

3. **Increase temperature slightly**:
   ```yaml
   local_search_temperature: 0.1
   # May improve format compliance
   ```

---

### Problem: No follow-up queries generated

**Symptoms**: Primer returns empty follow_up_queries

**Solutions**:
1. **Check primer prompt**:
   - Verify DRIFT_PRIMER_PROMPT clearly requests follow-ups
   - Ensure minimum count specified (5+)

2. **Increase primer folds**:
   ```yaml
   primer_folds: 15
   # More chances to generate follow-ups
   ```

3. **Check community reports**:
   - Verify reports have content
   - Ensure reports are relevant to query

4. **Use better model**:
   - Some models better at generating questions
   - Try Claude 3.5 Sonnet or GPT-4

---

### Problem: Memory issues

**Symptoms**: Out of memory errors

**Solutions**:
1. **Reduce QueryState size**:
   ```yaml
   n_depth: 1
   drift_k_followups: 2
   # Fewer actions in memory
   ```

2. **Reduce local search context**:
   ```yaml
   local_search_max_data_tokens: 4000
   ```

3. **Process serially** (if needed):
   - Reduce parallel primer folds
   - Execute actions sequentially

4. **Clear state between queries**:
   - Create new DRIFTSearch instance per query
   - Don't reuse QueryState

---

## Code References

### Key Files

- **Search Engine**: `packages/graphrag/graphrag/query/structured_search/drift_search/search.py:37-465`
- **Primer**: `packages/graphrag/graphrag/query/structured_search/drift_search/primer.py:123-227`
- **Action**: `packages/graphrag/graphrag/query/structured_search/drift_search/action.py:15-155`
- **QueryState**: `packages/graphrag/graphrag/query/structured_search/drift_search/state.py`
- **Context Builder**: `packages/graphrag/graphrag/query/structured_search/drift_search/drift_context.py`
- **Configuration**: `packages/graphrag/graphrag/config/models/drift_search_config.py:11-123`

### Key Classes

- `DRIFTSearch`: Main search orchestrator
- `DRIFTPrimer`: Primer phase handler
- `DriftAction`: Action/query/answer container
- `QueryState`: State management for iterative search
- `DRIFTSearchContextBuilder`: Combined context builder
- `PrimerResponse`: Structured primer output
- `LocalSearch`: Reused for action searches

### Key Functions

- `DRIFTSearch.search()`: Main entry point
- `DRIFTSearch._search_step()`: Single iteration
- `DRIFTSearch._reduce_response()`: Final synthesis
- `DRIFTPrimer.search()`: Primer execution
- `DRIFTPrimer.decompose_query()`: Single fold processing
- `DriftAction.search()`: Execute local search for action
- `QueryState.rank_incomplete_actions()`: Action prioritization

---

## Viewing the Diagram

### Option 1: PlantUML Online
1. Go to https://www.plantuml.com/plantuml/uml/
2. Paste contents of `drift_search_activity.puml`
3. Click "Submit"

### Option 2: VS Code Extension
1. Install "PlantUML" extension
2. Open `drift_search_activity.puml`
3. Press `Alt+D` to preview

### Option 3: Command Line
```bash
# Install PlantUML
brew install plantuml  # macOS
apt-get install plantuml  # Ubuntu

# Generate PNG
plantuml drift_search_activity.puml

# Generate SVG
plantuml -tsvg drift_search_activity.puml
```

---

## Related Documentation

- **Query Analysis**: See `query_analysis.md` for comprehensive overview of all search methods
- **Local Search Diagram**: See `local_search_activity.puml` - DRIFT reuses local search
- **Global Search Diagram**: See `global_search_activity.puml` - DRIFT uses similar primer concept
- **Basic Search Diagram**: See `basic_search_activity.puml` for simplest comparison
- **GraphRAG Docs**: https://microsoft.github.io/graphrag/

---

**Last Updated**: 2026-01-30
**Diagram Version**: 1.0
**Status**: Complete ✅
