# Current LLM Usage in GraphRAG - Analysis

**Date**: 2026-01-30
**Status**: Complete

---

## Executive Summary

GraphRAG currently uses **LiteLLM** as its LLM abstraction layer, with **OpenAI as the exclusive provider** for both text generation and embeddings. This architecture choice is excellent news for adding multi-provider support, as LiteLLM already supports 100+ providers including Claude (Anthropic).

**Key Discovery**: The foundation for multi-provider support already exists! Adding Claude support requires configuration changes and documentation, not a major architectural overhaul.

---

## Current Architecture

### LLM Abstraction Layer

GraphRAG uses a well-designed abstraction with two primary interfaces:

```
graphrag_llm Package (packages/graphrag-llm/)
├── completion/
│   ├── completion.py           # Abstract LLMCompletion interface
│   ├── lite_llm_completion.py  # LiteLLM implementation
│   └── completion_factory.py   # Factory pattern for instantiation
└── embedding/
    ├── embedding.py            # Abstract LLMEmbedding interface
    ├── lite_llm_embedding.py   # LiteLLM implementation
    └── embedding_factory.py    # Factory pattern for instantiation
```

**Implementation**: Both completion and embedding are implemented using [LiteLLM](https://docs.litellm.ai/), a unified API that supports 100+ LLM providers.

**Key Code Locations**:
- Completion: `packages/graphrag-llm/graphrag_llm/completion/lite_llm_completion.py:44`
- Embedding: `packages/graphrag-llm/graphrag_llm/embedding/lite_llm_embedding.py:34`
- Model Config: `packages/graphrag-llm/graphrag_llm/config/model_config.py:19`

---

## Current Provider Configuration

### Default Models

From `packages/graphrag/graphrag/config/defaults.py:31-37`:

```python
DEFAULT_COMPLETION_MODEL = "gpt-4.1"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
DEFAULT_MODEL_PROVIDER = "openai"
```

### Configuration Structure

From `packages/graphrag-llm/graphrag_llm/config/model_config.py:19`:

```python
class ModelConfig(BaseModel):
    model_provider: str  # e.g., "openai", "azure"
    model: str           # e.g., "gpt-4o", "gpt-3.5-turbo"
    api_key: str | None
    api_base: str | None
    api_version: str | None
    auth_method: AuthMethod  # api_key or azure_managed_identity
    call_args: dict[str, Any]  # Additional parameters
```

**LiteLLM Model ID Format**: `{model_provider}/{model}`

Example from `packages/graphrag-llm/graphrag_llm/completion/lite_llm_completion.py:250`:
```python
model = f"{model_provider}/{model}"  # e.g., "openai/gpt-4o"
```

### User Configuration (YAML)

From `packages/graphrag/graphrag/config/init_content.py:19-35`:

```yaml
completion_models:
  default_completion_model:
    model_provider: openai
    model: gpt-4o
    auth_method: api_key
    api_key: ${GRAPHRAG_API_KEY}
    retry:
      type: exponential_backoff

embedding_models:
  default_embedding_model:
    model_provider: openai
    model: text-embedding-3-large
    auth_method: api_key
    api_key: ${GRAPHRAG_API_KEY}
    retry:
      type: exponential_backoff
```

---

## LLM Operations in GraphRAG

### 1. Text Generation (Completion) Operations

#### 1.1 Entity Extraction
**File**: `packages/graphrag/graphrag/index/operations/extract_graph/extract_graph.py:22`

**Purpose**: Extract entities and relationships from text chunks

**LLM Call**:
```python
async def extract_graph(
    text_units: pd.DataFrame,
    model: "LLMCompletion",  # ← Uses completion model
    prompt: str,
    entity_types: list[str],
    max_gleanings: int,
    ...
)
```

**Prompts**: `prompts/extract_graph.txt`

**Token Estimates** (per text chunk ~1200 tokens):
- Input: 1200 tokens (text) + 500 tokens (prompt) = 1700 tokens
- Output: 500-1000 tokens (JSON structured output with entities/relationships)

**Call Frequency**: Once per text chunk (typically 100-10,000 chunks per index)

**Configuration**:
```yaml
extract_graph:
  completion_model_id: default_completion_model
  prompt: "prompts/extract_graph.txt"
  max_gleanings: 1
```

---

#### 1.2 Claims Extraction (Optional)
**File**: `packages/graphrag/graphrag/index/operations/extract_covariates/extract_covariates.py`

**Purpose**: Extract claims/facts from text chunks

**Token Estimates**:
- Input: 1200 tokens (text) + 400 tokens (prompt) = 1600 tokens
- Output: 300-500 tokens (claims)

**Call Frequency**: Once per text chunk (if enabled)

**Configuration**:
```yaml
extract_claims:
  enabled: false  # Disabled by default
  completion_model_id: default_completion_model
  max_gleanings: 1
```

---

#### 1.3 Description Summarization
**File**: `packages/graphrag/graphrag/index/operations/summarize_descriptions/summarize_descriptions.py`

**Purpose**: Summarize entity descriptions when merging duplicates

**Token Estimates**:
- Input: 500-2000 tokens (multiple descriptions)
- Output: 200-500 tokens (summary)

**Call Frequency**: Once per unique entity (~100-10,000 entities)

**Configuration**:
```yaml
summarize_descriptions:
  completion_model_id: default_completion_model
  prompt: "prompts/summarize_descriptions.txt"
  max_length: 500
```

---

#### 1.4 Community Report Generation
**Files**:
- `packages/graphrag/graphrag/index/workflows/create_community_reports.py`
- `packages/graphrag/graphrag/index/operations/summarize_communities/summarize_communities.py`

**Purpose**: Generate comprehensive reports for each detected community

**LLM Call**:
```python
async def summarize_communities(
    communities: pd.DataFrame,
    model: "LLMCompletion",  # ← Uses completion model
    ...
)
```

**Token Estimates** (per community):
- Input: 2000-8000 tokens (community context: entities, relationships, descriptions)
- Output: 500-2000 tokens (structured report)

**Call Frequency**: Once per community (typically 50-500 communities)

**Prompts**:
- `prompts/community_report_graph.txt` (for graph-based context)
- `prompts/community_report_text.txt` (for text-based context)

**Configuration**:
```yaml
community_reports:
  completion_model_id: default_completion_model
  graph_prompt: "prompts/community_report_graph.txt"
  text_prompt: "prompts/community_report_text.txt"
  max_length: 2000
  max_input_length: 8000
```

---

#### 1.5 Query Operations

GraphRAG supports 4 query methods, all using completion models:

##### Local Search
**File**: `packages/graphrag/graphrag/query/structured_search/local_search/`

**Purpose**: Answer questions using local graph context

**Token Estimates**:
- Input: 2000-12000 tokens (entities, relationships, text units)
- Output: 500-2000 tokens (answer)

**Configuration**:
```yaml
local_search:
  completion_model_id: default_completion_model
  prompt: "prompts/local_search_system_prompt.txt"
  max_context_tokens: 12000
```

---

##### Global Search
**File**: `packages/graphrag/graphrag/query/structured_search/global_search/`

**Purpose**: Answer questions using community reports (map-reduce pattern)

**Token Estimates**:
- Map phase: 1000-4000 tokens input per community
- Reduce phase: 5000-20000 tokens input (all map results)
- Output: 500-2000 tokens (final answer)

**Configuration**:
```yaml
global_search:
  completion_model_id: default_completion_model
  map_prompt: "prompts/global_search_map_system_prompt.txt"
  reduce_prompt: "prompts/global_search_reduce_system_prompt.txt"
```

---

##### DRIFT Search
**File**: `packages/graphrag/graphrag/query/structured_search/drift_search/`

**Purpose**: Dynamic reasoning and iterative follow-up traversal

**Token Estimates**: Variable (iterative, 3+ rounds)

**Configuration**:
```yaml
drift_search:
  completion_model_id: default_completion_model
  prompt: "prompts/drift_search_system_prompt.txt"
  reduce_prompt: "prompts/drift_search_reduce_prompt.txt"
```

---

##### Basic Search
**File**: Query factory

**Purpose**: Simple RAG without graph structure

**Token Estimates**:
- Input: 2000-10000 tokens
- Output: 500-1500 tokens

**Configuration**:
```yaml
basic_search:
  completion_model_id: default_completion_model
  prompt: "prompts/basic_search_system_prompt.txt"
```

---

### 2. Embedding Operations

#### 2.1 Text Embeddings
**File**: `packages/graphrag/graphrag/index/workflows/generate_text_embeddings.py`

**Purpose**: Generate vector embeddings for semantic search

**Embedding Targets**:
1. **Entity descriptions** - For entity similarity search
2. **Community summaries** - For community similarity search
3. **Text units** - For text chunk retrieval

**LLM Call**:
```python
async def embed_text(
    text_units: pd.DataFrame,
    model: "LLMEmbedding",  # ← Uses embedding model
    embedding_column_names: list[str],
    ...
)
```

**Token Estimates** (per batch):
- Input: 16 texts × ~200 tokens = 3200 tokens per batch
- Output: 16 embeddings × 1536 dimensions = 24,576 floats

**Call Frequency**:
- Once per entity description (~1,000-10,000)
- Once per community summary (~50-500)
- Once per text unit (~100-10,000)

**Configuration**:
```yaml
embed_text:
  embedding_model_id: default_embedding_model
  batch_size: 16
  batch_max_tokens: 8191
  names:
    - entity_description
    - community_summary
    - text_unit
```

**Batch Processing**: Embeddings are batched for efficiency
- Default batch size: 16 texts
- Max tokens per batch: 8191
- From `packages/graphrag/graphrag/config/defaults.py:122-124`

---

## Token Usage Analysis

### Example: Indexing 1000 Documents

Assumptions:
- 1000 documents
- 10,000 text chunks (avg 1200 tokens each)
- 5,000 unique entities
- 200 communities

#### Completion Tokens (Text Generation)

| Operation | Calls | Input Tokens (each) | Output Tokens (each) | Total Input | Total Output |
|-----------|-------|-------------------|---------------------|-------------|--------------|
| Entity Extraction | 10,000 | 1,700 | 750 | 17,000,000 | 7,500,000 |
| Description Summarization | 5,000 | 1,000 | 350 | 5,000,000 | 1,750,000 |
| Community Reports | 200 | 5,000 | 1,500 | 1,000,000 | 300,000 |
| **Total** | **15,200** | - | - | **23,000,000** | **9,550,000** |

**Cost Estimate (OpenAI GPT-4-turbo)**:
- Input: 23M tokens × $10/1M = $230
- Output: 9.55M tokens × $30/1M = $286.50
- **Total**: **$516.50**

---

#### Embedding Tokens

| Target | Count | Tokens (each) | Total Tokens |
|--------|-------|--------------|--------------|
| Entity Descriptions | 5,000 | 100 | 500,000 |
| Community Summaries | 200 | 400 | 80,000 |
| Text Units | 10,000 | 200 | 2,000,000 |
| **Total** | **15,200** | - | **2,580,000** |

**Cost Estimate (OpenAI text-embedding-3-large)**:
- 2.58M tokens × $0.13/1M = **$0.34**

---

#### Total Indexing Cost (OpenAI)

```
Completion:  $516.50
Embeddings:  $0.34
─────────────────────
Total:       $516.84
```

---

## Current Implementation Details

### 1. LiteLLM Integration

**Key Code**: `packages/graphrag-llm/graphrag_llm/completion/lite_llm_completion.py:276-278`

```python
response = litellm.completion(
    model=f"{model_provider}/{model}",  # e.g., "openai/gpt-4o"
    api_key=model_config.api_key,
    api_base=model_config.api_base,
    messages=messages,
    **kwargs
)
```

**Embedding**: `packages/graphrag-llm/graphrag_llm/embedding/lite_llm_embedding.py:188`

```python
response = litellm.embedding(
    model=f"{model_provider}/{model}",  # e.g., "openai/text-embedding-3-large"
    input=input_texts,
    **kwargs
)
```

---

### 2. Middleware Pipeline

GraphRAG wraps LLM calls with middleware for:
- **Caching**: Avoid redundant LLM calls
- **Rate Limiting**: Respect provider rate limits
- **Retries**: Exponential backoff for transient failures
- **Metrics**: Track token usage, latency, errors
- **Logging**: Debug LLM interactions

**Code**: `packages/graphrag-llm/graphrag_llm/middleware/with_middleware_pipeline.py`

---

### 3. Supported OpenAI Models

From LiteLLM documentation and GraphRAG defaults:

#### Completion Models
- `gpt-4o` (recommended, cost-effective)
- `gpt-4.1` (default in v3.x)
- `gpt-4-turbo`
- `gpt-3.5-turbo` (budget option)
- Azure OpenAI equivalents

#### Embedding Models
- `text-embedding-3-large` (default, 3072 dims)
- `text-embedding-3-small` (cost-effective, 1536 dims)
- `text-embedding-ada-002` (legacy)

---

## Current Limitations

### 1. Provider Lock-in
**Issue**: While LiteLLM supports 100+ providers, GraphRAG documentation and examples only show OpenAI.

**Impact**:
- Users assume OpenAI is required
- No guidance for using alternative providers
- Configuration examples don't show multi-provider setup

**Evidence**: Default configuration (`packages/graphrag/graphrag/config/init_content.py:19-35`) only shows OpenAI.

---

### 2. Embeddings Requirement
**Issue**: Claude (Anthropic) does not provide embedding models.

**Impact**:
- Cannot use Claude exclusively
- Must combine Claude (completion) with another provider (embeddings)
- Requires multi-provider configuration

**Current Workaround**: Use OpenAI/Voyage/Cohere for embeddings, or add SentenceTransformer support.

---

### 3. Model-Specific Prompts
**Issue**: Prompts may be optimized for OpenAI models.

**Impact**:
- Claude may interpret prompts differently
- Structured output formatting may differ
- JSON extraction reliability unclear

**Mitigation**: Test prompts with Claude, adjust if needed.

---

### 4. Cost Transparency
**Issue**: No built-in cost estimation for different providers.

**Impact**:
- Users don't know potential cost savings with Claude
- No guidance on cost-optimized model selection

**Example**: Using Claude 3 Haiku for extraction could reduce costs by 90%+.

---

## Key Findings

### ✅ Strengths

1. **LiteLLM Integration**
   - Already supports 100+ providers including Claude
   - Unified API for all providers
   - Drop-in replacement architecture

2. **Clean Abstraction**
   - `LLMCompletion` and `LLMEmbedding` interfaces
   - Factory pattern for instantiation
   - Middleware pipeline for cross-cutting concerns

3. **Flexible Configuration**
   - Model-specific settings
   - Per-operation model selection
   - Support for multiple model instances

4. **Production Features**
   - Caching, rate limiting, retries
   - Metrics and monitoring
   - Azure managed identity support

---

### ⚠️ Gaps for Multi-Provider Support

1. **Documentation**
   - No examples for non-OpenAI providers
   - Configuration guide only shows OpenAI
   - No provider comparison or selection guide

2. **Embeddings**
   - No local/open-source embedding option
   - Requires separate provider if using Claude
   - No SentenceTransformer support

3. **Validation**
   - Prompts not tested with Claude
   - Structured output format compatibility unknown
   - Quality comparison needed

4. **Cost Optimization**
   - No cost estimation tooling
   - No guidance on model selection per operation
   - No per-operation provider mixing examples

---

## Recommendations for Adding Claude Support

### 1. Configuration Changes (Minimal Effort) ✅

**Action**: Update documentation to show Claude configuration examples.

**Effort**: Low (1-2 days)

**Example**:
```yaml
completion_models:
  claude_completion:
    model_provider: anthropic  # ← Change to anthropic
    model: claude-3-5-sonnet-20241022  # ← Claude model
    api_key: ${ANTHROPIC_API_KEY}
    retry:
      type: exponential_backoff
```

**Why This Works**: LiteLLM already maps `anthropic/claude-3-5-sonnet-20241022` to Anthropic API!

---

### 2. Add SentenceTransformer Support (Medium Effort) ⚠️

**Action**: Implement `SentenceTransformerEmbedding` class for local embeddings.

**Effort**: Medium (1-2 weeks)

**Benefits**:
- Zero cost embeddings
- Full privacy (data never leaves machine)
- No API rate limits
- Offline capability

**Implementation**:
```python
class SentenceTransformerEmbedding(LLMEmbedding):
    def __init__(self, model_name: str, device: str = "cuda"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.device = device

    def embedding(self, input: list[str]) -> LLMEmbeddingResponse:
        embeddings = self.model.encode(input, device=self.device)
        return LLMEmbeddingResponse(data=embeddings)
```

**Configuration**:
```yaml
embedding_models:
  local_embedding:
    type: sentence_transformer  # ← New type
    model: BAAI/bge-large-en-v1.5
    device: cuda  # or cpu, mps
```

---

### 3. Prompt Validation (Medium Effort) ⚠️

**Action**: Test all GraphRAG prompts with Claude 3.5 Sonnet.

**Effort**: Medium (1 week)

**Testing Needed**:
- Entity extraction accuracy
- JSON output format compliance
- Relationship extraction quality
- Community report quality
- Query response quality

**Methodology**:
- Use same test dataset for OpenAI and Claude
- Compare extracted entities/relationships
- Measure precision/recall
- Evaluate report quality

---

### 4. Cost Optimization Guide (Low Effort) ✅

**Action**: Document cost-optimized configurations.

**Effort**: Low (2-3 days)

**Example Guide**:
```yaml
# Ultra-cheap extraction with Claude 3 Haiku
extract_graph:
  completion_model_id: haiku_completion

completion_models:
  haiku_completion:
    model_provider: anthropic
    model: claude-3-haiku-20240307  # $0.25/$1.25 per 1M tokens

# Quality reports with Claude 3.5 Sonnet
community_reports:
  completion_model_id: sonnet_completion

completion_models:
  sonnet_completion:
    model_provider: anthropic
    model: claude-3-5-sonnet-20241022  # $3/$15 per 1M tokens

# Free local embeddings
embed_text:
  embedding_model_id: local_embedding

embedding_models:
  local_embedding:
    type: sentence_transformer
    model: BAAI/bge-large-en-v1.5
    device: cuda

# Cost: ~$15 for 1000 docs (vs $517 with OpenAI)
# Savings: 97.1%
```

---

## Next Steps

1. **Document 02**: Analyze Claude capabilities and API compatibility
2. **Document 03**: Design multi-provider configuration patterns
3. **Document 04**: Benchmark Claude vs OpenAI quality and performance
4. **Document 05**: Cost-benefit analysis and GO/NO-GO decision
5. **Document 06**: Implementation plan for SentenceTransformer support
6. **Document 07**: User adoption and migration strategy

---

## Appendix: Code References

### Key Files for Multi-Provider Support

| File | Lines | Purpose |
|------|-------|---------|
| `packages/graphrag-llm/graphrag_llm/completion/lite_llm_completion.py` | 44-315 | LiteLLM completion implementation |
| `packages/graphrag-llm/graphrag_llm/embedding/lite_llm_embedding.py` | 34-199 | LiteLLM embedding implementation |
| `packages/graphrag-llm/graphrag_llm/config/model_config.py` | 19-112 | Model configuration schema |
| `packages/graphrag/graphrag/config/defaults.py` | 31-37 | Default model settings |
| `packages/graphrag/graphrag/config/init_content.py` | 19-135 | User-facing configuration template |

### LLM Usage Locations

| Operation | File | Function |
|-----------|------|----------|
| Entity Extraction | `packages/graphrag/graphrag/index/operations/extract_graph/extract_graph.py` | `extract_graph:22` |
| Claims Extraction | `packages/graphrag/graphrag/index/operations/extract_covariates/extract_covariates.py` | `extract_covariates` |
| Description Summarization | `packages/graphrag/graphrag/index/operations/summarize_descriptions/summarize_descriptions.py` | `summarize_descriptions` |
| Community Reports | `packages/graphrag/graphrag/index/operations/summarize_communities/summarize_communities.py` | `summarize_communities` |
| Text Embeddings | `packages/graphrag/graphrag/index/workflows/generate_text_embeddings.py` | `generate_text_embeddings` |
| Local Search | `packages/graphrag/graphrag/query/structured_search/local_search/` | `LocalSearch` |
| Global Search | `packages/graphrag/graphrag/query/structured_search/global_search/` | `GlobalSearch` |
| DRIFT Search | `packages/graphrag/graphrag/query/structured_search/drift_search/` | `DRIFTSearch` |

---

**Document Status**: Complete ✅
**Next Document**: `02_claude_capabilities.md` - Evaluate Claude API features and compatibility
