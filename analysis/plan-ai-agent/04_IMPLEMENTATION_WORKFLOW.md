# Implementation Workflow Guide

**Document**: 04_IMPLEMENTATION_WORKFLOW.md
**Status**: Ready for Execution
**Purpose**: Step-by-step execution guide for AI agents

---

## Overview

This document provides the detailed workflow that Implementation and Assessment AI Agents should follow during the Claude/SentenceTransformer implementation. Each step is concrete and actionable.

---

## Prerequisites

### Environment Setup

```bash
# Navigate to repository
cd /Users/aselaillayapparachchi/code/GraphRAG/Microsoft/graphrag-ms

# Verify Python version
python --version  # Should be 3.11-3.13

# Install dependencies
uv sync

# Verify tools work
uv run poe check  # Should run without errors
```

### Required Access
- [ ] Repository access (read/write)
- [ ] Anthropic API key (for testing Claude)
- [ ] OpenAI API key (for baseline comparisons)

### Knowledge Requirements
- [ ] Read `/analysis/claude-support/` documents (context)
- [ ] Read `/analysis/plan-ai-agent/README.md` (overview)
- [ ] Understand TDD principles (01_TDD_STRATEGY.md)
- [ ] Know coding standards (02_CODING_STANDARDS.md)

---

## General Workflow Pattern

Every task follows this pattern:

```
┌─────────────────────────────────────────────────┐
│ 1. READ TASK SPECIFICATION                     │
│    └─ Understand requirements and acceptance   │
│                                                 │
│ 2. WRITE TESTS FIRST (TDD)                     │
│    └─ Define expected behavior in tests        │
│                                                 │
│ 3. IMPLEMENT FEATURE                           │
│    └─ Write minimal code to pass tests         │
│                                                 │
│ 4. RUN LOCAL CHECKS                            │
│    └─ uv run poe check && uv run poe test_unit │
│                                                 │
│ 5. REQUEST ASSESSMENT                          │
│    └─ Signal readiness for review              │
│                                                 │
│ 6. ITERATE ON FEEDBACK                         │
│    └─ Address assessment feedback              │
│                                                 │
│ 7. GET APPROVAL                                │
│    └─ Move to next task                        │
└─────────────────────────────────────────────────┘
```

---

## Phase 1: Documentation & Claude Support (Weeks 1-2)

### Task 1.1: Create Claude Configuration Examples

**Duration**: 2-3 hours
**Complexity**: Low

#### Step 1: Read Requirements
```bash
# Read source documentation
cat analysis/claude-support/06_implementation_plan.md | grep -A 50 "Task 1.1"
```

**Requirements**:
- Create 3 example YAML configurations
- Must be syntactically valid
- Must follow existing config patterns
- Include comments explaining options

#### Step 2: Create Files

**File 1**: `docs/examples/claude-basic.yaml`
```yaml
# Basic Claude configuration for GraphRAG
# Replace OpenAI with Claude for text generation

completion_models:
  default_completion_model:
    model_provider: anthropic  # Use Anthropic's Claude
    model: claude-3-5-sonnet-20241022  # Latest Sonnet model
    api_key: ${ANTHROPIC_API_KEY}  # Set in environment

    # Optional: Configure retry behavior
    retry:
      type: exponential_backoff
      max_retries: 5
      max_wait: 60.0

embedding_models:
  default_embedding_model:
    model_provider: openai  # Keep OpenAI for embeddings
    model: text-embedding-3-small
    api_key: ${OPENAI_API_KEY}

# Expected cost savings: 70% compared to GPT-4 Turbo
# Context window: 200K tokens (vs GPT-4's 128K)
```

**File 2**: `docs/examples/claude-optimized.yaml`
```yaml
# Cost-optimized Claude configuration
# Uses Haiku for extraction, Sonnet for quality work

completion_models:
  # Fast, cheap extraction
  haiku_extraction:
    model_provider: anthropic
    model: claude-3-haiku-20240307  # $0.25/$1.25 per 1M tokens
    api_key: ${ANTHROPIC_API_KEY}
    rate_limit:
      type: sliding_window
      max_requests_per_minute: 1000

  # Quality reports and summaries
  sonnet_quality:
    model_provider: anthropic
    model: claude-3-5-sonnet-20241022  # $3/$15 per 1M tokens
    api_key: ${ANTHROPIC_API_KEY}

embedding_models:
  default_embedding_model:
    model_provider: openai
    model: text-embedding-3-small
    api_key: ${OPENAI_API_KEY}

# Workflow-specific model assignments
extract_graph:
  completion_model_id: haiku_extraction

summarize_descriptions:
  completion_model_id: sonnet_quality

community_reports:
  completion_model_id: sonnet_quality

# Expected cost savings: 95% compared to GPT-4 Turbo
```

**File 3**: `docs/examples/claude-local-embeddings.yaml`
```yaml
# Claude + Local Embeddings (Maximum cost savings + Privacy)
# Zero cost for embeddings, 95% cost reduction overall

completion_models:
  default_completion_model:
    model_provider: anthropic
    model: claude-3-haiku-20240307
    api_key: ${ANTHROPIC_API_KEY}

embedding_models:
  default_embedding_model:
    type: sentence_transformer  # Local embeddings
    model: BAAI/bge-large-en-v1.5  # High-quality model
    device: cuda  # or cpu, mps (Mac M1/M2)
    batch_size: 32
    normalize_embeddings: true

# Benefits:
# - Zero embedding costs (runs locally)
# - Full data privacy (never leaves your machine)
# - No rate limits
# - Offline capable
# Expected cost: ~$10-15 per 1000 docs (vs $330 with OpenAI)
```

#### Step 3: Validate Syntax

```bash
# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('docs/examples/claude-basic.yaml'))"
python -c "import yaml; yaml.safe_load(open('docs/examples/claude-optimized.yaml'))"
python -c "import yaml; yaml.safe_load(open('docs/examples/claude-local-embeddings.yaml'))"
```

#### Step 4: Manual Testing (Optional)

```bash
# Test that configuration loads (if time permits)
# Copy example to test location
cp docs/examples/claude-basic.yaml /tmp/test-config.yaml

# Try to load it (dry-run)
uv run python -c "
from graphrag.config import load_config
config = load_config('/tmp/test-config.yaml')
print(f'Config loaded: {config}')
"
```

#### Step 5: Request Assessment

```
ASSESSMENT REQUEST
Task: 1.1 - Create Claude Configuration Examples
Phase: 1 (Documentation)
Files Created:
  - docs/examples/claude-basic.yaml
  - docs/examples/claude-optimized.yaml
  - docs/examples/claude-local-embeddings.yaml
Validation: YAML syntax validated
Testing: Manual load test passed
Ready for Review: Yes
```

---

### Task 1.2: Write LLM Provider Guide

**Duration**: 4-6 hours
**Complexity**: Medium

#### Step 1: Create Document Structure

**File**: `docs/configuration/llm-providers.md`

```markdown
# LLM Provider Configuration

GraphRAG supports multiple LLM providers for both text generation and embeddings.

## Table of Contents
1. [Overview](#overview)
2. [Supported Providers](#supported-providers)
3. [Configuration Patterns](#configuration-patterns)
4. [Provider Comparison](#provider-comparison)
5. [Cost Optimization](#cost-optimization)
6. [Troubleshooting](#troubleshooting)

## Overview

GraphRAG uses LiteLLM as an abstraction layer, enabling support for 100+ LLM providers
with a unified configuration interface.

### Architecture

```
GraphRAG Application
    ↓
LiteLLM Abstraction Layer
    ↓
┌─────────────┬──────────────┬──────────────┐
│   OpenAI    │   Anthropic  │   Azure      │
│   Claude    │    Gemini    │   Ollama     │
└─────────────┴──────────────┴──────────────┘
```

## Supported Providers

### Text Generation (Completions)

| Provider | Models | Context | Cost Range | Notes |
|----------|--------|---------|------------|-------|
| OpenAI | GPT-4o, GPT-4 Turbo, GPT-3.5 | 128K | $0.50-30/1M | Default, battle-tested |
| Anthropic Claude | 3.5 Sonnet, 3 Opus, 3 Haiku | 200K | $0.25-75/1M | Best reasoning, longer context |
| Azure OpenAI | Same as OpenAI | 128K | Custom | Enterprise support |
| Gemini | Pro, Flash | 1M | $0.35-7/1M | Longest context |

### Embeddings

| Provider | Models | Dimensions | Cost | Notes |
|----------|--------|------------|------|-------|
| OpenAI | text-embedding-3-small/large | 512-3072 | $0.02-0.13/1M | Default |
| Voyage AI | voyage-large-2 | 1536 | $0.12/1M | Optimized for RAG |
| Cohere | embed-english-v3.0 | 1024 | $0.10/1M | Multilingual |
| SentenceTransformer | 100+ models | 384-1024 | FREE | Local, private |

## Configuration Patterns

### Pattern 1: OpenAI Only (Default)

[... detailed examples ...]

### Pattern 2: Claude + OpenAI Embeddings

[... detailed examples ...]

### Pattern 3: Claude + Local Embeddings

[... detailed examples ...]

## Provider Comparison

[... comparison tables ...]

## Cost Optimization

[... strategies and calculations ...]

## Troubleshooting

[... common issues and solutions ...]
```

#### Step 2: Fill in Sections

Work through each section, providing:
- Clear explanations
- Copy-paste ready examples
- Tables comparing options
- Troubleshooting guidance

#### Step 3: Request Assessment

```
ASSESSMENT REQUEST
Task: 1.2 - Write LLM Provider Guide
Phase: 1 (Documentation)
Files Created:
  - docs/configuration/llm-providers.md (~2000 words)
Validation: Markdown syntax valid
Completeness: All sections filled
Ready for Review: Yes
```

---

## Phase 2: SentenceTransformer Implementation (Weeks 3-4)

### Task 2.1: Implement SentenceTransformerEmbedding.__init__

**Duration**: 3-4 hours
**Complexity**: Medium-High

#### Step 1: Write Tests First (TDD - RED)

**File**: `tests/unit/graphrag_llm/embedding/test_sentence_transformer_embedding.py`

```python
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Unit tests for SentenceTransformerEmbedding."""

import pytest
from unittest.mock import Mock, patch
from graphrag_llm.config import ModelConfig


class TestSentenceTransformerEmbedding:
    """Test suite for SentenceTransformerEmbedding initialization."""

    @pytest.fixture
    def mock_config(self):
        """Fixture providing a mock ModelConfig."""
        return ModelConfig(
            model="all-MiniLM-L6-v2",
            model_provider="sentence_transformer",
            type="sentence_transformer",
        )

    @pytest.fixture
    def mock_dependencies(self):
        """Fixture providing mocked dependencies."""
        return {
            "tokenizer": Mock(),
            "metrics_store": Mock(),
            "cache_key_creator": Mock(),
        }

    @patch("graphrag_llm.embedding.sentence_transformer_embedding.SentenceTransformer")
    def test_initialization_with_default_device(
        self, mock_st_class, mock_config, mock_dependencies
    ):
        """Test SentenceTransformerEmbedding initializes with default device detection.

        Arrange: Mock SentenceTransformer and device detection
        Act: Create SentenceTransformerEmbedding instance
        Assert: Device is detected and model is loaded
        """
        # Arrange
        mock_model = Mock()
        mock_st_class.return_value = mock_model

        # Act
        from graphrag_llm.embedding.sentence_transformer_embedding import (
            SentenceTransformerEmbedding,
        )

        embedding = SentenceTransformerEmbedding(
            model_id="test/model",
            model_config=mock_config,
            **mock_dependencies,
        )

        # Assert
        assert embedding is not None
        assert embedding._device in ["cuda", "cpu", "mps"]
        assert embedding._model == mock_model
        mock_st_class.assert_called_once()

    @patch("graphrag_llm.embedding.sentence_transformer_embedding.SentenceTransformer")
    def test_initialization_with_explicit_cuda(
        self, mock_st_class, mock_config, mock_dependencies
    ):
        """Test initialization with explicit CUDA device."""
        # Arrange
        mock_model = Mock()
        mock_st_class.return_value = mock_model

        # Act
        from graphrag_llm.embedding.sentence_transformer_embedding import (
            SentenceTransformerEmbedding,
        )

        embedding = SentenceTransformerEmbedding(
            model_id="test/model",
            model_config=mock_config,
            device="cuda",
            **mock_dependencies,
        )

        # Assert
        assert embedding._device == "cuda"
        mock_st_class.assert_called_with("all-MiniLM-L6-v2", device="cuda")

    def test_initialization_raises_import_error_when_package_missing(
        self, mock_config, mock_dependencies
    ):
        """Test that ImportError is raised with helpful message when package missing."""
        # Arrange
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            # Act & Assert
            with pytest.raises(ImportError) as exc_info:
                from graphrag_llm.embedding.sentence_transformer_embedding import (
                    SentenceTransformerEmbedding,
                )

                SentenceTransformerEmbedding(
                    model_id="test/model",
                    model_config=mock_config,
                    **mock_dependencies,
                )

            assert "sentence-transformers not installed" in str(exc_info.value)
            assert "pip install sentence-transformers" in str(exc_info.value)
```

#### Step 2: Run Tests (Should FAIL - RED)

```bash
uv run pytest tests/unit/graphrag_llm/embedding/test_sentence_transformer_embedding.py -v

# Expected: ModuleNotFoundError: No module named 'graphrag_llm.embedding.sentence_transformer_embedding'
# This is correct! We haven't implemented it yet (TDD RED phase)
```

#### Step 3: Implement Minimal Code (TDD - GREEN)

**File**: `packages/graphrag-llm/graphrag_llm/embedding/sentence_transformer_embedding.py`

```python
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""SentenceTransformer-based embedding generation."""

from typing import TYPE_CHECKING, Any, Unpack

from graphrag_llm.embedding.embedding import LLMEmbedding
from graphrag_llm.middleware import with_middleware_pipeline
from graphrag_llm.types import LLMEmbeddingResponse

if TYPE_CHECKING:
    from graphrag_cache import Cache, CacheKeyCreator
    from graphrag_llm.config import ModelConfig
    from graphrag_llm.metrics import MetricsProcessor, MetricsStore
    from graphrag_llm.rate_limit import RateLimiter
    from graphrag_llm.retry import Retry
    from graphrag_llm.tokenizer import Tokenizer
    from graphrag_llm.types import (
        AsyncLLMEmbeddingFunction,
        LLMEmbeddingArgs,
        LLMEmbeddingFunction,
        Metrics,
    )


class SentenceTransformerEmbedding(LLMEmbedding):
    """Local embedding generation using SentenceTransformers.

    This class provides a local, free alternative to API-based embeddings
    using the sentence-transformers library. It supports CUDA, CPU, and MPS devices.

    Attributes
    ----------
    _model : SentenceTransformer
        The loaded SentenceTransformer model instance.
    _device : str
        The device being used: "cuda", "cpu", or "mps".
    _batch_size : int
        The batch size for encoding operations.
    _normalize_embeddings : bool
        Whether to L2-normalize the generated embeddings.

    Examples
    --------
    >>> from graphrag_llm.embedding import create_embedding
    >>> from graphrag_llm.config import ModelConfig
    >>> config = ModelConfig(
    ...     model="all-MiniLM-L6-v2",
    ...     model_provider="sentence_transformer",
    ...     type="sentence_transformer",
    ... )
    >>> embedding = create_embedding(config)
    >>> result = embedding.embedding(input=["Hello world"])
    >>> len(result.data[0]["embedding"])
    384
    """

    _model_config: "ModelConfig"
    _model_id: str
    _model: Any  # SentenceTransformer model
    _device: str
    _batch_size: int
    _normalize_embeddings: bool
    _track_metrics: bool
    _metrics_store: "MetricsStore"
    _metrics_processor: "MetricsProcessor | None"
    _cache: "Cache | None"
    _cache_key_creator: "CacheKeyCreator"
    _tokenizer: "Tokenizer"
    _rate_limiter: "RateLimiter | None"
    _retrier: "Retry | None"
    _embedding: "LLMEmbeddingFunction"
    _embedding_async: "AsyncLLMEmbeddingFunction"

    def __init__(
        self,
        *,
        model_id: str,
        model_config: "ModelConfig",
        tokenizer: "Tokenizer",
        metrics_store: "MetricsStore",
        metrics_processor: "MetricsProcessor | None" = None,
        rate_limiter: "RateLimiter | None" = None,
        retrier: "Retry | None" = None,
        cache: "Cache | None" = None,
        cache_key_creator: "CacheKeyCreator",
        device: str | None = None,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize SentenceTransformerEmbedding.

        Parameters
        ----------
        model_id : str
            The model identifier.
        model_config : ModelConfig
            The model configuration.
        tokenizer : Tokenizer
            The tokenizer instance.
        metrics_store : MetricsStore
            The metrics store instance.
        metrics_processor : MetricsProcessor | None, default=None
            Optional metrics processor.
        rate_limiter : RateLimiter | None, default=None
            Optional rate limiter (not typically needed for local).
        retrier : Retry | None, default=None
            Optional retry strategy.
        cache : Cache | None, default=None
            Optional cache instance.
        cache_key_creator : CacheKeyCreator
            Cache key creator function.
        device : str | None, default=None
            Device to run model on: "cuda", "cpu", or "mps".
            If None, automatically detects best available device.
        batch_size : int, default=32
            Batch size for encoding operations.
        normalize_embeddings : bool, default=True
            Whether to L2-normalize embeddings.
        **kwargs : Any
            Additional keyword arguments.

        Raises
        ------
        ImportError
            If sentence-transformers package is not installed.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            msg = (
                "sentence-transformers package not installed. "
                "Install it with: pip install sentence-transformers\n"
                "Or install graphrag with: pip install graphrag[local-embeddings]"
            )
            raise ImportError(msg) from e

        self._model_id = model_id
        self._model_config = model_config
        self._tokenizer = tokenizer
        self._metrics_store = metrics_store
        self._metrics_processor = metrics_processor
        self._track_metrics = metrics_processor is not None
        self._cache = cache
        self._cache_key_creator = cache_key_creator
        self._rate_limiter = rate_limiter
        self._retrier = retrier
        self._batch_size = batch_size
        self._normalize_embeddings = normalize_embeddings

        # Detect device if not specified
        self._device = device or _detect_device()

        # Load the SentenceTransformer model
        model_name = model_config.model
        self._model = SentenceTransformer(model_name, device=self._device)

        # Create base embedding functions
        self._embedding, self._embedding_async = _create_base_embeddings(
            model=self._model,
            device=self._device,
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize_embeddings,
        )

        # Wrap with middleware pipeline
        self._embedding, self._embedding_async = with_middleware_pipeline(
            model_config=self._model_config,
            model_fn=self._embedding,
            async_model_fn=self._embedding_async,
            request_type="embedding",
            cache=self._cache,
            cache_key_creator=self._cache_key_creator,
            tokenizer=self._tokenizer,
            metrics_processor=self._metrics_processor,
            rate_limiter=self._rate_limiter,
            retrier=self._retrier,
        )

    def embedding(
        self, /, **kwargs: Unpack["LLMEmbeddingArgs"]
    ) -> "LLMEmbeddingResponse":
        """Generate embeddings synchronously.

        Parameters
        ----------
        **kwargs : Unpack[LLMEmbeddingArgs]
            Keyword arguments including input texts.

        Returns
        -------
        LLMEmbeddingResponse
            Generated embeddings with usage statistics.
        """
        request_metrics: Metrics | None = kwargs.pop("metrics", None) or {}
        if not self._track_metrics:
            request_metrics = None

        try:
            return self._embedding(metrics=request_metrics, **kwargs)
        finally:
            if request_metrics:
                self._metrics_store.update_metrics(metrics=request_metrics)

    async def embedding_async(
        self, /, **kwargs: Unpack["LLMEmbeddingArgs"]
    ) -> "LLMEmbeddingResponse":
        """Generate embeddings asynchronously.

        Parameters
        ----------
        **kwargs : Unpack[LLMEmbeddingArgs]
            Keyword arguments including input texts.

        Returns
        -------
        LLMEmbeddingResponse
            Generated embeddings with usage statistics.
        """
        request_metrics: Metrics | None = kwargs.pop("metrics", None) or {}
        if not self._track_metrics:
            request_metrics = None

        try:
            return await self._embedding_async(metrics=request_metrics, **kwargs)
        finally:
            if request_metrics:
                self._metrics_store.update_metrics(metrics=request_metrics)

    @property
    def metrics_store(self) -> "MetricsStore":
        """Get metrics store."""
        return self._metrics_store

    @property
    def tokenizer(self) -> "Tokenizer":
        """Get tokenizer."""
        return self._tokenizer


def _detect_device() -> str:
    """Detect the best available device for embedding generation.

    Returns
    -------
    str
        One of "cuda", "mps", or "cpu".
    """
    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def _create_base_embeddings(
    *,
    model: Any,
    device: str,
    batch_size: int,
    normalize_embeddings: bool,
) -> tuple["LLMEmbeddingFunction", "AsyncLLMEmbeddingFunction"]:
    """Create base embedding functions for SentenceTransformer.

    Parameters
    ----------
    model : SentenceTransformer
        The loaded model instance.
    device : str
        The device to use.
    batch_size : int
        Batch size for encoding.
    normalize_embeddings : bool
        Whether to normalize embeddings.

    Returns
    -------
    tuple[LLMEmbeddingFunction, AsyncLLMEmbeddingFunction]
        Sync and async embedding functions.
    """
    import asyncio

    def _base_embedding(**kwargs: Any) -> LLMEmbeddingResponse:
        kwargs.pop("metrics", None)  # Remove metrics if present
        input_texts: list[str] = kwargs.get("input", [])

        if isinstance(input_texts, str):
            input_texts = [input_texts]

        # Generate embeddings
        embeddings = model.encode(
            input_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=normalize_embeddings,
            device=device,
        )

        # Format response to match LiteLLM structure
        data = [
            {"embedding": emb.tolist(), "index": i}
            for i, emb in enumerate(embeddings)
        ]

        return LLMEmbeddingResponse(
            data=data,
            model=model.model_card_data.model_name or "sentence-transformer",
            usage={
                "prompt_tokens": sum(len(t.split()) for t in input_texts),
                "total_tokens": sum(len(t.split()) for t in input_texts),
            },
        )

    async def _base_embedding_async(**kwargs: Any) -> LLMEmbeddingResponse:
        # Run synchronous embedding in thread pool
        return await asyncio.to_thread(_base_embedding, **kwargs)

    return _base_embedding, _base_embedding_async
```

#### Step 4: Run Tests (Should PASS - GREEN)

```bash
uv run pytest tests/unit/graphrag_llm/embedding/test_sentence_transformer_embedding.py -v

# Expected: All tests pass
```

#### Step 5: Run All Quality Checks

```bash
# Format code
uv run poe format

# Check all quality gates
uv run poe check

# Run all unit tests
uv run poe test_unit

# Check coverage
uv run coverage run -m pytest tests/unit/graphrag_llm/embedding/
uv run coverage report --include="**/sentence_transformer_embedding.py"
```

#### Step 6: Refactor (TDD - REFACTOR)

Review code for:
- [ ] Duplication
- [ ] Naming clarity
- [ ] Documentation completeness
- [ ] Error handling robustness

#### Step 7: Request Assessment

```
ASSESSMENT REQUEST
Task: 2.1 - Implement SentenceTransformerEmbedding.__init__
Phase: 2 (SentenceTransformer Implementation)
Files Modified:
  - packages/graphrag-llm/graphrag_llm/embedding/sentence_transformer_embedding.py (NEW, ~400 lines)
Tests Added:
  - tests/unit/graphrag_llm/embedding/test_sentence_transformer_embedding.py (NEW, ~150 lines)
All Tests Pass: Yes (3/3)
Coverage: 95%
Ruff: 0 violations
Pyright: 0 errors
Ready for Review: Yes
```

---

## Assessment Agent Workflow

### Step 1: Receive Signal

Wait for Implementation Agent to signal:
```
ASSESSMENT REQUEST
Task: [Task ID]
Phase: [Phase Number]
Files Modified: [List]
Tests Added: [List]
All Tests Pass: [Yes/No]
Coverage: [Percentage]
Ready for Review: [Yes/No]
```

### Step 2: Run Automated Checks

```bash
# Navigate to repo
cd /Users/aselaillayapparachchi/code/GraphRAG/Microsoft/graphrag-ms

# 1. Check formatting
uv run ruff format . --check
# Record: PASS/FAIL

# 2. Check linting
uv run ruff check .
# Record: Number of violations

# 3. Check types
uv run pyright
# Record: Number of errors

# 4. Run tests
uv run poe test_unit
# Record: Pass/fail count

# 5. Check coverage
uv run coverage run -m pytest tests/unit/
uv run coverage report
# Record: Coverage percentage
```

### Step 3: Manual Review

Review code against checklist from `10_CODE_REVIEW_CHECKLIST.md`:

- [ ] Factory pattern adherence
- [ ] Documentation complete
- [ ] Error messages helpful
- [ ] Tests comprehensive
- [ ] No code smells

### Step 4: Provide Feedback

Use structured feedback template from `03_AI_ASSESSMENT.md`.

**Example - Changes Requested**:
```
ASSESSMENT RESULT - Iteration 1
Task: 2.1 - Implement SentenceTransformerEmbedding.__init__
Status: ⚠️ CHANGES REQUESTED

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AUTOMATED CHECKS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Formatting: PASS
✅ Linting: PASS
✅ Type Checking: PASS
✅ Unit Tests: PASS (3/3)
⚠️  Coverage: WARNING (88%, target 90%)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COVERAGE ISSUES (SHOULD FIX)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Lines 145-150 uncovered (sentence_transformer_embedding.py)
   Missing test: Error handling when model name is invalid
   Fix: Add test_initialization_raises_error_for_invalid_model()

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Must Fix (blockers): 0 issues
Should Fix (quality): 1 issue
Nice to Have: 0 issues

Next Steps:
1. Add test for invalid model name (issue 1)
2. Re-run coverage check
3. Request re-assessment

Estimated Time: 15 minutes
```

**Example - Approved**:
```
ASSESSMENT RESULT - Iteration 2
Task: 2.1 - Implement SentenceTransformerEmbedding.__init__
Status: ✅ APPROVED

All checks passed:
✅ Formatting: PASS
✅ Linting: PASS (0 violations)
✅ Type Checking: PASS (0 errors)
✅ Unit Tests: PASS (4/4)
✅ Coverage: PASS (94%)
✅ Documentation: COMPLETE
✅ Tests: COMPREHENSIVE
✅ Code Quality: EXCELLENT

Ready to proceed to next task: Implement embedding() method

Excellent work! Implementation follows all patterns correctly.
```

### Step 5: Track Progress

Maintain iteration log:
```
Task: 2.1 - Implement SentenceTransformerEmbedding.__init__
├─ Iteration 1: 2024-01-30 10:00
│  └─ Result: CHANGES REQUESTED (1 coverage issue)
└─ Iteration 2: 2024-01-30 10:30
   └─ Result: ✅ APPROVED
```

---

## Escalation Procedure

If Implementation Agent is blocked or >5 iterations:

### Step 1: Document Blocker

```
ESCALATION REQUEST
Task: [Task ID]
Iterations: [Number]
Blocker Type: [Technical/Architectural/Ambiguous]

Description:
[Clear description of the blocker]

Attempted Solutions:
1. [What was tried]
2. [What was tried]

Assessment Agent Input:
[Assessment agent's perspective on the issue]

Recommended Action:
[What needs human decision]
```

### Step 2: Pause Implementation

Wait for human guidance before proceeding.

### Step 3: Resume After Clarification

Once blocker is resolved, resume normal workflow.

---

## Communication Protocol

### Implementation Agent Signals

**Ready for Assessment**:
```
ASSESSMENT REQUEST
Task: [ID]
Phase: [Number]
Files Modified: [List]
Tests Added: [List]
All Tests Pass: [Yes/No]
Coverage: [%]
Ready for Review: Yes
```

**Question/Blocker**:
```
QUESTION
Task: [ID]
Question: [Specific question]
Context: [Why asking]
Options Considered: [What was already tried]
```

### Assessment Agent Responses

**Approved**:
```
ASSESSMENT RESULT
Task: [ID]
Status: ✅ APPROVED
[Summary]
Next Task: [Task ID]
```

**Changes Requested**:
```
ASSESSMENT RESULT
Task: [ID]
Status: ⚠️ CHANGES REQUESTED
[Detailed feedback]
```

**Blocked**:
```
ASSESSMENT RESULT
Task: [ID]
Status: ❌ BLOCKED
[Explanation of blocker]
[Recommended escalation]
```

---

## Progress Tracking

Use this template to track overall progress:

```
PHASE 1: DOCUMENTATION (Weeks 1-2)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Task 1.1: Claude Config Examples      ✅ COMPLETE (2 iterations)
Task 1.2: LLM Provider Guide          ✅ COMPLETE (1 iteration)
Task 1.3: Migration Guide             ✅ COMPLETE (2 iterations)
Task 1.4: Cost Optimization Guide     🔄 IN PROGRESS (Iteration 1)

PHASE 2: SENTENCETRANSFORMER (Weeks 3-4)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Task 2.1: ST __init__                 ✅ COMPLETE (2 iterations)
Task 2.2: ST embedding()              🔄 IN PROGRESS (Iteration 1)
Task 2.3: ST embedding_async()        ⏳ PENDING
Task 2.4: Factory Updates             ⏳ PENDING
Task 2.5: Config Schema               ⏳ PENDING
Task 2.6: Integration Tests           ⏳ PENDING

PHASE 3: VALIDATION (Week 5)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Task 3.1: Prompt Validation           ⏳ PENDING
Task 3.2: Performance Benchmarks      ⏳ PENDING
Task 3.3: Example Notebooks           ⏳ PENDING

PHASE 4: RELEASE (Week 6)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Task 4.1: Multi-Platform Testing      ⏳ PENDING
Task 4.2: Release Preparation         ⏳ PENDING
Task 4.3: Final Release               ⏳ PENDING
```

---

## Summary Checklist

Before starting any task:
- [ ] Read task requirements
- [ ] Understand acceptance criteria
- [ ] Review relevant coding standards
- [ ] Set up test environment

During implementation:
- [ ] Write tests first (TDD)
- [ ] Implement minimal code
- [ ] Run local checks frequently
- [ ] Follow coding standards

Before requesting assessment:
- [ ] All tests pass
- [ ] All checks pass (format, lint, types)
- [ ] Coverage ≥90%
- [ ] Documentation complete
- [ ] Ready for review

---

**Next Document**: Phase-specific implementation plans (05-08)
