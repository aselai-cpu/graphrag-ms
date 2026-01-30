# Test-Driven Development Strategy

**Document**: 01_TDD_STRATEGY.md
**Status**: Ready for Implementation
**Approach**: Red-Green-Refactor with AI Agent Assessment

---

## Overview

This document defines the Test-Driven Development (TDD) strategy for implementing Claude and SentenceTransformer support in GraphRAG. TDD is critical for ensuring code quality, maintainability, and backward compatibility.

---

## TDD Principles

### Core Cycle: Red-Green-Refactor

```
┌──────────────────────────────────────────────────────────┐
│                                                          │
│  RED: Write Failing Test                                │
│  ├─ Define expected behavior                            │
│  ├─ Write test that fails (feature not implemented)     │
│  └─ Verify test fails for the right reason              │
│                                                          │
│  GREEN: Make Test Pass                                  │
│  ├─ Write minimal code to make test pass                │
│  ├─ Don't worry about elegance yet                      │
│  └─ All tests pass (including existing ones)            │
│                                                          │
│  REFACTOR: Improve Code Quality                         │
│  ├─ Remove duplication                                  │
│  ├─ Improve naming and structure                        │
│  ├─ Optimize performance if needed                      │
│  └─ All tests still pass                                │
│                                                          │
│  ASSESS: AI Agent Review                                │
│  ├─ Automated checks (ruff, pyright, coverage)          │
│  ├─ Manual review (design, patterns, docs)              │
│  └─ Feedback loop until approved                        │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## Test Pyramid

GraphRAG testing follows the standard test pyramid:

```
          ┌───────────┐
         /   E2E       \     ← 5% (Few, slow, expensive)
        /   Tests      \
       ┌─────────────────┐
      /   Integration    \   ← 20% (Some, medium speed)
     /      Tests         \
    ┌───────────────────────┐
   /    Unit Tests          \  ← 75% (Many, fast, cheap)
  /                          \
 └────────────────────────────┘
```

### Unit Tests (75%)
- **Focus**: Individual functions and classes
- **Speed**: Fast (<10ms per test)
- **Isolation**: Mock external dependencies
- **Coverage Target**: 90%+

### Integration Tests (20%)
- **Focus**: Component interactions
- **Speed**: Medium (100ms-1s per test)
- **Isolation**: Use real components, mock external APIs
- **Coverage Target**: Critical paths

### E2E Tests (5%)
- **Focus**: Full pipeline execution
- **Speed**: Slow (>1s per test)
- **Isolation**: Minimal mocking
- **Coverage Target**: Happy paths

---

## TDD for SentenceTransformer Implementation

### Phase 2: Core Implementation

#### Task 1: SentenceTransformerEmbedding.__init__

**Step 1: RED - Write Failing Test**

```python
# tests/unit/graphrag_llm/embedding/test_sentence_transformer_embedding.py

import pytest
from graphrag_llm.embedding.sentence_transformer_embedding import (
    SentenceTransformerEmbedding,
)
from graphrag_llm.config import ModelConfig

def test_initialization_with_default_device():
    """Test SentenceTransformerEmbedding initializes with default device."""
    # Arrange
    config = ModelConfig(
        model="all-MiniLM-L6-v2",
        model_provider="sentence_transformer",
        type="sentence_transformer",
    )

    # Act
    embedding = SentenceTransformerEmbedding(
        model_id="test/all-MiniLM-L6-v2",
        model_config=config,
        tokenizer=mock_tokenizer,
        metrics_store=mock_metrics_store,
        cache_key_creator=mock_cache_key_creator,
    )

    # Assert
    assert embedding is not None
    assert embedding._device in ["cuda", "cpu", "mps"]
    assert embedding._model is not None
```

**Step 2: GREEN - Implement Minimal Code**

```python
# packages/graphrag-llm/graphrag_llm/embedding/sentence_transformer_embedding.py

class SentenceTransformerEmbedding(LLMEmbedding):
    """LLMEmbedding based on SentenceTransformers."""

    def __init__(
        self,
        *,
        model_id: str,
        model_config: "ModelConfig",
        tokenizer: "Tokenizer",
        metrics_store: "MetricsStore",
        cache_key_creator: "CacheKeyCreator",
        device: str | None = None,
        **kwargs,
    ):
        """Initialize SentenceTransformerEmbedding."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            msg = (
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            raise ImportError(msg) from e

        # Detect device if not specified
        if device is None:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self._device = device
        self._model_id = model_id
        self._model_config = model_config
        self._tokenizer = tokenizer
        self._metrics_store = metrics_store
        self._cache_key_creator = cache_key_creator

        # Load model
        model_name = model_config.model
        self._model = SentenceTransformer(model_name, device=device)
```

**Step 3: REFACTOR - Improve Quality**

```python
# Extract device detection to separate function
def _detect_device() -> str:
    """Detect the best available device for embedding generation.

    Returns
    -------
        str: One of "cuda", "mps", or "cpu"
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

# Update __init__ to use helper
device = device or _detect_device()
```

**Step 4: ASSESS - Request AI Review**

```
ASSESSMENT REQUEST
Task: SentenceTransformerEmbedding.__init__
Phase: 2
Files Modified:
  - packages/graphrag-llm/graphrag_llm/embedding/sentence_transformer_embedding.py
Tests Added:
  - tests/unit/graphrag_llm/embedding/test_sentence_transformer_embedding.py::test_initialization_with_default_device
All Tests Pass: Yes
Coverage: 95% (function level)
Ready for Review: Yes
```

---

## Test Organization

### Directory Structure

```
tests/
├── unit/                                    # Unit tests (fast, isolated)
│   └── graphrag_llm/
│       └── embedding/
│           └── test_sentence_transformer_embedding.py
│
├── integration/                             # Integration tests
│   └── test_sentence_transformer.py
│
└── smoke/                                   # End-to-end smoke tests
    └── test_claude_indexing.py
```

### Naming Conventions

**Test Files**:
- Pattern: `test_<module_name>.py`
- Example: `test_sentence_transformer_embedding.py`

**Test Functions**:
- Pattern: `test_<feature>_<condition>_<expected_outcome>`
- Examples:
  - `test_initialization_with_default_device()`
  - `test_embedding_generation_with_single_text()`
  - `test_async_embedding_with_multiple_texts()`
  - `test_device_selection_when_cuda_unavailable()`

**Test Classes** (optional, for grouping):
- Pattern: `Test<ClassName>`
- Example: `TestSentenceTransformerEmbedding`

---

## Test Specifications

### Unit Test Template

```python
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Unit tests for <module_name>."""

import pytest
from unittest.mock import Mock, patch

from graphrag_llm.embedding.sentence_transformer_embedding import (
    SentenceTransformerEmbedding,
)


class TestSentenceTransformerEmbedding:
    """Test suite for SentenceTransformerEmbedding class."""

    @pytest.fixture
    def mock_config(self):
        """Fixture providing a mock ModelConfig."""
        return Mock(
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

    def test_<feature>_<condition>(self, mock_config, mock_dependencies):
        """Test <specific behavior>.

        Arrange: <setup description>
        Act: <action description>
        Assert: <expected outcome>
        """
        # Arrange
        <setup code>

        # Act
        result = <action>

        # Assert
        assert result == expected
        <additional assertions>

    def test_<feature>_raises_<exception>_when_<condition>(self):
        """Test that <feature> raises <exception> when <condition>."""
        # Arrange
        <setup invalid condition>

        # Act & Assert
        with pytest.raises(ExpectedException) as exc_info:
            <action that should fail>

        assert "expected error message" in str(exc_info.value)
```

### Integration Test Template

```python
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Integration tests for SentenceTransformer embedding."""

import pytest
from graphrag.index import create_pipeline_config, run_pipeline_with_config


def test_full_indexing_with_sentence_transformer():
    """Test complete indexing pipeline with SentenceTransformer embeddings.

    This test verifies:
    1. Pipeline configuration accepts SentenceTransformer config
    2. Embeddings are generated successfully
    3. Output format matches expected schema
    4. Embeddings have correct dimensionality
    """
    # Arrange
    config = create_pipeline_config(
        embedding_model={
            "type": "sentence_transformer",
            "model": "all-MiniLM-L6-v2",
            "device": "cpu",  # Use CPU for CI/CD
        }
    )
    test_data = load_test_documents()

    # Act
    result = run_pipeline_with_config(config, test_data)

    # Assert
    assert result.success
    assert len(result.entities) > 0

    # Check embeddings
    for entity in result.entities:
        assert entity.description_embedding is not None
        assert len(entity.description_embedding) == 384  # MiniLM dimension
        assert all(isinstance(v, float) for v in entity.description_embedding)
```

---

## Test Coverage Requirements

### Overall Target: ≥90%

**Branch Coverage**:
- Critical paths: 100%
- Error handling: 95%
- Edge cases: 90%
- Default paths: 85%

### Per-Module Requirements

**SentenceTransformerEmbedding**:
- `__init__`: 100% (critical)
- `embedding`: 100% (critical)
- `embedding_async`: 100% (critical)
- Helper functions: 95%
- Properties: 90%

**Factory Updates**:
- `create_embedding`: 100%
- Registration: 100%

**Config Updates**:
- Schema validation: 100%
- Default values: 95%

---

## Testing Strategy by Phase

### Phase 1: Documentation (Weeks 1-2)

**No Traditional Tests** - Documentation phase

**Validation Approach**:
1. Manual configuration testing
2. Example verification
3. Documentation review
4. Clarity assessment

**Acceptance Criteria**:
- [ ] All examples are syntactically correct YAML
- [ ] Examples successfully load and validate
- [ ] Claude configuration works with test credentials
- [ ] Documentation is clear and complete

---

### Phase 2: SentenceTransformer (Weeks 3-4)

**TDD Approach**: Write tests before implementation

#### Week 3: Core Implementation

**Day 1-2: Initialization Tests**
```python
# Write first
test_initialization_with_default_device()
test_initialization_with_explicit_cuda()
test_initialization_with_explicit_cpu()
test_initialization_with_explicit_mps()
test_initialization_raises_import_error_when_package_missing()
test_initialization_loads_model_successfully()

# Then implement
SentenceTransformerEmbedding.__init__()
```

**Day 3-4: Embedding Generation Tests**
```python
# Write first
test_embedding_generation_with_single_text()
test_embedding_generation_with_multiple_texts()
test_embedding_generation_with_empty_input()
test_embedding_generation_with_long_text()
test_embedding_output_format_matches_litellm()
test_embedding_dimensions_match_model()
test_embedding_normalization_when_enabled()
test_embedding_normalization_when_disabled()

# Then implement
SentenceTransformerEmbedding.embedding()
_create_base_embeddings()
```

**Day 5: Async and Factory Tests**
```python
# Write first
test_async_embedding_generation()
test_async_embedding_runs_in_thread_pool()
test_factory_creates_sentence_transformer_when_configured()
test_factory_registers_sentence_transformer_type()

# Then implement
SentenceTransformerEmbedding.embedding_async()
Update embedding_factory.py
```

#### Week 4: Integration Testing

**Day 1-2: Integration Tests**
```python
# Write first
test_full_indexing_with_sentence_transformer()
test_query_with_sentence_transformer_embeddings()
test_mixed_claude_completion_and_st_embeddings()
test_batch_processing_performance()

# Then implement
Any necessary adjustments to make integration work
```

**Day 3-4: Edge Cases and Error Handling**
```python
# Write first
test_device_fallback_when_cuda_unavailable()
test_model_download_on_first_use()
test_cache_integration()
test_metrics_tracking()
test_concurrent_embedding_generation()

# Then implement
Error handling and edge case logic
```

---

### Phase 3: Validation (Week 5)

**Test Approach**: Validation and benchmarking

**Day 1-2: Prompt Validation**
- Test all GraphRAG prompts with Claude
- Validate JSON output format
- Check quality against baselines

**Day 3-4: Performance Benchmarking**
- Embedding speed tests (GPU vs CPU)
- Indexing throughput tests
- Memory usage tests
- Cost calculation tests

---

### Phase 4: Release (Week 6)

**Test Approach**: Final validation

**Multi-Platform Testing Matrix**:

| Platform | Python | Provider Config | Status |
|----------|--------|----------------|--------|
| Linux | 3.10 | OpenAI | ✅ |
| Linux | 3.11 | Claude + OpenAI Embed | ⏳ |
| Linux | 3.12 | Claude + ST (GPU) | ⏳ |
| macOS | 3.10 | OpenAI | ✅ |
| macOS | 3.11 | Claude + ST (MPS) | ⏳ |
| macOS | 3.12 | Claude + ST (CPU) | ⏳ |
| Windows | 3.10 | OpenAI | ✅ |
| Windows | 3.11 | Claude + OpenAI Embed | ⏳ |
| Windows | 3.12 | Claude + ST (CPU) | ⏳ |

---

## Test Execution

### Running Tests Locally

```bash
# All tests
uv run poe test

# Unit tests only (fast, run frequently)
uv run poe test_unit

# Specific test file
uv run pytest tests/unit/graphrag_llm/embedding/test_sentence_transformer_embedding.py

# Specific test function
uv run pytest tests/unit/graphrag_llm/embedding/test_sentence_transformer_embedding.py::test_initialization_with_default_device

# With coverage
uv run coverage run -m pytest tests/unit
uv run coverage report

# Integration tests (slower)
uv run poe test_integration

# Watch mode (re-run on file changes)
uv run pytest-watch tests/unit
```

### CI/CD Pipeline

**On Every Commit**:
1. Run unit tests
2. Check code coverage (must be ≥90%)
3. Run ruff (formatting and linting)
4. Run pyright (type checking)
5. Build documentation

**On Pull Request**:
1. All commit checks
2. Run integration tests
3. Run smoke tests
4. Generate coverage report
5. Security scan (bandit)

**On Merge to Main**:
1. All PR checks
2. Build packages
3. Run full test suite
4. Generate release artifacts

---

## Mocking Strategy

### External Dependencies to Mock

**Unit Tests** - Mock Everything External:
- `sentence_transformers.SentenceTransformer`
- `torch.cuda.is_available()`
- `torch.backends.mps.is_available()`
- API clients (OpenAI, Anthropic)
- File system operations
- Network calls

**Integration Tests** - Mock Sparingly:
- External API calls (use mock API)
- Expensive operations (model downloads)
- Non-deterministic operations

**E2E Tests** - Minimal Mocking:
- Only mock external paid APIs
- Use real local components

### Mock Examples

```python
# Mock SentenceTransformer model
@patch("graphrag_llm.embedding.sentence_transformer_embedding.SentenceTransformer")
def test_initialization_loads_model(mock_st_class):
    mock_model = Mock()
    mock_st_class.return_value = mock_model

    embedding = SentenceTransformerEmbedding(...)

    mock_st_class.assert_called_once_with(
        "all-MiniLM-L6-v2",
        device="cuda"
    )

# Mock device detection
@patch("torch.cuda.is_available", return_value=False)
@patch("torch.backends.mps.is_available", return_value=True)
def test_device_detection_prefers_mps_when_cuda_unavailable(
    mock_mps, mock_cuda
):
    device = _detect_device()
    assert device == "mps"
```

---

## Test Data Management

### Fixtures and Test Data

**Location**: `tests/fixtures/`

**Organization**:
```
tests/fixtures/
├── configs/
│   ├── claude_basic.yaml
│   ├── sentence_transformer_cpu.yaml
│   └── mixed_providers.yaml
├── documents/
│   ├── sample_text_short.txt
│   ├── sample_text_long.txt
│   └── sample_documents.json
└── expected_outputs/
    ├── extracted_entities.json
    └── embeddings.npy
```

### pytest Fixtures

```python
# conftest.py

import pytest
from pathlib import Path

@pytest.fixture
def fixture_dir():
    """Return path to fixtures directory."""
    return Path(__file__).parent / "fixtures"

@pytest.fixture
def sample_config(fixture_dir):
    """Load sample configuration."""
    config_path = fixture_dir / "configs" / "sentence_transformer_cpu.yaml"
    return yaml.safe_load(config_path.read_text())

@pytest.fixture
def sample_texts():
    """Return sample texts for embedding."""
    return [
        "This is a short test sentence.",
        "GraphRAG uses knowledge graphs for retrieval.",
        "SentenceTransformer models generate dense embeddings.",
    ]

@pytest.fixture
def mock_sentence_transformer():
    """Return a mock SentenceTransformer model."""
    mock = Mock()
    mock.encode.return_value = np.random.rand(3, 384)
    return mock
```

---

## Continuous Improvement

### After Each Test Cycle

1. **Review Test Failures**: Understand why tests failed
2. **Update Tests**: If requirements changed, update tests first
3. **Refactor**: Improve test readability and maintainability
4. **Remove Duplication**: Extract common setup to fixtures
5. **Document**: Add docstrings explaining complex test scenarios

### Metrics to Track

- **Test Count**: Growing with features
- **Test Speed**: Keep unit tests fast (<10ms avg)
- **Coverage**: Maintain ≥90%
- **Flakiness**: Eliminate flaky tests immediately
- **Execution Time**: Optimize slow tests

---

## TDD Anti-Patterns to Avoid

### ❌ Don't Do This

1. **Writing Implementation First**
   ```python
   # Bad: Implement first, test later
   def embedding(self, input):
       return self._model.encode(input)

   # Then write test
   def test_embedding():
       assert embedding works...
   ```

2. **Testing Implementation Details**
   ```python
   # Bad: Testing internal method calls
   def test_embedding_calls_encode():
       embedding.embedding(text)
       embedding._model.encode.assert_called_once()  # Too specific!
   ```

3. **Large, Unfocused Tests**
   ```python
   # Bad: Testing too much at once
   def test_everything():
       # Tests initialization, embedding, async, caching, metrics...
       # 100 lines of test code
   ```

4. **No Assertions**
   ```python
   # Bad: Test that doesn't verify anything
   def test_embedding():
       result = embedding.embedding(text)
       # No assertions!
   ```

### ✅ Do This Instead

1. **Write Test First**
   ```python
   # Good: Test defines expected behavior
   def test_embedding_returns_correct_dimensions():
       result = embedding.embedding(["test"])
       assert len(result.data) == 1
       assert len(result.data[0]["embedding"]) == 384

   # Then implement to make it pass
   ```

2. **Test Behavior, Not Implementation**
   ```python
   # Good: Testing observable behavior
   def test_embedding_returns_normalized_vectors():
       result = embedding.embedding(["test"])
       vector = result.data[0]["embedding"]
       norm = np.linalg.norm(vector)
       assert abs(norm - 1.0) < 1e-6
   ```

3. **One Concept Per Test**
   ```python
   # Good: Focused tests
   def test_initialization_with_cuda():
       embedding = SentenceTransformerEmbedding(..., device="cuda")
       assert embedding._device == "cuda"

   def test_embedding_single_text():
       result = embedding.embedding(input=["test"])
       assert len(result.data) == 1
   ```

4. **Clear Assertions**
   ```python
   # Good: Multiple specific assertions
   def test_embedding_output_format():
       result = embedding.embedding(["test"])
       assert isinstance(result.data, list)
       assert len(result.data) == 1
       assert "embedding" in result.data[0]
       assert "index" in result.data[0]
       assert result.data[0]["index"] == 0
   ```

---

## Summary Checklist

Before requesting assessment, verify:

- [ ] Test written before implementation (TDD)
- [ ] Test name follows naming convention
- [ ] Test has clear docstring
- [ ] Arrange-Act-Assert structure used
- [ ] All assertions are specific and meaningful
- [ ] Edge cases tested
- [ ] Error cases tested
- [ ] Mock external dependencies appropriately
- [ ] Test is fast (unit tests <10ms)
- [ ] Test is deterministic (no flakiness)
- [ ] All tests pass locally
- [ ] Coverage ≥90% for new code

---

**Next Document**: [02_CODING_STANDARDS.md](./02_CODING_STANDARDS.md)
