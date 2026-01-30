# Coding Standards and Quality Requirements

**Document**: 02_CODING_STANDARDS.md
**Status**: Ready for Implementation
**Purpose**: Define code quality requirements for Claude/SentenceTransformer implementation

---

## Overview

This document specifies the coding standards, conventions, and quality requirements that all code must meet. These standards are enforced through automated tools (ruff, pyright) and manual review by assessment agents.

---

## Automated Quality Checks

### Running Quality Checks

```bash
# Run all checks
uv run poe check

# Individual checks
uv run ruff format . --check  # Check formatting
uv run ruff check .           # Check linting
uv run pyright               # Check types
```

###Fix Issues

```bash
# Auto-fix formatting
uv run poe format

# Auto-fix safe issues
uv run poe fix

# Auto-fix including unsafe fixes (use cautiously)
uv run poe fix_unsafe
```

---

## Formatting Standards (Ruff Format)

### Configuration

From `pyproject.toml`:
```toml
[tool.ruff]
target-version = "py310"
extend-include = ["*.ipynb"]

[tool.ruff.format]
preview = true
docstring-code-format = true
docstring-code-line-length = 20
```

### Key Rules

**Line Length**: 88 characters (PEP 8 default)
**Indentation**: 4 spaces (no tabs)
**Quotes**: Double quotes for strings
**Trailing Commas**: Required in multi-line structures
**Import Sorting**: Automatic via ruff

### Examples

**Good**:
```python
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from graphrag_llm.embedding import LLMEmbedding
from graphrag_llm.types import LLMEmbeddingResponse


class SentenceTransformerEmbedding(LLMEmbedding):
    """Local embedding generation using SentenceTransformers.

    This class provides a local, free alternative to API-based embeddings.
    """

    def __init__(
        self,
        *,
        model_id: str,
        model_config: "ModelConfig",
        device: str = "cuda",
    ) -> None:
        """Initialize SentenceTransformerEmbedding.

        Parameters
        ----------
        model_id : str
            The model identifier.
        model_config : ModelConfig
            The model configuration.
        device : str, default="cuda"
            Device to run model on: "cuda", "cpu", or "mps".
        """
        self._model_id = model_id
        self._device = device
```

**Bad**:
```python
# Missing imports organization
import asyncio
from graphrag_llm.embedding import LLMEmbedding
import torch
from typing import Any

# Inconsistent indentation
class SentenceTransformerEmbedding(LLMEmbedding):
  """Bad indentation."""

  def __init__(self,
               model_id: str,  # Don't align like this
               device: str = 'cuda'):  # Use double quotes
    pass  # Missing docstring
```

---

## Linting Standards (Ruff Check)

### Enabled Rule Sets

From `pyproject.toml`:
```toml
[tool.ruff.lint]
select = [
    "E4", "E7", "E9", "W291",  # pycodestyle
    "F",                        # pyflakes
    "I",                        # isort
    "N",                        # pep8-naming
    "D",                        # pydocstyle
    "UP",                       # pyupgrade
    "S",                        # bandit (security)
    "B",                        # bugbear
    "A",                        # builtins
    "C4",                       # comprehensions
    "PTH",                      # pathlib
    "RUF",                      # ruff-specific
    "ARG",                      # unused arguments
    "RET",                      # return statements
    "SIM",                      # simplify
    "TCH",                      # type-checking
    "PERF",                     # performance
    "CPY",                      # copyright
]
```

### Critical Rules

#### 1. Security (S)

**S101**: No `assert` in production code
```python
# Bad
def process(data):
    assert data is not None  # Removed in optimized bytecode!
    return process_data(data)

# Good
def process(data):
    if data is None:
        msg = "Data cannot be None"
        raise ValueError(msg)
    return process_data(data)
```

**S108**: No `try/except/pass`
```python
# Bad
try:
    risky_operation()
except Exception:
    pass  # Silently swallowing errors!

# Good
try:
    risky_operation()
except SpecificException as e:
    logger.warning(f"Operation failed: {e}")
    # or re-raise, or handle appropriately
```

#### 2. Type Checking (TCH)

**TCH001-003**: Proper `TYPE_CHECKING` imports
```python
# Good
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from graphrag_cache import Cache
    from graphrag_llm.config import ModelConfig

# This prevents circular imports and reduces runtime overhead
```

#### 3. Error Messages (EM)

**EM101-102**: Define exception messages as variables
```python
# Bad
raise ValueError("Model not found")

# Good
msg = "Model not found"
raise ValueError(msg)
```

#### 4. Path Handling (PTH)

**PTH118-123**: Use `pathlib.Path` instead of `os.path`
```python
# Bad
import os
config_path = os.path.join(root, "config.yaml")

# Good
from pathlib import Path
config_path = Path(root) / "config.yaml"
```

---

## Type Checking (Pyright)

### Configuration

From `pyproject.toml`:
```toml
[tool.pyright]
include = [
    "packages/graphrag/graphrag",
    "packages/graphrag-llm/graphrag_llm",
    "tests"
]
```

### Type Annotation Requirements

**All Functions**: Must have parameter and return type annotations
```python
# Good
def create_embedding(
    model_id: str,
    config: ModelConfig,
    *,
    device: str | None = None,
) -> LLMEmbedding:
    """Create an embedding instance."""
    ...

# Bad
def create_embedding(model_id, config, device=None):
    """Create an embedding instance."""
    ...
```

**Class Attributes**: Type annotate instance variables
```python
# Good
class SentenceTransformerEmbedding(LLMEmbedding):
    _model: Any  # SentenceTransformer type
    _device: str
    _batch_size: int
    _model_id: str

    def __init__(self, ...) -> None:
        self._model = SentenceTransformer(...)
        self._device = device
        self._batch_size = batch_size

# Bad
class SentenceTransformerEmbedding(LLMEmbedding):
    def __init__(self, ...):
        self._model = SentenceTransformer(...)  # No type hints
```

**Conditional Imports**: Use `TYPE_CHECKING`
```python
# Good
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

class SentenceTransformerEmbedding:
    _model: "SentenceTransformer"  # String annotation for TYPE_CHECKING import

# Bad
from sentence_transformers import SentenceTransformer  # Always imported

class SentenceTransformerEmbedding:
    _model: SentenceTransformer
```

---

## Documentation Standards

### Docstring Style: NumPy

GraphRAG uses NumPy-style docstrings as specified in `pyproject.toml`:
```toml
[tool.ruff.lint.pydocstyle]
convention = "numpy"
```

### Module Docstrings

```python
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""SentenceTransformer-based embedding generation.

This module provides a local, free alternative to API-based embeddings
using the sentence-transformers library. It supports CUDA, CPU, and MPS devices.
"""
```

### Class Docstrings

```python
class SentenceTransformerEmbedding(LLMEmbedding):
    """Local embedding generation using SentenceTransformers.

    This class implements the LLMEmbedding interface for local embedding
    generation. It downloads and caches models from HuggingFace and runs
    them locally on CUDA, CPU, or MPS devices.

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
    >>> from graphrag_llm.embedding import SentenceTransformerEmbedding
    >>> embedding = SentenceTransformerEmbedding(
    ...     model_id="test/model",
    ...     model_config=config,
    ...     device="cuda",
    ... )
    >>> result = embedding.embedding(input=["Hello world"])
    >>> len(result.data[0]["embedding"])
    384
    """
```

### Function/Method Docstrings

```python
def embedding(
    self,
    /,
    **kwargs: Unpack["LLMEmbeddingArgs"],
) -> "LLMEmbeddingResponse":
    """Generate embeddings for input texts.

    Parameters
    ----------
    **kwargs : Unpack[LLMEmbeddingArgs]
        Keyword arguments including:
        - input : list[str] or str
            The text(s) to generate embeddings for.
        - metrics : Metrics | None
            Optional metrics tracking dictionary.

    Returns
    -------
    LLMEmbeddingResponse
        Response containing:
        - data : list[dict]
            List of embedding dictionaries with "embedding" and "index" keys.
        - model : str
            The model identifier used.
        - usage : dict
            Token usage statistics.

    Raises
    ------
    ValueError
        If input is empty or invalid format.
    RuntimeError
        If model fails to generate embeddings.

    Examples
    --------
    >>> result = embedding.embedding(input=["Hello", "World"])
    >>> len(result.data)
    2
    >>> len(result.data[0]["embedding"])
    384

    Notes
    -----
    The embedding generation runs synchronously. For async operations,
    use `embedding_async()` instead.

    See Also
    --------
    embedding_async : Async version of this method
    """
```

### Docstring Requirements

- [ ] **Summary Line**: One-line summary (<80 chars)
- [ ] **Extended Summary**: Detailed description if needed
- [ ] **Parameters**: All parameters documented
- [ ] **Returns**: Return value(s) documented
- [ ] **Raises**: All exceptions documented
- [ ] **Examples**: Usage examples for complex functions
- [ ] **Notes**: Important implementation details
- [ ] **See Also**: Related functions/classes

---

## Code Organization

### Import Order

Enforced by ruff's `I` rules:

1. **Standard library** imports
2. **Third-party** imports
3. **Local** imports

Each group separated by a blank line:

```python
# Copyright header

"""Module docstring."""

# Standard library
import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Third-party
import numpy as np
import torch

# Local
from graphrag_llm.embedding import LLMEmbedding
from graphrag_llm.types import LLMEmbeddingResponse

# TYPE_CHECKING imports
if TYPE_CHECKING:
    from graphrag_cache import Cache
    from graphrag_llm.config import ModelConfig
```

### Class Organization

```python
class MyClass:
    """Class docstring."""

    # 1. Class variables
    _registry: dict[str, type] = {}

    # 2. __init__
    def __init__(self, ...) -> None:
        """Initialize."""
        ...

    # 3. Public methods (alphabetical)
    def embedding(self, ...) -> ...:
        """Public method."""
        ...

    def embedding_async(self, ...) -> ...:
        """Public async method."""
        ...

    # 4. Properties (alphabetical)
    @property
    def model_id(self) -> str:
        """Get model ID."""
        return self._model_id

    # 5. Private methods (alphabetical, prefixed with _)
    def _create_embedding(self, ...) -> ...:
        """Private helper."""
        ...

    # 6. Dunder methods (except __init__)
    def __repr__(self) -> str:
        """String representation."""
        return f"MyClass(model_id={self._model_id})"
```

---

## Naming Conventions

### General Rules (PEP 8)

| Type | Convention | Example |
|------|------------|---------|
| Module | `snake_case` | `sentence_transformer_embedding.py` |
| Class | `PascalCase` | `SentenceTransformerEmbedding` |
| Function | `snake_case` | `create_embedding()` |
| Method | `snake_case` | `embedding_async()` |
| Variable | `snake_case` | `model_id`, `batch_size` |
| Constant | `UPPER_SNAKE_CASE` | `DEFAULT_BATCH_SIZE` |
| Private | `_leading_underscore` | `_model`, `_detect_device()` |
| Type Variable | `PascalCase` | `ModelType`, `ConfigType` |

### Specific Naming Guidelines

**Boolean Variables**: Use `is_`, `has_`, `can_`, `should_` prefix
```python
# Good
is_initialized: bool = False
has_cuda: bool = torch.cuda.is_available()
can_use_mps: bool = hasattr(torch.backends, "mps")
should_normalize: bool = config.normalize_embeddings

# Bad
initialized: bool = False  # Ambiguous
cuda: bool = torch.cuda.is_available()  # Not clear it's boolean
```

**Collections**: Use plural nouns
```python
# Good
embeddings: list[float] = []
model_configs: dict[str, ModelConfig] = {}
input_texts: list[str] = ["text1", "text2"]

# Bad
embedding_list: list[float] = []  # Redundant
config: dict[str, ModelConfig] = {}  # Should be plural
```

**Factory Functions**: Start with `create_`
```python
# Good
def create_embedding(...) -> LLMEmbedding:
    """Create an embedding instance."""

def create_tokenizer(...) -> Tokenizer:
    """Create a tokenizer instance."""

# Bad
def get_embedding(...) -> LLMEmbedding:  # Not clear it creates new instance
def make_tokenizer(...) -> Tokenizer:  # Use "create" for consistency
```

---

## Error Handling

### Exception Guidelines

**Be Specific**: Raise specific exception types
```python
# Good
if model_id is None:
    msg = "model_id cannot be None"
    raise ValueError(msg)

if not Path(config_path).exists():
    msg = f"Configuration file not found: {config_path}"
    raise FileNotFoundError(msg)

# Bad
if model_id is None:
    raise Exception("Bad input")  # Too generic
```

**Provide Context**: Include helpful error messages
```python
# Good
try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    msg = (
        "sentence-transformers package not installed. "
        "Install it with: pip install sentence-transformers"
    )
    raise ImportError(msg) from e

# Bad
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("Package not found")  # No help for user
```

**Chain Exceptions**: Use `from e` to preserve stack trace
```python
# Good
try:
    result = risky_operation()
except OperationError as e:
    msg = "Failed to complete operation"
    raise RuntimeError(msg) from e  # Preserves original exception

# Bad
try:
    result = risky_operation()
except OperationError:
    raise RuntimeError("Failed")  # Loses original error context
```

---

## Performance Guidelines

### Avoid Premature Optimization

Focus on correctness first, optimize later if needed.

### Efficient Patterns

**Use List Comprehensions** (when readable):
```python
# Good
squares = [x**2 for x in range(10)]

# Bad
squares = []
for x in range(10):
    squares.append(x**2)
```

**Use `pathlib.Path`** for file operations:
```python
# Good
from pathlib import Path

config_file = Path("config.yaml")
if config_file.exists():
    content = config_file.read_text()

# Bad
import os

config_file = "config.yaml"
if os.path.exists(config_file):
    with open(config_file) as f:
        content = f.read()
```

**Avoid Mutable Default Arguments**:
```python
# Good
def process(items: list[str] | None = None) -> list[str]:
    if items is None:
        items = []
    return items

# Bad
def process(items: list[str] = []) -> list[str]:  # Mutable default!
    return items
```

---

## Factory Pattern Adherence

GraphRAG uses factory pattern extensively. New components must follow this pattern.

### Factory Registration

```python
# In factory file (e.g., embedding_factory.py)

class EmbeddingFactory(Factory["LLMEmbedding"]):
    """Factory for creating Embedding instances."""

embedding_factory = EmbeddingFactory()

def register_embedding(
    embedding_type: str,
    embedding_initializer: Callable[..., "LLMEmbedding"],
    scope: "ServiceScope" = "transient",
) -> None:
    """Register a custom embedding implementation."""
    embedding_factory.register(embedding_type, embedding_initializer, scope)

# Lazy registration in factory
def create_embedding(model_config: "ModelConfig", ...) -> "LLMEmbedding":
    strategy = model_config.type

    if strategy not in embedding_factory:
        match strategy:
            case LLMProviderType.SentenceTransformer:
                from graphrag_llm.embedding.sentence_transformer_embedding import (
                    SentenceTransformerEmbedding,
                )
                register_embedding(
                    embedding_type=LLMProviderType.SentenceTransformer,
                    embedding_initializer=SentenceTransformerEmbedding,
                    scope="singleton",  # Share instance
                )

    return embedding_factory.create(strategy, init_args={...})
```

### Why Singleton for SentenceTransformer?

Models are expensive to load, so we share instances:
- `scope="singleton"`: One instance per model_id
- `scope="transient"`: New instance every time

---

## Copyright Headers

All files must include:

```python
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
```

---

## Pre-Commit Checklist

Before requesting assessment, verify:

### Automated Checks
- [ ] `uv run ruff format . --check` passes
- [ ] `uv run ruff check .` passes (0 violations)
- [ ] `uv run pyright` passes (0 errors)
- [ ] `uv run poe test_unit` passes (all tests)
- [ ] `uv run coverage report` shows ≥90% coverage

### Manual Checks
- [ ] All functions have type annotations
- [ ] All functions have numpy-style docstrings
- [ ] No TODOs or FIXMEs in production code
- [ ] Error messages are helpful and specific
- [ ] No commented-out code
- [ ] No debug print statements
- [ ] Imports are organized correctly
- [ ] Factory pattern followed (if applicable)
- [ ] Backward compatible with existing code

### Documentation Checks
- [ ] README updated (if applicable)
- [ ] CHANGELOG entry added (semversioner)
- [ ] Examples tested and working
- [ ] Docstrings accurate and complete

---

## Assessment Failure Examples

### Will Fail Assessment

```python
# Missing copyright
"""Module without copyright header."""

# Missing type annotations
def embedding(self, input):  # No types!
    return self._model.encode(input)

# Missing docstring
def _helper(self, data):
    return process(data)

# Generic exception
if not data:
    raise Exception("Bad data")  # Too generic

# No error message variable
raise ValueError("Model not found")  # Should use msg variable

# Commented code
# def old_implementation():
#     ...  # Remove this!

# Debug prints
def process(data):
    print(f"Debug: {data}")  # Remove before commit
    return result
```

### Will Pass Assessment

```python
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""SentenceTransformer embedding implementation."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from graphrag_llm.types import LLMEmbeddingArgs, LLMEmbeddingResponse


def embedding(
    self,
    /,
    **kwargs: Unpack["LLMEmbeddingArgs"],
) -> "LLMEmbeddingResponse":
    """Generate embeddings for input texts.

    Parameters
    ----------
    **kwargs : Unpack[LLMEmbeddingArgs]
        Embedding arguments including input texts.

    Returns
    -------
    LLMEmbeddingResponse
        Generated embeddings with usage statistics.

    Raises
    ------
    ValueError
        If input is empty or invalid.
    """
    input_texts = kwargs.get("input", [])

    if not input_texts:
        msg = "Input texts cannot be empty"
        raise ValueError(msg)

    embeddings = self._model.encode(
        input_texts,
        batch_size=self._batch_size,
        normalize_embeddings=self._normalize_embeddings,
    )

    return self._format_response(embeddings)
```

---

**Next Document**: [03_AI_ASSESSMENT.md](./03_AI_ASSESSMENT.md)
