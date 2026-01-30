# Code Review Checklist for Assessment Agent

**Document**: 10_CODE_REVIEW_CHECKLIST.md
**Status**: Ready for Use
**Purpose**: Comprehensive checklist for AI Assessment Agent reviews

---

## How to Use This Checklist

1. **Run Automated Checks First** - Don't proceed to manual review if automated checks fail
2. **Check Items Systematically** - Go through each section in order
3. **Document Findings** - Note specific line numbers and issues
4. **Be Specific** - Provide actionable feedback, not just "fix this"
5. **Use Examples** - Show good vs bad code when providing feedback

---

## Level 1: Automated Checks (MUST PASS)

### ✅ 1.1 Code Formatting

```bash
uv run ruff format . --check
```

**Criteria**:
- [ ] No formatting violations
- [ ] Consistent style across all files
- [ ] Line length ≤88 characters
- [ ] Proper indentation (4 spaces)
- [ ] Trailing commas in multi-line structures

**If Failed**: Request `uv run poe format`

---

### ✅ 1.2 Linting

```bash
uv run ruff check .
```

**Check Results For**:

**Security (S rules)**:
- [ ] No `assert` in production code (S101)
- [ ] No hardcoded passwords/secrets (S105, S106)
- [ ] No `try/except/pass` (S108)
- [ ] Proper input validation (S301-S324)

**Import Organization (I rules)**:
- [ ] Imports properly sorted
- [ ] Standard lib → Third-party → Local order
- [ ] No unused imports

**Type Checking (TCH rules)**:
- [ ] Proper `TYPE_CHECKING` usage
- [ ] No circular imports
- [ ] Runtime vs type-checking imports separated

**Error Messages (EM rules)**:
- [ ] Exception messages in variables
- [ ] No f-strings in exception constructors

**Naming (N rules)**:
- [ ] snake_case for functions/variables
- [ ] PascalCase for classes
- [ ] UPPER_SNAKE_CASE for constants

**If Failed**: Review each violation, request specific fixes

---

### ✅ 1.3 Type Checking

```bash
uv run pyright
```

**Criteria**:
- [ ] Zero type errors
- [ ] All function parameters typed
- [ ] All return values typed
- [ ] Class attributes typed
- [ ] Proper use of Optional/Union types

**Common Issues to Check**:
```python
# Bad: Missing types
def process(data):
    return data.transform()

# Good: Fully typed
def process(data: pd.DataFrame) -> pd.DataFrame:
    return data.transform()
```

**If Failed**: Request type annotations for each missing type

---

### ✅ 1.4 Tests Pass

```bash
uv run poe test_unit
```

**Criteria**:
- [ ] All tests pass (0 failures)
- [ ] No tests skipped without justification
- [ ] Test execution time reasonable
- [ ] No test warnings

**If Failed**: Review test failures, identify root cause

---

### ✅ 1.5 Test Coverage

```bash
uv run coverage run -m pytest tests/unit/
uv run coverage report
```

**Criteria**:
- [ ] Overall coverage ≥90%
- [ ] New code coverage 100%
- [ ] No critical paths uncovered
- [ ] All branches covered

**Check Coverage Report**:
```
Name                                           Stmts   Miss  Cover
------------------------------------------------------------------
sentence_transformer_embedding.py                150      8    95%
```

**If Below 90%**: Identify uncovered lines and request additional tests

---

## Level 2: Manual Code Review (SHOULD PASS)

### 📐 2.1 Architecture & Design

#### Factory Pattern Compliance

**Check**: Does new code follow factory pattern?

```python
# ✅ Good: Factory registration
def create_embedding(model_config: ModelConfig, ...) -> LLMEmbedding:
    strategy = model_config.type

    if strategy not in embedding_factory:
        match strategy:
            case LLMProviderType.SentenceTransformer:
                register_embedding(
                    embedding_type=LLMProviderType.SentenceTransformer,
                    embedding_initializer=SentenceTransformerEmbedding,
                    scope="singleton",
                )

    return embedding_factory.create(strategy, ...)

# ❌ Bad: Direct instantiation
embedding = SentenceTransformerEmbedding(...)  # Bypasses factory!
```

**Checklist**:
- [ ] New classes registered via factory
- [ ] No direct instantiation in application code
- [ ] Proper use of singleton vs transient scope
- [ ] Factory pattern consistent with existing code

---

#### Dependency Injection

**Check**: Are dependencies injected, not hardcoded?

```python
# ✅ Good: Dependencies injected
class SentenceTransformerEmbedding(LLMEmbedding):
    def __init__(
        self,
        *,
        tokenizer: Tokenizer,        # Injected
        metrics_store: MetricsStore,  # Injected
        cache: Cache | None = None,   # Injected
        **kwargs,
    ):
        self._tokenizer = tokenizer
        self._metrics_store = metrics_store
        self._cache = cache

# ❌ Bad: Hardcoded dependencies
class SentenceTransformerEmbedding(LLMEmbedding):
    def __init__(self, model_name: str):
        self._tokenizer = LiteLLMTokenizer()  # Hardcoded!
        self._metrics_store = MetricsStore()  # Hardcoded!
```

**Checklist**:
- [ ] All dependencies passed via constructor
- [ ] No hardcoded service instantiation
- [ ] Dependencies are interface types (not concrete classes)
- [ ] Optional dependencies have defaults (None)

---

#### Interface Implementation

**Check**: Does class fully implement required interface?

```python
# Check that all abstract methods are implemented
class SentenceTransformerEmbedding(LLMEmbedding):  # Must implement LLMEmbedding
    def embedding(self, ...) -> LLMEmbeddingResponse:  # ✅ Required
        ...

    def embedding_async(self, ...) -> LLMEmbeddingResponse:  # ✅ Required
        ...

    @property
    def metrics_store(self) -> MetricsStore:  # ✅ Required
        ...

    @property
    def tokenizer(self) -> Tokenizer:  # ✅ Required
        ...
```

**Checklist**:
- [ ] All interface methods implemented
- [ ] Method signatures match interface exactly
- [ ] Return types match interface
- [ ] Properties implemented if required

---

#### Separation of Concerns

**Check**: Is code properly modularized?

**Checklist**:
- [ ] Classes have single responsibility
- [ ] Functions do one thing well
- [ ] No God classes (>500 lines)
- [ ] No God functions (>50 lines)
- [ ] Helper functions extracted appropriately

**Example Issues**:
```python
# ❌ Bad: Function doing too much
def embedding(self, input):
    # Device detection (separate concern)
    device = detect_device()

    # Model loading (separate concern)
    model = load_model(self.model_name, device)

    # Tokenization (separate concern)
    tokens = tokenize(input)

    # Embedding (actual concern)
    embeddings = model.encode(tokens)

    # Formatting (separate concern)
    return format_response(embeddings)

# ✅ Good: Single responsibility
def embedding(self, input):
    # Assumes model already loaded, uses injected tokenizer
    embeddings = self._model.encode(
        input,
        batch_size=self._batch_size,
    )
    return self._format_response(embeddings)
```

---

### 📝 2.2 Documentation Quality

#### Module Docstrings

**Required Elements**:
- [ ] Copyright header present
- [ ] Module-level docstring present
- [ ] Purpose clearly stated
- [ ] Key classes/functions mentioned

```python
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""SentenceTransformer-based embedding generation.

This module provides local, free embedding generation using the
sentence-transformers library. Supports CUDA, CPU, and MPS devices.

Key Classes
-----------
SentenceTransformerEmbedding
    Main embedding class implementing LLMEmbedding interface.
"""
```

---

#### Class Docstrings

**Required Elements** (NumPy style):
- [ ] One-line summary
- [ ] Extended description
- [ ] Attributes section
- [ ] Examples section (for complex classes)

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
    >>> from graphrag_llm.embedding import create_embedding
    >>> config = ModelConfig(model="all-MiniLM-L6-v2", type="sentence_transformer")
    >>> embedding = create_embedding(config)
    >>> result = embedding.embedding(input=["Hello world"])
    >>> len(result.data[0]["embedding"])
    384
    """
```

**Check**:
- [ ] Summary line ≤80 characters
- [ ] Extended description clear and helpful
- [ ] All instance attributes listed
- [ ] Attribute types specified
- [ ] Usage example provided

---

#### Function/Method Docstrings

**Required Elements**:
- [ ] One-line summary
- [ ] Parameters section (all parameters)
- [ ] Returns section
- [ ] Raises section (all exceptions)
- [ ] Examples (for complex functions)

```python
def embedding(
    self,
    /,
    **kwargs: Unpack["LLMEmbeddingArgs"],
) -> "LLMEmbeddingResponse":
    """Generate embeddings for input texts.

    This method encodes input texts into dense vector embeddings using
    the loaded SentenceTransformer model. Embeddings are generated in
    batches for efficiency.

    Parameters
    ----------
    **kwargs : Unpack[LLMEmbeddingArgs]
        Keyword arguments including:
        - input : list[str] or str
            Text(s) to generate embeddings for.
        - metrics : Metrics | None
            Optional metrics tracking dictionary.

    Returns
    -------
    LLMEmbeddingResponse
        Response containing:
        - data : list[dict]
            Embeddings with "embedding" and "index" keys.
        - model : str
            Model identifier used.
        - usage : dict
            Token usage statistics.

    Raises
    ------
    ValueError
        If input is empty or has invalid format.
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
    This method runs synchronously. For async operations,
    use `embedding_async()` instead.

    See Also
    --------
    embedding_async : Async version of this method
    """
```

**Check**:
- [ ] All parameters documented
- [ ] Parameter types clear
- [ ] Return value fully described
- [ ] All raised exceptions listed
- [ ] Example shows realistic usage

---

### 🚨 2.3 Error Handling

#### Error Message Quality

**Check**: Are error messages helpful?

```python
# ✅ Good: Helpful error with solution
try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    msg = (
        "sentence-transformers package not installed. "
        "Install it with: pip install sentence-transformers\n"
        "Or install graphrag with: pip install graphrag[local-embeddings]"
    )
    raise ImportError(msg) from e

# ❌ Bad: Unhelpful error
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("Package missing")
```

**Checklist**:
- [ ] Error messages describe what went wrong
- [ ] Error messages suggest how to fix
- [ ] Error messages are user-friendly
- [ ] Error context preserved with `from e`

---

#### Exception Specificity

**Check**: Are specific exception types used?

```python
# ✅ Good: Specific exceptions
if model_name is None:
    msg = "model_name cannot be None"
    raise ValueError(msg)

if not config_path.exists():
    msg = f"Configuration file not found: {config_path}"
    raise FileNotFoundError(msg)

# ❌ Bad: Generic exception
if not model_name:
    raise Exception("Bad input")

if not config_path.exists():
    raise Exception("File not found")
```

**Checklist**:
- [ ] No bare `Exception` raised
- [ ] Appropriate exception type for error
- [ ] Custom exceptions documented
- [ ] Exception hierarchy logical

---

#### Error Message Variables

**Check**: Are error messages in variables?

```python
# ✅ Good: Message in variable (EM101)
if not input_texts:
    msg = "Input texts cannot be empty"
    raise ValueError(msg)

# ❌ Bad: Inline string
if not input_texts:
    raise ValueError("Input texts cannot be empty")  # EM101 violation
```

**Checklist**:
- [ ] All exception messages use `msg` variable
- [ ] No f-strings directly in exceptions
- [ ] Consistent message format

---

### 🧪 2.4 Test Quality

#### Test Structure

**Check**: Are tests well-structured?

```python
# ✅ Good: Clear Arrange-Act-Assert
def test_embedding_generation_with_single_text():
    """Test embedding generation with a single text input.

    Arrange: Create embedding instance with mocked model
    Act: Generate embedding for single text
    Assert: Embedding has correct format and dimensions
    """
    # Arrange
    mock_model = Mock()
    mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
    embedding = SentenceTransformerEmbedding(...)
    embedding._model = mock_model

    # Act
    result = embedding.embedding(input=["Hello world"])

    # Assert
    assert len(result.data) == 1
    assert len(result.data[0]["embedding"]) == 3
    assert result.data[0]["index"] == 0

# ❌ Bad: Unclear structure
def test_embedding():
    embedding = SentenceTransformerEmbedding(...)
    result = embedding.embedding(input=["Hello"])
    assert result  # What are we testing?
```

**Checklist**:
- [ ] Clear Arrange-Act-Assert structure
- [ ] Test name describes what's tested
- [ ] Docstring explains test purpose
- [ ] One concept per test

---

#### Test Coverage

**Check**: Are all paths tested?

**Required Coverage**:
- [ ] Happy path tested
- [ ] Edge cases tested (empty input, None, etc.)
- [ ] Error paths tested
- [ ] Boundary conditions tested

```python
# Required tests for a method:
def test_method_with_valid_input():  # Happy path
    ...

def test_method_with_empty_input():  # Edge case
    ...

def test_method_with_none_input():  # Edge case
    ...

def test_method_raises_error_for_invalid_input():  # Error path
    ...

def test_method_with_boundary_value():  # Boundary
    ...
```

---

#### Test Independence

**Check**: Are tests independent?

```python
# ✅ Good: Independent tests using fixtures
@pytest.fixture
def embedding_instance():
    return SentenceTransformerEmbedding(...)

def test_embedding_generation(embedding_instance):
    result = embedding_instance.embedding(input=["test"])
    assert result

def test_async_embedding(embedding_instance):
    result = await embedding_instance.embedding_async(input=["test"])
    assert result

# ❌ Bad: Tests depend on shared state
class TestEmbedding:
    embedding = None  # Shared state!

    def test_initialization(self):
        self.embedding = SentenceTransformerEmbedding(...)

    def test_embedding(self):
        # Depends on test_initialization running first!
        result = self.embedding.embedding(input=["test"])
```

**Checklist**:
- [ ] Tests can run in any order
- [ ] No shared mutable state
- [ ] Fixtures used for setup
- [ ] Tests clean up after themselves

---

### 💅 2.5 Code Quality

#### No Code Duplication

**Check**: Is there duplicated code?

```python
# ❌ Bad: Duplication
def embedding(self, input):
    # Validation
    if not input:
        msg = "Input cannot be empty"
        raise ValueError(msg)
    if not isinstance(input, (str, list)):
        msg = "Input must be string or list"
        raise TypeError(msg)

    embeddings = self._model.encode(input)
    return self._format_response(embeddings)

async def embedding_async(self, input):
    # Same validation duplicated!
    if not input:
        msg = "Input cannot be empty"
        raise ValueError(msg)
    if not isinstance(input, (str, list)):
        msg = "Input must be string or list"
        raise TypeError(msg)

    embeddings = await asyncio.to_thread(self._model.encode, input)
    return self._format_response(embeddings)

# ✅ Good: Extract validation
def _validate_input(self, input):
    """Validate input texts."""
    if not input:
        msg = "Input cannot be empty"
        raise ValueError(msg)
    if not isinstance(input, (str, list)):
        msg = "Input must be string or list"
        raise TypeError(msg)

def embedding(self, input):
    self._validate_input(input)
    embeddings = self._model.encode(input)
    return self._format_response(embeddings)

async def embedding_async(self, input):
    self._validate_input(input)
    embeddings = await asyncio.to_thread(self._model.encode, input)
    return self._format_response(embeddings)
```

**Checklist**:
- [ ] No copy-pasted code blocks
- [ ] Common logic extracted to functions
- [ ] DRY principle followed

---

#### Clear Naming

**Check**: Are names clear and descriptive?

```python
# ✅ Good: Clear names
def _detect_device() -> str:
    """Detect the best available device for embedding generation."""
    ...

def _create_base_embeddings(...) -> tuple[LLMEmbeddingFunction, AsyncLLMEmbeddingFunction]:
    """Create base embedding functions for SentenceTransformer."""
    ...

# ❌ Bad: Unclear names
def _det() -> str:  # What does this detect?
    ...

def _cbe(...) -> tuple:  # What is cbe?
    ...
```

**Checklist**:
- [ ] Function names describe what they do
- [ ] Variable names are descriptive
- [ ] No single-letter names (except loop counters)
- [ ] Boolean names start with is/has/can/should

---

#### Function Length

**Check**: Are functions reasonably sized?

**Guidelines**:
- [ ] Functions <50 lines (ideal)
- [ ] Functions <100 lines (acceptable)
- [ ] Functions >100 lines (should be split)

**If Long**: Suggest splitting into helper functions

---

#### Magic Numbers

**Check**: Are magic numbers explained?

```python
# ❌ Bad: Magic numbers
if len(embeddings) > 384:
    ...

batch_size = 32

# ✅ Good: Named constants or explained
# Model-specific embedding dimension
MINILM_EMBEDDING_DIM = 384

if len(embeddings) > MINILM_EMBEDDING_DIM:
    ...

# Default batch size for efficient GPU utilization
DEFAULT_BATCH_SIZE = 32
batch_size = config.batch_size or DEFAULT_BATCH_SIZE
```

**Checklist**:
- [ ] No unexplained numeric literals
- [ ] Constants have meaningful names
- [ ] Magic numbers commented if inline

---

## Level 3: Best Practices (NICE TO HAVE)

### ⚡ 3.1 Performance

**Check for**:
- [ ] No obvious performance bottlenecks
- [ ] Efficient data structures used
- [ ] Appropriate use of batching
- [ ] Minimal copying of data

**Not Blockers**: Note for future optimization

---

### 🔒 3.2 Security

**Check for**:
- [ ] Input validation at boundaries
- [ ] No SQL injection risks
- [ ] No command injection risks
- [ ] Secrets not hardcoded

**Use**: `bandit` security scanner

---

### 📦 3.3 Maintainability

**Check for**:
- [ ] Code is easy to understand
- [ ] Complex logic is commented
- [ ] Consistent patterns used
- [ ] Easy to extend

---

## Assessment Result Template

Use this template for feedback:

```
ASSESSMENT RESULT - Iteration [N]
Task: [Task ID]
Status: [✅ APPROVED / ⚠️ CHANGES REQUESTED / ❌ BLOCKED]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AUTOMATED CHECKS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[✅/❌] Formatting: [PASS/FAIL]
[✅/❌] Linting: [PASS/FAIL - X violations]
[✅/❌] Type Checking: [PASS/FAIL - X errors]
[✅/❌] Unit Tests: [PASS/FAIL - X/Y passed]
[✅/⚠️ ] Coverage: [PASS/WARNING - X%]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ISSUES FOUND
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[Category - Priority]
1. [Issue description]
   Location: file.py:line_number
   Fix: [Specific actionable fix]

2. [Issue description]
   Location: file.py:line_number
   Fix: [Specific actionable fix]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Must Fix (blockers): X issues
Should Fix (quality): X issues
Nice to Have: X issues

Next Steps:
1. [Step 1]
2. [Step 2]

Estimated Time: [Time estimate]
```

---

## Summary

Before approving, verify ALL of:

**Level 1 (Automated) - MUST PASS**:
- [ ] Formatting: 0 violations
- [ ] Linting: 0 violations
- [ ] Type Checking: 0 errors
- [ ] Tests: All pass
- [ ] Coverage: ≥90%

**Level 2 (Manual) - SHOULD PASS**:
- [ ] Architecture: Factory pattern followed
- [ ] Documentation: Complete and clear
- [ ] Error Handling: Helpful and specific
- [ ] Tests: Comprehensive and independent
- [ ] Code Quality: No duplication, clear naming

**Level 3 (Best Practices) - NICE TO HAVE**:
- [ ] Performance: No obvious issues
- [ ] Security: Best practices followed
- [ ] Maintainability: Easy to understand and extend

---

**Use this checklist for every assessment iteration!**
