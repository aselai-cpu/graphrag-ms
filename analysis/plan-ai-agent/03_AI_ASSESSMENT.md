# AI Agent Assessment Framework

**Document**: 03_AI_ASSESSMENT.md
**Status**: Ready for Implementation
**Purpose**: Define iterative assessment process and criteria for AI-to-AI collaboration

---

## Overview

This document defines how Assessment AI Agents review Implementation AI Agent work through multiple iterative cycles. The goal is to ensure high code quality through automated checks and structured feedback loops before human review.

---

## Assessment Agent Role

The Assessment Agent acts as an automated code reviewer that:
1. **Runs Automated Checks** - Executes tests, linting, type-checking
2. **Reviews Code Quality** - Evaluates design, patterns, documentation
3. **Provides Specific Feedback** - Actionable improvement suggestions
4. **Tracks Progress** - Monitors quality metrics across iterations
5. **Approves or Requests Changes** - Clear go/no-go decisions

---

## Iterative Assessment Cycle

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Iteration 1                                                    │
│  ├─ Implementation Agent: Initial implementation                │
│  ├─ Assessment Agent: First review                             │
│  └─ Result: Feedback provided → Changes needed                 │
│                                                                 │
│  Iteration 2                                                    │
│  ├─ Implementation Agent: Address feedback                      │
│  ├─ Assessment Agent: Second review                            │
│  └─ Result: Minor issues → One more iteration                  │
│                                                                 │
│  Iteration 3                                                    │
│  ├─ Implementation Agent: Final fixes                           │
│  ├─ Assessment Agent: Final review                             │
│  └─ Result: ✅ APPROVED → Move to next task                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Expected Iterations per Task**: 2-3 cycles
**Maximum Iterations**: 5 before escalation to human

---

## Assessment Criteria Hierarchy

###Level 1: Automated Checks (MUST PASS)

These checks must pass before manual review begins.

#### 1.1 Test Execution
```bash
uv run poe test_unit
```

**Criteria**:
- [ ] All tests pass (0 failures)
- [ ] No test skipped without justification
- [ ] Test execution time reasonable (<5min for unit tests)

**Failure Action**: Request fixes, do not proceed to manual review

#### 1.2 Code Formatting
```bash
uv run ruff format . --check
```

**Criteria**:
- [ ] All files properly formatted
- [ ] Consistent style across all files
- [ ] Line length ≤88 characters

**Failure Action**: Request auto-fix: `uv run poe format`

#### 1.3 Linting
```bash
uv run ruff check .
```

**Criteria**:
- [ ] Zero ruff violations
- [ ] All security warnings addressed (S rules)
- [ ] All import sorting correct (I rules)

**Failure Action**: Request fixes (auto-fix safe issues first)

#### 1.4 Type Checking
```bash
uv run pyright
```

**Criteria**:
- [ ] Zero type errors
- [ ] All functions have type annotations
- [ ] Proper use of TYPE_CHECKING imports

**Failure Action**: Request type annotation additions

#### 1.5 Test Coverage
```bash
uv run coverage run -m pytest tests/unit
uv run coverage report
```

**Criteria**:
- [ ] Overall coverage ≥90%
- [ ] New code coverage 100%
- [ ] No critical paths uncovered

**Failure Action**: Request additional tests for uncovered code

---

### Level 2: Manual Code Review (SHOULD PASS)

After automated checks pass, assess code quality manually.

#### 2.1 Architecture & Design

**Factory Pattern Adherence**:
- [ ] New components use existing factory pattern
- [ ] Registration follows lazy-loading approach
- [ ] Scope (singleton/transient) appropriate
- [ ] No direct instantiation in application code

**Separation of Concerns**:
- [ ] Clear single responsibility per class/function
- [ ] No God classes or functions (>100 lines)
- [ ] Proper abstraction layers maintained
- [ ] Dependencies injected, not hardcoded

**Interface Compliance**:
- [ ] New classes implement required interfaces fully
- [ ] Method signatures match interface definitions
- [ ] Return types consistent with interface

**Example Good**:
```python
# Good: Factory pattern, dependency injection, clear interfaces
class SentenceTransformerEmbedding(LLMEmbedding):  # Implements interface
    def __init__(
        self,
        *,
        model_id: str,
        model_config: ModelConfig,  # Injected
        tokenizer: Tokenizer,       # Injected
        metrics_store: MetricsStore,  # Injected
        cache_key_creator: CacheKeyCreator,  # Injected
        **kwargs,
    ):
        # Clean initialization with injected dependencies
        ...

# Registration via factory (lazy-loaded)
def create_embedding(model_config: ModelConfig, ...) -> LLMEmbedding:
    if strategy not in embedding_factory:
        register_embedding(
            embedding_type=LLMProviderType.SentenceTransformer,
            embedding_initializer=SentenceTransformerEmbedding,
            scope="singleton",
        )
    return embedding_factory.create(strategy, ...)
```

**Example Bad**:
```python
# Bad: Direct instantiation, no factory, hardcoded dependencies
class SentenceTransformerEmbedding:  # Doesn't implement interface!
    def __init__(self, model_name: str):
        # Hardcoded dependencies
        self.tokenizer = LiteLLMTokenizer()  # Should be injected!
        self.metrics = MetricsStore()  # Should be injected!

# Direct instantiation (bypasses factory)
embedding = SentenceTransformerEmbedding("model")  # Bad!
```

---

#### 2.2 Documentation Quality

**Module Docstrings**:
- [ ] Present at file top (after copyright)
- [ ] Describes module purpose clearly
- [ ] Mentions key classes/functions

**Class Docstrings**:
- [ ] Describes class purpose and usage
- [ ] Lists key attributes
- [ ] Includes usage example for complex classes
- [ ] Follows NumPy style

**Function/Method Docstrings**:
- [ ] One-line summary present
- [ ] All parameters documented
- [ ] Return value documented
- [ ] Exceptions documented
- [ ] Examples for complex functions

**Example Good**:
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

---

#### 2.3 Error Handling

**Error Message Quality**:
- [ ] Error messages are helpful and actionable
- [ ] Include suggestions for fixes when possible
- [ ] Messages defined as variables (not inline strings)
- [ ] Context preserved with `from e`

**Exception Specificity**:
- [ ] Specific exception types used (not generic `Exception`)
- [ ] Appropriate exception for error type
- [ ] Custom exceptions documented

**Example Good**:
```python
try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    msg = (
        "sentence-transformers package not installed. "
        "Install it with: pip install sentence-transformers\n"
        "Or install graphrag with: pip install graphrag[local-embeddings]"
    )
    raise ImportError(msg) from e  # Preserves stack trace

if model_name is None or model_name == "":
    msg = (
        f"Invalid model name: {model_name!r}. "
        "Expected a valid HuggingFace model identifier like "
        "'sentence-transformers/all-MiniLM-L6-v2'"
    )
    raise ValueError(msg)
```

**Example Bad**:
```python
try:
    from sentence_transformers import SentenceTransformer
except:  # Bad: Bare except
    raise Exception("Import failed")  # Bad: Generic exception, no context

if not model_name:
    raise ValueError("Bad model")  # Bad: Unhelpful message
```

---

#### 2.4 Test Quality

**Test Coverage**:
- [ ] All public methods tested
- [ ] Edge cases covered
- [ ] Error paths tested
- [ ] Happy path tested

**Test Structure**:
- [ ] Clear Arrange-Act-Assert structure
- [ ] One concept per test
- [ ] Descriptive test names
- [ ] Clear failure messages

**Test Independence**:
- [ ] Tests don't depend on execution order
- [ ] Tests clean up after themselves
- [ ] No shared mutable state between tests

**Example Good**:
```python
def test_embedding_generation_with_empty_input_raises_value_error():
    """Test that embedding generation raises ValueError for empty input.

    Arrange: Create embedding instance with valid config
    Act: Call embedding() with empty input list
    Assert: ValueError is raised with helpful message
    """
    # Arrange
    embedding = SentenceTransformerEmbedding(...)

    # Act & Assert
    with pytest.raises(ValueError) as exc_info:
        embedding.embedding(input=[])

    assert "cannot be empty" in str(exc_info.value).lower()
```

---

### Level 3: Implementation Best Practices (NICE TO HAVE)

These are not blockers but should be addressed when feasible.

#### 3.1 Performance Considerations
- [ ] No obvious performance issues
- [ ] Appropriate use of batching
- [ ] Efficient data structures used
- [ ] Avoids unnecessary copying

#### 3.2 Code Readability
- [ ] Clear variable names
- [ ] Functions not too long (<50 lines ideal)
- [ ] Complex logic commented
- [ ] Magic numbers explained

#### 3.3 Maintainability
- [ ] No code duplication
- [ ] Consistent patterns across codebase
- [ ] Easy to extend
- [ ] Configuration over hardcoding

---

## Assessment Workflow

### Step 1: Receive Implementation Signal

Implementation Agent signals readiness:
```
ASSESSMENT REQUEST
Task: Implement SentenceTransformerEmbedding.__init__
Phase: 2
Files Modified:
  - packages/graphrag-llm/graphrag_llm/embedding/sentence_transformer_embedding.py
Tests Added:
  - tests/unit/graphrag_llm/embedding/test_sentence_transformer_embedding.py::test_initialization_with_default_device
  - tests/unit/graphrag_llm/embedding/test_sentence_transformer_embedding.py::test_initialization_with_explicit_device
All Tests Pass: Yes
Coverage: 95%
Ready for Review: Yes
```

### Step 2: Run Automated Checks

```bash
# Navigate to repo
cd /path/to/graphrag-ms

# Run all automated checks
uv run poe check      # Format, lint, type-check
uv run poe test_unit  # Unit tests
uv run coverage report  # Coverage check
```

**Record Results**:
```
Automated Checks:
├─ Formatting: ✅ PASS
├─ Linting: ❌ FAIL (3 violations found)
│  └─ S108: try/except/pass at line 45
│  └─ D102: Missing docstring for _helper method
│  └─ UP035: Deprecated typing.List, use list instead
├─ Type Checking: ✅ PASS
├─ Unit Tests: ✅ PASS (12/12)
└─ Coverage: ⚠️ WARNING (88% - below 90% threshold)
   └─ Uncovered: lines 67-72 in sentence_transformer_embedding.py
```

### Step 3: Manual Code Review

If automated checks pass, perform manual review:

**Check Factory Pattern**:
```python
# ✅ Good: Lazy registration in factory
if strategy not in embedding_factory:
    match strategy:
        case LLMProviderType.SentenceTransformer:
            from graphrag_llm.embedding.sentence_transformer_embedding import ...
            register_embedding(...)

# Find Issues:
# - None, factory pattern correctly implemented
```

**Check Documentation**:
```python
# ⚠️ Issue: Missing Examples section in class docstring
class SentenceTransformerEmbedding(LLMEmbedding):
    """Local embedding generation using SentenceTransformers.

    ... (good description) ...

    Attributes
    ----------
    ... (properly documented) ...

    # MISSING: Examples section!
    """
```

**Check Error Handling**:
```python
# ✅ Good: Helpful error message with fix suggestion
try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    msg = (
        "sentence-transformers not installed. "
        "Install with: pip install sentence-transformers"
    )
    raise ImportError(msg) from e
```

**Check Tests**:
```python
# ⚠️ Issue: Missing test for error case
# Found test: test_initialization_with_default_device()
# Found test: test_initialization_with_explicit_device()
# MISSING test: test_initialization_raises_import_error_when_package_missing()
```

### Step 4: Provide Structured Feedback

```
ASSESSMENT RESULT - Iteration 1
Task: Implement SentenceTransformerEmbedding.__init__
Status: ⚠️ CHANGES REQUESTED

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AUTOMATED CHECKS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Formatting: PASS
❌ Linting: FAIL (3 violations)
✅ Type Checking: PASS
✅ Unit Tests: PASS
⚠️  Coverage: WARNING (88%, target 90%)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LINTING ISSUES (MUST FIX)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. S108 at line 45 (sentence_transformer_embedding.py)
   Issue: try/except/pass silently swallows errors
   Fix: Add logging or re-raise specific exceptions

2. D102 at line 89 (sentence_transformer_embedding.py)
   Issue: Missing docstring for _helper method
   Fix: Add numpy-style docstring

3. UP035 at line 12 (sentence_transformer_embedding.py)
   Issue: Deprecated typing.List
   Fix: Change `from typing import List` to use `list` (Python 3.10+)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COVERAGE ISSUES (SHOULD FIX)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. Lines 67-72 uncovered (sentence_transformer_embedding.py)
   Missing test: Device fallback logic when CUDA unavailable
   Fix: Add test_device_detection_fallback_to_cpu()

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DOCUMENTATION ISSUES (SHOULD FIX)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5. Missing Examples section in SentenceTransformerEmbedding docstring
   Fix: Add usage example showing basic initialization

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TEST ISSUES (SHOULD FIX)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
6. Missing test for ImportError case
   Fix: Add test_initialization_raises_import_error_when_package_missing()

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Must Fix (blockers): 3 issues
Should Fix (quality): 3 issues
Nice to Have: 0 issues

Next Steps:
1. Fix linting violations (issues 1-3)
2. Add missing test (issue 6)
3. Improve coverage to 90%+ (issue 4)
4. Add documentation example (issue 5)
5. Re-run checks and request re-assessment

Estimated Time: 30-45 minutes
```

### Step 5: Track Iteration

**Iteration Log**:
```
Task: Implement SentenceTransformerEmbedding.__init__
├─ Iteration 1: 2024-01-30 10:00
│  └─ Result: CHANGES REQUESTED (3 must-fix, 3 should-fix)
├─ Iteration 2: 2024-01-30 11:00
│  └─ Result: CHANGES REQUESTED (1 should-fix remaining)
└─ Iteration 3: 2024-01-30 11:30
   └─ Result: ✅ APPROVED
```

### Step 6: Approve or Escalate

**Approval**:
```
ASSESSMENT RESULT - Iteration 3
Task: Implement SentenceTransformerEmbedding.__init__
Status: ✅ APPROVED

All checks passed:
✅ Formatting: PASS
✅ Linting: PASS (0 violations)
✅ Type Checking: PASS
✅ Unit Tests: PASS (15/15)
✅ Coverage: PASS (94%)
✅ Documentation: COMPLETE
✅ Tests: COMPREHENSIVE
✅ Code Quality: EXCELLENT

Ready to proceed to next task: Implement embedding() method

Great work! The implementation follows all patterns correctly.
```

**Escalation** (if >5 iterations):
```
ESCALATION REQUIRED
Task: Implement SentenceTransformerEmbedding.__init__
Iterations: 5
Status: NOT RESOLVED

Persistent Issues:
- Type checking errors not resolving
- Design pattern concern: direct instantiation vs factory

Recommended Actions:
1. Human review of type annotation approach
2. Architecture discussion on factory pattern usage

Implementation Agent Notes:
[Agent's explanation of challenges]

Assessment Agent Notes:
[Agent's concerns and suggestions]
```

---

## Assessment Templates

### Quick Pass Template

For simple, well-executed tasks:

```
ASSESSMENT RESULT
Task: [Task Name]
Status: ✅ APPROVED (Quick Pass)

All automated checks passed:
✅ Tests, ✅ Formatting, ✅ Linting, ✅ Types, ✅ Coverage

Code quality excellent:
✅ Factory pattern correct
✅ Documentation complete
✅ Error handling appropriate
✅ Tests comprehensive

No issues found. Excellent work!
Ready for next task.
```

### Detailed Feedback Template

For tasks needing improvements:

```
ASSESSMENT RESULT - Iteration [N]
Task: [Task Name]
Status: ⚠️ CHANGES REQUESTED

[Automated Checks Section]
[Issues by Category]
[Summary]
[Next Steps]
```

---

## Quality Metrics Tracking

Track these metrics across all tasks:

### Per-Task Metrics
- Iterations to approval (target: ≤3)
- Issues found per iteration
- Time to resolution
- Test coverage achieved
- Documentation completeness

### Aggregate Metrics
- Average iterations per task
- Most common issue types
- Improvement trends over time
- Bottleneck identification

---

## Continuous Improvement

After each phase, Assessment Agent should:

1. **Identify Patterns**: Common issues across tasks
2. **Update Guidelines**: Refine assessment criteria based on learnings
3. **Suggest Process Improvements**: Optimize workflow
4. **Share Learnings**: Document best practices discovered

---

## Summary Checklist for Assessment Agent

Before approving any task:

- [ ] All automated checks pass
- [ ] Code follows factory pattern
- [ ] Documentation complete and clear
- [ ] Tests comprehensive (90%+ coverage)
- [ ] Error handling appropriate
- [ ] No code smells or anti-patterns
- [ ] Backward compatibility maintained
- [ ] Performance considerations addressed
- [ ] Security best practices followed

---

**Next Document**: [04_IMPLEMENTATION_WORKFLOW.md](./04_IMPLEMENTATION_WORKFLOW.md)
