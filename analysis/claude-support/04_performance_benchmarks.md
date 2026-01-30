# Performance Benchmarks and Quality Comparison

**Date**: 2026-01-30
**Status**: Complete (Planning - POC Implementation Required)

---

## Executive Summary

This document outlines the benchmarking methodology for comparing OpenAI and Claude (+ SentenceTransformer) for GraphRAG tasks. **Actual benchmarks require POC implementation** and will be conducted during Phase 1 of the implementation plan.

**Planned Benchmarks**:
1. **Entity Extraction Quality** - Precision, recall, F1 score
2. **Community Report Quality** - Human evaluation, coherence metrics
3. **Query Response Quality** - Accuracy, relevance, completeness
4. **Embedding Quality** - Retrieval performance (P@k, R@k, NDCG)
5. **Performance** - Latency, throughput, cost per operation
6. **Reliability** - JSON parsing success rate, error rates

---

## Benchmark Methodology

### Test Dataset

**Dataset Selection**:
- Use existing GraphRAG test dataset: `tests/fixtures/test_data/`
- Or create standardized benchmark: 100 documents from diverse domains

**Domains**:
1. **News Articles** (25 docs) - Current events, factual information
2. **Research Papers** (25 docs) - Technical, citation-heavy
3. **Corporate Documents** (25 docs) - Business communications
4. **Wikipedia Articles** (25 docs) - Encyclopedia entries

**Size**: ~100,000 words total (~150K tokens)

---

### Benchmark 1: Entity Extraction Quality

**Metrics**:
- **Precision**: Correct entities / Total extracted entities
- **Recall**: Correct entities / Ground truth entities
- **F1 Score**: Harmonic mean of precision and recall
- **Entity Type Accuracy**: Correct entity type assignments
- **Relationship Accuracy**: Correct relationship identification

**Ground Truth**: Manual annotation of entities and relationships

**Models to Test**:
1. OpenAI GPT-4 Turbo (baseline)
2. OpenAI GPT-4o
3. Claude 3.5 Sonnet
4. Claude 3 Haiku

**Test Procedure**:
```python
# For each test document:
for document in test_documents:
    # Extract entities with each model
    entities_gpt4 = extract_entities(document, model="gpt-4-turbo")
    entities_gpt4o = extract_entities(document, model="gpt-4o")
    entities_sonnet = extract_entities(document, model="claude-3-5-sonnet")
    entities_haiku = extract_entities(document, model="claude-3-haiku")

    # Compare against ground truth
    metrics_gpt4 = evaluate(entities_gpt4, ground_truth)
    metrics_gpt4o = evaluate(entities_gpt4o, ground_truth)
    metrics_sonnet = evaluate(entities_sonnet, ground_truth)
    metrics_haiku = evaluate(entities_haiku, ground_truth)
```

**Expected Results** (based on model benchmarks):

| Model | Precision | Recall | F1 Score | Type Accuracy |
|-------|-----------|--------|----------|---------------|
| GPT-4 Turbo | 85-90% | 80-85% | 82-87% | 90% |
| GPT-4o | 87-92% | 82-87% | 84-89% | 92% |
| **Claude 3.5 Sonnet** | **88-93%** | **83-88%** | **85-90%** | **93%** ✅ |
| Claude 3 Haiku | 80-85% | 75-80% | 77-82% | 85% |

**Prediction**: Claude 3.5 Sonnet will match or slightly exceed GPT-4o, Claude 3 Haiku will be slightly lower but acceptable.

---

### Benchmark 2: JSON Output Reliability

**Metrics**:
- **JSON Parse Success Rate**: Valid JSON outputs / Total attempts
- **Schema Compliance**: Outputs matching expected schema
- **Retry Rate**: Requests requiring retry due to invalid output

**Models to Test**:
1. OpenAI GPT-4 Turbo (with native JSON mode)
2. OpenAI GPT-4o (with native JSON mode)
3. Claude 3.5 Sonnet (with prompting)
4. Claude 3 Haiku (with prompting)

**Test Procedure**:
```python
# For each model, attempt 1000 entity extractions
for i in range(1000):
    response = model.extract_entities(test_text)

    try:
        parsed = json.loads(response)
        json_valid += 1

        if validate_schema(parsed):
            schema_valid += 1
    except json.JSONDecodeError:
        json_invalid += 1
        retries += 1
```

**Expected Results**:

| Model | JSON Valid | Schema Valid | Retry Rate |
|-------|------------|--------------|------------|
| GPT-4 Turbo (JSON mode) | 100% | 99% | 0% |
| GPT-4o (JSON mode) | 100% | 99% | 0% |
| **Claude 3.5 Sonnet** | **99.9%** | **99%** | **0.1%** ✅ |
| Claude 3 Haiku | 99.5% | 98% | 0.5% |

**Prediction**: Claude will be slightly lower than OpenAI's native JSON mode but still highly reliable (99.9%+).

---

### Benchmark 3: Community Report Quality

**Metrics**:
- **Coherence**: Logical flow and readability
- **Completeness**: Coverage of important information
- **Accuracy**: Factual correctness
- **Conciseness**: Information density

**Evaluation Method**: Human evaluation + automated metrics

**Automated Metrics**:
- Perplexity (lower = more coherent)
- BLEU score (vs reference summaries)
- ROUGE score (recall-oriented)
- Readability scores (Flesch-Kincaid)

**Human Evaluation**:
- 3 evaluators rate each report on 1-5 scale
- Criteria: Coherence, Completeness, Accuracy, Usefulness

**Models to Test**:
1. OpenAI GPT-4 Turbo
2. Claude 3.5 Sonnet
3. Claude 3 Opus

**Test Procedure**:
```python
# Generate community reports for 50 communities
for community in test_communities:
    report_gpt4 = generate_report(community, model="gpt-4-turbo")
    report_sonnet = generate_report(community, model="claude-3-5-sonnet")
    report_opus = generate_report(community, model="claude-3-opus")

    # Human evaluation
    scores_gpt4 = human_evaluate(report_gpt4)
    scores_sonnet = human_evaluate(report_sonnet)
    scores_opus = human_evaluate(report_opus)
```

**Expected Results**:

| Model | Coherence | Completeness | Accuracy | Overall |
|-------|-----------|--------------|----------|---------|
| GPT-4 Turbo | 4.2/5 | 4.0/5 | 4.1/5 | 4.1/5 |
| **Claude 3.5 Sonnet** | **4.5/5** | **4.3/5** | **4.4/5** | **4.4/5** ✅ |
| Claude 3 Opus | 4.6/5 | 4.4/5 | 4.5/5 | 4.5/5 |

**Prediction**: Claude 3.5 Sonnet will exceed GPT-4 Turbo in report quality (superior reasoning).

---

### Benchmark 4: Query Response Quality

**Metrics**:
- **Accuracy**: Correct information in response
- **Relevance**: On-topic and addresses question
- **Completeness**: Covers all aspects of question
- **Clarity**: Easy to understand

**Evaluation Method**: Human evaluation with standardized questions

**Test Set**: 100 questions across difficulty levels:
- 40 Simple factual questions
- 40 Analytical questions
- 20 Complex reasoning questions

**Models to Test**:
1. OpenAI GPT-4 Turbo
2. OpenAI GPT-4o
3. Claude 3.5 Sonnet
4. Claude 3 Opus

**Expected Results**:

| Model | Simple | Analytical | Complex | Overall |
|-------|--------|------------|---------|---------|
| GPT-4 Turbo | 4.3/5 | 4.0/5 | 3.8/5 | 4.0/5 |
| GPT-4o | 4.4/5 | 4.1/5 | 3.9/5 | 4.1/5 |
| **Claude 3.5 Sonnet** | **4.5/5** | **4.4/5** | **4.3/5** | **4.4/5** ✅ |
| Claude 3 Opus | 4.6/5 | 4.5/5 | 4.5/5 | 4.5/5 |

**Prediction**: Claude will excel at complex reasoning questions due to superior reasoning capabilities.

---

### Benchmark 5: Embedding Quality

**Metrics**:
- **Precision@k**: Relevant results in top k
- **Recall@k**: Coverage of relevant results
- **NDCG@k**: Normalized discounted cumulative gain
- **Mean Reciprocal Rank (MRR)**: Average rank of first relevant result

**Test Procedure**:
1. Generate embeddings for all entities and text units
2. Create test queries with known relevant results
3. Perform vector similarity search
4. Compare retrieval quality across embedding models

**Models to Test**:
1. OpenAI text-embedding-3-large (3072 dims)
2. OpenAI text-embedding-3-small (1536 dims)
3. Voyage voyage-large-2 (1536 dims)
4. BAAI/bge-large-en-v1.5 (1024 dims)
5. intfloat/e5-large-v2 (1024 dims)
6. all-mpnet-base-v2 (768 dims)

**Test Queries**: 50 queries with manually annotated relevant results

**Expected Results** (based on MTEB benchmarks):

| Model | P@10 | R@10 | NDCG@10 | MRR |
|-------|------|------|---------|-----|
| OpenAI text-embedding-3-large | 0.78 | 0.62 | 0.81 | 0.72 |
| OpenAI text-embedding-3-small | 0.75 | 0.59 | 0.78 | 0.69 |
| Voyage voyage-large-2 | 0.79 | 0.63 | 0.82 | 0.73 |
| **BAAI/bge-large-en-v1.5** | **0.80** | **0.64** | **0.83** | **0.74** ✅ |
| intfloat/e5-large-v2 | 0.79 | 0.63 | 0.82 | 0.73 |
| all-mpnet-base-v2 | 0.72 | 0.56 | 0.75 | 0.66 |

**Prediction**: BGE-large and E5-large will match or exceed OpenAI embeddings (per MTEB leaderboard).

---

### Benchmark 6: Performance and Latency

**Metrics**:
- **Latency**: Time to complete single request
- **Throughput**: Requests per minute
- **Cost per 1K tokens**: Input and output costs
- **Time to index**: Total time for 100-document corpus

**Models to Test**:
- Completions: GPT-4 Turbo, GPT-4o, Claude 3.5 Sonnet, Claude 3 Haiku
- Embeddings: OpenAI, Voyage, BGE-large (CPU), BGE-large (GPU)

**Test Procedure**:
```python
import time

# Completion latency
start = time.time()
response = model.completion(messages=test_messages, max_tokens=1000)
latency = time.time() - start

# Throughput
start = time.time()
responses = []
for i in range(100):
    responses.append(model.completion(messages=test_messages))
throughput = 100 / (time.time() - start)

# Full indexing time
start = time.time()
index = create_index(documents=test_documents, model=model)
indexing_time = time.time() - start
```

**Expected Results - Completion Latency**:

| Model | 100 token output | 1000 token output | Tokens/sec |
|-------|------------------|-------------------|------------|
| GPT-4 Turbo | 2.5s | 25s | 40 tok/s |
| GPT-4o | 1.3s | 12.5s | 80 tok/s |
| **Claude 3.5 Sonnet** | **1.2s** | **12s** | **85 tok/s** ✅ |
| **Claude 3 Haiku** | **0.8s** | **8s** | **120 tok/s** ✅ |

**Expected Results - Embedding Latency**:

| Model | 100 texts | Device | Time | Texts/sec |
|-------|-----------|--------|------|-----------|
| OpenAI text-embedding-3-small | 100 | API | 2-3s | 40/s |
| **BGE-large** | 100 | GPU (A100) | **0.5s** | **200/s** ✅ |
| BGE-large | 100 | CPU (16 cores) | 5s | 20/s |
| E5-large | 100 | GPU (A100) | 0.6s | 167/s |

**Expected Results - Full Indexing Time** (100 documents):

| Configuration | Extraction | Embeddings | Total | Cost |
|---------------|------------|------------|-------|------|
| GPT-4 Turbo + OpenAI Embed | 45 min | 2 min | 47 min | $52 |
| GPT-4o + OpenAI Embed | 25 min | 2 min | 27 min | $12 |
| **Claude 3 Haiku + BGE (GPU)** | **15 min** | **0.5 min** | **15.5 min** ✅ | **$0.80** ✅ |
| Claude 3.5 Sonnet + BGE (GPU) | 20 min | 0.5 min | 20.5 min | $6 |

**Prediction**: Claude 3 Haiku + SentenceTransformer will be **3x faster** and **65x cheaper** than GPT-4 Turbo.

---

### Benchmark 7: Cost Efficiency

**Scenario**: Index 1000 documents (realistic production workload)

**Breakdown**:
- 10,000 text chunks for entity extraction
- 5,000 entities for summarization
- 200 communities for reports
- 15,000 embeddings (entities, communities, text units)

**Cost Calculation**:

| Configuration | Extraction | Summarization | Reports | Embeddings | **Total** |
|---------------|------------|---------------|---------|------------|-----------|
| GPT-4 Turbo + OpenAI | $230 | $80 | $20 | $0.34 | **$330.34** |
| GPT-4o + OpenAI | $57 | $20 | $5 | $0.34 | **$82.34** |
| Claude 3.5 Sonnet + OpenAI | $69 | $24 | $6 | $0.34 | **$99.34** |
| Claude 3 Haiku + OpenAI | $3.45 | $1.20 | $6 | $0.34 | **$10.99** ✅ |
| **Claude 3 Haiku (extract) + Sonnet (reports) + BGE** | **$3.45** | **$1.20** | **$6** | **$0** | **$10.65** ✅ |

**Cost Savings**:
- Claude Haiku + BGE vs GPT-4 Turbo: **96.8% savings** ($10.65 vs $330.34)
- Claude Haiku + BGE vs GPT-4o: **87.1% savings** ($10.65 vs $82.34)

---

## Benchmark Implementation Plan

### Phase 1: Setup (Week 1)

**Tasks**:
1. Create standardized test dataset (100 documents)
2. Manually annotate ground truth entities and relationships
3. Create evaluation scripts
4. Set up API keys for all providers

**Deliverables**:
- `tests/benchmarks/data/test_corpus.json`
- `tests/benchmarks/data/ground_truth.json`
- `tests/benchmarks/scripts/evaluate_extraction.py`
- `tests/benchmarks/scripts/evaluate_reports.py`
- `tests/benchmarks/scripts/evaluate_embeddings.py`

---

### Phase 2: Entity Extraction Benchmarks (Week 2)

**Tasks**:
1. Run entity extraction with all completion models
2. Calculate precision, recall, F1 scores
3. Measure JSON reliability
4. Compare latency and throughput

**Deliverables**:
- `results/entity_extraction_quality.csv`
- `results/json_reliability.csv`
- `results/extraction_performance.csv`

---

### Phase 3: Report and Query Benchmarks (Week 3)

**Tasks**:
1. Generate community reports with each model
2. Conduct human evaluations
3. Test query responses
4. Calculate coherence metrics

**Deliverables**:
- `results/report_quality.csv`
- `results/query_quality.csv`
- `results/human_evaluations.csv`

---

### Phase 4: Embedding Benchmarks (Week 4)

**Tasks**:
1. Generate embeddings with all providers
2. Run retrieval experiments
3. Calculate P@k, R@k, NDCG
4. Measure embedding speed (GPU vs CPU)

**Deliverables**:
- `results/embedding_quality.csv`
- `results/embedding_performance.csv`
- `results/retrieval_metrics.csv`

---

### Phase 5: Full Pipeline Benchmarks (Week 5)

**Tasks**:
1. Run complete indexing with each configuration
2. Measure end-to-end time and cost
3. Validate index quality
4. Test query performance on full indexes

**Deliverables**:
- `results/end_to_end_performance.csv`
- `results/cost_analysis.csv`
- Final benchmark report

---

## Benchmark Scripts

### Script 1: Entity Extraction Evaluation

**File**: `tests/benchmarks/scripts/evaluate_extraction.py`

```python
"""Evaluate entity extraction quality across models."""

import json
from typing import Any

from graphrag.index.operations.extract_graph import extract_graph
from graphrag_llm.completion.completion_factory import create_llm_completion


def load_test_data():
    """Load test corpus and ground truth."""
    with open("tests/benchmarks/data/test_corpus.json") as f:
        corpus = json.load(f)
    with open("tests/benchmarks/data/ground_truth.json") as f:
        ground_truth = json.load(f)
    return corpus, ground_truth


def evaluate_model(model_config: dict[str, Any], test_data: list):
    """Evaluate entity extraction for a given model."""
    model = create_llm_completion(**model_config)

    results = []
    for doc in test_data:
        # Extract entities
        entities, relationships = extract_entities(
            doc["text"],
            model=model,
        )

        # Calculate metrics
        precision = calculate_precision(entities, doc["ground_truth"])
        recall = calculate_recall(entities, doc["ground_truth"])
        f1 = 2 * (precision * recall) / (precision + recall)

        results.append({
            "doc_id": doc["id"],
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    return results


def calculate_precision(extracted, ground_truth):
    """Calculate precision: correct / extracted."""
    correct = len(set(extracted) & set(ground_truth))
    total = len(extracted)
    return correct / total if total > 0 else 0


def calculate_recall(extracted, ground_truth):
    """Calculate recall: correct / ground_truth."""
    correct = len(set(extracted) & set(ground_truth))
    total = len(ground_truth)
    return correct / total if total > 0 else 0


if __name__ == "__main__":
    corpus, ground_truth = load_test_data()

    models = [
        {"model_provider": "openai", "model": "gpt-4-turbo"},
        {"model_provider": "openai", "model": "gpt-4o"},
        {"model_provider": "anthropic", "model": "claude-3-5-sonnet-20241022"},
        {"model_provider": "anthropic", "model": "claude-3-haiku-20240307"},
    ]

    for model_config in models:
        print(f"Evaluating {model_config['model']}...")
        results = evaluate_model(model_config, corpus)

        # Save results
        with open(f"results/extraction_{model_config['model']}.json", "w") as f:
            json.dump(results, f, indent=2)
```

---

### Script 2: Embedding Quality Evaluation

**File**: `tests/benchmarks/scripts/evaluate_embeddings.py`

```python
"""Evaluate embedding quality across providers."""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def evaluate_embedding_model(model_config: dict, test_queries: list):
    """Evaluate retrieval quality for embedding model."""
    # Generate embeddings for corpus
    corpus_embeddings = generate_embeddings(corpus, model_config)

    # For each test query
    results = []
    for query in test_queries:
        # Generate query embedding
        query_emb = generate_embedding(query["text"], model_config)

        # Find top-k most similar
        similarities = cosine_similarity([query_emb], corpus_embeddings)[0]
        top_k_indices = np.argsort(similarities)[::-1][:10]

        # Calculate metrics
        relevant = set(query["relevant_docs"])
        retrieved = set(top_k_indices[:10])

        precision_at_10 = len(relevant & retrieved) / 10
        recall_at_10 = len(relevant & retrieved) / len(relevant)

        results.append({
            "query_id": query["id"],
            "p@10": precision_at_10,
            "r@10": recall_at_10,
        })

    return results


if __name__ == "__main__":
    models = [
        {"provider": "openai", "model": "text-embedding-3-large"},
        {"provider": "openai", "model": "text-embedding-3-small"},
        {"provider": "sentence_transformer", "model": "BAAI/bge-large-en-v1.5"},
        {"provider": "sentence_transformer", "model": "intfloat/e5-large-v2"},
    ]

    for model_config in models:
        print(f"Evaluating {model_config['model']}...")
        results = evaluate_embedding_model(model_config, test_queries)

        # Calculate averages
        avg_p10 = np.mean([r["p@10"] for r in results])
        avg_r10 = np.mean([r["r@10"] for r in results])

        print(f"  P@10: {avg_p10:.3f}")
        print(f"  R@10: {avg_r10:.3f}")
```

---

## Expected Benchmark Results Summary

### Quality Comparison

| Metric | GPT-4 Turbo | GPT-4o | Claude 3.5 Sonnet | Claude 3 Haiku |
|--------|-------------|---------|-------------------|----------------|
| **Entity F1** | 82-87% | 84-89% | **85-90%** ✅ | 77-82% |
| **JSON Reliability** | 100% | 100% | 99.9% | 99.5% |
| **Report Quality** | 4.1/5 | 4.2/5 | **4.4/5** ✅ | 3.8/5 |
| **Query Quality** | 4.0/5 | 4.1/5 | **4.4/5** ✅ | 3.9/5 |

### Performance Comparison

| Metric | GPT-4 Turbo | GPT-4o | Claude 3.5 Sonnet | Claude 3 Haiku |
|--------|-------------|---------|-------------------|----------------|
| **Speed (tok/s)** | 40 | 80 | 85 | **120** ✅ |
| **Indexing (100 docs)** | 47 min | 27 min | 20 min | **15 min** ✅ |

### Cost Comparison (1000 docs)

| Configuration | Completion Cost | Embedding Cost | **Total Cost** | **Savings** |
|---------------|----------------|----------------|----------------|-------------|
| GPT-4 Turbo + OpenAI | $330 | $0.34 | **$330.34** | 0% |
| GPT-4o + OpenAI | $82 | $0.34 | **$82.34** | 75% |
| Claude 3.5 Sonnet + OpenAI | $99 | $0.34 | **$99.34** | 70% |
| Claude 3 Haiku + OpenAI | $11 | $0.34 | **$11.34** | 97% |
| **Claude Haiku/Sonnet + BGE** | **$10.65** | **$0** | **$10.65** ✅ | **97%** ✅ |

### Embedding Comparison

| Model | P@10 | R@10 | NDCG@10 | Speed (GPU) | Cost |
|-------|------|------|---------|-------------|------|
| OpenAI text-embedding-3-large | 0.78 | 0.62 | 0.81 | 40/s (API) | $0.13/1M |
| OpenAI text-embedding-3-small | 0.75 | 0.59 | 0.78 | 40/s (API) | $0.02/1M |
| **BAAI/bge-large-en-v1.5** | **0.80** ✅ | **0.64** ✅ | **0.83** ✅ | **200/s** ✅ | **FREE** ✅ |
| intfloat/e5-large-v2 | 0.79 | 0.63 | 0.82 | 167/s | FREE |

---

## Validation Criteria

### Must Pass ✅

1. **Entity Extraction Quality**: Claude 3.5 Sonnet F1 ≥ 85%
2. **JSON Reliability**: Claude ≥ 99.5% valid JSON
3. **Embedding Quality**: BGE-large P@10 ≥ 0.75
4. **Cost Savings**: Claude + BGE ≥ 90% cheaper than GPT-4 Turbo

### Should Pass ✅

1. **Report Quality**: Claude 3.5 Sonnet ≥ 4.0/5 average
2. **Query Quality**: Claude 3.5 Sonnet ≥ 4.0/5 average
3. **Speed**: Claude 3 Haiku ≥ 2x faster than GPT-4 Turbo
4. **Embedding Speed**: BGE (GPU) ≥ 100 embeddings/sec

### Nice to Have

1. Claude 3.5 Sonnet exceeds GPT-4 Turbo quality
2. BGE-large matches OpenAI embedding quality
3. End-to-end indexing < 20 minutes (100 docs)

---

## Risk Assessment

### Low Risk ✅

- Benchmark methodology is standard
- Metrics are well-established
- Test automation is straightforward

### Medium Risk ⚠️

- Human evaluation requires 3 evaluators + time
- Ground truth annotation is labor-intensive
- POC implementation needed for actual testing

### Mitigation

- Start with automated metrics (faster)
- Use smaller test set for human evaluation
- Prioritize critical benchmarks (quality, cost)

---

## Next Steps

1. **Document 05**: Benefits and trade-offs analysis with GO/NO-GO recommendation
2. **Document 06**: Implementation plan with timeline and resources
3. **Document 07**: Adoption strategy and user migration guide
4. **POC Phase**: Implement benchmarks and gather real data

---

## Appendix: Test Data Structure

### Test Corpus Format

```json
{
  "documents": [
    {
      "id": "doc_001",
      "title": "Microsoft announces AI partnership",
      "text": "Microsoft Corporation announced...",
      "domain": "news",
      "ground_truth": {
        "entities": [
          {"name": "Microsoft Corporation", "type": "organization"},
          {"name": "Sam Altman", "type": "person"}
        ],
        "relationships": [
          {"source": "Microsoft Corporation", "target": "OpenAI", "type": "PARTNERS_WITH"}
        ]
      }
    }
  ]
}
```

### Query Test Format

```json
{
  "queries": [
    {
      "id": "q_001",
      "text": "What companies are partnering with AI startups?",
      "relevant_docs": ["doc_001", "doc_015", "doc_032"],
      "difficulty": "simple"
    }
  ]
}
```

---

**Document Status**: Complete (Planning) ✅
**POC Required**: Yes - benchmarks will be conducted during implementation
**Next Document**: `05_benefits_tradeoffs.md` - GO/NO-GO analysis
