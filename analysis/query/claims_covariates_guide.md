# Claims (Covariates) in GraphRAG - Complete Guide

## Overview

**Claims** (also called **Covariates**) are structured factual assertions extracted from source documents and associated with specific entities. They provide an additional layer of context beyond entity descriptions and relationships.

## What Are Claims?

Claims are metadata assertions about entities that capture:
- **Facts** about an entity (e.g., "Company A was fined for anti-competitive practices")
- **Events** involving an entity (e.g., "Person C was suspected of corruption in 2015")
- **Status information** (TRUE, FALSE, or SUSPECTED)
- **Temporal data** (when the claim occurred)
- **Evidence** (source text supporting the claim)

### Claim Structure

Each claim contains:

| Field | Description | Example |
|-------|-------------|---------|
| **subject_id** | Entity the claim is about | "COMPANY A" |
| **object_id** | Entity affected/related (or NONE) | "GOVERNMENT AGENCY B" |
| **type** | Category of claim | "ANTI-COMPETITIVE PRACTICES" |
| **status** | Verification status | TRUE / FALSE / SUSPECTED |
| **start_date** | When claim began | "2022-01-10T00:00:00" |
| **end_date** | When claim ended | "2022-01-10T00:00:00" |
| **description** | Detailed explanation with evidence | "Company A was found to engage in..." |
| **source_text** | Original quotes from source | "According to an article..." |
| **text_unit_ids** | Which text chunks contain this claim | ["unit_42", "unit_87"] |

---

## Configuration

### Enabling Claims Extraction

Claims are **DISABLED by default**. To enable:

```yaml
# settings.yaml
extract_claims:
  enabled: true  # Default: false

  # Model for claim extraction
  completion_model_id: default_completion_model

  # Claim extraction prompt (optional custom path)
  prompt: null  # Uses built-in prompt by default

  # What type of claims to extract
  description: "Any claims or facts that could be relevant to information discovery"

  # Extraction thoroughness
  max_gleanings: 1  # Number of extraction passes
```

### Recommended Claim Descriptions

The `description` parameter determines what types of claims are extracted:

**General Purpose** (default):
```yaml
description: "Any claims or facts that could be relevant to information discovery"
```

**Risk/Compliance Focus**:
```yaml
description: "red flags, controversies, risks, or compliance issues associated with entities"
```

**Relationship Focus**:
```yaml
description: "significant relationships, partnerships, or interactions between entities"
```

**Event Focus**:
```yaml
description: "important events, milestones, or incidents involving entities"
```

---

## How Claims Are Extracted (Indexing)

### Extraction Process

During indexing, when `extract_claims.enabled = true`:

1. **For each text unit** (document chunk):
   - LLM analyzes the text for claims about entities
   - Extracts structured claim data
   - Links claims to subject and object entities
   - Assigns claim types and status

2. **Claim fields populated**:
   ```
   Subject → Object → Type → Status → Dates → Description → Source
   ```

3. **Output saved** to `covariates.parquet`

### Extraction Prompt

The LLM is prompted with:
- **Target entities** to look for
- **Claim description** (what types of claims to extract)
- **Source text** from the document

Example extraction:

**Input Text**:
```
"According to an article on 2022/01/10, Company A was fined for bid rigging
while participating in multiple public tenders published by Government Agency B."
```

**Extracted Claim**:
```
Subject: COMPANY A
Object: GOVERNMENT AGENCY B
Type: ANTI-COMPETITIVE PRACTICES
Status: TRUE
Start Date: 2022-01-10
End Date: 2022-01-10
Description: Company A was found to engage in anti-competitive practices
             because it was fined for bid rigging in multiple public tenders
             published by Government Agency B according to an article published
             on 2022/01/10
Source: "According to an article published on 2022/01/10, Company A was fined..."
```

---

## How Claims Are Used (Query Time)

### Local Search with Claims

Claims provide additional context in **Local Search** queries.

#### Claim Retrieval Process:

1. **Entity Mapping**: Vector search finds relevant entities
2. **Graph Traversal**: Expand to entity neighborhoods
3. **Context Building**: Gather multiple sources:
   - ✅ Entities
   - ✅ Relationships
   - ✅ Community Reports
   - ✅ Text Units
   - ✅ **Claims** ← Additional context layer

4. **Claim Selection**:
   ```python
   # For each selected entity:
   selected_claims = [
       claim for claim in all_claims
       if claim.subject_id == entity.title
   ]
   ```

5. **Token Budget**:
   - Claims use **remaining token budget** after:
     - Entities (no limit - small)
     - Relationships (no limit - small)
     - Community reports (50% of total budget)
     - Text units (50% of total budget)
   - Claims fill any leftover space up to `max_context_tokens`

#### Formatted Claims Context:

```
-----Claims-----
id|entity|type|status|description|start_date|end_date|source_text
claim_1|COMPANY A|ANTI-COMPETITIVE PRACTICES|TRUE|Company A was fined...|2022-01-10|2022-01-10|According to...
claim_2|PERSON C|CORRUPTION|SUSPECTED|Person C was suspected...|2015-01-01|2015-12-30|The company is owned...
```

### DRIFT Search with Claims

DRIFT Search inherits Local Search functionality, so claims are automatically included in the local search iterations.

**Claim Usage**:
- **Primer Phase**: No claims (uses community reports only)
- **Local Search Iterations**: Claims included for each action
- **REDUCE Phase**: Claims context is part of synthesized answers

### Global and Basic Search

- **Global Search**: Does NOT use claims (community reports only)
- **Basic Search**: Does NOT use claims (text units only)

**Only Local and DRIFT search methods use claims.**

---

## Data Requirements

### With Claims Enabled

**Required Files**:
- ✅ `entities.parquet`
- ✅ `relationships.parquet`
- ✅ `communities.parquet`
- ✅ `community_reports.parquet`
- ✅ `text_units.parquet`
- ✅ `covariates.parquet` ← **Must exist**

### With Claims Disabled (Default)

**Required Files**:
- ✅ `entities.parquet`
- ✅ `relationships.parquet`
- ✅ `communities.parquet`
- ✅ `community_reports.parquet`
- ✅ `text_units.parquet`
- ⚠️ `covariates.parquet` ← **Optional** (ignored if exists)

---

## Use Cases for Claims

### When to Enable Claims

✅ **Compliance and Risk Analysis**:
- Extract regulatory violations, controversies, legal issues
- Example: "What red flags are associated with Company X?"

✅ **Event Tracking**:
- Capture temporal events and milestones
- Example: "When did significant incidents occur involving Entity Y?"

✅ **Fact Verification**:
- Track TRUE vs SUSPECTED vs FALSE assertions
- Example: "What verified facts exist about Person Z?"

✅ **Evidence-Based Analysis**:
- Link claims to source text for traceability
- Example: "Show me claims about Topic A with supporting evidence"

✅ **Relationship Context**:
- Capture relationship-specific facts
- Example: "What claims exist about the connection between A and B?"

### When to Disable Claims (Default)

❌ **General Knowledge Graphs**:
- Standard entity-relationship analysis doesn't need claims
- Relationships and community reports provide sufficient context

❌ **Cost-Sensitive Scenarios**:
- Claim extraction adds LLM calls during indexing
- Significant cost increase for large datasets

❌ **Speed Priority**:
- Claim extraction slows down indexing
- Adds processing time per text unit

❌ **Simple Use Cases**:
- Basic search or global search (don't use claims anyway)
- Entity descriptions and relationships sufficient

---

## Performance Impact

### Indexing Time

**With Claims Enabled**:
- ⏱️ **Indexing Time**: 30-50% slower
- 💰 **Indexing Cost**: 20-40% higher
- 💾 **Storage**: Additional `covariates.parquet` file

**Example** (1000 text units):
```
Without Claims: 10 minutes, $5.00
With Claims:    14 minutes, $6.50  (+40% time, +30% cost)
```

### Query Time

**With Claims Enabled**:
- ⏱️ **Query Speed**: Negligible impact (<5%)
  - Claims use remaining token budget
  - Typically only 10-50 claims per query
- 💰 **Query Cost**: Minimal increase
  - Claims replace some text unit tokens
  - Net cost impact: 0-5%

### Token Usage

**Typical Local Search Context** (12,000 tokens):
```
Entities:           ~500 tokens (descriptions)
Relationships:      ~800 tokens (connections)
Community Reports: 6000 tokens (50% budget)
Text Units:        4200 tokens (35% budget)
Claims:             500 tokens (4% - remaining space)
```

**Claims use leftover budget** after higher-priority context sources.

---

## Example: Claims in Action

### Scenario: Risk Analysis Query

**Query**: "What controversies or red flags are associated with Company X?"

**With Claims Disabled**:
```
Context includes:
- Entity: Company X description
- Relationships: Company X connections
- Community Reports: Industry overview
- Text Units: Raw text mentioning Company X

Answer: "Company X is mentioned in connection with Government Agency B.
The text discusses various business activities..."
```

**With Claims Enabled**:
```
Context includes:
- Entity: Company X description
- Relationships: Company X connections
- Community Reports: Industry overview
- Text Units: Raw text mentioning Company X
- Claims:
  * ANTI-COMPETITIVE PRACTICES (TRUE, 2022-01-10)
  * REGULATORY VIOLATION (TRUE, 2021-05-15)
  * CORRUPTION (SUSPECTED, 2020-03-20)

Answer: "Company X has confirmed anti-competitive practices violations.
On 2022-01-10, the company was fined for bid rigging in public tenders
with Government Agency B. Additionally, there was a suspected corruption
incident in 2020..."
```

**Claims provide**:
- ✅ **Structured facts** instead of unstructured text
- ✅ **Status verification** (TRUE vs SUSPECTED)
- ✅ **Temporal context** (when events occurred)
- ✅ **Evidence traceability** (source text)
- ✅ **Categorization** (types of issues)

---

## Code References

### Claim Extraction (Indexing)

- **Prompt**: `packages/graphrag/graphrag/prompts/index/extract_claims.py`
- **Workflow**: `packages/graphrag/graphrag/index/workflows/extract_covariates.py`
- **Configuration**: `packages/graphrag/graphrag/config/models/extract_claims_config.py`

### Claim Retrieval (Query)

- **Retrieval Logic**: `packages/graphrag/graphrag/query/input/retrieval/covariates.py`
- **Context Building**: `packages/graphrag/graphrag/query/context_builder/local_context.py:93-150`
- **Local Search Integration**: `packages/graphrag/graphrag/query/structured_search/local_search/mixed_context.py:432-444`
- **Data Model**: `packages/graphrag/graphrag/data_model/covariate.py`

---

## Configuration Examples

### Minimal (Claims Disabled - Default)

```yaml
# settings.yaml
# Claims disabled by default - no configuration needed
```

### Basic Claims Enabled

```yaml
# settings.yaml
extract_claims:
  enabled: true
  completion_model_id: default_completion_model
```

### Advanced Claims Configuration

```yaml
# settings.yaml
extract_claims:
  enabled: true

  # Use specific model for claim extraction
  completion_model_id: gpt_4o_completion

  # Custom claim description for risk analysis
  description: "red flags, controversies, regulatory violations, legal issues, or risks associated with entities"

  # More thorough extraction (2 passes)
  max_gleanings: 2

  # Custom prompt (optional)
  prompt: "prompts/custom_claim_extraction.txt"
```

### Cost-Optimized Claims

```yaml
# settings.yaml
extract_claims:
  enabled: true

  # Use cheaper model for cost savings
  completion_model_id: haiku_completion

  # Simpler claim description (faster)
  description: "major events or facts about entities"

  # Single pass (faster, cheaper)
  max_gleanings: 1
```

---

## Troubleshooting

### Problem: Claim extraction is slow

**Solutions**:
1. Reduce `max_gleanings` to 1 (single pass)
2. Use faster model (e.g., GPT-4o instead of GPT-4)
3. Simplify claim description (fewer claim types)
4. Process smaller document batches

### Problem: Claim extraction is expensive

**Solutions**:
1. Use cheaper model (Claude Haiku, GPT-3.5)
2. Reduce `max_gleanings` to 1
3. Only enable claims for critical use cases
4. Consider if claims are necessary (try without first)

### Problem: Too few claims extracted

**Solutions**:
1. Broaden `description` parameter
2. Increase `max_gleanings` to 2-3
3. Check source documents have factual content
4. Verify entities are being extracted correctly

### Problem: Too many irrelevant claims

**Solutions**:
1. Narrow `description` parameter (be more specific)
2. Review extraction prompt
3. Check if source data quality is poor
4. Consider using claim filtering in queries

### Problem: Claims not appearing in query results

**Verification**:
1. Check `covariates.parquet` exists and has data
2. Verify using Local or DRIFT search (Global/Basic don't use claims)
3. Confirm entities in claims match query entities
4. Check token budget - claims use remaining space

---

## Best Practices

### 1. Start Without Claims

- ❌ Don't enable claims immediately
- ✅ First build and test with claims disabled
- ✅ Evaluate if entity descriptions + relationships + community reports are sufficient
- ✅ Only enable claims if specific use case benefits

### 2. Choose Right Claim Description

- ❌ Don't use vague descriptions like "anything relevant"
- ✅ Be specific about what claims you want
- ✅ Tailor to your domain (legal, medical, business, etc.)
- ✅ Examples:
  - Legal: "legal issues, violations, or regulatory matters"
  - Medical: "diagnoses, treatments, or adverse events"
  - Business: "partnerships, acquisitions, or financial events"

### 3. Monitor Extraction Quality

- ✅ Sample extracted claims regularly
- ✅ Verify claim types are useful
- ✅ Check status assignments (TRUE/FALSE/SUSPECTED)
- ✅ Ensure temporal data is accurate

### 4. Balance Cost vs Value

- ✅ Claims add 20-40% to indexing cost
- ✅ Evaluate ROI for your use case
- ✅ Consider claims for subsets of data (not entire corpus)
- ✅ Use cost-effective models (Haiku) for claim extraction

### 5. Test Query Impact

- ✅ Run same queries with/without claims
- ✅ Compare answer quality
- ✅ Measure if claims improve relevance
- ✅ Verify claims don't dilute other context

---

## Comparison: With vs Without Claims

| Aspect | Without Claims (Default) | With Claims Enabled |
|--------|--------------------------|---------------------|
| **Indexing Speed** | Baseline | 30-50% slower |
| **Indexing Cost** | Baseline | 20-40% higher |
| **Storage** | 5 parquet files | 6 parquet files |
| **Context Richness** | Good | Better (structured facts) |
| **Temporal Info** | Limited | Precise (start/end dates) |
| **Evidence Links** | Text units | Direct source quotes |
| **Fact Verification** | Unclear | Status (TRUE/FALSE/SUSPECTED) |
| **Setup Complexity** | Simple | Moderate |
| **Best For** | General use | Compliance, risk, events |

---

## Summary

**Claims (Covariates)**:
- ✅ **Optional** feature (disabled by default)
- ✅ **Structured facts** extracted from text
- ✅ **Used by Local and DRIFT search** only
- ✅ **Best for**: Compliance, risk, events, fact verification
- ✅ **Trade-off**: Higher indexing cost for richer query context

**Enable claims when**:
- You need structured factual assertions
- Temporal context is important
- Evidence traceability matters
- Working with compliance/risk/legal data

**Keep claims disabled when**:
- Standard entity-relationship analysis sufficient
- Cost or speed is a priority
- Using Global or Basic search primarily
- General knowledge graph use case

---

**Last Updated**: 2026-01-30
**Applies to**: GraphRAG v3.0+
**Status**: Complete ✅
