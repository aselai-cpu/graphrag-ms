# Why Extract Covariates (Claims) in GraphRAG?

## Understanding the Knowledge Graph Layers

GraphRAG constructs multiple layers of knowledge representation, each capturing different aspects of information. This document explains why **covariates (claims)** are extracted separately from entities and relationships, and what unique value they provide.

---

## The Three Knowledge Layers

### 1. Entities (WHO/WHAT)
**What they represent**: The "things" in the world - people, organizations, places, concepts.

```
Example entities:
- Microsoft (ORGANIZATION)
- Bill Gates (PERSON)
- Seattle (GEO)
- Windows (PRODUCT)
```

**Characteristics**:
- ✅ Relatively static (entities exist over time)
- ✅ Identified by name and type
- ✅ Have descriptions (who/what they are)

### 2. Relationships (HOW CONNECTED)
**What they represent**: Connections and associations between entities.

```
Example relationships:
- Bill Gates → founded → Microsoft
- Microsoft → headquartered_in → Seattle
- Microsoft → develops → Windows
```

**Characteristics**:
- ✅ Structural connections (graph edges)
- ✅ Often bidirectional or symmetric
- ✅ Describe general associations
- ❌ Limited temporal information
- ❌ Limited factual specificity

### 3. Covariates/Claims (WHAT HAPPENED/WHAT'S TRUE)
**What they represent**: Specific, factual statements about entities with temporal and contextual bounds.

```
Example claims:
- CLAIM: "Microsoft reported $50B revenue in Q4 2023"
  - Subject: Microsoft
  - Type: FINANCIAL
  - Status: CONFIRMED
  - Start Date: 2023-10-01
  - End Date: 2023-12-31

- CLAIM: "Bill Gates stepped down as CEO in 2000"
  - Subject: Bill Gates
  - Object: Microsoft
  - Type: EMPLOYMENT
  - Status: CONFIRMED
  - End Date: 2000-01-13

- CLAIM: "Microsoft plans to invest $10B in OpenAI"
  - Subject: Microsoft
  - Object: OpenAI
  - Type: INVESTMENT
  - Status: ANNOUNCED (not yet confirmed)
  - Start Date: 2023-01-23
```

**Characteristics**:
- ✅ Temporal bounds (when it happened/was true)
- ✅ Verification status (confirmed, disputed, speculative)
- ✅ Specific factual content
- ✅ Context-dependent (claims can change or expire)

---

## Why We Need Claims in Addition to Entities and Relationships

### Problem 1: Relationships Are Too Coarse-Grained

**Without Claims**:
```
Entity: Bill Gates (PERSON)
Entity: Microsoft (ORGANIZATION)
Relationship: Bill Gates → associated_with → Microsoft
```

**What's missing?**
- What is the nature of this association?
- When did it start and end?
- Has the relationship changed over time?
- Is this current information or historical?

**With Claims**:
```
CLAIM 1: "Bill Gates co-founded Microsoft in 1975"
  - Type: FOUNDING
  - Status: CONFIRMED
  - Date: 1975-04-04

CLAIM 2: "Bill Gates served as CEO of Microsoft from 1975 to 2000"
  - Type: EMPLOYMENT
  - Status: CONFIRMED
  - Start: 1975-04-04
  - End: 2000-01-13

CLAIM 3: "Bill Gates stepped down from Microsoft board in 2020"
  - Type: GOVERNANCE
  - Status: CONFIRMED
  - Date: 2020-03-13

CLAIM 4: "Bill Gates remains a technical advisor to Microsoft"
  - Type: ADVISORY
  - Status: CURRENT
  - Start: 2020-03-13
```

Now we have **temporal precision** and **evolving relationships**.

### Problem 2: Facts Need Temporal Context

**Example: Company Metrics**

Without claims, you might have:
```
Entity: Tesla (ORGANIZATION)
Description: "Electric vehicle manufacturer with strong growth"
```

**Question**: "What was Tesla's revenue last quarter?"
**Answer**: Cannot be answered from entity description alone.

With claims:
```
CLAIM: "Tesla reported $25.2B revenue in Q3 2023"
  - Subject: Tesla
  - Type: FINANCIAL_METRIC
  - Metric: REVENUE
  - Value: 25.2B USD
  - Period: Q3 2023
  - Status: CONFIRMED
  - Source: Earnings report

CLAIM: "Tesla reported $21.3B revenue in Q2 2023"
  - Subject: Tesla
  - Type: FINANCIAL_METRIC
  - Metric: REVENUE
  - Value: 21.3B USD
  - Period: Q2 2023
  - Status: CONFIRMED
```

Now we can:
- ✅ Track metrics over time
- ✅ Compare periods
- ✅ Identify trends
- ✅ Cite specific sources

### Problem 3: Events vs. States

**Entities and relationships describe states** (what IS):
- "Microsoft is a technology company"
- "Bill Gates is connected to Microsoft"

**Claims describe events and changes** (what HAPPENED):
- "Microsoft acquired LinkedIn on June 13, 2016 for $26.2 billion"
- "Microsoft's stock price increased 40% in 2023"
- "Microsoft announced layoffs of 10,000 employees in January 2023"

### Problem 4: Uncertainty and Speculation

Relationships are binary (they exist or don't exist).

Claims can capture **degrees of certainty**:

```
CLAIM: "Microsoft is reportedly in talks to acquire Discord"
  - Status: SPECULATIVE
  - Source: Bloomberg report
  - Date: 2021-03-22
  - Confidence: LOW

CLAIM: "Microsoft confirmed acquisition of Activision Blizzard"
  - Status: CONFIRMED
  - Date: 2023-10-13
  - Confidence: HIGH
```

This allows queries like:
- "What are confirmed acquisitions?"
- "What are rumored deals?"
- "Which claims are disputed?"

### Problem 5: Multi-hop Reasoning

**Without Claims**:
```
Q: "Did Microsoft's CEO change affect stock price?"
A: Cannot answer - no temporal linkage between events
```

**With Claims**:
```
CLAIM 1: "Satya Nadella became CEO on February 4, 2014"
CLAIM 2: "Microsoft stock was $38.31 on February 4, 2014"
CLAIM 3: "Microsoft stock was $85.23 on February 4, 2015" (+122%)

A: "Yes, Microsoft stock increased 122% in the first year
    after Satya Nadella became CEO"
```

Claims enable **temporal reasoning** and **causal inference**.

---

## Real-World Use Cases Where Claims Are Critical

### 1. Financial Analysis
**Need**: Track financial metrics over time, compare periods, identify trends.

**Claims capture**:
- Quarterly/annual revenue, profit, expenses
- Stock prices at specific dates
- Analyst ratings and price targets (with dates)
- Dividend announcements and amounts
- Debt levels and credit ratings

**Example Query**: "How has Microsoft's revenue growth compared to its competitors over the last 5 years?"

### 2. Legal and Compliance
**Need**: Track legal events, regulatory filings, compliance status changes.

**Claims capture**:
- Lawsuits filed and resolved (with dates)
- Regulatory fines and penalties
- Patent applications and grants
- Contract signings and expirations
- Compliance certifications (valid from/to dates)

**Example Query**: "What legal disputes has Microsoft been involved in since 2020?"

### 3. News and Journalism
**Need**: Track evolving stories, updates, corrections.

**Claims capture**:
- Initial reports (with publication dates)
- Updates and corrections
- Confirmations and denials
- Status changes (rumored → confirmed → completed)

**Example Query**: "What was initially reported vs. what was later confirmed about Microsoft's AI investment?"

### 4. Competitive Intelligence
**Need**: Monitor competitor activities, announcements, strategic moves.

**Claims capture**:
- Product launches and announcements
- Partnership agreements
- Market expansions
- Leadership changes
- Strategic initiatives

**Example Query**: "What new products did Microsoft announce in 2023?"

### 5. Academic Research
**Need**: Track research findings, citations, retractions, updates.

**Claims capture**:
- Research findings (publication date)
- Methodology claims
- Retractions or corrections
- Citation relationships (who cited when)
- Peer review outcomes

**Example Query**: "Has this research finding been replicated or disputed?"

### 6. Supply Chain and Logistics
**Need**: Track inventory levels, shipments, delays, disruptions.

**Claims capture**:
- Shipment events (dispatched, arrived, delayed)
- Inventory levels at specific dates
- Supply disruptions (cause, date, duration)
- Supplier contract terms (valid from/to)

**Example Query**: "What supply chain disruptions affected Microsoft's Surface production in 2023?"

---

## The Trade-offs: Why Claims Are Optional

GraphRAG makes claim extraction **optional** (`extract_claims.enabled = false` by default) because:

### 1. Computational Cost
**Claims extraction is expensive**:
- Requires additional LLM calls (one per text unit)
- More complex prompts (temporal extraction, status inference)
- Increases indexing time by 30-50%
- Increases token usage significantly

**Example cost impact**:
```
Without claims:
- 1GB text → ~$100-200 (entities + relationships)

With claims:
- 1GB text → ~$150-300 (entities + relationships + claims)
```

### 2. Query Complexity
**Claims add query complexity**:
- Need temporal reasoning
- Need status filtering (confirmed vs. speculative)
- Need to handle claim conflicts (contradictory claims)
- More sophisticated retrieval logic

### 3. Domain Dependency
**Claims are more valuable in certain domains**:

**High value**:
- ✅ Financial data (metrics change frequently)
- ✅ News/journalism (events and updates)
- ✅ Legal documents (temporal precision critical)
- ✅ Scientific research (findings evolve)

**Lower value**:
- ❌ Static reference material (encyclopedias)
- ❌ Literature and fiction (fewer factual claims)
- ❌ Technical documentation (focus on current state)

### 4. Maintenance Burden
**Claims require updates**:
- Claims can become outdated
- Status changes (speculative → confirmed)
- Need periodic refresh/validation
- Conflict resolution logic needed

---

## Examples: With vs. Without Claims

### Example 1: Product Launch

**Without Claims (Entities + Relationships only)**:
```
Entities:
- iPhone 15 (PRODUCT)
- Apple (ORGANIZATION)
- California (GEO)

Relationships:
- Apple → develops → iPhone 15
- Apple → based_in → California

Query: "When was iPhone 15 released?"
Answer: Cannot determine from available data
```

**With Claims**:
```
CLAIM: "Apple announced iPhone 15 on September 12, 2023"
  - Subject: Apple
  - Object: iPhone 15
  - Type: PRODUCT_ANNOUNCEMENT
  - Date: 2023-09-12
  - Status: CONFIRMED

CLAIM: "iPhone 15 became available on September 22, 2023"
  - Subject: iPhone 15
  - Type: PRODUCT_AVAILABILITY
  - Date: 2023-09-22
  - Status: CONFIRMED

Query: "When was iPhone 15 released?"
Answer: "iPhone 15 was announced on September 12, 2023
         and became available on September 22, 2023"
```

### Example 2: Corporate Acquisition

**Without Claims**:
```
Entities:
- Microsoft (ORGANIZATION)
- LinkedIn (ORGANIZATION)

Relationships:
- Microsoft → owns → LinkedIn

Query: "How much did Microsoft pay for LinkedIn?"
Answer: Cannot determine
```

**With Claims**:
```
CLAIM 1: "Microsoft announced acquisition of LinkedIn on June 13, 2016"
  - Type: ACQUISITION_ANNOUNCED
  - Date: 2016-06-13

CLAIM 2: "Microsoft completed LinkedIn acquisition on December 8, 2016 for $26.2B"
  - Type: ACQUISITION_COMPLETED
  - Date: 2016-12-08
  - Amount: 26.2B USD
  - Status: CONFIRMED

Query: "How much did Microsoft pay for LinkedIn?"
Answer: "Microsoft acquired LinkedIn for $26.2 billion,
         completing the deal on December 8, 2016"
```

### Example 3: Leadership Changes

**Without Claims**:
```
Entities:
- Satya Nadella (PERSON)
- Microsoft (ORGANIZATION)

Relationships:
- Satya Nadella → works_for → Microsoft

Query: "When did Satya Nadella become CEO?"
Answer: Cannot determine
```

**With Claims**:
```
CLAIM 1: "Satya Nadella appointed CEO of Microsoft on February 4, 2014"
  - Subject: Satya Nadella
  - Object: Microsoft
  - Type: LEADERSHIP_APPOINTMENT
  - Role: CEO
  - Date: 2014-02-04
  - Status: CONFIRMED

CLAIM 2: "Satya Nadella succeeded Steve Ballmer as CEO"
  - Subject: Satya Nadella
  - Predecessor: Steve Ballmer
  - Type: SUCCESSION
  - Date: 2014-02-04

Query: "When did Satya Nadella become CEO?"
Answer: "Satya Nadella became CEO of Microsoft on
         February 4, 2014, succeeding Steve Ballmer"
```

---

## Architectural Design: Why Separate Claims?

### 1. Different Extraction Patterns

**Entities/Relationships**:
- Extracted from noun phrases and verb structures
- Relatively straightforward NLP patterns
- Focus on "who" and "what"

**Claims**:
- Extracted from complete sentences or paragraphs
- Require understanding of:
  - Temporal expressions ("in Q3 2023", "since January")
  - Epistemic modality ("confirmed", "reportedly", "allegedly")
  - Quantitative values ("$50B", "40% increase")
  - Causality ("due to", "as a result of")
- More complex linguistic analysis

### 2. Different Storage Requirements

**Entities/Relationships**:
```python
entity = {
    "id": "e_001",
    "name": "Microsoft",
    "type": "ORGANIZATION",
    "description": "Technology company..."
}

relationship = {
    "id": "r_001",
    "source": "Bill Gates",
    "target": "Microsoft",
    "type": "FOUNDED"
}
```

**Claims**:
```python
claim = {
    "id": "c_001",
    "subject_id": "e_001",  # Microsoft
    "object_id": "e_042",
    "claim_type": "FINANCIAL",
    "description": "Reported $50B revenue",
    "status": "CONFIRMED",  # or DISPUTED, SPECULATIVE
    "start_date": "2023-10-01",
    "end_date": "2023-12-31",
    "source_text": "Microsoft reported...",
    "confidence": 0.95
}
```

Claims need **temporal fields** and **status tracking** that entities/relationships don't require.

### 3. Different Query Patterns

**Entity/Relationship queries** (structural):
```cypher
// Who is connected to Microsoft?
MATCH (n)-[r]-(Microsoft)
RETURN n, r
```

**Claim queries** (temporal + factual):
```cypher
// What financial claims about Microsoft are confirmed for 2023?
MATCH (Microsoft)-[:HAS_CLAIM]->(c:Claim)
WHERE c.type = 'FINANCIAL'
  AND c.status = 'CONFIRMED'
  AND c.date >= '2023-01-01'
  AND c.date <= '2023-12-31'
RETURN c
```

### 4. Different Update Frequencies

**Entities**: Relatively stable
- Person exists for decades
- Company exists for years
- Updates: name changes, mergers

**Relationships**: Moderately stable
- Associations persist over time
- Updates: new connections, dissolved partnerships

**Claims**: Highly dynamic
- New claims added continuously
- Status changes frequently (rumor → confirmed)
- Claims expire or become outdated
- Requires active maintenance

---

## Research Background and Theoretical Foundation

### Knowledge Representation in AI

Claims/covariates in GraphRAG are inspired by several research areas:

#### 1. **Temporal Knowledge Graphs**
Traditional knowledge graphs (KG) represent static facts:
```
(subject, predicate, object)
(Microsoft, founded_by, Bill Gates)
```

Temporal KGs add time dimensions:
```
(subject, predicate, object, time)
(Microsoft, CEO, Steve Ballmer, [2000, 2014])
(Microsoft, CEO, Satya Nadella, [2014, present])
```

**Key Papers**:
- Leblay & Chekol (2018): "Deriving Validity Time in Knowledge Graph"
- Trivedi et al. (2017): "Know-Evolve: Deep Temporal Reasoning for Dynamic Knowledge Graphs"
- García-Durán et al. (2018): "Learning Sequence Encoders for Temporal Knowledge Graph Completion"

#### 2. **Event Extraction**
Events are specific occurrences with temporal bounds:
```
Event: Microsoft_LinkedIn_Acquisition
- Type: Acquisition
- Agent: Microsoft
- Patient: LinkedIn
- Time: 2016-12-08
- Location: USA
- Cost: $26.2B
```

**Key Papers**:
- ACE (Automatic Content Extraction) framework
- Doddington et al. (2004): "The Automatic Content Extraction (ACE) Program"
- Chen et al. (2015): "Event Extraction via Dynamic Multi-Pooling Convolutional Neural Networks"

#### 3. **Claim Verification / Fact Checking**
Claims need verification and status tracking:
```
Claim: "Microsoft will acquire Discord"
Status: REFUTED (Microsoft ended talks in April 2021)
Evidence: Bloomberg, WSJ reports
```

**Key Papers**:
- Thorne et al. (2018): "FEVER: a large-scale dataset for Fact Extraction and VERification"
- Augenstein et al. (2019): "MultiFC: A Real-World Multi-Domain Dataset for Evidence-Based Fact Checking"

#### 4. **Epistemic Logic**
Different modalities of truth:
- **Alethic**: necessarily true, possibly true
- **Deontic**: obligatory, permitted, forbidden
- **Epistemic**: known, believed, certain, uncertain

Claims capture epistemic states:
```
CONFIRMED: High certainty (known to be true)
DISPUTED: Conflicting evidence (uncertain)
SPECULATIVE: Low certainty (believed but unconfirmed)
REFUTED: Known to be false
```

#### 5. **Contextual Embeddings and RAG**
Modern RAG systems need context beyond static facts:
- **RAG (Retrieval-Augmented Generation)**: Retrieve relevant context for LLM generation
- **Temporal RAG**: Retrieve context specific to a time period
- **Claim-aware RAG**: Distinguish facts from speculation

**Key Papers**:
- Lewis et al. (2020): "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
- Edge et al. (2024): "From Local to Global: A Graph RAG Approach to Query-Focused Summarization" (GraphRAG paper)

---

## When to Enable Claims Extraction

### ✅ Enable Claims When:

1. **Your domain involves temporal facts**
   - Financial data, news, legal documents
   - "When did X happen?" queries are common

2. **Verification status matters**
   - Need to distinguish confirmed vs. rumored information
   - Tracking claim evolution (announcement → confirmation)

3. **Metrics and measurements are important**
   - Quarterly reports, performance metrics
   - Need to track values over time

4. **Event tracking is critical**
   - Product launches, acquisitions, incidents
   - Timeline reconstruction needed

5. **You need citation/provenance at fact level**
   - Each claim cites specific source text
   - Enables fine-grained fact checking

### ❌ Skip Claims When:

1. **Your domain is mostly static**
   - Reference materials, encyclopedias
   - Historical documents (already dated)

2. **Temporal precision isn't critical**
   - General knowledge questions
   - "What is X?" rather than "When did X happen?"

3. **Budget is constrained**
   - Claims add 30-50% to indexing cost
   - Entities + relationships may be sufficient

4. **Simple retrieval is adequate**
   - Basic RAG with text chunks works well
   - Don't need structured fact extraction

---

## Best Practices for Using Claims

### 1. Claim Granularity
**Too coarse**:
```
CLAIM: "Microsoft had a good year in 2023"
❌ Vague, not actionable
```

**Too fine**:
```
CLAIM: "On October 1st at 9:00 AM, Microsoft's stock price was $331.67"
❌ Too specific, creates noise
```

**Just right**:
```
CLAIM: "Microsoft reported $56.5B revenue in Q4 2023"
✅ Specific, relevant, actionable
```

### 2. Status Tracking
Always include verification status:
```
CONFIRMED: Officially announced/documented
SPECULATIVE: Reported but unconfirmed
DISPUTED: Conflicting information exists
REFUTED: Proven false
```

### 3. Source Attribution
Link claims to source text:
```
claim = {
    "description": "Microsoft acquired LinkedIn for $26.2B",
    "source_text_unit_id": "doc_042_chunk_15",
    "source_document": "SEC Filing 8-K, dated 2016-12-08"
}
```

### 4. Temporal Precision
Use appropriate temporal granularity:
```
Event: Use specific dates (2023-09-12)
Period: Use ranges (Q4 2023 = 2023-10-01 to 2023-12-31)
Ongoing: Use start date + "present" (2020-03-13 to present)
```

### 5. Claim Updates
Design for claim evolution:
```
Initial: CLAIM status=SPECULATIVE (rumor)
Update:  CLAIM status=CONFIRMED (official announcement)
Final:   CLAIM status=COMPLETED (deal closed)
```

---

## Further Reading

### Academic Papers

**Knowledge Graphs & Temporal Reasoning**:
1. **"Temporal Knowledge Graph Completion: A Survey"** - Cai et al. (2021)
   - Comprehensive survey of temporal KG research
   - [Link](https://arxiv.org/abs/2201.08236)

2. **"Know-Evolve: Deep Temporal Reasoning for Dynamic Knowledge Graphs"** - Trivedi et al. (2017)
   - ICML 2017 paper on temporal KG embeddings
   - [Link](https://arxiv.org/abs/1705.05742)

3. **"Recurrent Event Network: Autoregressive Structure Inference over Temporal Knowledge Graphs"** - Jin et al. (2020)
   - EMNLP 2020 paper
   - [Link](https://arxiv.org/abs/1904.05530)

**Event Extraction**:
4. **"Event Extraction: A Survey"** - Xiang & Wang (2019)
   - Survey of event extraction methods
   - [Link](https://arxiv.org/abs/1910.03836)

5. **"MAVEN: A Massive General Domain Event Detection Dataset"** - Wang et al. (2020)
   - ACL 2020, large-scale event dataset
   - [Link](https://arxiv.org/abs/2004.13590)

**Fact Verification & Claims**:
6. **"FEVER: a Large-scale Dataset for Fact Extraction and VERification"** - Thorne et al. (2018)
   - NAACL 2018, foundational claim verification dataset
   - [Link](https://arxiv.org/abs/1803.05355)

7. **"Explainable Automated Fact-Checking: A Survey"** - Kotonya & Toni (2020)
   - COLING 2020 survey
   - [Link](https://arxiv.org/abs/2011.03870)

**GraphRAG**:
8. **"From Local to Global: A Graph RAG Approach to Query-Focused Summarization"** - Edge et al. (2024)
   - The original GraphRAG paper from Microsoft Research
   - [Link](https://arxiv.org/abs/2404.16130)

### Books

9. **"Knowledge Graphs"** - Hogan et al. (2021)
   - Comprehensive textbook on KG theory and practice
   - Covers temporal and contextual aspects

10. **"Foundations of Semantic Web Technologies"** - Hitzler et al. (2009)
    - Classical reference for semantic web and knowledge representation

### Blog Posts & Technical Articles

11. **Microsoft Research Blog**: GraphRAG announcement
    - [Link](https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/)

12. **"Temporal Knowledge Graphs: A Survey"** - Towards Data Science
    - Practical introduction to temporal KGs

13. **Neo4j Blog**: "Modeling Time in Neo4j"
    - Practical patterns for temporal data in graphs

### Standards & Frameworks

14. **W3C Time Ontology**
    - Standard for representing temporal information
    - [Link](https://www.w3.org/TR/owl-time/)

15. **Schema.org Event**
    - Structured data markup for events
    - [Link](https://schema.org/Event)

### Tools & Libraries

16. **PyKEEN**: Python library for knowledge graph embeddings
    - Includes temporal KG models
    - [Link](https://github.com/pykeen/pykeen)

17. **ampligraph**: Library for graph representation learning
    - Supports temporal reasoning
    - [Link](https://docs.ampligraph.org/)

---

## Conclusion

**Covariates (claims) in GraphRAG serve a distinct purpose**:

1. **Entities** answer "WHO/WHAT"
2. **Relationships** answer "HOW CONNECTED"
3. **Claims** answer "WHAT HAPPENED WHEN" and "WHAT'S TRUE"

Claims are **optional but powerful** - they enable:
- ✅ Temporal reasoning
- ✅ Fact verification
- ✅ Event tracking
- ✅ Uncertainty handling
- ✅ Fine-grained provenance

**Enable claims when**:
- Temporal precision matters
- Facts evolve over time
- Verification status is important
- You need structured event extraction

**Skip claims when**:
- Domain is mostly static
- Budget is limited
- Simple entity/relationship extraction suffices

The decision should be based on your **domain requirements** and **query patterns**, not on technical complexity alone.

---

**Last Updated**: 2026-01-29
**Related Documents**:
- `pipeline_step_by_step.md`: Implementation details of claim extraction
- `entity_text_unit_linking.md`: How claims link to source text
