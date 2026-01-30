# Query Prompts

Query prompts are used to answer user questions by leveraging the constructed knowledge graph. GraphRAG supports multiple search strategies, each with specialized prompts.

## Table of Contents
1. [Local Search](#local-search)
2. [Global Search](#global-search)
3. [Basic Search](#basic-search)
4. [DRIFT Search](#drift-search)
5. [Question Generation](#question-generation)

---

## Local Search

**Location**: `packages/graphrag/graphrag/prompts/query/local_search_system_prompt.py`

**Purpose**: Answers questions using localized context from nearby entities, relationships, and community reports in the knowledge graph.

**Key Features**:
- Incorporates general knowledge alongside graph data
- References multiple dataset types (Sources, Reports, Entities, Relationships, Claims)
- Supports markdown formatting with sections and commentary
- Enforces data citation for all claims

### Prompt Structure

```python
LOCAL_SEARCH_SYSTEM_PROMPT = """
---Role---

You are a helpful assistant responding to questions about data in the tables provided.


---Goal---

Generate a response of the target length and format that responds to the user's question,
summarizing all information in the input data tables appropriate for the response length
and format, and incorporating any relevant general knowledge.

If you don't know the answer, just say so. Do not make anything up.

Points supported by data should list their data references as follows:

"This is an example sentence supported by multiple data references [Data: <dataset name> (record ids); <dataset name> (record ids)]."

Do not list more than 5 record ids in a single reference. Instead, list the top 5 most
relevant record ids and add "+more" to indicate that there are more.

For example:

"Person X is the owner of Company Y and subject to many allegations of wrongdoing
[Data: Sources (15, 16), Reports (1), Entities (5, 7); Relationships (23); Claims (2, 7, 34, 46, 64, +more)]."

where 15, 16, 1, 5, 7, 23, 2, 7, 34, 46, and 64 represent the id (not the index) of the relevant data record.

Do not include information where the supporting evidence for it is not provided.


---Target response length and format---

{response_type}


---Data tables---

{context_data}


---Goal---

Generate a response of the target length and format that responds to the user's question,
summarizing all information in the input data tables appropriate for the response length
and format, and incorporating any relevant general knowledge.

Add sections and commentary to the response as appropriate for the length and format.
Style the response in markdown.
"""
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `response_type` | string | Desired output format (e.g., "Multiple Paragraphs", "Single Paragraph", "List of 3-7 Points") |
| `context_data` | string | Formatted tables containing Entities, Relationships, Reports, Sources, and Claims |

### Sample Input Context

```
# Entities

id,entity,description,rank
5,ALICE SMITH,Software engineer at TechCorp with 10 years experience,8.5
7,TECHCORP,Technology company specializing in AI solutions,9.2

# Relationships

id,source,target,description,rank
23,ALICE SMITH,TECHCORP,Alice Smith is employed as a senior engineer at TechCorp,7.8

# Reports

id,report
1,TechCorp is a leading AI company with over 500 employees [Data: Entities (7)]

# Claims

id,subject,object,type,status,description
2,ALICE SMITH,AI MODEL,CONTRIBUTION,TRUE,Alice Smith led development of TechCorp's flagship AI model
```

### Sample Output

**User Query**: "What is Alice Smith's role at TechCorp?"

**Response**:
```markdown
Alice Smith is a senior software engineer at TechCorp, where she has been working for
10 years [Data: Entities (5); Relationships (23)]. She has made significant contributions
to the company's AI initiatives, notably leading the development of TechCorp's flagship
AI model [Data: Claims (2)].

TechCorp is a prominent technology company that specializes in AI solutions and employs
over 500 people [Data: Entities (7); Reports (1)]. Alice's extensive experience and
leadership in AI development make her a key contributor to the company's success.
```

---

## Global Search

**Location**: `packages/graphrag/graphrag/prompts/query/global_search_*_system_prompt.py`

**Purpose**: Answers broad, high-level questions by synthesizing information across the entire knowledge graph using a map-reduce approach.

Global search uses three coordinated prompts:

### 1. Map Phase Prompt

**File**: `global_search_map_system_prompt.py`

**Purpose**: Each community report is analyzed to extract key points relevant to the question.

```python
MAP_SYSTEM_PROMPT = """
---Role---

You are a helpful assistant responding to questions about data in the tables provided.


---Goal---

Generate a response consisting of a list of key points that responds to the user's question,
summarizing all relevant information in the input data tables.

You should use the data provided in the data tables below as the primary context for generating
the response.
If you don't know the answer or if the input data tables do not contain sufficient information
to provide an answer, just say so. Do not make anything up.

Each key point in the response should have the following element:
- Description: A comprehensive description of the point.
- Importance Score: An integer score between 0-100 that indicates how important the point is
  in answering the user's question. An 'I don't know' type of response should have a score of 0.

The response should be JSON formatted as follows:
{
    "points": [
        {"description": "Description of point 1 [Data: Reports (report ids)]", "score": score_value},
        {"description": "Description of point 2 [Data: Reports (report ids)]", "score": score_value}
    ]
}

The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".

Points supported by data should list the relevant reports as references as follows:
"This is an example sentence supported by data references [Data: Reports (report ids)]"

**Do not list more than 5 record ids in a single reference**. Instead, list the top 5 most relevant
record ids and add "+more" to indicate that there are more.

For example:
"Person X is the owner of Company Y and subject to many allegations of wrongdoing
[Data: Reports (2, 7, 64, 46, 34, +more)]. He is also CEO of company X [Data: Reports (1, 3)]"

where 1, 2, 3, 7, 34, 46, and 64 represent the id (not the index) of the relevant data report in
the provided tables.

Do not include information where the supporting evidence for it is not provided.

Limit your response length to {max_length} words.

---Data tables---

{context_data}
"""
```

**Sample Output**:
```json
{
    "points": [
        {
            "description": "TechCorp employs over 500 people specializing in AI solutions [Data: Reports (15, 23)]",
            "score": 85
        },
        {
            "description": "The company has raised $50M in Series B funding led by VentureCapital [Data: Reports (8, 12, +more)]",
            "score": 78
        }
    ]
}
```

### 2. Reduce Phase Prompt

**File**: `global_search_reduce_system_prompt.py`

**Purpose**: Synthesizes the key points from multiple analysts (map phase) into a coherent final answer.

```python
REDUCE_SYSTEM_PROMPT = """
---Role---

You are a helpful assistant responding to questions about a dataset by synthesizing perspectives
from multiple analysts.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarize
all the reports from multiple analysts who focused on different parts of the dataset.

Note that the analysts' reports provided below are ranked in the descending order of importance.

If you don't know the answer or if the provided reports do not contain sufficient information to
provide an answer, just say so. Do not make anything up.

The final response should remove all irrelevant information from the analysts' reports and merge
the cleaned information into a comprehensive answer that provides explanations of all the key points
and implications appropriate for the response length and format.

Add sections and commentary to the response as appropriate for the length and format. Style the
response in markdown.

The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".

The response should also preserve all the data references previously included in the analysts' reports,
but do not mention the roles of multiple analysts in the analysis process.

**Do not list more than 5 record ids in a single reference**. Instead, list the top 5 most relevant
record ids and add "+more" to indicate that there are more.

For example:

"Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Reports (2, 7, 34, 46, 64, +more)].
He is also CEO of company X [Data: Reports (1, 3)]"

where 1, 2, 3, 7, 34, 46, and 64 represent the id (not the index) of the relevant data record.

Do not include information where the supporting evidence for it is not provided.


---Target response length and format---

{response_type}

---Analyst Reports---

{report_data}


---Goal---

Generate a response of the target length and format that responds to the user's question, summarize
all the reports from multiple analysts who focused on different parts of the dataset.
"""
```

### 3. Knowledge Augmentation Prompt

**File**: `global_search_knowledge_system_prompt.py`

**Purpose**: Allows incorporation of real-world knowledge with verification tags.

```python
GENERAL_KNOWLEDGE_INSTRUCTION = """
The response may also include relevant real-world knowledge outside the dataset, but it must be
explicitly annotated with a verification tag [LLM: verify]. For example:

"This is an example sentence supported by real-world knowledge [LLM: verify]."
"""
```

**Sample Output with Knowledge**:
```markdown
## TechCorp's Business Operations

TechCorp is a leading AI solutions company with over 500 employees [Data: Reports (15, 23)].
The company recently secured $50M in Series B funding from VentureCapital [Data: Reports (8, 12, 19)].

### Industry Context

AI companies typically use Series B funding to scale operations and expand market reach [LLM: verify].
This funding round positions TechCorp competitively in the growing AI solutions market [Data: Reports (15)].
```

---

## Basic Search

**Location**: `packages/graphrag/graphrag/prompts/query/basic_search_system_prompt.py`

**Purpose**: Simple question answering based on provided data tables without complex graph traversal.

```python
BASIC_SEARCH_SYSTEM_PROMPT = """
---Role---

You are a helpful assistant responding to questions about data in the tables provided.


---Goal---

Generate a response of the target length and format that responds to the user's question,
summarizing all relevant information in the input data tables.

If you don't know the answer, just say so. Do not make anything up.

Points supported by data should list their data references as follows:

"This is an example sentence supported by data references [Data: Sources (record ids)]."

Do not list more than 5 record ids in a single reference. Instead, list the top 5 most relevant
record ids and add "+more" to indicate that there are more.

For example:
"Person X is the owner of Company Y and subject to many allegations of wrongdoing
[Data: Sources (15, 16, 1, 5, 7, +more)]."

where 15, 16, 1, 5, and 7 represent the id (not the index) of the relevant data record.

Do not include information where the supporting evidence for it is not provided.


---Data tables---

{context_data}

---Target response length and format---

{response_type}

Add sections and commentary to the response as appropriate for the length and format.
"""
```

**Key Differences from Local Search**:
- Only references "Sources" dataset, not the full range of graph data
- Simpler context without relationships or community reports
- More straightforward table-based question answering

---

## DRIFT Search

**Location**: `packages/graphrag/graphrag/prompts/query/drift_search_system_prompt.py`

**Purpose**: Dynamic Recursive Iterative Filtering for Topics - an advanced search that iteratively explores the knowledge graph.

### DRIFT Primer Prompt

```python
DRIFT_PRIMER_PROMPT = """
---Role---

You are a helpful agent designed to reason over a knowledge graph in response to a user query.

This is a unique knowledge graph where the edges are freeform text rather than verb operators
(e.g., instead of edges like "OWNS" or "WORKS_FOR", edges might be full sentences like
"Person X works for Company Y" or "Company Y acquired Company Z in 2020").


---Goal---

Generate an intermediate reasoning step toward answering the user's query that includes:

1. A score (0-100) indicating how well your current intermediate answer addresses the query
2. An intermediate answer (up to 2000 characters) based on the information you have
3. A list of follow-up queries to explore relevant parts of the knowledge graph


---Output Format---

Return JSON formatted as:
{
    "score": <confidence_score_0_to_100>,
    "intermediate_answer": "<detailed_analysis_up_to_2000_chars>",
    "follow_up_queries": [
        "query 1",
        "query 2",
        "query 3"
    ]
}


---Knowledge Graph Context---

{context_data}


---User Query---

{query}


---Instructions---

1. Analyze the provided knowledge graph context
2. Provide an intermediate answer based on what you know
3. Identify gaps in your knowledge
4. Generate 2-5 follow-up queries to explore those gaps
5. Score your confidence in the current answer
"""
```

### DRIFT Local Prompt

```python
DRIFT_LOCAL_SYSTEM_PROMPT = """
---Role---

You are a helpful assistant responding to questions about data in the tables provided.


---Goal---

Generate a response that summarizes the data relevant to the follow-up query. Focus especially
on information from the Sources dataset that contains the raw text data.

The response should be detailed and comprehensive, as it will be used in a later reasoning step.

Points supported by data should list their data references as follows:
"This is an example sentence [Data: Sources (record ids)]."


---Data tables---

{context_data}


---Follow-up Query---

{query}


Generate a comprehensive response summarizing all relevant information.
"""
```

### DRIFT Reduce Prompt

```python
DRIFT_REDUCE_PROMPT = """
---Role---

You are a helpful assistant synthesizing information from multiple search results.


---Goal---

Given the original user question and multiple intermediate search results, generate a final
comprehensive answer.


---Original Question---

{original_query}


---Search Results---

{search_results}


---Instructions---

1. Synthesize information from all search results
2. Maintain all data references [Data: Sources (ids)]
3. Provide a comprehensive, well-structured answer
4. Use markdown formatting with appropriate sections


Generate the final answer:
"""
```

**DRIFT Search Flow**:
1. **Primer**: Generate initial reasoning and follow-up queries
2. **Local Searches**: Execute each follow-up query against the graph
3. **Iteration**: Repeat primer with enriched context until confidence threshold met
4. **Reduce**: Synthesize all findings into final answer

---

## Question Generation

**Location**: `packages/graphrag/graphrag/prompts/query/question_gen_system_prompt.py`

**Purpose**: Generates follow-up questions based on example questions and available data.

```python
QUESTION_GENERATION_PROMPT = """
---Role---

You are a helpful assistant that generates insightful questions about a dataset.


---Goal---

Given a series of example questions provided by the user, generate a bulleted list of
{question_count} candidates for the next question.

The candidate questions should represent the most important or urgent information themes in the dataset.

Use the following example questions as a reference for the types of questions to generate:

{example_questions}


---Dataset Context---

{context_data}


---Instructions---

1. Analyze the dataset and identify key themes
2. Review the example questions to understand the desired question style
3. Generate {question_count} new questions that:
   - Address important information in the dataset
   - Follow similar style and complexity to the examples
   - Cover diverse aspects of the data
   - Are specific and answerable from the dataset


Generate the candidate questions as a bulleted markdown list:
"""
```

**Sample Input**:
```
Example Questions:
- What are the main activities of TechCorp?
- Who are the key people associated with VentureCapital?

Context: [Entity and relationship data about tech companies and investors]
```

**Sample Output**:
```markdown
- What is the relationship between TechCorp and VentureCapital?
- How much funding has TechCorp raised in total?
- Which AI technologies does TechCorp specialize in?
- What are the notable achievements of TechCorp's leadership team?
- What strategic partnerships has TechCorp formed?
```

---

## Common Parameters Across Query Prompts

| Parameter | Type | Description | Common Values |
|-----------|------|-------------|---------------|
| `context_data` | string | Input data tables (Entities, Relationships, Reports, etc.) | Formatted CSV/table text |
| `response_type` | string | Desired output format | "Multiple Paragraphs", "Single Paragraph", "Single Sentence", "List of 3-7 Points", "Single Page Report", "Multi-Page Report" |
| `query` | string | User's question | Any natural language question |
| `max_length` | integer | Maximum response length in words | 100-2000 |
| `language` | string | Output language | "English" (default), or any language |

## Response Format Options

All query prompts support flexible response formats:

1. **Multiple Paragraphs**: Detailed explanation with multiple sections
2. **Single Paragraph**: Concise summary in one paragraph
3. **Single Sentence**: Brief one-sentence answer
4. **List of 3-7 Points**: Bulleted key points
5. **Single Page Report**: Comprehensive single-page document
6. **Multi-Page Report**: Extended detailed report

## Data Citation Rules

All query prompts enforce strict citation rules:

- **Format**: `[Data: <dataset> (id1, id2, id3, id4, id5, +more)]`
- **Max IDs per reference**: 5 (use "+more" if additional records exist)
- **Dataset types**: Sources, Reports, Entities, Relationships, Claims
- **Multiple datasets**: Separate with semicolons

**Example**:
```
"Company X raised $50M from Investor Y [Data: Reports (1, 5); Entities (12, 15); Relationships (34, +more)]."
```

## Configuration

Query prompts can be customized in `settings.yaml`:

```yaml
local_search:
  prompt: null  # Use built-in prompt
  # OR
  prompt: "./custom_prompts/my_local_search.txt"  # Load custom prompt

global_search:
  map_prompt: null
  reduce_prompt: null
  knowledge_prompt: null

drift_search:
  prompt: null
  reduce_prompt: null

basic_search:
  prompt: null
```

## References

- Source files: `packages/graphrag/graphrag/prompts/query/`
- Configuration: `packages/graphrag/graphrag/config/models/*_search_config.py`
- Defaults: `packages/graphrag/graphrag/config/defaults.py`
