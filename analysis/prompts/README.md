# GraphRAG LLM Prompts Documentation

This directory contains comprehensive documentation of all LLM prompts used in GraphRAG, including their purpose, structure, parameters, and sample outputs.

## Overview

GraphRAG uses Large Language Models (LLMs) throughout its pipeline for various tasks including:

1. **Query Processing** - Answering user questions using the knowledge graph
2. **Knowledge Graph Construction** - Extracting entities, relationships, and communities from text
3. **Prompt Tuning** - Dynamically generating and optimizing prompts

## Documentation Structure

### [Query Prompts](./query_prompts.md)
Prompts used for answering user questions:
- **Local Search** - Context-aware search using nearby graph data
- **Global Search** - Map-reduce search across entire knowledge graph
- **Basic Search** - Simple table-based question answering
- **DRIFT Search** - Dynamic Recursive Iterative Filtering for Topics
- **Question Generation** - Generating follow-up questions

### [Index Prompts](./index_prompts.md)
Prompts used for building the knowledge graph:
- **Graph Extraction** - Extracting entities and relationships from text
- **Claim Extraction** - Identifying factual claims and assertions
- **Community Reports** - Generating summaries of entity communities
- **Description Summarization** - Consolidating entity descriptions

### [Prompt Tuning](./prompt_tuning.md)
Dynamic prompt generation and optimization:
- **Prompt Templates** - Parameterized prompt structures
- **Prompt Generators** - Dynamic prompt creation utilities
- **Domain Adaptation** - Customizing prompts for specific domains
- **Configuration Management** - How prompts are loaded and resolved

## Key Concepts

### Data Reference Format
GraphRAG prompts enforce a standardized citation format for grounding responses in data:

```
[Data: <dataset_name> (record_ids)]
```

Example:
```
"Person X is the owner of Company Y [Data: Entities (5, 7); Relationships (23); Claims (2, 7, 34, +more)]."
```

### Response Types
Many prompts support multiple response formats:
- **Multiple Paragraphs** - Detailed explanatory text
- **Single Paragraph** - Concise summary
- **Single Sentence** - Brief answer
- **List of 3-7 Points** - Bullet point format
- **Single Page Report** - Comprehensive document
- **JSON** - Structured data output

### Prompt Parameters
Common parameters used across prompts:
- `{context_data}` - Input data tables/text
- `{response_type}` - Desired output format
- `{entity_types}` - Types of entities to extract
- `{max_length}` - Maximum response length in words
- `{language}` - Output language (default: English)
- `{persona}` - Role/expertise context

## Location in Codebase

### Production Prompts
```
packages/graphrag/graphrag/prompts/
├── query/               # Query processing prompts
│   ├── local_search_system_prompt.py
│   ├── global_search_map_system_prompt.py
│   ├── global_search_reduce_system_prompt.py
│   ├── basic_search_system_prompt.py
│   ├── drift_search_system_prompt.py
│   └── question_gen_system_prompt.py
└── index/              # Knowledge graph construction prompts
    ├── extract_graph.py
    ├── extract_claims.py
    ├── community_report.py
    └── summarize_descriptions.py
```

### Prompt Tuning System
```
packages/graphrag/graphrag/prompt_tune/
├── template/           # Parameterized templates
│   ├── extract_graph.py
│   ├── entity_summarization.py
│   └── community_report_summarization.py
└── generator/          # Dynamic prompt generators
    ├── extract_graph_prompt.py
    ├── entity_summarization_prompt.py
    └── domain.py
```

### Configuration
```
packages/graphrag/graphrag/config/
├── models/            # Configuration models
│   ├── extract_graph_config.py
│   ├── local_search_config.py
│   └── global_search_config.py
└── defaults.py        # Default prompt values
```

## Usage Examples

### Using Built-in Prompts
```python
from graphrag.prompts.query import LOCAL_SEARCH_SYSTEM_PROMPT

# Built-in prompt with default entity types
prompt = LOCAL_SEARCH_SYSTEM_PROMPT.format(
    response_type="Multiple Paragraphs",
    context_data=data_tables
)
```

### Custom Prompt Loading
```yaml
# settings.yaml
entity_extraction:
  prompt: "./custom_prompts/my_extraction_prompt.txt"
```

### Prompt Tuning
```bash
# Auto-generate optimized prompts for your domain
graphrag prompt-tune \
  --root ./project \
  --domain "medical research" \
  --language "English"
```

## Prompt Design Principles

1. **Role Definition** - Clear specification of assistant's role and expertise
2. **Goal Statement** - Explicit description of desired output
3. **Format Specification** - Precise output format requirements (JSON, markdown, etc.)
4. **Grounding Rules** - Instructions for citing sources and avoiding hallucination
5. **Examples** - Few-shot learning examples for complex tasks
6. **Length Constraints** - Token/word limits for response management
7. **Iterative Refinement** - Multi-turn prompts for comprehensive extraction

## Statistics

- **Total Distinct Prompt Files**: 40+
- **Total Prompt Variables/Functions**: 60+
- **Query Prompts**: 7 variants
- **Index Prompts**: 5 core prompts
- **Prompt Templates**: 8 parameterized templates
- **Prompt Generators**: 7+ dynamic generators

## References

- [Manual Prompt Tuning Guide](../../docs/prompt_tuning/manual_prompt_tuning.md)
- [Auto Prompt Tuning Guide](../../docs/prompt_tuning/auto_prompt_tuning.md)
- [GraphRAG Configuration Documentation](../../docs/config/configuration.md)
