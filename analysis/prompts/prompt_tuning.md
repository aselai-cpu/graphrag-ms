# Prompt Tuning and Customization

GraphRAG provides a sophisticated prompt tuning system that allows you to automatically generate and customize prompts for your specific domain, language, and use case.

## Table of Contents
1. [Overview](#overview)
2. [Prompt Templates](#prompt-templates)
3. [Prompt Generators](#prompt-generators)
4. [Auto Prompt Tuning](#auto-prompt-tuning)
5. [Manual Prompt Customization](#manual-prompt-customization)
6. [Configuration and Resolution](#configuration-and-resolution)

---

## Overview

The prompt tuning system consists of three main components:

1. **Templates** - Parameterized prompt structures with placeholders
2. **Generators** - Functions that create customized prompts based on data samples
3. **Resolution** - Mechanism for loading built-in or custom prompts

### Why Prompt Tuning?

- **Domain Adaptation** - Adjust prompts for medical, legal, financial, or other specialized domains
- **Language Support** - Generate prompts in different languages
- **Entity Type Optimization** - Customize entity types for your data
- **Few-Shot Learning** - Generate relevant examples from your documents
- **Persona Customization** - Adjust the analyst role and perspective

---

## Prompt Templates

**Location**: `packages/graphrag/graphrag/prompt_tune/template/`

Templates are parameterized prompt structures used as bases for generation.

### Graph Extraction Templates

**File**: `extract_graph.py`

#### Standard Extraction Template

```python
GRAPH_EXTRACTION_PROMPT = """
-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types,
identify all entities of those types from the text and all relationships among the identified entities.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"<|><entity_name><|><entity_type><|><entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity)
   that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target
  entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the
  source entity and target entity
Format each relationship as ("relationship"<|><source_entity><|><target_entity><|><relationship_description><|><relationship_strength>)

3. Return output in {language} as a single list of all the entities and relationships identified
   in steps 1 and 2. Use **##** as the list delimiter.

4. When finished, output <|COMPLETE|>

######################
-Examples-
######################
{examples}

######################
-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:"""
```

**Parameters**:
- `{entity_types}` - Comma-separated entity type list
- `{language}` - Output language
- `{examples}` - Few-shot examples
- `{input_text}` - Text to extract from

#### JSON Output Template

```python
GRAPH_EXTRACTION_JSON_PROMPT = """
-Goal-
Given a text document, identify all entities and relationships. Return the output as a JSON list.

-Steps-
1. Identify all entities of the following types: [{entity_types}]
2. Identify all relationships between entities
3. Return as JSON array

-Output Format-
{{
    "entities": [
        {{
            "name": "ENTITY_NAME",
            "type": "ENTITY_TYPE",
            "description": "Description of entity"
        }}
    ],
    "relationships": [
        {{
            "source": "SOURCE_ENTITY",
            "target": "TARGET_ENTITY",
            "description": "Relationship description",
            "strength": 8
        }}
    ]
}}

-Examples-
{examples}

-Real Data-
Text: {input_text}

Output:"""
```

#### Untyped Extraction Template

```python
UNTYPED_GRAPH_EXTRACTION_PROMPT = """
-Goal-
Given a text document, identify all entities mentioned and relationships between them,
WITHOUT being constrained to predefined entity types.

-Steps-
1. Identify all entities. For each entity, determine:
   - entity_name: Name of the entity
   - entity_type: The type of entity (determine this from context)
   - entity_description: Comprehensive description
Format each entity as ("entity"<|><entity_name><|><entity_type><|><entity_description>)

2. Identify all relationships between entities
Format each relationship as ("relationship"<|><source_entity><|><target_entity><|><relationship_description><|><relationship_strength>)

3. Return output in {language} as a single list using **##** as delimiter

4. Output <|COMPLETE|> when finished

-Examples-
{examples}

-Real Data-
Text: {input_text}
######################
Output:"""
```

### Entity Summarization Template

**File**: `entity_summarization.py`

```python
ENTITY_SUMMARIZATION_PROMPT = """
{persona}

# Goal
Generate a comprehensive summary of the entity described below.
Given the entity name and a list of descriptions, all related to the same entity,
create a single, comprehensive description.

# Instructions
- Resolve any contradictory information
- Write in third person
- Include the entity name for full context
- Be comprehensive and detailed
- Output in {language}

#######
-Data-
Entity: {{entity_name}}
Description List:
{{description_list}}
#######

Generate a comprehensive summary:
"""
```

**Parameters**:
- `{persona}` - Expert role context
- `{language}` - Output language
- `{{entity_name}}` - Entity being summarized (runtime)
- `{{description_list}}` - Descriptions to merge (runtime)

### Community Report Template

**File**: `community_report_summarization.py`

```python
COMMUNITY_REPORT_SUMMARIZATION_PROMPT = """
{persona}

# Goal
Write a comprehensive assessment report of a community in a network graph, following the structure below.

{role}

# Report Structure
- TITLE: Community name representing key entities
- SUMMARY: Executive summary of the community
- IMPACT SEVERITY RATING: {report_rating_description}
- RATING EXPLANATION: Single sentence explaining the rating
- DETAILED FINDINGS: List of 5-10 key insights with comprehensive explanations

# Grounding Rules
- Cite data as: [Data: <dataset name> (record ids)]
- Maximum 5 record ids per reference, use "+more" if needed
- Example: "Entity X leads Project Y [Data: Entities (1, 5); Relationships (23)]"

# Output Format
Return JSON:
{{
    "title": "<report_title>",
    "summary": "<executive_summary>",
    "rating": <severity_rating>,
    "rating_explanation": "<explanation>",
    "findings": [
        {{
            "summary": "<insight_summary>",
            "explanation": "<detailed_explanation>"
        }}
    ]
}}

# Community Data
{{input_text}}

Generate the report in {language}:
"""
```

**Parameters**:
- `{persona}` - Analyst perspective
- `{role}` - Specific role instructions
- `{report_rating_description}` - Rating scale description
- `{language}` - Output language
- `{{input_text}}` - Community data (runtime)

---

## Prompt Generators

**Location**: `packages/graphrag/graphrag/prompt_tune/generator/`

Generators dynamically create optimized prompts based on data samples.

### Graph Extraction Prompt Generator

**File**: `extract_graph_prompt.py`

```python
async def create_extract_graph_prompt(
    entity_types: list[str] | str | None,
    docs: list[str],
    examples: list[str],
    language: str,
    max_token_count: int = 2000,
    json_mode: bool = False,
    output_path: Path | None = None,
    min_examples_required: int = 2,
) -> str:
    """
    Create a graph extraction prompt with examples generated from sample documents.

    Args:
        entity_types: List of entity types to extract, or None for untyped
        docs: Sample documents to generate examples from
        examples: Pre-existing examples to include
        language: Output language
        max_token_count: Token budget for the prompt
        json_mode: Use JSON output format
        output_path: Where to save the generated prompt
        min_examples_required: Minimum number of examples to include

    Returns:
        Generated prompt string
    """
    # Select appropriate template
    if entity_types is None:
        template = UNTYPED_GRAPH_EXTRACTION_PROMPT
    elif json_mode:
        template = GRAPH_EXTRACTION_JSON_PROMPT
    else:
        template = GRAPH_EXTRACTION_PROMPT

    # Generate examples from sample docs if needed
    if len(examples) < min_examples_required:
        generated_examples = await generate_examples_from_docs(
            docs=docs,
            entity_types=entity_types,
            num_examples=min_examples_required - len(examples)
        )
        examples.extend(generated_examples)

    # Format examples
    formatted_examples = format_examples(examples, entity_types)

    # Fill template
    prompt = template.format(
        entity_types=", ".join(entity_types) if entity_types else "ANY",
        language=language,
        examples=formatted_examples
    )

    # Trim to token budget if needed
    prompt = trim_to_token_budget(prompt, max_token_count)

    # Save if output path provided
    if output_path:
        output_path.write_text(prompt, encoding="utf-8")

    return prompt
```

### Entity Summarization Generator

**File**: `entity_summarization_prompt.py`

```python
def create_entity_summarization_prompt(
    persona: str,
    language: str = "English",
    output_path: Path | None = None,
) -> str:
    """
    Create an entity summarization prompt with specified persona.

    Args:
        persona: Expert role and perspective
        language: Output language
        output_path: Where to save the prompt

    Returns:
        Generated prompt string
    """
    prompt = ENTITY_SUMMARIZATION_PROMPT.format(
        persona=persona,
        language=language
    )

    if output_path:
        output_path.write_text(prompt, encoding="utf-8")

    return prompt
```

### Domain-Specific Generators

**File**: `domain.py`

```python
async def generate_domain_prompt(
    domain: str,
    docs: list[str],
    language: str = "English",
) -> str:
    """
    Generate a domain-specific extraction prompt.

    Args:
        domain: Target domain (e.g., "medical", "legal", "financial")
        docs: Sample documents from the domain
        language: Output language

    Returns:
        Domain-optimized prompt
    """
    # Analyze documents to identify domain-specific patterns
    domain_patterns = await analyze_domain_patterns(docs)

    # Generate domain-specific entity types
    entity_types = await infer_entity_types(docs, domain)

    # Generate domain-specific examples
    examples = await generate_domain_examples(docs, entity_types, domain)

    # Create persona based on domain
    persona = create_domain_persona(domain)

    # Generate prompt
    return await create_extract_graph_prompt(
        entity_types=entity_types,
        docs=docs,
        examples=examples,
        language=language
    )
```

**File**: `entity_types.py`

```python
async def generate_entity_types(
    docs: list[str],
    max_types: int = 10,
    language: str = "English",
) -> list[str]:
    """
    Automatically infer entity types from sample documents.

    Args:
        docs: Sample documents
        max_types: Maximum number of entity types to generate
        language: Output language

    Returns:
        List of inferred entity types
    """
    # Use LLM to analyze documents and identify entity types
    prompt = f"""
    Analyze the following documents and identify the {max_types} most important
    types of entities mentioned. Return as a comma-separated list.

    Documents:
    {format_docs_sample(docs)}

    Entity types:
    """

    response = await llm_call(prompt)
    entity_types = [t.strip() for t in response.split(",")]

    return entity_types[:max_types]
```

**File**: `persona.py`

```python
def create_persona(
    domain: str | None = None,
    role: str | None = None,
    expertise: list[str] | None = None,
) -> str:
    """
    Create a persona string for prompts.

    Args:
        domain: Target domain
        role: Specific role (e.g., "data analyst", "investigator")
        expertise: Areas of expertise

    Returns:
        Formatted persona string
    """
    if domain:
        domain_expertise = {
            "medical": "You are a medical data analyst with expertise in clinical terminology, disease classifications, and healthcare systems.",
            "legal": "You are a legal analyst with expertise in case law, contracts, and regulatory compliance.",
            "financial": "You are a financial analyst with expertise in markets, securities, and corporate finance.",
            "technology": "You are a technology analyst with expertise in software systems, networks, and IT infrastructure.",
        }
        base_persona = domain_expertise.get(domain, f"You are an expert analyst in the {domain} domain.")
    else:
        base_persona = "You are an expert data analyst."

    if role:
        base_persona += f" Your role is to act as a {role}."

    if expertise:
        base_persona += f" Your areas of expertise include: {', '.join(expertise)}."

    return base_persona
```

---

## Auto Prompt Tuning

**CLI Command**: `graphrag prompt-tune`

Auto prompt tuning generates optimized prompts automatically based on your data.

### Basic Usage

```bash
# Basic auto-tuning
graphrag prompt-tune \
  --root ./ragtest \
  --domain "medical research" \
  --language "English"

# With custom settings
graphrag prompt-tune \
  --root ./ragtest \
  --config ./settings.yaml \
  --domain "financial services" \
  --language "English" \
  --max-tokens 2000 \
  --chunk-size 1000 \
  --output ./prompts
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--root` | Project root directory | Required |
| `--domain` | Target domain for optimization | None |
| `--language` | Output language | "English" |
| `--max-tokens` | Token budget for prompts | 2000 |
| `--chunk-size` | Document chunk size for sampling | 1000 |
| `--output` | Output directory for generated prompts | `./prompts` |
| `--config` | Path to settings file | `./settings.yaml` |
| `--discover-entity-types` | Auto-discover entity types | True |
| `--skip-entity-types` | Entity types to skip | None |

### Tuning Process

1. **Document Sampling**
   - Loads sample documents from your corpus
   - Selects representative chunks across the dataset

2. **Entity Type Discovery**
   - Analyzes documents to identify relevant entity types
   - Uses LLM to infer domain-specific entities
   - Can be overridden with manual entity type list

3. **Example Generation**
   - Runs extraction on sample chunks
   - Generates 2-5 high-quality examples
   - Validates examples for correctness

4. **Persona Creation**
   - Generates domain-specific analyst persona
   - Customizes role and expertise descriptions

5. **Prompt Assembly**
   - Combines template + examples + persona
   - Optimizes token usage
   - Validates prompt structure

6. **Output Generation**
   - Saves prompts to output directory
   - Creates configuration entries
   - Generates tuning report

### Output Structure

```
./prompts/
├── entity_extraction.txt
├── entity_summarization.txt
├── community_report.txt
├── claim_extraction.txt
└── tuning_report.json
```

**tuning_report.json**:
```json
{
    "domain": "medical research",
    "language": "English",
    "entity_types": [
        "DISEASE",
        "DRUG",
        "PROTEIN",
        "GENE",
        "CLINICAL_TRIAL",
        "MEDICAL_PROCEDURE",
        "SYMPTOM",
        "ORGANIZATION"
    ],
    "persona": "You are a medical research analyst with expertise in clinical studies, pharmacology, and molecular biology.",
    "num_examples": 3,
    "token_counts": {
        "entity_extraction": 1850,
        "entity_summarization": 450,
        "community_report": 1200
    },
    "sample_documents_used": 10,
    "timestamp": "2024-01-30T10:30:00Z"
}
```

### Using Auto-Tuned Prompts

Update `settings.yaml` to use the generated prompts:

```yaml
entity_extraction:
  prompt: "./prompts/entity_extraction.txt"
  entity_types: ${file:prompts/tuning_report.json#entity_types}

summarize_descriptions:
  prompt: "./prompts/entity_summarization.txt"

community_reports:
  prompt: "./prompts/community_report.txt"

claim_extraction:
  prompt: "./prompts/claim_extraction.txt"
```

---

## Manual Prompt Customization

You can manually customize any prompt by creating a text file and referencing it in configuration.

### Step 1: Create Custom Prompt File

**custom_extraction.txt**:
```
-Goal-
Extract software components and their dependencies from technical documentation.

-Steps-
1. Identify all software entities:
   - LIBRARY: Software libraries and packages
   - SERVICE: Microservices and APIs
   - DATABASE: Data storage systems
   - TOOL: Development and deployment tools

2. Identify dependencies and relationships between components

3. Format output as:
   ("entity"<|><name><|><type><|><description>)
   ##
   ("relationship"<|><source><|><target><|><relationship><|><strength>)

4. Output <|COMPLETE|> when done

-Example-
Text: "The authentication service uses PostgreSQL for user storage and Redis for session caching."

Output:
("entity"<|>AUTHENTICATION SERVICE<|>SERVICE<|>Service handling user authentication)
##
("entity"<|>POSTGRESQL<|>DATABASE<|>Relational database storing user data)
##
("entity"<|>REDIS<|>DATABASE<|>In-memory cache for session storage)
##
("relationship"<|>AUTHENTICATION SERVICE<|>POSTGRESQL<|>Uses PostgreSQL for persistent user storage<|>9)
##
("relationship"<|>AUTHENTICATION SERVICE<|>REDIS<|>Uses Redis for session caching<|>8)
<|COMPLETE|>

-Real Data-
Text: {input_text}
Output:
```

### Step 2: Reference in Configuration

**settings.yaml**:
```yaml
entity_extraction:
  prompt: "./custom_extraction.txt"
  entity_types:
    - LIBRARY
    - SERVICE
    - DATABASE
    - TOOL
```

### Customization Tips

1. **Keep Structure Consistent**
   - Maintain delimiter format (`<|>`, `##`, `<|COMPLETE|>`)
   - Preserve JSON structure for community reports
   - Keep data citation format for query prompts

2. **Include Good Examples**
   - 2-3 examples for complex extractions
   - Show edge cases and difficult scenarios
   - Demonstrate proper formatting

3. **Be Specific**
   - Clear entity type definitions
   - Explicit relationship strength guidelines
   - Precise output format requirements

4. **Test Iteratively**
   - Run on sample documents
   - Check extraction quality
   - Refine based on results

---

## Configuration and Resolution

### Prompt Resolution Mechanism

**How GraphRAG loads prompts**:

```python
# From config models (e.g., extract_graph_config.py)
@dataclass
class ExtractGraphConfig:
    prompt: str | None = None  # Path to custom prompt file
    entity_types: list[str] | None = None

    def resolved_prompts(self) -> ExtractGraphPrompts:
        """Get the resolved graph extraction prompts."""
        if self.prompt:
            # Load custom prompt from file
            prompt_text = Path(self.prompt).read_text(encoding="utf-8")
        else:
            # Use built-in default prompt
            from graphrag.prompts.index import GRAPH_EXTRACTION_PROMPT
            prompt_text = GRAPH_EXTRACTION_PROMPT

        return ExtractGraphPrompts(extraction_prompt=prompt_text)
```

**Resolution Priority**:
1. Custom prompt file (if `prompt` is specified in config)
2. Built-in default prompt (if `prompt` is None)
3. Environment-specific overrides (if configured)

### Configuration Examples

**Full settings.yaml with custom prompts**:

```yaml
# Entity Extraction
entity_extraction:
  enabled: true
  prompt: "./prompts/entity_extraction.txt"
  entity_types:
    - organization
    - person
    - location
    - technology
  max_gleanings: 1

# Claim Extraction
claim_extraction:
  enabled: true
  prompt: "./prompts/claim_extraction.txt"
  description: "allegations, partnerships, and achievements"
  max_gleanings: 1

# Entity Summarization
summarize_descriptions:
  prompt: "./prompts/entity_summarization.txt"
  max_length: 500

# Community Reports
community_reports:
  prompt: "./prompts/community_report.txt"
  max_length: 1500

# Local Search
local_search:
  prompt: "./prompts/local_search.txt"
  text_unit_prop: 0.5
  community_prop: 0.25

# Global Search
global_search:
  map_prompt: "./prompts/global_map.txt"
  reduce_prompt: "./prompts/global_reduce.txt"
  knowledge_prompt: "./prompts/global_knowledge.txt"
  max_data_tokens: 8000
```

**Mixing built-in and custom prompts**:

```yaml
entity_extraction:
  prompt: "./prompts/entity_extraction.txt"  # Custom

summarize_descriptions:
  prompt: null  # Use built-in default

community_reports:
  prompt: "./prompts/community_report.txt"  # Custom
```

### Default Prompt Values

**File**: `packages/graphrag/graphrag/config/defaults.py`

```python
@dataclass
class ExtractGraphDefaults:
    prompt: str | None = None  # None = use built-in
    entity_types: list[str] = field(
        default_factory=lambda: ["organization", "person", "geo", "event"]
    )
    max_gleanings: int = 1

@dataclass
class LocalSearchDefaults:
    prompt: str | None = None
    text_unit_prop: float = 0.5
    community_prop: float = 0.25

@dataclass
class GlobalSearchDefaults:
    map_prompt: str | None = None
    reduce_prompt: str | None = None
    knowledge_prompt: str | None = None
    max_data_tokens: int = 8000
```

---

## Advanced Tuning Techniques

### Multi-Language Support

```bash
# Generate prompts in Spanish
graphrag prompt-tune \
  --root ./ragtest \
  --domain "legal" \
  --language "Spanish"
```

**Generated prompt excerpt**:
```
-Objetivo-
Dado un documento de texto, identificar todas las entidades y relaciones entre ellas.

-Pasos-
1. Identificar todas las entidades de los siguientes tipos: [ORGANIZACIÓN, PERSONA, UBICACIÓN]
...
3. Devolver la salida en español como una lista única utilizando **##** como delimitador
```

### Domain-Specific Entity Types

**Medical Domain**:
```yaml
entity_extraction:
  entity_types:
    - DISEASE
    - DRUG
    - PROTEIN
    - GENE
    - CLINICAL_TRIAL
    - MEDICAL_PROCEDURE
    - SYMPTOM
    - MEDICAL_DEVICE
    - BIOMARKER
```

**Legal Domain**:
```yaml
entity_extraction:
  entity_types:
    - CASE
    - STATUTE
    - REGULATION
    - COURT
    - JUDGE
    - LAW_FIRM
    - CONTRACT
    - LEGAL_DOCTRINE
```

**Financial Domain**:
```yaml
entity_extraction:
  entity_types:
    - COMPANY
    - SECURITY
    - CURRENCY
    - MARKET
    - FINANCIAL_INSTRUMENT
    - REGULATORY_BODY
    - ECONOMIC_INDICATOR
    - TRANSACTION
```

### Custom Rating Scales

**community_report.txt** with custom severity scale:
```
# Impact Severity Rating
Rate the business impact of this community on a scale of 0-10:
- 0-2: Minimal business impact, routine operations
- 3-4: Low impact, supporting functions
- 5-6: Moderate impact, important but not critical
- 7-8: High impact, critical business functions
- 9-10: Mission critical, core revenue or strategic importance
```

### Persona Variations

**Investigative Persona**:
```
You are an investigative analyst focused on identifying risks, conflicts of interest,
and potential compliance violations. Your expertise includes fraud detection, regulatory
compliance, and forensic analysis. Approach the data with skepticism and highlight any
red flags or unusual patterns.
```

**Strategic Persona**:
```
You are a strategic business analyst focused on identifying opportunities, partnerships,
and competitive advantages. Your expertise includes market analysis, competitive intelligence,
and strategic planning. Look for patterns that indicate strategic positioning and growth potential.
```

---

## Testing and Validation

### Prompt Quality Checklist

- [ ] Output format is parseable by GraphRAG
- [ ] Examples are clear and representative
- [ ] Entity types match domain requirements
- [ ] Relationship definitions are precise
- [ ] Token count is within LLM limits
- [ ] Language is consistent throughout
- [ ] Citations follow data reference format
- [ ] Completionality signals are present

### Validation Commands

```bash
# Test extraction on sample documents
graphrag index \
  --root ./ragtest \
  --config ./settings.yaml \
  --verbose \
  --resume ./test_run

# Check extraction quality
graphrag validate \
  --root ./ragtest \
  --output ./validation_report.json
```

### Iterative Refinement

1. Run indexing on sample documents
2. Review extracted entities and relationships
3. Identify misclassifications or missed entities
4. Update prompt with better examples or instructions
5. Re-run and compare results
6. Iterate until quality threshold met

---

## References

- [Manual Prompt Tuning Guide](../../docs/prompt_tuning/manual_prompt_tuning.md)
- [Auto Prompt Tuning Guide](../../docs/prompt_tuning/auto_prompt_tuning.md)
- Template Source: `packages/graphrag/graphrag/prompt_tune/template/`
- Generator Source: `packages/graphrag/graphrag/prompt_tune/generator/`
- Configuration Models: `packages/graphrag/graphrag/config/models/`
