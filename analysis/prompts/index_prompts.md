# Index Prompts

Index prompts are used during the knowledge graph construction phase to extract structured information from unstructured text documents. These prompts power the indexing pipeline that builds the graph database.

## Table of Contents
1. [Graph Extraction](#graph-extraction)
2. [Claim Extraction](#claim-extraction)
3. [Community Report Generation](#community-report-generation)
4. [Description Summarization](#description-summarization)

---

## Graph Extraction

**Location**: `packages/graphrag/graphrag/prompts/index/extract_graph.py`

**Purpose**: Extracts entities and relationships from text documents to build the knowledge graph.

**Key Features**:
- Identifies entities with names, types, and descriptions
- Extracts relationships with strength scores (1-10)
- Uses custom delimiters (`<|>` and `##`) for parsing
- Supports iterative extraction with CONTINUE and LOOP prompts
- Returns completion signal (`<|COMPLETE|>`)

### Main Extraction Prompt

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

3. Return output in English as a single list of all the entities and relationships identified
   in steps 1 and 2. Use **##** as the list delimiter.

4. When finished, output <|COMPLETE|>

######################
-Examples-
######################
Example 1:
Entity_types: ORGANIZATION,PERSON
Text:
The Verdantis's Central Institution is scheduled to meet on Monday and Thursday, with the
institution planning to release its latest policy decision on Thursday at 1:30 p.m. PDT,
followed by a press conference where Central Institution Chair Martin Smith will take questions.
Investors expect the Market Strategy Committee to hold its benchmark interest rate steady in
a range of 3.5%-3.75%.
######################
Output:
("entity"<|>CENTRAL INSTITUTION<|>ORGANIZATION<|>The Central Institution is the Federal Reserve of Verdantis, which is setting interest rates on Monday and Thursday)
##
("entity"<|>MARTIN SMITH<|>PERSON<|>Martin Smith is the chair of the Central Institution)
##
("entity"<|>MARKET STRATEGY COMMITTEE<|>ORGANIZATION<|>The Central Institution committee makes key decisions about interest rates and the growth of Verdantis's money supply)
##
("relationship"<|>MARTIN SMITH<|>CENTRAL INSTITUTION<|>Martin Smith is the Chair of the Central Institution and will answer questions at a press conference<|>9)
<|COMPLETE|>

######################
Example 2:
Entity_types: ORGANIZATION
Text:
TechGlobal's (TG) stock skyrocketed in its opening day on the Global Exchange Thursday. But IPO
experts warn that the semiconductor corporation's debut on the public markets isn't indicative of
how other newly listed companies may perform.

TechGlobal, a formerly public company, was taken private by Vision Holdings in 2014. The
well-established chip designer says it powers 85% of premium smartphones.
######################
Output:
("entity"<|>TECHGLOBAL<|>ORGANIZATION<|>TechGlobal is a stock now listed on the Global Exchange which powers 85% of premium smartphones)
##
("entity"<|>VISION HOLDINGS<|>ORGANIZATION<|>Vision Holdings is a firm that previously owned TechGlobal)
##
("relationship"<|>TECHGLOBAL<|>VISION HOLDINGS<|>Vision Holdings formerly owned TechGlobal from 2014 until present<|>5)
<|COMPLETE|>

######################
Example 3:
Entity_types: ORGANIZATION,GEO,PERSON
Text:
Five Aurelians jailed for 8 years in Firuzabad and widely regarded as hostages are on their way
home to Aurelia.

The swap orchestrated by Quintara was finalized when $8bn of Firuzi funds were transferred to
financial institutions in Krohaara, the capital of Quintara.

The exchange initiated in Firuzabad's capital, Tiruzia, led to the four men and one woman, who
are also Firuzi nationals, boarding a chartered flight to Krohaara.

They were welcomed by senior Aurelian officials and are now on their way to Aurelia's capital, Cashion.

The Aurelians include 39-year-old businessman Samuel Namara, who has been held in Tiruzia's
Alhamia Prison, as well as journalist Durke Bataglani, 59, and environmentalist Meggie Tazbah, 53,
who also holds Bratinas nationality.
######################
Output:
("entity"<|>FIRUZABAD<|>GEO<|>Firuzabad held Aurelians as hostages)
##
("entity"<|>AURELIA<|>GEO<|>Country seeking to release hostages)
##
("entity"<|>QUINTARA<|>GEO<|>Country that negotiated a swap of money in exchange for hostages)
##
("entity"<|>TIRUZIA<|>GEO<|>Capital of Firuzabad where the Aurelians were being held)
##
("entity"<|>KROHAARA<|>GEO<|>Capital city in Quintara)
##
("entity"<|>CASHION<|>GEO<|>Capital city in Aurelia)
##
("entity"<|>SAMUEL NAMARA<|>PERSON<|>Aurelian who spent time in Tiruzia's Alhamia Prison)
##
("entity"<|>ALHAMIA PRISON<|>GEO<|>Prison in Tiruzia)
##
("entity"<|>DURKE BATAGLANI<|>PERSON<|>Aurelian journalist who was held hostage)
##
("entity"<|>MEGGIE TAZBAH<|>PERSON<|>Bratinas national and environmentalist who was held hostage)
##
("relationship"<|>FIRUZABAD<|>AURELIA<|>Firuzabad negotiated a hostage exchange with Aurelia<|>2)
##
("relationship"<|>QUINTARA<|>AURELIA<|>Quintara brokered the hostage exchange between Firuzabad and Aurelia<|>2)
##
("relationship"<|>QUINTARA<|>FIRUZABAD<|>Quintara brokered the hostage exchange between Firuzabad and Aurelia<|>2)
##
("relationship"<|>SAMUEL NAMARA<|>ALHAMIA PRISON<|>Samuel Namara was a prisoner at Alhamia prison<|>8)
##
("relationship"<|>SAMUEL NAMARA<|>MEGGIE TAZBAH<|>Samuel Namara and Meggie Tazbah were exchanged in the same hostage release<|>2)
##
("relationship"<|>SAMUEL NAMARA<|>DURKE BATAGLANI<|>Samuel Namara and Durke Bataglani were exchanged in the same hostage release<|>2)
##
("relationship"<|>MEGGIE TAZBAH<|>DURKE BATAGLANI<|>Meggie Tazbah and Durke Bataglani were exchanged in the same hostage release<|>2)
##
("relationship"<|>SAMUEL NAMARA<|>FIRUZABAD<|>Samuel Namara was a hostage in Firuzabad<|>2)
##
("relationship"<|>MEGGIE TAZBAH<|>FIRUZABAD<|>Meggie Tazbah was a hostage in Firuzabad<|>2)
##
("relationship"<|>DURKE BATAGLANI<|>FIRUZABAD<|>Durke Bataglani was a hostage in Firuzabad<|>2)
<|COMPLETE|>

######################
-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:"""
```

### Iterative Extraction Prompts

```python
# Used when initial extraction may have missed entities
CONTINUE_PROMPT = """
MANY entities and relationships were missed in the last extraction. Remember to ONLY emit
entities that match any of the previously extracted types. Add them below using the same format:
"""

# Used to check if extraction is complete
LOOP_PROMPT = """
It appears some entities and relationships may have still been missed. Answer Y if there are
still entities or relationships that need to be added, or N if there are none. Please answer
with a single letter Y or N.
"""
```

### Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `entity_types` | string | Comma-separated list of entity types to extract | "organization, person, geo, event" |
| `input_text` | string | Source text to extract from | Required |

### Output Format

**Entity Format**:
```
("entity"<|><ENTITY_NAME><|><ENTITY_TYPE><|><entity description>)
```

**Relationship Format**:
```
("relationship"<|><SOURCE_ENTITY><|><TARGET_ENTITY><|><relationship description><|><strength_score>)
```

**Delimiter**: `##` between each entity/relationship

**Completion Signal**: `<|COMPLETE|>`

### Sample Extraction Flow

**Input Text**:
```
Microsoft announced the acquisition of OpenAI for $10 billion. CEO Satya Nadella praised
the deal as transformative for AI development.
```

**Entity Types**: `ORGANIZATION, PERSON, EVENT`

**Output**:
```
("entity"<|>MICROSOFT<|>ORGANIZATION<|>Microsoft is a technology company that acquired OpenAI)
##
("entity"<|>OPENAI<|>ORGANIZATION<|>OpenAI is an AI research company acquired by Microsoft for $10 billion)
##
("entity"<|>SATYA NADELLA<|>PERSON<|>Satya Nadella is the CEO of Microsoft)
##
("entity"<|>ACQUISITION<|>EVENT<|>Microsoft's acquisition of OpenAI for $10 billion)
##
("relationship"<|>MICROSOFT<|>OPENAI<|>Microsoft acquired OpenAI for $10 billion<|>10)
##
("relationship"<|>SATYA NADELLA<|>MICROSOFT<|>Satya Nadella is the CEO of Microsoft<|>9)
##
("relationship"<|>SATYA NADELLA<|>ACQUISITION<|>Satya Nadella praised the acquisition as transformative<|>7)
<|COMPLETE|>
```

### Relationship Strength Guidelines

| Score | Description |
|-------|-------------|
| 1-3 | Weak/incidental connection |
| 4-6 | Moderate relationship |
| 7-8 | Strong relationship |
| 9-10 | Critical/defining relationship |

---

## Claim Extraction

**Location**: `packages/graphrag/graphrag/prompts/index/extract_claims.py`

**Purpose**: Extracts factual claims, assertions, and accusations from text with status indicators and temporal information.

**Key Features**:
- Identifies claims with TRUE/FALSE/SUSPECTED status
- Includes temporal information (claim dates in ISO-8601 format)
- Links claims to subject and object entities
- Provides source text quotes as evidence
- Categorizes claims by type

### Claim Extraction Prompt

```python
CLAIM_EXTRACTION_PROMPT = """
-Target activity-
You are an intelligent assistant that helps a human analyst to analyze claims against certain
entities presented in a text document.

-Goal-
Given a text document that is potentially relevant to this activity, an entity specification,
and a claim description, extract all entities that match the entity specification and all claims
against those entities.

-Steps-
1. Extract all named entities that match the predefined entity specification. Entity specification
   can either be a list of entity names or a list of entity types.
2. For each entity identified in step 1, extract all claims associated with the entity. Claims
   need to match the specified claim description, and the entity should be the subject of the claim.
   For each claim, extract the following information:
   - Subject: name of the entity that is subject of the claim, capitalized. The subject entity
     is one that committed the action described in the claim. Subject needs to be one of the
     named entities identified in step 1.
   - Object: name of the entity that is object of the claim, capitalized. The object entity is
     one that received the action described in the claim. If object entity is unknown, use
     **NONE**.
   - Claim Type: overall category of the claim, capitalized. Name it in a way that can be
     repeated across multiple text inputs, so that similar claims share the same claim type
   - Claim Status: **TRUE**, **FALSE**, or **SUSPECTED**. TRUE means the claim is confirmed,
     FALSE means the claim is found to be False, SUSPECTED means the claim is not verified.
   - Claim Description: Detailed description explaining the reasoning behind the claim, together
     with all the related evidence and references.
   - Claim Date: Period (start_date, end_date) when the claim was made. Both start_date and
     end_date should be in ISO-8601 format. If the claim was made on a single date rather than
     a date range, set the same date for both start_date and end_date. If date is unknown, return
     **NONE**.
   - Claim Source Text: List of **all** quotes from the original text that are relevant to the claim.

Format each claim as (<claim_type>, <subject_entity>, <object_entity>, <claim_status>,
<claim_start_date>, <claim_end_date>, <claim_description>, <claim_source_text>)

3. Return output in English as a single list of all claims identified in steps 1 and 2. Use
   **##** as the list delimiter.

4. When finished, output <|COMPLETE|>

-Examples-
######################
Example 1:
Entity specification: ORGANIZATION
Claim description: red flags associated with an entity
Text:
According to an article on 2022/01/10, Company A was fined for breaching labor regulations.

######################
Output:
(REGULATORY_VIOLATION, COMPANY A, GOVERNMENT, TRUE, 2022-01-10, 2022-01-10, Company A was found to have violated labor regulations and was fined by the government, "According to an article on 2022/01/10, Company A was fined for breaching labor regulations.")
<|COMPLETE|>

######################
Example 2:
Entity specification: COMPANY A, PERSON B
Claim description: red flags associated with an entity
Text:
On 2022/02/15, Person B was suspected of engaging in insider trading with Company A. The
investigation is ongoing as of 2023/03/01.

######################
Output:
(INSIDER_TRADING, PERSON B, COMPANY A, SUSPECTED, 2022-02-15, 2023-03-01, Person B is under investigation for suspected insider trading involving Company A. The investigation began on 2022/02/15 and is still ongoing as of 2023/03/01, "On 2022/02/15, Person B was suspected of engaging in insider trading with Company A. The investigation is ongoing as of 2023/03/01.")
<|COMPLETE|>

######################
-Real Data-
######################
Entity specification: {entity_specs}
Claim description: {claim_description}
Text: {input_text}
######################
Output:
"""
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `entity_specs` | string | Entity names or types to look for claims about |
| `claim_description` | string | Type of claims to extract (e.g., "red flags", "achievements", "allegations") |
| `input_text` | string | Source text to extract claims from |

### Output Format

```
(<CLAIM_TYPE>, <SUBJECT_ENTITY>, <OBJECT_ENTITY>, <STATUS>, <START_DATE>, <END_DATE>, <description>, <source_quotes>)
```

**Delimiters**:
- Between claims: `##`
- Completion signal: `<|COMPLETE|>`

### Sample Claim Extraction

**Input**:
```
Entity specification: TECHCORP, JOHN DOE
Claim description: business activities and partnerships
Text: On January 15, 2024, TechCorp announced a strategic partnership with DataSystems Inc.
CEO John Doe stated the partnership would accelerate AI development. Some analysts question
whether TechCorp has the resources to fulfill partnership obligations.
```

**Output**:
```
(STRATEGIC_PARTNERSHIP, TECHCORP, DATASYSTEMS INC, TRUE, 2024-01-15, 2024-01-15, TechCorp formed a strategic partnership with DataSystems Inc. to accelerate AI development, "On January 15, 2024, TechCorp announced a strategic partnership with DataSystems Inc.")
##
(LEADERSHIP_STATEMENT, JOHN DOE, DATASYSTEMS INC, TRUE, 2024-01-15, 2024-01-15, CEO John Doe publicly endorsed the partnership and stated it would accelerate AI development, "CEO John Doe stated the partnership would accelerate AI development.")
##
(RESOURCE_CONCERN, TECHCORP, NONE, SUSPECTED, 2024-01-15, 2024-01-15, Analysts have raised questions about whether TechCorp has sufficient resources to meet partnership obligations, "Some analysts question whether TechCorp has the resources to fulfill partnership obligations.")
<|COMPLETE|>
```

### Claim Status Guidelines

| Status | Description | Use When |
|--------|-------------|----------|
| TRUE | Confirmed fact | Explicitly stated as fact, officially announced |
| FALSE | Disproven claim | Explicitly refuted or proven false |
| SUSPECTED | Unverified | Alleged, rumored, under investigation, questioned |

---

## Community Report Generation

**Location**: `packages/graphrag/graphrag/prompts/index/community_report.py`

**Purpose**: Generates comprehensive analytical reports about communities of related entities in the knowledge graph.

**Key Features**:
- Creates structured JSON reports
- Includes impact severity rating (0-10)
- Lists 5-10 key insights with explanations
- Limits report length to configurable word count
- Enforces data citation rules

### Community Report Prompt

```python
COMMUNITY_REPORT_PROMPT = """
You are an AI assistant that helps a human analyst to perform general information discovery.
Information discovery is the process of identifying and assessing relevant information associated
with certain entities (e.g., organizations and individuals) within a network.

# Goal
Write a comprehensive report of a community, given a list of entities that belong to the community
as well as their relationships and optional associated claims. The report will be used to inform
decision-makers about information associated with the community and their potential impact. The
content of this report includes an overview of the community's key entities, their legal compliance,
technical capabilities, reputation, and noteworthy claims.

# Report Structure

The report should include the following sections:

- TITLE: community's name that represents its key entities - title should be short but specific.
  When possible, include representative named entities in the title.
- SUMMARY: An executive summary of the community's overall structure, how its entities are related
  to each other, and significant information associated with its entities.
- IMPACT SEVERITY RATING: a float score between 0-10 that represents the severity of IMPACT posed
  by entities within the community. IMPACT is the scored importance of a community.
- RATING EXPLANATION: Give a single sentence explanation of the IMPACT severity rating.
- DETAILED FINDINGS: A list of 5-10 key insights about the community. Each insight should have a
  short summary followed by multiple paragraphs of explanatory text grounded according to the
  grounding rules below. Be comprehensive.

Return output as a well-formed JSON-formatted string with the following format:
{
    "title": <report_title>,
    "summary": <executive_summary>,
    "rating": <impact_severity_rating>,
    "rating_explanation": <rating_explanation>,
    "findings": [
        {
            "summary": <insight_1_summary>,
            "explanation": <insight_1_explanation>
        },
        {
            "summary": <insight_2_summary>,
            "explanation": <insight_2_explanation>
        }
    ]
}

# Grounding Rules

Points supported by data should list their data references as follows:

"This is an example sentence supported by multiple data references [Data: <dataset name> (record ids); <dataset name> (record ids)]."

Do not list more than 5 record ids in a single reference. Instead, list the top 5 most relevant
record ids and add "+more" to indicate that there are more.

For example:
"Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Reports (1), Entities (5, 7); Relationships (23); Claims (7, 2, 34, 64, 46, +more)]."

where 1, 5, 7, 23, 2, 34, 46, and 64 represent the id (not the index) of the relevant data record.

Do not include information where the supporting evidence for it is not provided.

Limit the total report length to {max_report_length} words.


# Example Input
-----------
Text:

Entities

human_readable_id,title,description
5,VERDANT OASIS PLAZA,Verdant Oasis Plaza is the location of the Unity March
6,HARMONY ASSEMBLY,Harmony Assembly is an organization that is holding a march at Verdant Oasis Plaza

Relationships

human_readable_id,source,target,description
37,VERDANT OASIS PLAZA,UNITY MARCH,Verdant Oasis Plaza is the location of the Unity March
38,VERDANT OASIS PLAZA,HARMONY ASSEMBLY,Harmony Assembly is holding a march at Verdant Oasis Plaza
39,VERDANT OASIS PLAZA,UNITY MARCH,The Unity March is taking place at Verdant Oasis Plaza
40,VERDANT OASIS PLAZA,TRIBUNE SPOTLIGHT,Tribune Spotlight is reporting on the Unity march taking place at Verdant Oasis Plaza
41,VERDANT OASIS PLAZA,BAILEY ASADI,Bailey Asadi is speaking at Verdant Oasis Plaza about the march
43,HARMONY ASSEMBLY,UNITY MARCH,Harmony Assembly is organizing the Unity March

Output:
{
    "title": "Verdant Oasis Plaza and Unity March",
    "summary": "The community revolves around the Verdant Oasis Plaza, which is the location of the Unity March. The plaza has relationships with the Harmony Assembly, Unity March, and Tribune Spotlight, all of which are associated with the march event.",
    "rating": 5.0,
    "rating_explanation": "The impact severity rating is moderate due to the potential for unrest or conflict during the Unity March.",
    "findings": [
        {
            "summary": "Verdant Oasis Plaza as the central location",
            "explanation": "Verdant Oasis Plaza is the central entity in this community, serving as the location for the Unity March. This plaza is the common link between all other entities, suggesting its significance in the community. The plaza's association with the march could potentially lead to issues such as public disorder or conflict, depending on the nature of the march and the reactions it provokes. [Data: Entities (5), Relationships (37, 38, 39, 40, 41,+more)]"
        },
        {
            "summary": "Harmony Assembly's role in the community",
            "explanation": "Harmony Assembly is another key entity in this community, being the organizer of the march at Verdant Oasis Plaza. The nature of Harmony Assembly and its march could be a potential source of threat, depending on their objectives and the reactions they provoke. The relationship between Harmony Assembly and the plaza is crucial in understanding the dynamics of this community. [Data: Entities(6), Relationships (38, 43)]"
        },
        {
            "summary": "Unity March as a significant event",
            "explanation": "The Unity March is a significant event taking place at Verdant Oasis Plaza. This event is a key factor in the community's dynamics and could be a potential source of threat, depending on the nature of the march and the reactions it provokes. The relationship between the march and the plaza is crucial in understanding the dynamics of this community. [Data: Relationships (39)]"
        },
        {
            "summary": "Role of Tribune Spotlight",
            "explanation": "Tribune Spotlight is reporting on the Unity March taking place in Verdant Oasis Plaza. This suggests that the event has attracted media attention, which could amplify its impact on the community. The role of Tribune Spotlight could be significant in shaping public perception of the event and the entities involved. [Data: Relationships (40)]"
        }
    ]
}


# Real Data

Use the following text for your answer. Do not make anything up in your answer.

Text:
{input_text}

Output:"""
```

### Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `input_text` | string | Community data (entities, relationships, claims) in CSV format | Required |
| `max_report_length` | integer | Maximum report length in words | 1500 |

### Output Format (JSON)

```json
{
    "title": "Community Name with Key Entities",
    "summary": "Executive summary of the community",
    "rating": 7.5,
    "rating_explanation": "Single sentence explaining the impact rating",
    "findings": [
        {
            "summary": "Key insight title",
            "explanation": "Multiple paragraphs with detailed analysis and data citations [Data: Entities (1, 2, 3); Relationships (5, 7)]"
        }
    ]
}
```

### Impact Severity Rating Guidelines

| Rating | Description |
|--------|-------------|
| 0-2 | Minimal impact - routine, low-significance community |
| 3-4 | Low impact - limited influence or scope |
| 5-6 | Moderate impact - notable but contained influence |
| 7-8 | High impact - significant influence or potential concerns |
| 9-10 | Critical impact - major influence, critical infrastructure, or serious concerns |

### Sample Community Report

**Input Data**:
```
Entities:
1,ALICE CHEN,Chief AI Scientist at NeuroTech with 15 years ML experience
2,NEUROTECH,Leading AI research company developing neural interfaces
3,BRAINLINK PROJECT,Flagship neural interface project at NeuroTech

Relationships:
10,ALICE CHEN,NEUROTECH,Alice Chen leads the AI research team at NeuroTech
11,ALICE CHEN,BRAINLINK PROJECT,Alice Chen is the principal investigator for BrainLink
12,NEUROTECH,BRAINLINK PROJECT,NeuroTech is funding and developing the BrainLink Project

Claims:
5,BRAINLINK PROJECT,FDA,REGULATORY_APPROVAL,SUSPECTED,2024-01-15,2024-06-30,BrainLink Project is seeking FDA approval for human trials
```

**Output**:
```json
{
    "title": "NeuroTech and BrainLink Neural Interface Development",
    "summary": "This community centers around NeuroTech, a leading AI research company, and its flagship BrainLink Project focused on neural interfaces. Alice Chen serves as the Chief AI Scientist and principal investigator. The community is pursuing regulatory approval for human trials, indicating advancement toward commercialization.",
    "rating": 7.5,
    "rating_explanation": "The high impact rating reflects the potentially transformative nature of neural interface technology and the regulatory scrutiny it attracts.",
    "findings": [
        {
            "summary": "NeuroTech's leadership in neural interface research",
            "explanation": "NeuroTech is positioned as a leading AI research company specializing in neural interface development [Data: Entities (2)]. The company is funding and developing the BrainLink Project, its flagship initiative in this space [Data: Relationships (12)]. This investment demonstrates NeuroTech's commitment to advancing brain-computer interface technology, a field with significant implications for healthcare and human augmentation."
        },
        {
            "summary": "Alice Chen's critical role in AI research leadership",
            "explanation": "Alice Chen serves dual critical roles within this community as both Chief AI Scientist at NeuroTech and principal investigator for the BrainLink Project [Data: Entities (1); Relationships (10, 11)]. Her 15 years of machine learning experience positions her as a key technical leader. The concentration of leadership responsibility in one individual creates both a center of excellence and a potential single point of dependency for the project's success."
        },
        {
            "summary": "Regulatory pathway for BrainLink human trials",
            "explanation": "The BrainLink Project is currently seeking FDA approval for human trials, with activities spanning from January to June 2024 [Data: Claims (5)]. This regulatory pursuit indicates the project has progressed beyond preliminary research stages toward clinical validation. FDA approval for neural interface devices requires rigorous safety and efficacy demonstrations, suggesting substantial technical maturity. However, the 'SUSPECTED' status indicates this approval is not yet confirmed, representing a critical uncertainty for the project timeline."
        }
    ]
}
```

---

## Description Summarization

**Location**: `packages/graphrag/graphrag/prompts/index/summarize_descriptions.py`

**Purpose**: Consolidates multiple descriptions of the same entity into a single, coherent summary.

**Key Features**:
- Resolves contradictory information
- Maintains third-person voice
- Includes entity names in summary
- Configurable length limit

### Summarization Prompt

```python
SUMMARIZE_PROMPT = """
You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or more entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include the entity name.

If the provided descriptions have contradictory information, please resolve the contradictions and provide
a single, coherent summary.

Make sure it is written in third person, and include the entity names so we the have full context.

Limit the summary to {max_length} words.

#######
-Data-
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""
```

### Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `entity_name` | string | Name(s) of the entity/entities being summarized | Required |
| `description_list` | string | Newline-separated list of descriptions to merge | Required |
| `max_length` | integer | Maximum summary length in words | 500 |

### Sample Summarization

**Input**:
```
Entity: MICROSOFT
Description List:
- Technology company based in Redmond, Washington
- Develops software products including Windows and Office
- Founded by Bill Gates and Paul Allen in 1975
- Major cloud computing provider through Azure platform
- Acquired GitHub in 2018 for $7.5 billion
- Acquired LinkedIn in 2016
- One of the world's most valuable companies
```

**Output**:
```
Microsoft is a technology company headquartered in Redmond, Washington, founded in 1975 by
Bill Gates and Paul Allen. The company develops major software products including the Windows
operating system and Office productivity suite. Microsoft has expanded into cloud computing
through its Azure platform and has made significant acquisitions including LinkedIn in 2016
and GitHub in 2018 for $7.5 billion. Microsoft is one of the world's most valuable companies.
```

### Handling Contradictions

**Input with Contradictions**:
```
Entity: TECHCORP
Description List:
- TechCorp was founded in 2010 by John Smith
- The company was established in 2012 by Jane Doe
- TechCorp employs 500 people
- TechCorp has 300 employees
- Based in San Francisco
- Headquartered in New York
```

**Output**:
```
TechCorp is a technology company. There are conflicting accounts of its founding, with sources
indicating establishment between 2010-2012 by either John Smith or Jane Doe. The company employs
between 300-500 people according to different reports. Location information varies between
San Francisco and New York.
```

Note: The summarizer acknowledges contradictions when they cannot be definitively resolved.

---

## Common Index Prompt Features

### Delimiters
- **Field separator**: `<|>` (within entities/relationships/claims)
- **Record separator**: `##` (between entities/relationships/claims)
- **Completion signal**: `<|COMPLETE|>`

### Parsing Format
All index prompts use structured, parseable output formats to enable automated processing:

```
("record_type"<|>field1<|>field2<|>field3<|>field4)
##
("record_type"<|>field1<|>field2<|>field3<|>field4)
<|COMPLETE|>
```

### Language Support
All prompts can be parameterized with language specifications:
- Default: English
- Supports any language via `{language}` parameter
- Output format remains English-compatible (JSON, delimiters)

### Few-Shot Learning
Complex extraction prompts (graph extraction, claim extraction) include 2-3 detailed examples to improve accuracy and consistency.

## Configuration

Index prompts can be customized in `settings.yaml`:

```yaml
entity_extraction:
  prompt: null  # Use built-in prompt
  entity_types: ["organization", "person", "geo", "event"]
  max_gleanings: 1  # Number of follow-up extraction rounds

claim_extraction:
  enabled: true
  prompt: null
  description: "Any claims or facts that could be relevant"
  max_gleanings: 1

community_reports:
  prompt: null
  max_length: 1500

summarize_descriptions:
  prompt: null
  max_length: 500
```

## Pipeline Integration

Index prompts are used in the following pipeline stages:

1. **Text Chunking** → Text divided into processable chunks
2. **Graph Extraction** → Entities and relationships extracted
3. **Entity Resolution** → Duplicate entities merged
4. **Description Summarization** → Entity descriptions consolidated
5. **Claim Extraction** → Claims extracted (if enabled)
6. **Community Detection** → Graph clustering algorithms applied
7. **Community Reports** → Summaries generated for each community
8. **Embedding Generation** → Vector embeddings created for search

## References

- Source files: `packages/graphrag/graphrag/prompts/index/`
- Configuration: `packages/graphrag/graphrag/config/models/`
- Pipeline code: `packages/graphrag/graphrag/index/`
