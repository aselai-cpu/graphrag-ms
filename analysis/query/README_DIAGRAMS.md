# GraphRAG Query Search Diagrams

Visual documentation of all four GraphRAG search methods.

## Diagram Files (✅ All Valid & Rendering)

| Diagram | File | Size | Status |
|---------|------|------|--------|
| **Local Search** | `local_search_activity.puml` | 9.3K | ✅ Valid |
| **Global Search** | `global_search_activity.puml` | 12K | ✅ Valid |
| **DRIFT Search** | `drift_search_activity.puml` | 9.5K | ✅ Valid |
| **Basic Search** | `basic_search_activity.puml` | 5.2K | ✅ Valid |
| Query Sequence | `query_sequence.puml` | 7.1K | ✅ Valid |

## PNG Files (Pre-rendered)

High-resolution PNG files have been generated for immediate viewing:
- `Local Search Activity Diagram.png` (285K)
- `Global Search Activity Diagram.png` (326K)
- `DRIFT Search Activity Diagram.png` (300K)
- `Basic Search Activity Diagram.png` (368K)

## Viewing the Diagrams

### Option 1: View PNG Files (Easiest)
The PNG files are ready to view in any image viewer or directly in VS Code.

### Option 2: VS Code PlantUML Plugin
1. Install "PlantUML" extension (jebbs.plantuml)
2. Open any `.puml` file
3. Press `Alt+D` (Windows/Linux) or `Option+D` (Mac)

**Note**: All syntax issues have been fixed. Diagrams should render correctly.

### Option 3: Command Line
```bash
cd analysis/query

# Generate PNG (default)
plantuml local_search_activity.puml

# Generate SVG (scalable)
plantuml -tsvg local_search_activity.puml

# Generate all at once
plantuml *_search_activity.puml
```

### Option 4: PlantUML Online
1. Go to https://www.plantuml.com/plantuml/uml/
2. Copy/paste diagram content
3. Click "Submit"

## Companion Guide Documents

Each diagram has a comprehensive guide document:

| Search Method | Guide Document | Lines |
|---------------|----------------|-------|
| Local Search | `local_search_diagram_guide.md` | 519 |
| Global Search | `global_search_diagram_guide.md` | 855 |
| DRIFT Search | `drift_search_diagram_guide.md` | 1025 |
| Basic Search | `basic_search_diagram_guide.md` | 671 |
| **Claims/Covariates** | `claims_covariates_guide.md` | Complete |

**Special Topic Guides**:
- `claims_covariates_guide.md` - Comprehensive guide on how claims (covariates) work in GraphRAG
  - What claims are and when to use them
  - How to enable/configure claim extraction
  - How claims are used in Local and DRIFT search
  - Examples and best practices

Each guide includes:
- Detailed phase-by-phase breakdown
- Configuration parameters
- Performance characteristics (speed/cost/quality)
- Use case recommendations
- Optimization tips
- Troubleshooting guidance
- Code references

## Quick Comparison

| Method | Speed | Cost | LLM Calls | Best For |
|--------|-------|------|-----------|----------|
| **Basic** | ⚡⚡⚡⚡⚡ | $ | 1 | Simple factoid questions |
| **Local** | ⚡⚡⚡⚡ | $$ | 1 | Entity relationships |
| **Global** | ⚡⚡ | $$$ | 10+ | Broad themes & summaries |
| **DRIFT** | ⚡ | $$$$ | 17+ | Complex multi-faceted analysis |

## Fixes Applied

### Issues Found and Fixed:
1. ✅ Removed `#Pink:` colored activity syntax (unsupported)
2. ✅ Removed `note right of` / `note left of` positioning (caused errors)
3. ✅ Validated all diagrams with PlantUML
4. ✅ Generated PNG files for immediate viewing
5. ✅ Cleaned up duplicate and backup files

### What Was Removed:
- Positioned notes (caused rendering errors)
- Summary notes at diagram edges
- Color highlighting on specific activities

### What Remains:
- Complete activity flow diagrams
- Swim lanes with color coding
- Inline notes on activities
- Partitions and forks
- Decision points and loops
- All essential flow information

## File Structure

```
analysis/query/
├── basic_search_activity.puml          # Diagram source
├── Basic Search Activity Diagram.png   # Pre-rendered image
├── basic_search_diagram_guide.md       # Comprehensive guide
├── drift_search_activity.puml
├── DRIFT Search Activity Diagram.png
├── drift_search_diagram_guide.md
├── global_search_activity.puml
├── Global Search Activity Diagram.png
├── global_search_diagram_guide.md
├── local_search_activity.puml
├── Local Search Activity Diagram.png
├── local_search_diagram_guide.md
├── query_sequence.puml                 # Pre-existing sequence diagram
├── query_analysis.md                   # Overall query analysis
├── claims_covariates_guide.md          # Claims/covariates documentation
└── README_DIAGRAMS.md                  # This file
```

---

**Last Updated**: 2026-01-30
**Status**: ✅ All diagrams validated and rendering correctly
**PlantUML Version**: Tested with PlantUML & Graphviz 14.1.1
