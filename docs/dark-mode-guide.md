# Dark Mode Guide: Safe Neo4j Migration

This guide explains how to use GraphRAG's Dark Mode framework to safely migrate from NetworkX to Neo4j for graph operations.

## What is Dark Mode?

Dark Mode enables parallel execution of two graph backends:
- **Primary backend** (NetworkX): Serves production traffic, results returned to users
- **Shadow backend** (Neo4j): Runs for validation, failures don't affect primary

This allows you to validate Neo4j implementation in production without risk.

## Quick Start

### 1. Enable Dark Mode

Create or update your `settings.yaml`:

```yaml
dark_mode:
  enabled: true
  primary_backend: networkx      # Production backend
  shadow_backend: neo4j          # Validation backend
  log_path: output/dark_mode_metrics.jsonl

# Neo4j connection (required when shadow_backend: neo4j)
neo4j:
  uri: bolt://localhost:7687
  username: neo4j
  password: ${NEO4J_PASSWORD}
  database: graphrag
```

### 2. Run Your Pipeline

```bash
graphrag index --root . --config settings.yaml
```

Dark mode runs transparently - your pipeline works exactly as before, but both backends execute in parallel.

### 3. Analyze Metrics

After running your pipeline, analyze the metrics:

```bash
graphrag dark-mode analyze output/dark_mode_metrics.jsonl
```

Example output:

```
================================================================================
DARK MODE METRICS ANALYSIS
================================================================================

📊 OPERATIONS SUMMARY
  Total operations: 1250
    - load_graph: 250
    - detect_communities: 250
    - compute_node_degrees: 250
    - export_graph: 500

✅ CORRECTNESS METRICS
  Shadow errors: 5 (0.40%)
  Comparisons passed: 1200
  Comparisons failed: 45
  Pass rate: 96.39%

⚡ PERFORMANCE METRICS
  Avg primary duration: 125.3ms
  Avg shadow duration: 178.9ms
  Avg latency ratio: 1.43x
  P50 latency ratio: 1.39x
  P95 latency ratio: 1.89x
  P99 latency ratio: 2.15x

🚀 CUTOVER READINESS
  ✅ READY FOR CUTOVER
  All criteria met - shadow backend validated successfully!
```

### 4. Check Cutover Readiness

Quick check if ready for production cutover:

```bash
graphrag dark-mode check-cutover output/dark_mode_metrics.jsonl
```

Returns exit code 0 if ready, 1 if not ready.

## CLI Commands

### `graphrag dark-mode analyze`

Comprehensive analysis of dark mode execution:

```bash
# Basic analysis
graphrag dark-mode analyze metrics.jsonl

# Custom cutover criteria
graphrag dark-mode analyze metrics.jsonl \
  --min-operations 500 \
  --max-error-rate 0.02 \
  --min-pass-rate 0.90 \
  --max-latency-ratio 3.0

# Export to CSV for further analysis
graphrag dark-mode analyze metrics.jsonl --export-csv analysis.csv
```

**Options:**
- `--min-operations`: Minimum operations required (default: 1000)
- `--max-error-rate`: Maximum shadow error rate (default: 0.01 = 1%)
- `--min-pass-rate`: Minimum comparison pass rate (default: 0.95 = 95%)
- `--max-latency-ratio`: Maximum latency ratio (default: 2.0x)
- `--export-csv`: Export metrics to CSV file

### `graphrag dark-mode summary`

Quick summary without full analysis:

```bash
graphrag dark-mode summary metrics.jsonl
```

Output:
```
📊 Total operations: 1250
✅ Pass rate: 96.39%
⚠️  Error rate: 0.40%
⚡ Latency ratio: 1.43x

✅ Ready for cutover!
```

### `graphrag dark-mode check-cutover`

Programmatic cutover readiness check (useful for CI/CD):

```bash
# Basic check
graphrag dark-mode check-cutover metrics.jsonl
echo $?  # 0 if ready, 1 if not

# Verbose output
graphrag dark-mode check-cutover metrics.jsonl --verbose

# Custom criteria
graphrag dark-mode check-cutover metrics.jsonl \
  --min-operations 500 \
  --max-latency-ratio 3.0
```

## Configuration Reference

### Dark Mode Configuration

```yaml
dark_mode:
  # Enable/disable dark mode
  enabled: true

  # Primary backend (serves production)
  primary_backend: networkx

  # Shadow backend (validation only)
  shadow_backend: neo4j

  # Metrics log file path
  log_path: output/dark_mode_metrics.jsonl

  # Comparison thresholds
  comparison:
    entity_match_threshold: 0.99     # 99% entity F1 score
    community_match_threshold: 0.95  # 95% clustering similarity
    degree_tolerance: 0.1            # 10% degree difference

  # Cutover criteria
  cutover_criteria:
    min_operations: 1000        # Minimum operations to validate
    max_error_rate: 0.01        # Maximum 1% error rate
    min_pass_rate: 0.95         # Minimum 95% pass rate
    max_latency_ratio: 2.0      # Shadow can be 2x slower
```

## Metrics File Format

Metrics are written in JSON Lines format (one JSON object per line):

```json
{
  "operation": "detect_communities",
  "timestamp": "2026-01-31T10:15:30.123456",
  "primary_duration_ms": 125.3,
  "shadow_duration_ms": 178.9,
  "shadow_error": null,
  "comparison": {
    "operation": "compare_communities",
    "passed": true,
    "metrics": {
      "primary_level_count": 3,
      "shadow_level_count": 3,
      "common_levels": 3,
      "avg_similarity": 0.9875
    },
    "differences": []
  }
}
```

## Programmatic Usage

### Python API

```python
from graphrag.index.graph.dark_mode import MetricsAnalyzer

# Load and analyze metrics
analyzer = MetricsAnalyzer("output/dark_mode_metrics.jsonl")
analyzer.load_metrics()

analysis = analyzer.analyze(
    min_operations=1000,
    max_error_rate=0.01,
    min_pass_rate=0.95,
    max_latency_ratio=2.0,
)

# Check readiness
if analysis.ready_for_cutover:
    print("✅ Ready for cutover!")
    print(f"Pass rate: {analysis.comparison_pass_rate:.2%}")
    print(f"Latency: {analysis.avg_latency_ratio:.2f}x")
else:
    print("❌ Not ready:")
    for reason in analysis.blocking_reasons:
        print(f"  - {reason}")

# Export to CSV
analyzer.export_to_csv("analysis.csv")
```

### Factory API

```python
from graphrag.config.models.dark_mode_config import DarkModeConfig
from graphrag.index.graph import create_graph_backend_with_dark_mode

# Configure dark mode
config = DarkModeConfig(
    enabled=True,
    primary_backend="networkx",
    shadow_backend="neo4j",
    log_path="metrics.jsonl",
)

# Create backend (returns DarkModeOrchestrator)
backend = create_graph_backend_with_dark_mode(
    config,
    shadow_backend_kwargs={
        "uri": "bolt://localhost:7687",
        "username": "neo4j",
        "password": "password",
    }
)

# Use like any GraphBackend
backend.load_graph(entities, relationships)
communities = backend.detect_communities(max_cluster_size=10)

# Check cutover readiness
ready, reasons = backend.check_cutover_readiness()
```

## Cutover Process

### Phase 1: Validation (Dark Mode Enabled)

1. Enable dark mode with NetworkX primary, Neo4j shadow
2. Run your pipeline in production
3. Monitor metrics continuously
4. Analyze results with CLI tools

### Phase 2: Readiness Check

Check all cutover criteria:

```bash
graphrag dark-mode check-cutover metrics.jsonl --verbose
```

Criteria:
- ✅ Minimum operations executed (validate at scale)
- ✅ Low shadow error rate (< 1%)
- ✅ High comparison pass rate (> 95%)
- ✅ Acceptable latency ratio (< 2x)

### Phase 3: Cutover

Once ready, update configuration:

```yaml
dark_mode:
  enabled: false              # Disable dark mode
  primary_backend: neo4j      # Switch to Neo4j
  # shadow_backend removed - no longer needed
```

### Phase 4: Monitoring

After cutover, monitor Neo4j performance:
- Query latency
- Memory usage
- Error rates

## Troubleshooting

### High Error Rate

If shadow error rate is high:

1. Check shadow backend logs for errors
2. Verify Neo4j connection and resources
3. Check for data compatibility issues
4. Review comparison framework thresholds

### Low Pass Rate

If comparison pass rate is low:

1. Check `differences` in metrics log
2. Verify comparison thresholds are appropriate
3. Check for algorithmic differences (e.g., community detection randomness)
4. Investigate specific failing operations

### High Latency

If shadow latency is too high:

1. Check Neo4j server resources (CPU, memory)
2. Verify database indexes are created
3. Check network latency between app and Neo4j
4. Consider increasing `max_latency_ratio` if functionality is correct

## Best Practices

1. **Start with dev/staging** - Validate dark mode in non-production first
2. **Monitor continuously** - Check metrics regularly during dark mode
3. **Set realistic thresholds** - Don't require perfection, allow small differences
4. **Collect enough data** - Run at scale (1000+ operations) before cutover
5. **Document issues** - Track and resolve any comparison failures
6. **Plan rollback** - Have a plan to switch back to NetworkX if needed

## Next Steps

- See [Configuration Guide](configuration.md) for full config options
- See [Neo4j Setup Guide](neo4j-setup.md) for Neo4j installation
- See [API Reference](api-reference.md) for programmatic usage
