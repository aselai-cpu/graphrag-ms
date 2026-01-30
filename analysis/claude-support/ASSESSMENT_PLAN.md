# GraphRAG Claude Support Assessment Plan

**Date**: 2026-01-29
**Objective**: Assess feasibility and approach for adding Anthropic Claude as an LLM provider and SentenceTransformer as a local embedding option alongside OpenAI in GraphRAG
**Status**: Planning Phase

---

## Executive Summary

This assessment evaluates adding:
1. **Anthropic Claude** as an alternative LLM provider for text generation
2. **SentenceTransformer** as a local, open-source embedding option

GraphRAG currently uses OpenAI exclusively for both text generation and embeddings. The goal is to understand technical requirements, benefits, challenges, and implementation strategy for multi-provider LLM and embedding support.

---

## Current State (Baseline)

### How GraphRAG Currently Uses OpenAI

GraphRAG uses OpenAI's API for multiple LLM-powered operations:

1. **Entity Extraction** (Step 4)
   - Extract entities from text chunks
   - Model: GPT-4, GPT-4-turbo, or GPT-3.5-turbo
   - Prompt: Entity extraction template
   - Output: Structured entity list

2. **Relationship Extraction** (Step 4)
   - Extract relationships between entities
   - Model: GPT-4, GPT-4-turbo, or GPT-3.5-turbo
   - Prompt: Relationship extraction template
   - Output: Structured relationship list

3. **Covariate/Claims Extraction** (Step 6)
   - Extract temporal claims and facts
   - Model: GPT-4, GPT-4-turbo, or GPT-3.5-turbo
   - Prompt: Claims extraction template
   - Output: Structured claims list

4. **Community Report Generation** (Step 8)
   - Summarize communities
   - Model: GPT-4, GPT-4-turbo
   - Prompt: Summarization template
   - Output: Community report text

5. **Query Response Generation** (Query phase)
   - MAP phase: Generate answers per community
   - REDUCE phase: Aggregate answers
   - Model: GPT-4, GPT-4-turbo
   - Prompt: Query-specific templates

6. **Embeddings Generation** (Step 9)
   - Generate vector embeddings
   - Model: text-embedding-ada-002, text-embedding-3-small, text-embedding-3-large
   - Output: 1536 or 3072-dimensional vectors

### Current Architecture

```
GraphRAG Pipeline
    ↓
LLM Operations (OpenAI Only)
    ├── Text Generation (GPT-4/3.5)
    │   ├── Entity extraction
    │   ├── Relationship extraction
    │   ├── Claims extraction
    │   ├── Community reports
    │   └── Query responses
    └── Embeddings (Ada/text-embedding-3)
        ├── Entity embeddings
        ├── Community embeddings
        └── Text unit embeddings
```

**Location**: `packages/graphrag-llm/`

---

## Proposed Architecture (Multi-Provider Support)

### Vision: Multi-Provider LLM and Embedding Support

```
GraphRAG Pipeline
    ↓
┌─────────────────────────────────────────────────────────┐
│         LLM Abstraction Layer (NEW)                     │
└─────────────────────────────────────────────────────────┘
    ↓                                    ↓
Text Generation Providers      Embedding Providers
    │                                    │
    ├── OpenAI                          ├── OpenAI (API)
    │   ├── GPT-4o                      │   ├── text-embedding-ada-002
    │   ├── GPT-4-turbo                 │   ├── text-embedding-3-small
    │   └── GPT-3.5-turbo               │   └── text-embedding-3-large
    │                                   │
    ├── Claude (NEW)                    ├── Voyage AI (API)
    │   ├── Claude 3.5 Sonnet           │   ├── voyage-large-2
    │   ├── Claude 3 Opus               │   └── voyage-2
    │   ├── Claude 3 Sonnet             │
    │   └── Claude 3 Haiku              ├── Cohere (API)
    │                                   │   └── embed-english-v3.0
    └── Future: Gemini, etc.            │
                                        └── SentenceTransformer (LOCAL, NEW)
                                            ├── all-MiniLM-L6-v2 (384 dims)
                                            ├── all-mpnet-base-v2 (768 dims)
                                            ├── BGE-large-en-v1.5 (1024 dims)
                                            ├── E5-large-v2 (1024 dims)
                                            └── Custom models (user-provided)
```

**Key Innovation**: SentenceTransformer runs **locally** (no API, no cost, full privacy)

---

## Assessment Areas

### 1. Technical Feasibility

**Questions to Answer**:
- ✅ Can Claude handle all GraphRAG LLM operations?
- ✅ Does Claude support structured output (JSON mode)?
- ✅ What are Claude's token limits vs OpenAI?
- ✅ How do Claude's APIs differ from OpenAI?
- ✅ What about embeddings (Claude doesn't have native embeddings)?
- ✅ Are there feature gaps?

**Assessment Tasks**:
1. Map all LLM operations to Claude capabilities
2. Test Claude API with GraphRAG prompts
3. Compare JSON output quality
4. Evaluate context window sizes
5. Test prompt compatibility
6. Identify missing features
7. Test SentenceTransformer models for embedding quality
8. Benchmark local vs API embedding performance
9. Evaluate embedding dimension compatibility
10. Test GPU acceleration for local embeddings

---

### 2. Architecture Analysis

**Questions to Answer**:
- How to abstract LLM provider interface?
- Where does provider selection happen?
- How to handle provider-specific features?
- How to manage API credentials for multiple providers?
- How to handle embeddings (different providers)?

**Assessment Tasks**:
1. Design LLM provider abstraction layer
2. Define common interface for text generation
3. Define common interface for embeddings
4. Plan configuration schema
5. Design provider factory pattern
6. Handle provider-specific optimizations

---

### 3. Performance Implications

**Questions to Answer**:
- Is Claude faster or slower than OpenAI?
- What are latency differences?
- What are cost differences?
- What are rate limit differences?
- How does output quality compare?

**Assessment Tasks**:
1. Benchmark entity extraction (Claude vs OpenAI)
2. Benchmark relationship extraction
3. Benchmark community reports
4. Benchmark query responses
5. Compare output quality
6. Compare costs

---

### 4. Benefits Analysis

**Potential Benefits**:
- ✅ **Provider Choice**: Users can choose based on cost, quality, availability
- ✅ **Cost Optimization**: Claude may be cheaper for some operations
- ✅ **Quality Options**: Claude 3.5 Sonnet competitive with GPT-4
- ✅ **Resilience**: Fallback if one provider down
- ✅ **Regional Availability**: Different providers available in different regions
- ✅ **Vendor Diversity**: Not locked into single provider
- ✅ **Claude Strengths**: Longer context (200K), better instruction following

**Assessment Tasks**:
1. Quantify cost savings potential
2. Evaluate quality on GraphRAG tasks
3. Test fallback scenarios
4. Assess regional availability
5. Measure context window benefits

---

### 5. Trade-offs and Challenges

**Potential Challenges**:
- ❌ **Embeddings**: Claude has no native embeddings (need Voyage AI or similar)
- ❌ **API Differences**: Different API structure than OpenAI
- ❌ **Prompt Tuning**: May need different prompts for optimal results
- ❌ **Testing Complexity**: Must test with multiple providers
- ❌ **Configuration Complexity**: More options for users
- ❌ **Maintenance**: Support multiple API versions
- ❌ **Rate Limits**: Different limits per provider

**Assessment Tasks**:
1. Evaluate embeddings integration (Voyage AI, Cohere, etc.)
2. Test prompt compatibility
3. Estimate development effort
4. Design testing strategy
5. Plan configuration UI/UX
6. Design rate limit handling

---

### 6. Implementation Strategy

**Phases**:

**Phase 1: Foundation** (2-3 weeks)
- Design LLM provider abstraction
- Implement Claude provider
- Test basic operations
- Validate output quality

**Phase 2: Integration** (3-4 weeks)
- Integrate with indexing pipeline
- Add configuration options
- Implement embeddings alternatives
- Comprehensive testing

**Phase 3: Optimization** (2-3 weeks)
- Prompt optimization for Claude
- Performance tuning
- Cost optimization
- Documentation

**Phase 4: Rollout** (2-3 weeks)
- Beta release
- User feedback
- Stable release
- Make Claude a recommended option

**Assessment Tasks**:
1. Define detailed implementation plan
2. Identify code changes required
3. Estimate development time
4. Plan testing strategy
5. Design rollout approach

---

### 7. Use Case Validation

**Target Use Cases**:
1. **Cost-Sensitive Projects**: Use Claude 3 Haiku for extraction (cheaper)
2. **Quality-First Projects**: Use Claude 3.5 Sonnet for reports (high quality)
3. **Long Documents**: Use Claude's 200K context window
4. **Provider Redundancy**: Fallback to Claude if OpenAI down
5. **Regional Deployments**: Use provider available in region

**Assessment Tasks**:
1. Design use case patterns
2. Test each use case
3. Measure value
4. Document best practices

---

## Assessment Deliverables

### Documents to Create

1. **Current Architecture Analysis** (`01_current_llm_usage.md`)
   - All LLM operations in GraphRAG
   - OpenAI API usage patterns
   - Prompt engineering details
   - Token usage and costs

2. **Claude Capabilities Assessment** (`02_claude_capabilities.md`)
   - Feature comparison matrix
   - API differences
   - Model comparison (Claude vs GPT)
   - Embeddings alternatives

3. **Architecture Design** (`03_architecture_design.md`)
   - LLM provider abstraction
   - Configuration schema
   - Integration points
   - Code structure

4. **Performance Benchmarks** (`04_performance_benchmarks.md`)
   - Extraction quality comparison
   - Latency comparison
   - Cost comparison
   - Context window tests

5. **Benefits and Trade-offs** (`05_benefits_tradeoffs.md`)
   - Quantified benefits
   - Risk analysis
   - Cost analysis
   - Decision matrix

6. **Implementation Plan** (`06_implementation_plan.md`)
   - Phase breakdown
   - Task list with estimates
   - Resource requirements
   - Timeline

7. **Migration Strategy** (`07_adoption_strategy.md`)
   - User adoption path
   - Configuration guides
   - Best practices
   - Troubleshooting

8. **Proof of Concept Code** (`poc/`)
   - Claude provider implementation
   - Test scripts
   - Benchmark code
   - Example configurations

---

## Success Criteria

### Must Have
- ✅ Claude can perform all LLM operations (extraction, summarization, query)
- ✅ Output quality comparable to OpenAI
- ✅ Performance within 2x of OpenAI latency
- ✅ Backward compatibility (OpenAI remains default)
- ✅ Clear documentation for provider selection

### Should Have
- ✅ Cost savings for specific use cases
- ✅ Seamless provider switching
- ✅ Fallback mechanism if provider fails
- ✅ Embeddings integration (Voyage AI or similar)

### Nice to Have
- ✅ Output quality better than OpenAI for some tasks
- ✅ Automatic provider selection based on task
- ✅ Cost optimization recommendations
- ✅ Support for additional providers (Gemini, etc.)

---

## Timeline

### Week 1: Research & Analysis
- Study current LLM usage in GraphRAG
- Research Claude API and capabilities
- Test Claude with sample prompts
- Create architecture diagrams

**Deliverables**:
- `01_current_llm_usage.md`
- `02_claude_capabilities.md`

### Week 2: Design & Prototyping
- Design LLM provider abstraction
- Implement basic Claude provider
- Test entity extraction
- Test relationship extraction

**Deliverables**:
- `03_architecture_design.md`
- `poc/claude_provider.py`

### Week 3: Testing & Benchmarking
- Run comprehensive benchmarks
- Compare quality and performance
- Test embeddings integration
- Document findings

**Deliverables**:
- `04_performance_benchmarks.md`
- Benchmark data and scripts

### Week 4: Analysis & Decision
- Complete benefits/trade-offs analysis
- Create implementation plan
- Design adoption strategy
- Present findings

**Deliverables**:
- `05_benefits_tradeoffs.md`
- `06_implementation_plan.md`
- `07_adoption_strategy.md`
- Final recommendation

---

## Key Questions to Answer

### Technical Questions

1. **JSON Mode**: Does Claude support structured JSON output like OpenAI's JSON mode?
   - OpenAI: Native JSON mode
   - Claude: Structured output via prompting

2. **Context Window**: How do context limits compare?
   - OpenAI GPT-4-turbo: 128K tokens
   - Claude 3.5 Sonnet: 200K tokens
   - Claude 3 Opus: 200K tokens

3. **Function Calling**: Does Claude support tool/function calling?
   - OpenAI: Native function calling
   - Claude: Tool use feature (similar)

4. **Streaming**: Can Claude stream responses?
   - Both: Yes

5. **Embeddings**: How to handle embeddings?
   - OpenAI: Native embeddings API
   - Claude: Partner with Voyage AI, or use Cohere, or continue using OpenAI embeddings

### Business Questions

6. **Cost**: Which provider is more cost-effective?
   - GPT-4-turbo: $10/1M input tokens, $30/1M output tokens
   - Claude 3.5 Sonnet: $3/1M input tokens, $15/1M output tokens
   - Claude 3 Haiku: $0.25/1M input tokens, $1.25/1M output tokens

7. **Quality**: Which produces better results for GraphRAG tasks?
   - Need empirical testing

8. **Availability**: Which is more reliable/available?
   - Both have high uptime
   - Different rate limits

9. **Regional**: Which works in more regions?
   - OpenAI: Most regions
   - Claude: US, UK, EU (expanding)

### Strategic Questions

10. **Vendor Lock-in**: Does multi-provider reduce risk?
    - Yes, provides flexibility

11. **User Demand**: Do users want Claude support?
    - Need to survey community

12. **Maintenance**: Is multi-provider worth the complexity?
    - Need to evaluate effort

---

## Open Questions

### Embeddings Strategy

**Question**: Claude has no native embeddings. What's the strategy?

**Options**:

1. **Keep OpenAI Embeddings**: Use OpenAI for embeddings even with Claude for text
   - Pros: Works with existing code, high quality
   - Cons: Still depends on OpenAI, ongoing API costs

2. **Use Voyage AI**: Anthropic's recommended embeddings partner
   - Pros: Optimized for Claude, good quality
   - Cons: Another API to integrate, additional cost

3. **Use Cohere Embeddings**: Alternative embeddings provider
   - Pros: High quality, good support
   - Cons: Another provider, additional cost

4. **Use SentenceTransformer (LOCAL)**: Open-source, runs locally ⭐ **NEW**
   - Pros:
     - **Zero cost** (no API calls)
     - **Full privacy** (data never leaves local machine)
     - **No rate limits** (unlimited throughput)
     - **Offline capable** (no internet required)
     - **GPU acceleration** (CUDA support for speed)
     - **Many models**: all-MiniLM, BGE, E5, custom models
     - **Open source** (MIT/Apache licensed)
   - Cons:
     - Requires local compute (CPU/GPU)
     - May be slower than API without GPU
     - Quality varies by model selection
     - Requires managing model downloads

5. **Support Multiple Embedding Providers**: Abstract embeddings too
   - Pros: Maximum flexibility, users choose best option
   - Cons: Most complex to implement

**Recommendation**: Support all options (multi-provider abstraction), with SentenceTransformer as recommended default for cost-conscious users

**SentenceTransformer Models** (popular choices):
- `all-MiniLM-L6-v2`: 384 dims, fastest, good quality
- `all-mpnet-base-v2`: 768 dims, better quality, still fast
- `BAAI/bge-large-en-v1.5`: 1024 dims, SOTA quality
- `intfloat/e5-large-v2`: 1024 dims, excellent performance
- Custom: User can provide any HuggingFace model

---

### Prompt Compatibility

**Question**: Will existing OpenAI prompts work with Claude?

**Known Differences**:
- Claude uses `\n\nHuman:` and `\n\nAssistant:` format (legacy)
- Claude 3+ uses Messages API (similar to OpenAI)
- Claude prefers XML tags for structure
- Claude has different system message handling

**Strategy**:
1. Test existing prompts with Claude
2. Tune prompts if needed
3. Consider provider-specific prompt templates
4. Document best practices per provider

---

### Configuration Complexity

**Question**: How do users choose provider?

**Options**:

**Option 1: Simple (per-pipeline)**:
```yaml
llm:
  provider: claude  # or openai
  model: claude-3-5-sonnet-20241022
```

**Option 2: Granular (per-operation)**:
```yaml
llm:
  extraction:
    provider: claude
    model: claude-3-haiku-20240307  # Cheap for extraction
  summarization:
    provider: claude
    model: claude-3-5-sonnet-20241022  # Quality for reports
  embeddings:
    provider: openai
    model: text-embedding-3-small
```

**Option 3: Auto (recommendation engine)**:
```yaml
llm:
  strategy: cost-optimized  # or quality-first, balanced
  # GraphRAG selects best provider per operation
```

**Recommendation**: TBD after UX testing

---

## Risk Assessment

### High Priority Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **Output quality worse** | Medium | High | Thorough testing, prompt tuning |
| **Embeddings integration complex** | High | Medium | Use OpenAI embeddings initially |
| **Performance regression** | Low | High | Benchmark early and often |
| **User confusion** | Medium | Medium | Clear documentation, good defaults |

### Medium Priority Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **Maintenance burden** | Medium | Medium | Good abstraction, comprehensive tests |
| **API changes** | Medium | Medium | Version pinning, deprecation handling |
| **Cost optimization complex** | Medium | Low | Provide recommendations, simple defaults |

### Contingency Plans

**If Claude quality unacceptable**:
- Make Claude opt-in (not default)
- Continue improving prompts
- Document use cases where Claude works well

**If embeddings integration too complex**:
- Phase 1: Use OpenAI embeddings only
- Phase 2: Add Voyage AI later
- Keep embeddings separate from text generation

**If development too complex**:
- Start with simple provider switching
- Add advanced features (per-operation, auto-select) later
- Focus on core use cases first

---

## Resource Plan

### Team Composition

**Weeks 1-2** (Research & Design):
- 1 Backend Developer (full-time)
- 0.5 LLM Engineer (part-time, prompt testing)

**Weeks 3-6** (Implementation):
- 1 Backend Developer (full-time)
- 0.5 LLM Engineer (part-time, quality testing)

**Weeks 7-8** (Testing & Documentation):
- 0.5 Backend Developer (part-time)
- 0.25 Technical Writer (part-time)
- 0.25 QA Engineer (part-time)

**Total Effort**: ~8-10 person-weeks

### Infrastructure

**Development**:
- Claude API access (Anthropic)
- OpenAI API access (comparison)
- Test datasets
- Benchmark environment
- GPU instance for SentenceTransformer testing (optional)

**Testing**:
- Multiple API keys (rate limit testing)
- Cost tracking tools
- Quality evaluation framework
- Local compute for SentenceTransformer benchmarks
- Model download cache (~1-5 GB per model)

---

## Budget

**API Costs** (testing):
- Claude API: ~$200 (testing)
- OpenAI API: ~$100 (comparison)
- Voyage AI: ~$50 (embeddings testing)
- **Total**: ~$350

**Development** (internal):
- 8-10 person-weeks @ $10K/week
- **Total**: ~$80-100K

**Operational** (ongoing):
- No additional infrastructure costs
- Users pay for their own API usage

---

## Next Steps

### Immediate Actions (This Week)
1. ✅ Create assessment plan (this document)
2. ⏳ Analyze current LLM usage in GraphRAG
3. ⏳ Set up Claude API access
4. ⏳ Test Claude with sample GraphRAG prompts
5. ⏳ Set up SentenceTransformer environment
6. ⏳ Create comparison framework

### Week 2 Actions
1. ⏳ Design LLM provider abstraction
2. ⏳ Design embedding provider abstraction
3. ⏳ Implement basic Claude provider POC
4. ⏳ Implement SentenceTransformer provider POC
5. ⏳ Test entity extraction with Claude
6. ⏳ Test embeddings with SentenceTransformer models
7. ⏳ Document findings

### Week 3 Actions
1. ⏳ Run comprehensive benchmarks (text generation)
2. ⏳ Run comprehensive benchmarks (embeddings)
3. ⏳ Compare quality and performance
4. ⏳ Test GPU acceleration for local embeddings
5. ⏳ Evaluate different SentenceTransformer models
6. ⏳ Document results

### Week 4 Actions
1. ⏳ Complete benefits/trade-offs analysis
2. ⏳ Create implementation plan
3. ⏳ Design adoption strategy
4. ⏳ Present recommendation

---

## Success Indicators

### Assessment Success
- [ ] All LLM operations mapped to Claude capabilities
- [ ] Quality comparison complete with data
- [ ] Performance benchmarks collected
- [ ] Cost analysis quantified
- [ ] Clear GO/NO-GO recommendation

### Implementation Success (if GO)
- [ ] Claude provider working for all operations
- [ ] Backward compatibility maintained
- [ ] Performance acceptable
- [ ] Users successfully using Claude
- [ ] Documentation complete

---

## References

### Anthropic Documentation
- [Claude API](https://docs.anthropic.com/claude/reference/getting-started-with-the-api)
- [Claude Models](https://docs.anthropic.com/claude/docs/models-overview)
- [Prompt Engineering](https://docs.anthropic.com/claude/docs/prompt-engineering)
- [Tool Use](https://docs.anthropic.com/claude/docs/tool-use)

### SentenceTransformer Documentation
- [SentenceTransformers Library](https://www.sbert.net/)
- [Pretrained Models](https://www.sbert.net/docs/pretrained_models.html)
- [Model Hub](https://huggingface.co/models?library=sentence-transformers)
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) - Embedding benchmarks

### GraphRAG Codebase
- `packages/graphrag-llm/` - LLM operations
- `packages/graphrag/graphrag/prompts/` - Prompt templates
- `packages/graphrag/graphrag/index/operations/` - LLM-using operations

### Embedding Providers
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- [Voyage AI](https://www.voyageai.com/) - Anthropic partner
- [Cohere Embeddings](https://cohere.com/embeddings)

---

## Conclusion

This assessment will provide a comprehensive evaluation of adding:
1. **Claude support** for text generation (cost savings + quality options)
2. **SentenceTransformer support** for local embeddings (zero cost + full privacy)

The goal is to make an **informed, data-driven decision** on whether multi-provider LLM and embedding support is beneficial and feasible.

**Expected Outcome**: Clear recommendation (GO/NO-GO) with implementation plan, benefits analysis, and adoption strategy.

**Key Value Proposition**:
- **Cost**: Up to 90% savings (Claude Haiku + SentenceTransformer)
- **Privacy**: Local embeddings keep data on-premises
- **Flexibility**: Users choose best provider for their needs
- **Resilience**: Multiple providers reduce vendor lock-in

---

**Status**: ✅ Planning Complete - Ready to begin assessment
**Next Update**: Week 1 completion - Current LLM Usage Analysis & Provider Capabilities

