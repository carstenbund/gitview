# GitHub GraphQL API Investigation for GitView

## Executive Summary

Adopting GitHub's GraphQL API for GitView could provide significant advantages for private repository access and collaboration insights, but comes with meaningful implementation costs and limitations. This investigation covers the technical feasibility, performance implications, and architectural considerations.

---

## Current Architecture vs. GraphQL Approach

### Current Architecture
- **Data Source**: Local git repository analysis via GitPython
- **Scope**: Commit history, diffs, file changes, code metrics
- **Authentication**: System-level git config (SSH keys or credentials)
- **GitHub Integration**: Minimal—only uses standard `git clone` operations
- **Limitations**:
  - Cannot access collaboration metadata (PRs, reviews, discussions)
  - Private repo access depends on SSH key configuration
  - No insights into team workflow or review processes

### GraphQL Approach
- **Data Source**: GitHub's GraphQL API
- **Scope**: Git history + collaboration data (PRs, reviews, discussions, insights)
- **Authentication**: GitHub Personal Access Token (PAT)
- **GitHub Integration**: Direct API queries for all data
- **Advantages**: Richer context, fine-grained access control, unified authentication

---

## Advantages of Using GitHub GraphQL API

### 1. **Private Repository Access with Fine-Grained Control** ✓

**Current Limitation**: Private repo access relies entirely on system SSH configuration.

**GraphQL Solution**: GitHub Personal Access Tokens (PATs) can be:
- **Scoped to specific repositories** (even single repos in an organization)
- **Limited to read-only access** via fine-grained permissions
- **Time-limited** with configurable expiration dates
- **Revocable without affecting other tools** (no SSH key management concerns)

**Implementation**:
```python
# Personal Access Token with `repo` scope
headers = {
    "Authorization": f"Bearer {github_token}",
    "Content-Type": "application/json"
}
```

**Benefits for Multi-User Scenarios**:
- Each user can provide their own token scoped to their repositories
- No shared SSH key management
- Audit trail of token usage
- Revoke access instantly without SSH reconfiguration

### 2. **Access to Collaboration & Workflow Metadata**

Git history alone cannot capture:

| Data Type | Git History | GraphQL API |
|-----------|-------------|------------|
| Commits | ✓ | ✓ |
| Branches | ✓ | ✓ |
| Diffs | ✓ | ✓ |
| **Pull Requests** | ✗ | ✓ |
| **Code Reviews** | ✗ | ✓ |
| **Review Threads** | ✗ | ✓ |
| **Discussions** | ✗ | ✓ |
| **Issue Tracking** | ✗ | ✓ |
| **Assignees/Reviewers** | ✗ | ✓ |
| **Timing Metrics** | Time to review, merge cycle time, reviewer assignment | ✗ |

**Example: PR-Enhanced Story Context**
```python
# Could supplement commit narrative with:
# - Which PR merged each commit
# - Who reviewed it and for how long
# - Any blocking reviews or discussions
# - Integration with related issues
```

### 3. **Superior Performance for Complex Queries**

**REST API Approach** (fetching 2,100 repos):
- 50 repos in 30 seconds = ~0.6 seconds per repo
- Requires ~42 sequential requests (with pagination)

**GraphQL Approach** (same 2,100 repos):
- 2,100 repos in ~8 seconds
- Single optimized query with batching

**Rate Limit Efficiency**:
- **REST API**: Counted per request (5,000 requests/hour)
  - Each API call = 1 request unit
  - Getting full commit history across multiple endpoints = many requests

- **GraphQL API**: Counted per query complexity in "points" (5,000 points/hour)
  - Each query costs 1-10+ points depending on complexity
  - Single complex query replaces 5-10 REST calls
  - Result: **More data per rate limit unit**

**Example Calculation**:
```
REST API (fetching repo with 100 commits + PRs):
- GET /repos/{owner}/{repo} = 1 request
- GET /repos/{owner}/{repo}/commits = 1 request (paginated)
- GET /repos/{owner}/{repo}/pulls = 1 request (paginated)
- Total = 3+ requests, all hit rate limit equally

GraphQL (same data in single query):
- 1 complex query = 3-5 points
- Net result: Fetch more data with fewer rate limit units used
```

### 4. **Incremental Enhancement Without Breaking Changes**

GraphQL data can be **layered on top** of existing git analysis:
- Start with local git history (current approach)
- Optionally use GraphQL for additional context when token is available
- Graceful fallback if API fails
- No need to migrate all data sources at once

**Example Hybrid Approach**:
```python
# Phase 1: Use git history as primary source
commits = extract_from_local_repo()

# Phase 2: Enhance with GraphQL if available
if github_token:
    pr_data = fetch_pr_context_from_graphql()
    commits = enrich_with_pr_metadata(commits, pr_data)

# Falls back gracefully if no token or API unavailable
```

---

## Costs & Challenges

### 1. **Fine-Grained PATs Have Limited GraphQL Support** ⚠️

**Current Issue** (as of 2024-2025):
- Fine-grained PATs were designed primarily for REST API
- GraphQL support for fine-grained PATs is **incomplete and problematic**
- Known errors: `Resource not accessible by personal access token`
- Missing permission scopes for certain GraphQL queries
- Workarounds require broader permissions than intended

**Example Problem**:
```python
# Fine-grained PAT with "repo" read-only + "contents" read-only
# Still fails with 403 errors on some GraphQL queries
error: "Resource not accessible by personal access token"
```

**Practical Solution**:
- Must use **classic Personal Access Tokens** (broader scopes) for reliable GraphQL
- These have less granular access control but work reliably
- For private repos: Need `repo` scope (full control of private repos)

**Implication**: The advertised "fine-grained access control" benefit is **partially unrealized for GraphQL usage**. Fine-grained tokens work better with REST API.

### 2. **New Authentication Requirement**

**Current**: SSH key management (system-level)
**GraphQL**: Requires GitHub token storage and management

**Security Considerations**:
- Token must be stored securely (environment variable, secure config, etc.)
- If token is compromised, attacker has API access
- If SSH key is compromised, attacker can clone repos
- **Token has broader API access** than SSH (can query more than just git data)

**User Experience Impact**:
- Users must generate tokens explicitly
- Token management workflow (generation, storage, rotation)
- vs. current: Just use existing SSH setup

### 3. **Rate Limiting Complexity**

**REST API**: Simple—5,000 requests/hour
```
If you make 5,001 requests, you're rate limited. Easy to calculate.
```

**GraphQL**: Points-based calculation
```
Simple query: 1 point
Complex query with deep nesting: 10+ points
Need to understand query cost before executing
Some operations have minimum costs (e.g., pagination setup)
```

**Secondary Rate Limits**:
- 100 concurrent requests (shared with REST)
- 2,000 points per minute
- 90 seconds CPU time per 60 seconds real time
- 80 content-generating requests per minute
- **NEW (2025)**: Request timeouts now count against rate limit

**Practical Impact**:
- Must implement query cost estimation
- Need to monitor rate limit points (not just requests)
- Risk of hitting limits unexpectedly with complex queries
- More sophisticated rate limiting strategy needed

### 4. **Code Complexity Increase**

**Current Implementation** (~3,968 lines):
- GitPython: Simple, well-known library
- Clear data flow: Git → Analysis → Story
- Minimal external dependencies

**GraphQL Integration**:
- New dependency: GraphQL library (e.g., `gql`, `sgqlc`)
- New module: GraphQL query builder
- Error handling: API errors, timeouts, rate limits
- Testing: Mocking GraphQL responses
- Documentation: New authentication flow

**Estimated Additional Code**:
- GraphQL query builder: ~150-200 lines
- API client wrapper: ~200-300 lines
- Error handling & rate limiting: ~150-200 lines
- Authentication & token management: ~100-150 lines
- Tests: ~200-300 lines
- **Total**: ~800-1,150 additional lines

**Complexity Trade-offs**:
- Git extraction becomes optional
- New dependency chain
- Increased maintenance burden
- More testing scenarios needed

### 5. **API Dependency & Availability**

**Current**: Works offline after initial clone
**GraphQL**: Requires live API access

**Risks**:
- GitHub API downtime affects analysis
- Network latency affects performance
- Cannot analyze in offline/disconnected scenarios
- Rate limits can stop analysis mid-run

**Mitigation Strategies**:
- Cache GraphQL results
- Implement retry logic
- Add fallback to pure git analysis
- Rate limit monitoring before queries

### 6. **Data Model Changes**

GitView currently models history as:
```python
CommitRecord: {
    hash, timestamp, author, message,
    insertions, deletions, language_breakdown,
    readme_state, comments, large_change_flag
}
```

GraphQL would add:
```python
EnhancedCommit: {
    ...CommitRecord,
    pull_request_id, pr_title, pr_labels,
    reviewers: [Reviewer],
    review_comments: [Comment],
    merged_by: Author,
    review_duration: timedelta,
    discussion_threads: [Thread]
}
```

**Migration Effort**:
- Update data structures
- Modify extractor to populate new fields
- Update chunker/summarizer logic
- Adjust narrative generation (LLM prompts)
- Backward compatibility for existing cached data

### 7. **Learning Curve & Debugging**

GraphQL debugging is different from REST:
- No simple URL testing in browser (need GraphQL explorer)
- Query language is different from REST concepts
- Error messages can be cryptic
- Rate limit errors appear differently

---

## Technical Feasibility Assessment

### Compatibility Matrix

| Feature | Current Git | GraphQL | Notes |
|---------|------------|---------|-------|
| Public repos | ✓ | ✓ | Both work well |
| Private repos (SSH) | ✓ | - | Git only, no API |
| Private repos (token) | - | ✓ | GraphQL only |
| Commit history | ✓ | ✓ | Full compatibility |
| Diffs | ✓ | Limited | GraphQL has limited diff support |
| PR context | - | ✓ | GraphQL only |
| Review metadata | - | ✓ | GraphQL only |
| Offline analysis | ✓ | - | API requires connectivity |
| Fine-grained access | - | Partial | Limitations documented |

### Recommended Hybrid Implementation

**Phase 1**: Enhanced private repo support with standard PAT
- Add optional `--github-token` parameter
- Use GraphQL only for explicit context enrichment
- Maintain pure git analysis as fallback

**Phase 2**: PR/Review integration (if valuable)
- Integrate PR timeline with commit history
- Add review metrics to narrative
- Requires significant narrative logic changes

**Phase 3**: Advanced insights
- Team collaboration metrics
- Review efficiency analysis
- Workflow bottleneck detection

---

## Rate Limiting: Deep Dive

### Current gitview Usage Pattern

Analyzing a single repository:
```python
1. Clone repo (git, not API) = no rate limit
2. Analyze commits = local analysis = no rate limit
3. Generate summary with LLM = LLM API rate limit (not GitHub)
4. Write output = local I/O = no rate limit

Total GitHub API calls: 0
```

### With GraphQL for Enrichment

Analyzing single repository with PR context:
```python
1. Query repo metadata = 1 point
2. Query commits (paginated) = ~2-3 points per 100 commits
3. Query PRs with reviews = 5-10 points
4. Query discussions = 2-3 points

Total for single repo: 10-20 points
Rate limit: 5,000 points/hour = ~250-500 repos before hitting limit
```

**Practical Impact**:
- Batch analysis of 10 repos: ~100-200 points = no problem
- Large-scale analysis of 1,000+ repos: Would need to implement:
  - Query batching
  - Rate limit monitoring
  - Progressive/incremental processing
  - Caching strategy

---

## Decision Matrix: When to Use GraphQL

### ✓ Good Use Cases for GraphQL

1. **Private Repository Analysis**
   - User provides personal token
   - Don't have access via SSH
   - Want audit trail and time-limited access

2. **Team Workflow Analysis**
   - Understanding review processes
   - Measuring PR velocity
   - Identifying bottlenecks
   - Integration with team metrics

3. **Enhanced Narratives**
   - Contextualize commits with PRs
   - Mention reviewers who provided feedback
   - Highlight blocked/long-review scenarios
   - Cross-reference with discussions/issues

4. **Small-to-Medium Scale**
   - Analyzing 10-100 repos
   - Rate limits not a concern
   - User provides explicit tokens

### ✗ Poor Use Cases for GraphQL

1. **Bulk Analysis at Scale**
   - 1,000+ repos
   - Rate limiting becomes problematic
   - Cost (in API calls) becomes significant

2. **Offline/Disconnected Analysis**
   - Air-gapped environments
   - Low-bandwidth situations
   - Requires always-on connectivity

3. **Public Repository Analysis**
   - Current git-based approach is simpler
   - No authentication complications
   - Better offline support

4. **One-Off Analysis**
   - User doesn't want to manage tokens
   - Prefer simplicity over feature richness

---

## Cost-Benefit Summary

### Implementation Costs

| Item | Effort | Notes |
|------|--------|-------|
| GraphQL library integration | 1-2 days | gql, sgqlc, or similar |
| Query builder & client | 2-3 days | Reusable component |
| Authentication flow | 1 day | Token generation, validation |
| Rate limiting & caching | 2-3 days | Critical for reliability |
| Error handling | 1-2 days | API-specific error scenarios |
| Data model updates | 2-3 days | Schema changes, migrations |
| Testing & documentation | 2-3 days | Mocking, fixtures, docs |
| **Total Estimated** | **11-19 days** | For full implementation |

### Benefits Quantified

| Benefit | Value | Applicability |
|---------|-------|----------------|
| Private repo access diversity | High | Users without SSH setup |
| Fine-grained access control | Medium | Security-conscious orgs |
| PR/review context | High | Team-focused analysis |
| API efficiency | Medium | Large-scale analysis |
| Offline capability | Negative | Loss of offline mode |
| Simplicity | Negative | More complex codebase |

---

## Recommendations

### **Recommendation 1: Hybrid Approach (Low-Risk, High-Value)**

**Implementation**:
- Add optional `--github-token` parameter
- Use GraphQL ONLY when token provided
- Fall back to pure git analysis if no token
- Start with simple PR timeline enrichment

**Advantages**:
- Incremental implementation
- No breaking changes
- Users opt-in to new features
- Leverages existing git analysis

**Timeline**: 1-2 weeks for MVP

**Example Usage**:
```bash
# Current approach (still works)
gitview analyze org/repo

# Enhanced with PR context
gitview analyze org/repo --github-token $GITHUB_TOKEN
```

### **Recommendation 2: Use Standard PAT, Not Fine-Grained**

**Rationale**:
- Fine-grained PATs have documented GraphQL issues
- Standard PATs (classic) work reliably with GraphQL
- Fine-grained tokens better suited for REST API
- Simpler user experience

**Token Scopes**:
- For private repos: `repo` (full control, read-only works)
- For public repos: `public_repo`
- Recommendation: Use `repo` scope for flexibility

**Documentation**:
```markdown
## Generating a GitHub Token

1. Go to https://github.com/settings/tokens
2. Create "Personal access token (classic)"
3. Scopes: Select "repo" for private repos
4. Set expiration (e.g., 90 days)
5. Copy token and use: gitview analyze org/repo --github-token $TOKEN
```

### **Recommendation 3: Start with Single-Repo Analysis**

**Phase 1** (MVP):
- Support for 1-10 repos per run
- Simple PR timeline overlay
- Rate limiting: not a concern at this scale

**Phase 2** (Enhancement):
- Batch support for 10-100 repos
- Caching of API responses
- Rate limit monitoring

**Phase 3** (Advanced):
- Distributed analysis for 1,000+ repos
- Query optimization
- Rate limit queue management

### **Recommendation 4: Cache Aggressively**

GraphQL responses should be cached to:
- Reduce API calls and rate limit usage
- Improve performance for repeated analysis
- Support offline access for cached data

**Suggested Caching**:
```python
# Cache key: (owner, repo, branch, commit_hash)
# TTL: 24 hours (commits don't change)
# Size: Conservative (10-50MB per 100 repos)

cache_dir = ~/.gitview/cache/github/
```

---

## Open Questions & Risks

### Unresolved Questions

1. **GraphQL Query Performance**
   - What's the optimal query structure?
   - Should we batch requests or paginate?
   - How to handle very large repositories (100k+ commits)?

2. **Privacy & Security**
   - How to securely store/pass tokens?
   - Should tokens be in environment variables only?
   - Audit logging for token usage?

3. **Narrative Integration**
   - How to incorporate PR/review data into LLM prompts?
   - Will it improve narrative quality?
   - How much context is too much for LLM?

4. **Backwards Compatibility**
   - Should existing reports be regenerated if now optional GraphQL data?
   - How to handle cache invalidation?
   - Migration strategy for existing users?

### Key Risks

1. **GraphQL API Changes**
   - GitHub may change schema or rate limits
   - Requires adaptive error handling

2. **Token Leakage**
   - Tokens in environment/config could be exposed
   - Need secure token handling patterns

3. **Rate Limit Surprises**
   - Complex queries might cost more than expected
   - Secondary rate limits could block analysis mid-run

4. **User Friction**
   - Requires token generation step
   - Additional complexity in documentation
   - Potential confusion about when to use

---

## Conclusion

### Should GitView Adopt GraphQL?

**Short Answer**: **Yes, but incrementally and optionally.**

**Rationale**:
- ✓ Significant value for private repo access and workflow insights
- ✓ Can be implemented as optional feature (hybrid approach)
- ✓ No impact on existing git-based analysis
- ✗ Non-trivial implementation cost (~2-3 weeks)
- ✗ Fine-grained tokens not yet reliable for GraphQL
- ✗ Rate limiting complexity requires careful handling

### Recommended Path Forward

1. **Immediate** (Week 1-2):
   - Prototype GraphQL client with standard PAT
   - Test PR timeline enrichment
   - Validate rate limit behavior

2. **Short-term** (Month 1):
   - Implement hybrid approach with `--github-token` flag
   - Add PR context to narratives
   - Document authentication & usage

3. **Medium-term** (Month 2-3):
   - Add review/discussion insights
   - Implement caching layer
   - Batch processing for multiple repos

4. **Long-term** (Future):
   - Fine-grained token support (when GitHub fixes issues)
   - Advanced team analytics
   - Integration with CI/CD insights

---

## References

### GitHub GraphQL API Documentation
- [Rate Limits and Query Limits](https://docs.github.com/en/graphql/overview/rate-limits-and-query-limits-for-the-graphql-api)
- [Forming Calls with GraphQL](https://docs.github.com/en/graphql/guides/forming-calls-with-graphql)
- [GraphQL Improvements for Fine-Grained PATs](https://github.blog/changelog/2023-04-27-graphql-improvements-for-fine-grained-pats-and-github-apps/)

### Key Resources
- [GitHub GraphQL API vs REST API Comparison](https://docs.github.com/en/rest/about-the-rest-api/comparing-githubs-rest-api-and-graphql-api)
- [Fine-Grained Personal Access Tokens Blog Post](https://github.blog/security/application-security/introducing-fine-grained-personal-access-tokens-for-github/)
- [GraphQL Performance Benchmarks](https://www.stevemar.net/github-graphql-vs-rest/)

### Community Issues
- [Fine-Grained PAT GraphQL Support Issues](https://github.com/actions/add-to-project/issues/289)
- [GitHub CLI Fine-Grained Token Support](https://github.com/cli/cli/issues/6680)
