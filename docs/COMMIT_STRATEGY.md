# GitView Commit Strategy Guide

A guide for writing commits and PRs that maximize GitView's ability to extract meaningful storylines and track development progress.

## Overview

GitView uses multi-signal detection to identify and track "storylines" - narrative threads that span multiple phases of development. By following these conventions, you help GitView:

- Automatically detect and categorize development initiatives
- Track progress across commits and phases
- Generate accurate narratives about your project's evolution
- Provide confidence-scored storyline extraction

---

## Storyline Categories

GitView recognizes these development categories:

| Category | Description | Trigger Keywords |
|----------|-------------|------------------|
| `feature` | New functionality | add, implement, new, feature, create |
| `refactor` | Code restructuring | refactor, restructure, reorganize, cleanup, modularize |
| `bugfix` | Bug fixes | fix, bug, issue, error, crash, problem |
| `debt` | Technical debt | debt, technical debt, legacy, deprecate |
| `infrastructure` | Build/deploy systems | infra, ci/cd, build, deploy, pipeline, config |
| `docs` | Documentation | doc, documentation, readme, comment, wiki |
| `test` | Testing | test, spec, coverage, unit test |

---

## Storyline Statuses

Track initiative lifecycle with these statuses:

| Status | Meaning | When to Use |
|--------|---------|-------------|
| `new` | Starting a new initiative | First commit of a feature/effort |
| `continued` | Work in progress | Ongoing development |
| `completed` | Initiative finished | Final commit completing the work |
| `stalled` | Work paused/blocked | When stopping temporarily |

---

## Commit Message Conventions

### Basic Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### GitView-Optimized Format

For maximum storyline extraction, use this enhanced format:

```
<type>(<scope>): <subject>

<body>

Storyline: [STATUS:category] Initiative Title
```

### Examples

#### Starting a New Feature
```
feat(auth): implement OAuth2 login flow

Add OAuth2 authentication with Google and GitHub providers.
Includes token refresh logic and secure session handling.

Storyline: [new:feature] OAuth2 Authentication System
```

#### Continuing Work
```
feat(auth): add OAuth callback handlers

Implement callback endpoints for OAuth providers.
Handle authorization codes and exchange for tokens.

Storyline: [continued:feature] OAuth2 Authentication System
```

#### Completing an Initiative
```
feat(auth): finalize OAuth2 integration with tests

Add comprehensive test coverage and documentation.
OAuth2 system is now production-ready.

Storyline: [completed:feature] OAuth2 Authentication System
```

#### Bug Fix
```
fix(auth): resolve token expiry race condition

Token refresh was sometimes failing due to timing issues.
Added mutex lock around refresh operations.

Storyline: [continued:bugfix] Authentication Token Handling Issues
```

#### Refactoring
```
refactor(core): modularize CLI command structure

Extract commands to separate files for maintainability.
Reduces cli.py from 2800 to 400 lines.

Storyline: [new:refactor] CLI Modularization Initiative
```

---

## PR Title Patterns

GitView detects storylines from PR titles. Use these patterns:

### Feature PRs
```
feat: Add user authentication system
feature: Implement dark mode toggle
[Feature] Shopping cart functionality
```

### Fix PRs
```
fix: Resolve memory leak in cache handler
bugfix: Correct timezone handling
[Bug] Fix crash on empty input
```

### Refactor PRs
```
refactor: Modularize payment processing
cleanup: Remove deprecated API endpoints
[Refactor] Simplify database queries
```

### Infrastructure PRs
```
ci: Add automated deployment pipeline
infra: Migrate to Kubernetes
[CI/CD] Setup GitHub Actions workflow
```

---

## PR Labels for Signal Boosting

GitView assigns higher confidence to storylines detected from PR labels:

| Label | Confidence | Category Mapping |
|-------|------------|------------------|
| `feature` | 0.9 | feature |
| `enhancement` | 0.9 | feature |
| `bug` | 0.9 | bugfix |
| `fix` | 0.9 | bugfix |
| `refactor` | 0.9 | refactor |
| `tech-debt` | 0.9 | debt |
| `documentation` | 0.9 | docs |
| `infrastructure` | 0.9 | infrastructure |

**Recommendation**: Always label your PRs for better storyline tracking.

---

## File Clustering for Detection

GitView detects storylines by analyzing which files change together:

### Best Practices

1. **Cohesive commits**: Keep related changes together
   ```
   # Good: All auth-related files in one commit
   auth/oauth.py
   auth/providers.py
   auth/tokens.py
   tests/test_auth.py

   # Bad: Mixed concerns
   auth/oauth.py
   payments/stripe.py
   docs/readme.md
   ```

2. **Consistent file patterns**: Use consistent naming
   ```
   # Good: Clear module structure
   features/user_auth/
   features/payments/
   features/notifications/

   # Bad: Scattered files
   auth.py
   pay.py
   notify.py
   ```

3. **Test files with implementation**: Include tests
   ```
   # Signals a complete feature implementation
   src/feature.py
   tests/test_feature.py
   ```

---

## Phase Summary Markers

When GitView generates phase summaries, it looks for storyline sections. You can influence this by including structured comments in significant commits:

### Commit Body Format
```
Major changes in this commit:

## Storylines

- [new:feature] User Dashboard: Starting implementation of analytics dashboard
- [continued:refactor] API Cleanup: Continuing REST endpoint standardization
- [completed:bugfix] Memory Issues: Final fix for memory leak in cache layer
```

---

## Initiative Tracking Tips

### 1. Use Consistent Titles

Keep storyline titles consistent across commits:

```
# Good: Same title throughout
Storyline: [new:feature] Payment Gateway Integration
Storyline: [continued:feature] Payment Gateway Integration
Storyline: [completed:feature] Payment Gateway Integration

# Bad: Varying titles
Storyline: [new:feature] Add Payments
Storyline: [continued:feature] Payment System
Storyline: [completed:feature] Stripe Integration
```

### 2. Mark Phase Boundaries

Explicitly mark when initiatives transition:

```
# Starting
Storyline: [new:feature] Dark Mode Support

# Major milestone
Storyline: [continued:feature] Dark Mode Support - Theme engine complete

# Completion
Storyline: [completed:feature] Dark Mode Support
```

### 3. Link Related Work

Reference related initiatives in commit bodies:

```
feat(ui): add theme toggle component

Implements the UI toggle for dark/light mode switching.

Related to: Dark Mode Support initiative
Depends on: Theme Engine (completed in phase 3)

Storyline: [continued:feature] Dark Mode Support
```

---

## Confidence Scoring

GitView assigns confidence scores based on detection source:

| Source | Confidence | Notes |
|--------|------------|-------|
| PR Labels | 0.9 | Highest confidence |
| PR Title Patterns | 0.8 | Title keywords |
| File Clusters | 0.7 | Related file changes |
| Commit Messages | 0.6 | Message patterns |
| LLM Extraction | 0.5 | Natural language analysis |

**To maximize confidence**: Use PR labels AND structured commit messages.

---

## Example: Full Feature Lifecycle

### Phase 1: Initiation
```
feat(search): begin advanced search implementation

Starting work on full-text search with filters.

Storyline: [new:feature] Advanced Search System
```

### Phase 2: Development
```
feat(search): add search index builder

Implement Elasticsearch indexing for products.

Storyline: [continued:feature] Advanced Search System
```

```
feat(search): implement filter components

Add UI components for search filters.

Storyline: [continued:feature] Advanced Search System
```

### Phase 3: Testing
```
test(search): add search integration tests

Comprehensive test coverage for search functionality.

Storyline: [continued:feature] Advanced Search System
```

### Phase 4: Completion
```
feat(search): finalize search with documentation

Complete implementation with user docs and API reference.

Storyline: [completed:feature] Advanced Search System
```

---

## Quick Reference Card

```
COMMIT FORMAT:
<type>(<scope>): <subject>
Storyline: [STATUS:CATEGORY] Title

TYPES: feat, fix, refactor, docs, test, chore, ci
STATUS: new, continued, completed, stalled
CATEGORY: feature, refactor, bugfix, debt, infrastructure, docs, test

EXAMPLES:
feat(auth): add login endpoint
Storyline: [new:feature] User Authentication

fix(api): resolve timeout issue
Storyline: [continued:bugfix] API Performance Issues

refactor(db): optimize query patterns
Storyline: [completed:refactor] Database Optimization
```

---

## Integration with GitView

Run GitView to see your storylines:

```bash
# Analyze repository
gitview analyze

# List detected storylines
gitview storyline list

# Show specific storyline details
gitview storyline show <id>

# Generate storyline report
gitview storyline report
```

---

## Additional Resources

- [Conventional Commits](https://www.conventionalcommits.org/)
- [GitView Documentation](../README.md)
- [Storyline Tracker API](../gitview/storyline/README.md)
