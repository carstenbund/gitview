# GitView Project Guidelines

## Commit Message Format

Use this format for all commits to enable GitView storyline tracking:

```
<type>(<scope>): <subject>

<body>

Storyline: [STATUS:CATEGORY] Initiative Title
```

### Types
- `feat` - New feature
- `fix` - Bug fix
- `refactor` - Code restructuring
- `docs` - Documentation
- `test` - Tests
- `chore` - Maintenance
- `ci` - CI/CD changes

### Storyline Status
- `new` - Starting an initiative
- `continued` - Work in progress
- `completed` - Initiative finished
- `stalled` - Work paused

### Storyline Categories
- `feature` - New functionality
- `refactor` - Code restructuring
- `bugfix` - Bug fixes
- `debt` - Technical debt
- `infrastructure` - Build/deploy systems
- `docs` - Documentation
- `test` - Testing

### Examples

```bash
# Starting a new feature
feat(auth): implement OAuth2 login flow

Add OAuth2 authentication with Google and GitHub providers.

Storyline: [new:feature] OAuth2 Authentication System
```

```bash
# Continuing work
feat(auth): add OAuth callback handlers

Implement callback endpoints for OAuth providers.

Storyline: [continued:feature] OAuth2 Authentication System
```

```bash
# Completing an initiative
feat(auth): finalize OAuth2 with tests

Complete implementation with full test coverage.

Storyline: [completed:feature] OAuth2 Authentication System
```

```bash
# Bug fix
fix(api): resolve timeout on large requests

Storyline: [new:bugfix] API Performance Issues
```

```bash
# Refactoring
refactor(cli): modularize command structure

Extract commands to separate files.

Storyline: [new:refactor] CLI Modularization
```

### Key Rules

1. **Consistent titles**: Keep storyline titles identical across related commits
2. **One initiative per commit**: Focus each commit on a single storyline when possible
3. **Mark completions**: Always use `[completed:...]` when finishing an initiative
4. **Cohesive changes**: Group related file changes in the same commit
