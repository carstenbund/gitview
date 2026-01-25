# use GitView Commit Format
```
<type>(<scope>): <subject>

<body>

Storyline: [STATUS:CATEGORY] Initiative Title
```

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

