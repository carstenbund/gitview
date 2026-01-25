This project is about analyzing git projects. The gitview/cli.py orchestrates a series of actions: 

 - 1. retrieval of the git in question,
 - 2. read and chunk it up in different phases (based on preference)
 - 3. send each phase to a LLM for evaluation
 - 4. summarize the phases
 - 5. create a report
  

# use GitView Commit Message Format
```
<type>(<scope>): <subject>

<body>

<footer>

Storyline: [status:category] Initiative Title
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

---

Quick Reference:
Status: new, continued, completed, stalled
Category: feature, refactor, bugfix, debt, infrastructure, docs, test
Key Rules:
Keep storyline titles consistent across commits
One initiative per commit
Mark completions explicitly
Group related changes together
