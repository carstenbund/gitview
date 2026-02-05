# Adaptive Review Agent Architecture

## Overview

The Adaptive Review Agent transforms GitView from a procedural pipeline into a discovery-driven analysis system. Instead of mechanically executing fixed steps, the agent observes, reasons, and adapts its workflow based on what it discovers during analysis.

## The Problem with Procedural Pipelines

The current pipeline executes sequentially:

```
Extract → Chunk → Summarize → Track Storylines → Generate Narrative → Write Output
```

**Limitations:**
- Every repository gets the same treatment regardless of content
- No feedback loops - can't revisit phases based on discoveries
- Can't adjust depth based on significance
- Discoveries in later steps can't influence earlier analysis
- No goal-oriented reasoning about what matters

## The OODA Loop Architecture

The Adaptive Review Agent implements an **OODA Loop** (Observe-Orient-Decide-Act):

```
    ┌─────────────────────────────────────────────────────────┐
    │                                                         │
    v                                                         │
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐     │
│ OBSERVE │───>│ ORIENT  │───>│ DECIDE  │───>│  ACT    │─────┘
└─────────┘    └─────────┘    └─────────┘    └─────────┘
    │              │              │              │
    │              │              │              │
    v              v              v              v
  Extract       Evaluate       Choose        Execute
  Patterns      Significance   Strategy      Analysis
  & Signals     & Risks        & Focus       & Record
```

### Observe Phase

Gather raw information:
- Extract commits and metadata
- Detect patterns (file clusters, commit patterns)
- Identify signals (large changes, refactors, security keywords)
- Collect metrics (LOC, churn, author diversity)

### Orient Phase

Evaluate what we've observed:
- **Significance scoring**: How important is this finding?
- **Risk assessment**: Security concerns, breaking changes
- **Goal alignment**: Does this relate to user's stated goals?
- **Novelty detection**: Is this unexpected or unusual?
- **Confidence calibration**: How certain are we?

### Decide Phase

Choose the next action based on evaluation:
- **Deepen**: Investigate further (drill into a phase, file, or pattern)
- **Broaden**: Expand scope (include related files, branches)
- **Pivot**: Shift focus to a discovery
- **Skip**: De-prioritize low-significance areas
- **Conclude**: Mark investigation complete
- **Clarify**: Flag for user attention (future enhancement)

### Act Phase

Execute the chosen action:
- Perform targeted analysis
- Record findings as discoveries
- Update internal state
- Loop back to Observe

## Discovery Model

Discoveries are first-class objects that capture findings:

```python
@dataclass
class Discovery:
    """A finding that may influence subsequent analysis."""
    id: str
    discovery_type: DiscoveryType  # PATTERN, ANOMALY, RISK, INSIGHT, QUESTION
    source: str                     # Which component found this
    phase_number: Optional[int]     # Where in the timeline
    confidence: float               # 0.0 - 1.0
    significance: float             # 0.0 - 1.0 (how important)
    title: str
    description: str
    evidence: Dict[str, Any]        # Supporting data
    implications: List[str]         # What this suggests
    suggested_actions: List[str]    # Recommended follow-ups
    timestamp: datetime
```

### Discovery Types

| Type | Description | Example |
|------|-------------|---------|
| `PATTERN` | Recurring behavior or structure | "Tests added after every feature commit" |
| `ANOMALY` | Unexpected deviation | "Sudden 80% LOC deletion in stable module" |
| `RISK` | Potential concern | "Security-related files modified without review" |
| `INSIGHT` | Meaningful observation | "Architecture shifted from monolith to services" |
| `QUESTION` | Needs clarification | "Purpose of 15 commits with no message unclear" |

## Decision Engine

The Decision Engine evaluates discoveries and chooses actions:

```python
class DecisionEngine:
    def evaluate(self, discoveries: List[Discovery], context: AnalysisContext) -> Decision:
        """Decide what to do next based on accumulated discoveries."""

        # Score each potential action
        actions = []

        # High-significance discoveries → deepen investigation
        high_sig = [d for d in discoveries if d.significance > 0.7]
        for discovery in high_sig:
            actions.append(DeepenAction(target=discovery, priority=discovery.significance))

        # Risk discoveries → immediate attention
        risks = [d for d in discoveries if d.discovery_type == DiscoveryType.RISK]
        for risk in risks:
            actions.append(InvestigateRiskAction(target=risk, priority=1.0))

        # Goal-relevant discoveries → prioritize
        if context.user_goals:
            relevant = self._filter_goal_relevant(discoveries, context.user_goals)
            for discovery in relevant:
                actions.append(ExploreGoalAction(target=discovery, priority=0.8))

        # Choose highest priority action, or conclude if none compelling
        return self._select_best_action(actions, context)
```

## Adaptive Behaviors

### 1. Significance-Based Depth

Spend more effort on significant phases:

```python
def determine_phase_depth(self, phase: Phase) -> AnalysisDepth:
    """Decide how deeply to analyze a phase."""
    signals = []

    # Large LOC changes suggest significant work
    if abs(phase.loc_delta_percent) > 30:
        signals.append(("loc_change", 0.8))

    # Security-related keywords
    if self._has_security_keywords(phase):
        signals.append(("security", 1.0))

    # Many authors = coordination, interesting
    if phase.unique_authors > 5:
        signals.append(("collaboration", 0.6))

    # Breaking changes
    if self._detects_breaking_changes(phase):
        signals.append(("breaking", 0.9))

    score = self._aggregate_signals(signals)

    if score > 0.8:
        return AnalysisDepth.DEEP      # Full hierarchical analysis
    elif score > 0.5:
        return AnalysisDepth.STANDARD  # Normal analysis
    else:
        return AnalysisDepth.LIGHT     # Quick summary
```

### 2. Discovery-Triggered Re-Analysis

When summarization reveals something unexpected, investigate further:

```python
async def summarize_with_discovery(self, phase: Phase) -> Tuple[str, List[Discovery]]:
    """Summarize phase and extract discoveries for potential re-analysis."""
    summary = await self.summarizer.summarize_phase(phase)

    discoveries = self.discovery_extractor.extract(summary, phase)

    # Check for triggers that warrant deeper investigation
    for discovery in discoveries:
        if discovery.discovery_type == DiscoveryType.ANOMALY:
            if discovery.significance > 0.7:
                # Trigger deeper file-level analysis
                yield DeepenAction(
                    target=phase,
                    reason=f"Anomaly detected: {discovery.title}",
                    depth=AnalysisDepth.DEEP
                )

        if discovery.discovery_type == DiscoveryType.RISK:
            # Always investigate risks
            yield InvestigateAction(
                target=discovery,
                reason=f"Risk requires investigation: {discovery.title}"
            )

    return summary, discoveries
```

### 3. Goal-Oriented Focus

When user provides goals, prioritize relevant discoveries:

```python
def filter_by_goals(self, discoveries: List[Discovery], goals: List[str]) -> List[Discovery]:
    """Prioritize discoveries that relate to user's stated goals."""
    goal_keywords = self._extract_keywords(goals)

    for discovery in discoveries:
        relevance = self._compute_relevance(discovery, goal_keywords)
        discovery.goal_relevance = relevance

        if relevance > 0.7:
            discovery.significance *= 1.5  # Boost significance
            discovery.suggested_actions.insert(0, "Prioritize: relates to stated goals")

    return sorted(discoveries, key=lambda d: d.significance * d.goal_relevance, reverse=True)
```

### 4. Adaptive Chunking

Adjust phase boundaries based on content analysis:

```python
def adaptive_rechunk(self, phases: List[Phase], discoveries: List[Discovery]) -> List[Phase]:
    """Re-chunk phases if discoveries suggest better boundaries."""

    # Find phases that should be split
    for discovery in discoveries:
        if discovery.discovery_type == DiscoveryType.ANOMALY:
            if "boundary" in discovery.title.lower():
                # Split this phase at the anomaly point
                phase = phases[discovery.phase_number]
                split_point = discovery.evidence.get("commit_index")
                if split_point:
                    new_phases = self._split_phase(phase, split_point)
                    phases = self._replace_phase(phases, phase, new_phases)

    # Find phases that should be merged
    for i, phase in enumerate(phases[:-1]):
        if phase.commit_count < 5 and phases[i+1].commit_count < 5:
            if self._are_thematically_similar(phase, phases[i+1]):
                merged = self._merge_phases(phase, phases[i+1])
                phases = self._replace_phases(phases, [phase, phases[i+1]], [merged])

    return phases
```

## Workflow Orchestration

The agent orchestrates analysis through a state machine:

```python
class AdaptiveReviewAgent:
    """Orchestrates adaptive git history analysis."""

    async def analyze(self, repo_path: str, config: AnalysisConfig) -> AnalysisResult:
        """Run adaptive analysis with discovery-driven workflow."""

        context = AnalysisContext(repo_path=repo_path, config=config)
        discoveries: List[Discovery] = []

        # Phase 1: Initial extraction and orientation
        context.state = AnalysisState.EXTRACTING
        records = await self._extract_history(context)
        discoveries.extend(self._observe_extraction_patterns(records))

        # Phase 2: Adaptive chunking
        context.state = AnalysisState.CHUNKING
        phases = await self._chunk_adaptively(records, context, discoveries)
        discoveries.extend(self._observe_phase_patterns(phases))

        # Phase 3: Iterative summarization with discovery
        context.state = AnalysisState.SUMMARIZING
        for phase in phases:
            depth = self.decision_engine.determine_depth(phase, discoveries, context)
            summary, new_discoveries = await self._summarize_adaptive(phase, depth)
            phase.summary = summary
            discoveries.extend(new_discoveries)

            # Check for re-analysis triggers
            actions = self.decision_engine.evaluate(new_discoveries, context)
            for action in actions:
                if isinstance(action, DeepenAction):
                    await self._deepen_analysis(action.target, context)
                elif isinstance(action, PivotAction):
                    context.focus = action.new_focus

        # Phase 4: Discovery synthesis
        context.state = AnalysisState.SYNTHESIZING
        synthesis = self._synthesize_discoveries(discoveries, context)

        # Phase 5: Narrative generation with discovery context
        context.state = AnalysisState.NARRATING
        narrative = await self._generate_narrative(phases, discoveries, synthesis, context)

        # Phase 6: Compile results
        return AnalysisResult(
            phases=phases,
            discoveries=discoveries,
            synthesis=synthesis,
            narrative=narrative,
            context=context
        )
```

## Discovery Persistence

Discoveries are persisted for incremental analysis:

```json
{
  "discoveries": [
    {
      "id": "disc_001",
      "type": "ANOMALY",
      "source": "phase_summarizer",
      "phase_number": 3,
      "confidence": 0.85,
      "significance": 0.9,
      "title": "Major architecture shift detected",
      "description": "Phase 3 shows migration from monolithic to microservices",
      "evidence": {
        "files_added": ["services/auth/", "services/api/"],
        "files_removed": ["src/monolith/"],
        "loc_delta": -15000
      },
      "implications": [
        "Team invested heavily in decoupling",
        "May have temporary performance impact",
        "Enables independent scaling"
      ],
      "suggested_actions": [
        "Investigate service boundaries",
        "Check for incomplete migrations"
      ]
    }
  ]
}
```

## Configuration

```yaml
# adaptive_review.yaml
agent:
  # Significance thresholds
  significance_threshold_high: 0.7
  significance_threshold_medium: 0.4

  # Maximum iterations to prevent infinite loops
  max_iterations: 10
  max_deepening_per_phase: 2

  # Discovery extraction
  extract_discoveries: true
  discovery_confidence_threshold: 0.5

  # Adaptive behaviors
  enable_adaptive_chunking: true
  enable_discovery_triggers: true
  enable_goal_focusing: true

  # Depth settings
  default_depth: standard
  force_deep_on_security: true
  force_deep_on_breaking: true
```

## Example: Adaptive vs Procedural

### Scenario: Repository with Security Incident

**Procedural Pipeline:**
1. Extracts all 500 commits ✓
2. Chunks into 10 phases ✓
3. Summarizes each phase equally ✓
4. Generates narrative ✓
5. **Misses**: The security fix in Phase 7 gets same treatment as routine updates

**Adaptive Agent:**
1. Extracts all 500 commits ✓
2. Chunks into 10 phases ✓
3. Starts summarizing Phase 1...
4. **Discovery in Phase 7**: "Emergency security patch" detected
5. **Decision**: Pivot - prioritize Phase 7 analysis
6. **Deep Analysis**:
   - Traces affected files
   - Identifies vulnerability timeline
   - Finds related commits in other phases
7. **Re-Orient**: Updates significance scores
8. **Narrative**: Includes dedicated security timeline section
9. **Result**: Comprehensive security incident analysis

## Implementation Plan

### Phase 1: Discovery Infrastructure (Week 1)
- [ ] Discovery model and types
- [ ] Discovery extractor (from summaries)
- [ ] Discovery persistence

### Phase 2: Decision Engine (Week 2)
- [ ] Decision model and actions
- [ ] Significance scoring
- [ ] Goal relevance computation

### Phase 3: Adaptive Behaviors (Week 3)
- [ ] Significance-based depth
- [ ] Discovery-triggered re-analysis
- [ ] Goal-oriented focus

### Phase 4: Agent Orchestration (Week 4)
- [ ] AdaptiveReviewAgent class
- [ ] State machine workflow
- [ ] Configuration system

### Phase 5: CLI Integration (Week 5)
- [ ] New `analyze --adaptive` flag
- [ ] Discovery reporting
- [ ] Interactive mode (optional)

## Success Metrics

| Metric | Procedural | Adaptive (Target) |
|--------|------------|-------------------|
| Security issues detected | ~60% | >90% |
| User goal alignment | N/A | >80% relevant |
| Analysis depth variance | 0 (uniform) | High (focused) |
| Re-analysis triggers | 0 | ~2-5 per repo |
| User satisfaction | Baseline | +40% |

## Future Enhancements

1. **Interactive Mode**: Pause and ask user for clarification
2. **Learning**: Adjust thresholds based on user feedback
3. **Multi-Repository**: Cross-repo pattern detection
4. **Team Insights**: Author collaboration analysis
5. **Predictive**: Forecast likely storyline outcomes
