"""Derive version events and phases for a branch from a descriptor."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from .descriptor import find_descriptor, git_root
from .models import (
    ROLES,
    SEALING_ROLES,
    UNRESOLVED,
    UNVERSIONED_PHASE_ID,
    Descriptor,
    DescriptorError,
    Problem,
    SourceSpec,
    VersionEvent,
    VersionPhase,
    VersionTimeline,
)
from .sources import BranchHistory, Observation, observe_source, render_label

Key = Tuple[Optional[int], Optional[int], Optional[int]]


def effective_roles(source: SourceSpec) -> Dict[str, str]:
    """Field name → role, using ``suggested`` for unresolved roles so a draft can be previewed."""
    roles = {}
    for f in source.fields:
        role = f.role if f.role != UNRESOLVED else f.suggested
        if role in ROLES:
            roles[f.name] = role
    return roles


def _key(values: Dict[str, Optional[int]], roles: Dict[str, str]) -> Key:
    by_role = {role: values.get(name) for name, role in roles.items() if role != 'label'}
    return by_role.get('generation'), by_role.get('boundary'), by_role.get('step')


def load_timeline(project_dir: Union[str, Path], branch: str = 'HEAD',
                  descriptor: Optional[Descriptor] = None) -> VersionTimeline:
    """Load the descriptor in ``project_dir`` (unless given) and replay it over ``branch``."""
    project_dir = Path(project_dir).resolve()
    if descriptor is None:
        descriptor = find_descriptor(project_dir)
        if descriptor is None:
            raise DescriptorError(f"no version descriptor in {project_dir}")
    history = BranchHistory(git_root(project_dir), branch)
    return build_timeline(history, descriptor)


def build_timeline(history: BranchHistory, descriptor: Descriptor) -> VersionTimeline:
    problems: List[Problem] = []
    for source, f in descriptor.unresolved_fields():
        hint = f" (suggested: {f.suggested})" if f.suggested else ''
        problems.append(Problem('error', f"field {f.name} has no role yet{hint}", source=source.describe()))

    roles_per_source = [effective_roles(s) for s in descriptor.sources]
    for source, roles in zip(descriptor.sources, roles_per_source):
        for role in ('generation', 'boundary', 'step'):
            names = [n for n, r in roles.items() if r == role]
            if len(names) > 1:
                problems.append(Problem('error', f"fields {', '.join(names)} share the {role} role",
                                        source=source.describe()))
    if not any(r in SEALING_ROLES for roles in roles_per_source for r in roles.values()):
        problems.append(Problem('error', "no field has a generation or boundary role, so nothing seals a phase"))

    per_source: List[List[Observation]] = []
    for source in descriptor.sources:
        observations, source_problems = observe_source(history, source, descriptor.scope)
        problems.extend(source_problems)
        if not observations:
            problems.append(Problem('warning', "source produced no versions on this branch",
                                    source=source.describe()))
        per_source.append(observations)

    events = _merge(history, descriptor, per_source, roles_per_source, problems)
    phases = _phases(history, events) if history.chain else []
    if history.chain and not events:
        problems.append(Problem('error', f"no versions found on {history.branch}"))
    return VersionTimeline(descriptor=descriptor, branch=history.branch,
                           tip=history.chain[-1] if history.chain else '',
                           events=events, phases=phases, problems=problems)


def _merge(history: BranchHistory, descriptor: Descriptor, per_source: List[List[Observation]],
           roles_per_source: List[Dict[str, str]], problems: List[Problem]) -> List[VersionEvent]:
    """Walk the chain; the first source with a version in effect decides the version."""
    by_position: Dict[int, List[Tuple[int, Observation]]] = {}
    for s, observations in enumerate(per_source):
        for obs in observations:
            by_position.setdefault(obs.position, []).append((s, obs))

    current: List[Optional[Observation]] = [None] * len(per_source)
    events: List[VersionEvent] = []
    previous: Optional[Tuple[Key, str]] = None
    for position in sorted(by_position):
        for s, obs in by_position[position]:
            current[s] = obs
        sha = history.chain[position]
        _check_agreement(descriptor, current, roles_per_source, by_position[position], sha, problems)

        s = next(i for i, obs in enumerate(current) if obs is not None)
        obs, source, roles = current[s], descriptor.sources[s], roles_per_source[s]
        key = _key(obs.values, roles)
        label = render_label(source, obs.values, obs.raw)
        if previous is not None and (key, label) == previous:
            continue
        level = _level(previous[0] if previous else None, key)
        if previous is not None:
            _check_order(previous[0], key, sha, source, problems)
        events.append(VersionEvent(
            commit=sha, timestamp=history.timestamps[position], source=source.describe(),
            raw=obs.raw, label=label, fields={k: v for k, v in obs.values.items() if v is not None},
            level=level, key=key, message=obs.message))
        previous = (key, label)
    return events


def _level(previous: Optional[Key], key: Key) -> str:
    names = ('generation', 'boundary', 'step')
    if previous is None:
        return next((names[i] for i in range(3) if key[i] is not None), 'label')
    for i in range(3):
        if key[i] != previous[i]:
            return names[i]
    return 'label'


def _check_order(previous: Key, key: Key, sha: str, source: SourceSpec, problems: List[Problem]) -> None:
    names = ('generation', 'boundary', 'step')
    for i in range(3):
        a, b = previous[i], key[i]
        if a is None or b is None or a == b:
            continue
        if b < a:
            problems.append(Problem('error', f"{names[i]} went backwards ({a} → {b}) "
                                    f"without a higher component increasing",
                                    commit=sha, source=source.describe()))
        return


def _check_agreement(descriptor: Descriptor, current: List[Optional[Observation]],
                     roles_per_source: List[Dict[str, str]], arrived: List[Tuple[int, Observation]],
                     sha: str, problems: List[Problem]) -> None:
    """Sources in effect at a commit where one of them changed must describe the same version."""
    active = [(i, obs) for i, obs in enumerate(current) if obs is not None]
    arrived_ids = {i for i, _ in arrived}
    for n, (i, a) in enumerate(active):
        for j, b in active[n + 1:]:
            if i not in arrived_ids and j not in arrived_ids:
                continue
            ka, kb = _key(a.values, roles_per_source[i]), _key(b.values, roles_per_source[j])
            if any(x is not None and y is not None and x != y for x, y in zip(ka, kb)):
                problems.append(Problem(
                    'error', f"{descriptor.sources[i].describe()} says {a.raw} but "
                             f"{descriptor.sources[j].describe()} says {b.raw}", commit=sha))


def phase_id(key: Key) -> str:
    return '.'.join(str(v) for v in key[:2] if v is not None) or str(key[2])


def _phases(history: BranchHistory, events: List[VersionEvent]) -> List[VersionPhase]:
    sealing = [e for e in events if e.level in SEALING_ROLES]
    if not sealing:
        return []
    positions = [history.index[e.commit] for e in sealing]
    last = len(history.chain) - 1
    phases: List[VersionPhase] = []

    def make(pid, level, start, end, event, sealed):
        prev_end = history.chain[start - 1] if start > 0 else None
        steps = [e for e in events if e.level == 'step' and start <= history.index[e.commit] <= end]
        phases.append(VersionPhase(
            id=pid, level=level, start=history.chain[start], end=history.chain[end],
            start_timestamp=history.timestamps[start], end_timestamp=history.timestamps[end],
            commits=history.count_range(prev_end, history.chain[end]),
            sealed=sealed, event=event, steps=steps))

    if positions[0] > 0:
        make(UNVERSIONED_PHASE_ID, 'unversioned', 0, positions[0] - 1, None, True)
    for n, (event, start) in enumerate(zip(sealing, positions)):
        sealed = n + 1 < len(sealing)
        end = positions[n + 1] - 1 if sealed else last
        make(phase_id(event.key), event.level, start, end, event, sealed)
    return phases
