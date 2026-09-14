"""Versions commands — the repository's version descriptor and the phases it yields (no LLM)."""

import json
import sys
from pathlib import Path

from rich.markup import escape

from .base import BaseCommand
from ..versioning import (
    DescriptorError,
    GitError,
    detect,
    find_descriptor,
    load_timeline,
    render_draft,
    write_draft,
)
from ..versioning.descriptor import LOCATIONS

NO_DESCRIPTOR_HINT = (
    "No version descriptor found. Looked for "
    + ", ".join(f"{name} {where}" for name, _, where in LOCATIONS)
    + ".\nRun `gitview versions detect` to draft one."
)


class _VersionsCommand(BaseCommand):
    def validate(self) -> None:
        pass

    def project_dir(self) -> Path:
        path = Path(self.get_option("repo", ".")).resolve()
        if not path.is_dir():
            self.print_error(f"Error: {path} is not a directory")
            sys.exit(1)
        return path

    def timeline(self):
        project = self.project_dir()
        try:
            if find_descriptor(project) is None:
                self.print_error(escape(NO_DESCRIPTOR_HINT))
                sys.exit(1)
            return load_timeline(project, self.get_option("branch", "HEAD"))
        except (DescriptorError, GitError) as exc:
            self.print_error(escape(f"Error: {exc}"))
            sys.exit(1)

    def print_problems(self, problems, limit=None) -> None:
        shown = problems if limit is None else problems[:limit]
        for p in shown:
            style = 'red' if p.severity == 'error' else 'yellow'
            where = ' '.join(x for x in (p.commit[:8], p.source) if x)
            where = f" [dim]({escape(where)})[/dim]" if where else ''
            self.console.print(f"[{style}]{p.severity}:[/{style}] {escape(p.message)}{where}")
        if limit is not None and len(problems) > limit:
            self.console.print(f"[dim]… {len(problems) - limit} more; run `gitview versions check`[/dim]")


class VersionsListCommand(_VersionsCommand):
    """List the phases the descriptor yields on a branch."""

    def execute(self):
        tl = self.timeline()
        if self.get_option("json_output", False):
            print(json.dumps(tl.to_dict(), indent=2))
            return tl

        d = tl.descriptor
        self.console.print("[dim]" + escape(f"Descriptor: {d.origin}{' in ' + d.scope if d.scope else ''} — "
                                             f"{', '.join(s.describe() for s in d.sources)}") + "[/dim]")
        if d.description:
            self.console.print(f"[dim]{escape(d.description)}[/dim]")
        table = self.create_table()
        for column, justify in (("Phase", "left"), ("Level", "left"), ("From", "left"), ("To", "left"),
                                ("Commits", "right"), ("Steps", "right"), ("State", "left"), ("Version", "left")):
            table.add_column(column, justify=justify, no_wrap=column in ('Phase', 'From', 'To', 'Version'))
        for p in tl.phases:
            table.add_row(p.id, p.level, p.start_timestamp[:10], p.end_timestamp[:10], str(p.commits),
                          str(len(p.steps)), 'sealed' if p.sealed else 'open',
                          p.event.label if p.event else '')
        self.console.print(table)
        if tl.problems:
            self.console.print(f"{len(tl.errors)} error(s), {len(tl.warnings)} warning(s):")
            self.print_problems(tl.problems, limit=5)
        return tl


class VersionsCheckCommand(_VersionsCommand):
    """Validate the descriptor against the whole branch history; exit 1 on errors."""

    def execute(self):
        tl = self.timeline()
        if self.get_option("json_output", False):
            print(json.dumps({'ok': not tl.errors, **tl.to_dict()}, indent=2))
        else:
            sealed = sum(1 for p in tl.phases if p.sealed)
            self.console.print(f"{tl.descriptor.origin}: {len(tl.descriptor.sources)} source(s), "
                               f"{len(tl.events)} version change(s), {len(tl.phases)} phase(s) "
                               f"({sealed} sealed) on {tl.branch}")
            self.print_problems(tl.problems)
            if tl.errors:
                self.print_error(f"Check failed: {len(tl.errors)} error(s)")
            else:
                self.print_success("Check passed" + (f" with {len(tl.warnings)} warning(s)" if tl.warnings else ''))
        if tl.errors:
            sys.exit(1)
        return tl


class VersionsDetectCommand(_VersionsCommand):
    """Draft a descriptor from the repository and print it (or append it with --write)."""

    def execute(self):
        project = self.project_dir()
        try:
            detection = detect(project, self.get_option("branch", "HEAD"))
            existing = find_descriptor(project)
        except (DescriptorError, GitError) as exc:
            self.print_error(escape(f"Error: {exc}"))
            sys.exit(1)

        if not detection.candidates:
            self.print_error("No version source found: no version-shaped tags, version files, "
                             "project-file versions or numbered migrations.")
            for note in detection.notes:
                self.console.print(f"[dim]{escape(note)}[/dim]")
            sys.exit(1)

        draft = render_draft(detection)
        if not self.get_option("write", False):
            print(draft, end='')
            if existing is not None:
                self.print_warning(f"{existing.origin} already has a descriptor; this draft was not written.")
            return detection

        if existing is not None:
            self.print_error(f"Error: {existing.origin} already has a descriptor; not writing.")
            sys.exit(1)
        path = write_draft(detection)
        self.print_success(f"Wrote draft descriptor to {path}")
        self.console.print("Replace every role = \"?\" (see the suggestions and evidence), "
                           "then run `gitview versions check`.")
        return detection
