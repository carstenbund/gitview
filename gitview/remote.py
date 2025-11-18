"""Remote git repository handling with cloning and cleanup."""

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass
from urllib.parse import urlparse

from rich.console import Console

console = Console()


@dataclass
class RepoInfo:
    """Information about a git repository."""
    url: str  # Full git URL for cloning
    host: str  # e.g., "github.com"
    org: str   # Organization/user
    repo: str  # Repository name

    @property
    def full_path(self) -> str:
        """Full path component for organizing outputs."""
        return f"{self.host}/{self.org}/{self.repo}"

    @property
    def short_name(self) -> str:
        """Short name for display."""
        return f"{self.org}/{self.repo}"


class RemoteRepoHandler:
    """Handle remote git repository operations."""

    def __init__(self, repo_spec: str):
        """Initialize with a repository specification.

        Args:
            repo_spec: Can be:
                - Local path: /path/to/repo or ./repo
                - GitHub shortcut: org/repo or user/repo
                - Full HTTPS URL: https://github.com/org/repo.git
                - SSH URL: git@github.com:org/repo.git
        """
        self.repo_spec = repo_spec
        self.is_local = self._is_local_path(repo_spec)
        self.temp_dir: Optional[Path] = None

        if not self.is_local:
            self.repo_info = self._parse_remote_url(repo_spec)
        else:
            self.repo_info = None

    def _is_local_path(self, spec: str) -> bool:
        """Check if the spec is a local path."""
        # Check if it's an existing directory
        if Path(spec).exists():
            return True

        # Check if it looks like a path (contains / or \ or is . or ..)
        if spec in ['.', '..']:
            return True

        # If it starts with / or ./ or ../ or ~/, it's a path
        if spec.startswith(('/', './', '../', '~/')):
            return True

        # If it contains a colon NOT followed by // (to distinguish from URLs)
        # and not matching git@host: pattern, it might be a Windows path
        if ':' in spec and '://' not in spec and not spec.startswith('git@'):
            return True

        return False

    def _parse_remote_url(self, spec: str) -> RepoInfo:
        """Parse a remote repository URL or shortcut.

        Supports:
        - org/repo -> https://github.com/org/repo.git
        - https://github.com/org/repo.git
        - git@github.com:org/repo.git
        - https://gitlab.com/org/repo
        """
        # GitHub shortcut: org/repo
        if '/' in spec and not any(x in spec for x in ['://', '@', '\\']):
            parts = spec.split('/')
            if len(parts) == 2:
                org, repo = parts
                repo = repo.rstrip('.git')
                return RepoInfo(
                    url=f"https://github.com/{org}/{repo}.git",
                    host="github.com",
                    org=org,
                    repo=repo
                )

        # SSH URL: git@host:org/repo.git
        ssh_match = re.match(r'git@([^:]+):([^/]+)/(.+?)(?:\.git)?$', spec)
        if ssh_match:
            host, org, repo = ssh_match.groups()
            return RepoInfo(
                url=f"git@{host}:{org}/{repo}.git",
                host=host,
                org=org,
                repo=repo
            )

        # HTTPS/HTTP URL: https://host/org/repo.git
        if '://' in spec:
            parsed = urlparse(spec)
            host = parsed.netloc
            path = parsed.path.strip('/')
            if path.endswith('.git'):
                path = path[:-4]  # Remove .git suffix
            path_parts = path.split('/')

            if len(path_parts) >= 2:
                org = path_parts[-2]
                repo = path_parts[-1]

                # Normalize URL to include .git
                clean_path = '/'.join(path_parts)
                url = f"{parsed.scheme}://{host}/{clean_path}"
                if not url.endswith('.git'):
                    url += '.git'

                return RepoInfo(
                    url=url,
                    host=host,
                    org=org,
                    repo=repo
                )

        # If we can't parse it, try to use it as-is and extract what we can
        raise ValueError(f"Could not parse repository specification: {spec}\n"
                        f"Supported formats:\n"
                        f"  - GitHub shortcut: org/repo\n"
                        f"  - HTTPS URL: https://github.com/org/repo.git\n"
                        f"  - SSH URL: git@github.com:org/repo.git")

    def clone(self, progress_callback=None) -> Path:
        """Clone remote repository to temporary directory.

        Args:
            progress_callback: Optional function to call with progress updates

        Returns:
            Path to cloned repository
        """
        if self.is_local:
            raise ValueError("Cannot clone local repository")

        # Create temp directory
        self.temp_dir = Path(tempfile.mkdtemp(prefix='gitview_'))
        target_path = self.temp_dir / self.repo_info.repo

        console.print(f"[cyan]Cloning remote repository:[/cyan] {self.repo_info.url}")
        console.print(f"[dim]Target: {target_path}[/dim]")

        try:
            # Clone the repository
            cmd = ['git', 'clone', '--progress', self.repo_info.url, str(target_path)]

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Git writes progress to stderr
            for line in process.stderr:
                if progress_callback:
                    progress_callback(line.strip())

            process.wait()

            if process.returncode != 0:
                raise subprocess.CalledProcessError(
                    process.returncode, cmd,
                    output=process.stdout.read() if process.stdout else None,
                    stderr=process.stderr.read() if process.stderr else None
                )

            # Check repository size
            size_mb = self._get_directory_size(target_path)

            if size_mb > 1000:  # > 1GB
                console.print(f"[yellow]Warning: Large repository ({size_mb:.1f} MB)[/yellow]")
                console.print("[yellow]Analysis may take a while...[/yellow]")
            elif size_mb > 100:  # > 100MB
                console.print(f"[yellow]Note: Repository size is {size_mb:.1f} MB[/yellow]")

            console.print(f"[green]Clone complete[/green] ({size_mb:.1f} MB)\n")

            return target_path

        except subprocess.CalledProcessError as e:
            self.cleanup()
            console.print(f"[red]Error cloning repository:[/red] {e}")
            if e.stderr:
                console.print(f"[red]{e.stderr}[/red]")
            raise
        except Exception as e:
            self.cleanup()
            raise

    def _get_directory_size(self, path: Path) -> float:
        """Get directory size in MB."""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
        except Exception:
            return 0.0

        return total_size / (1024 * 1024)  # Convert to MB

    def cleanup(self):
        """Remove temporary clone directory."""
        if self.temp_dir and self.temp_dir.exists():
            try:
                console.print(f"[dim]Cleaning up temporary clone: {self.temp_dir}[/dim]")
                shutil.rmtree(self.temp_dir)
                self.temp_dir = None
            except Exception as e:
                console.print(f"[yellow]Warning: Failed to cleanup {self.temp_dir}: {e}[/yellow]")

    def get_local_path(self) -> Path:
        """Get the local path to work with.

        For local repos, returns the resolved path.
        For remote repos, returns None (must clone first).
        """
        if self.is_local:
            return Path(self.repo_spec).resolve()
        return None

    def get_default_output_path(self, base_dir: str = "~/Documents/gitview") -> Path:
        """Get the default output path for this repository.

        Args:
            base_dir: Base directory for all gitview outputs

        Returns:
            Path like ~/Documents/gitview/github.com/org/repo
        """
        base = Path(base_dir).expanduser()

        if self.is_local:
            # For local repos, use the directory name
            local_path = self.get_local_path()
            return base / "local" / local_path.name
        else:
            # For remote repos, use full path with host
            return base / self.repo_info.full_path


def resolve_repo_spec(repo_spec: str) -> RemoteRepoHandler:
    """Resolve a repository specification to a handler.

    Args:
        repo_spec: Repository specification (path, URL, or shortcut)

    Returns:
        RemoteRepoHandler instance
    """
    return RemoteRepoHandler(repo_spec)
