"""Base command class for GitView CLI commands."""

from abc import ABC, abstractmethod
from typing import Any, Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table


class BaseCommand(ABC):
    """Base class for all CLI commands.

    Provides common functionality for console output, progress tracking,
    and command execution lifecycle.
    """

    def __init__(self, **kwargs):
        """Initialize command with options.

        Args:
            **kwargs: Command options passed from CLI
        """
        self.options = kwargs
        self.console = Console()

    @abstractmethod
    def validate(self) -> None:
        """Validate command options.

        Raises:
            click.UsageError: If validation fails
        """
        pass

    @abstractmethod
    def execute(self) -> Any:
        """Execute the command logic.

        Returns:
            Command result (varies by command type)
        """
        pass

    def run(self) -> Any:
        """Template method: validate then execute.

        Returns:
            Result from execute()
        """
        self.validate()
        return self.execute()

    def print_header(self, title: str) -> None:
        """Print a styled header."""
        self.console.print(f"\n[bold blue]{title}[/bold blue]\n")

    def print_success(self, message: str) -> None:
        """Print a success message."""
        self.console.print(f"[green]{message}[/green]")

    def print_warning(self, message: str) -> None:
        """Print a warning message."""
        self.console.print(f"[yellow]{message}[/yellow]")

    def print_error(self, message: str) -> None:
        """Print an error message."""
        self.console.print(f"[red]{message}[/red]")

    def print_info(self, message: str) -> None:
        """Print an info message."""
        self.console.print(f"[cyan]{message}[/cyan]")

    def create_progress(self) -> Progress:
        """Create a progress context manager."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        )

    def create_table(self, title: Optional[str] = None) -> Table:
        """Create a Rich table with optional title."""
        return Table(title=title, show_header=True, header_style="bold cyan")

    def get_option(self, key: str, default: Any = None) -> Any:
        """Get a command option value."""
        return self.options.get(key, default)
