"""Base handler class for analysis pipeline.

All specialized handlers inherit from BaseHandler to ensure
consistent interface and shared utility methods.
"""

from abc import ABC, abstractmethod
from rich.console import Console

from ..analyzer import AnalysisConfig, AnalysisContext


class HandlerError(Exception):
    """Exception raised when a handler execution fails."""
    pass


class BaseHandler(ABC):
    """Base class for all analysis handlers.

    Handlers are responsible for a single aspect of the analysis pipeline.
    They operate on a shared AnalysisContext, modifying it in place.
    """

    def __init__(self, config: AnalysisConfig, console: Console):
        """Initialize handler with configuration and console.

        Args:
            config: Immutable analysis configuration
            console: Rich console for output
        """
        self.config = config
        self.console = console

    @abstractmethod
    def execute(self, context: AnalysisContext) -> None:
        """Execute this handler's responsibility.

        This is the main entry point for handler execution. Implementations
        should modify the context in place with their results.

        Args:
            context: Shared analysis context (modified in-place)

        Raises:
            HandlerError: If handler execution fails
        """
        pass

    def should_execute(self, context: AnalysisContext) -> bool:
        """Check if this handler should execute.

        Default: always execute. Override for conditional handlers
        (e.g., GitHub enrichment only if token is provided).

        Args:
            context: Current analysis context

        Returns:
            True if handler should execute, False to skip
        """
        return True

    def _log_info(self, message: str) -> None:
        """Helper to log info messages.

        Args:
            message: Message to log
        """
        self.console.print(f"[cyan]{message}[/cyan]")

    def _log_success(self, message: str) -> None:
        """Helper to log success messages.

        Args:
            message: Message to log
        """
        self.console.print(f"[green]{message}[/green]")

    def _log_warning(self, message: str) -> None:
        """Helper to log warnings.

        Args:
            message: Message to log
        """
        self.console.print(f"[yellow]{message}[/yellow]")

    def _log_error(self, message: str) -> None:
        """Helper to log error messages.

        Args:
            message: Message to log
        """
        self.console.print(f"[red]{message}[/red]")

    def _log_bold(self, message: str) -> None:
        """Helper to log bold messages.

        Args:
            message: Message to log
        """
        self.console.print(f"[bold]{message}[/bold]")
