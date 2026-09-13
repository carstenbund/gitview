"""Provider seam: how GitView asks an external analyser for structural evidence.

A provider is *optional* infrastructure. GitView never imports a concrete
provider at module load; ``get_provider`` resolves one by name on demand so
the rest of the system works with no analyser installed.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

from .models import StructuralSnapshot


class StructuralProviderError(RuntimeError):
    """The provider could not produce an observation (missing tool, bad input, …)."""


class StructuralProvider(ABC):
    """Produce a :class:`StructuralSnapshot` for a repository at a commit."""

    #: Short identifier used on the CLI (``--structural <name>``) and stored as provenance.
    name: str = ''

    @abstractmethod
    def available(self) -> Tuple[bool, str]:
        """Return ``(True, '')`` when the provider can run, else ``(False, reason)``."""

    @abstractmethod
    def snapshot(
        self,
        repo_path: Union[str, Path],
        sha: str,
        *,
        source: Optional[Union[str, Path]] = None,
        refresh: bool = False,
    ) -> StructuralSnapshot:
        """Observe ``repo_path`` and return a snapshot.

        ``sha`` is the commit GitView believes the working tree is at; a
        provider that records its own commit must prefer that and let the
        caller detect drift. ``source`` overrides the provider's default
        input location; ``refresh`` asks the provider to re-run its analysis
        instead of reusing an existing output.
        """


_REGISTRY: Dict[str, Callable[[], StructuralProvider]] = {}


def register_provider(name: str, factory: Callable[[], StructuralProvider]) -> None:
    _REGISTRY[name] = factory


def provider_names() -> List[str]:
    _load_builtin()
    return sorted(_REGISTRY)


def get_provider(name: str) -> StructuralProvider:
    _load_builtin()
    try:
        factory = _REGISTRY[name]
    except KeyError:
        known = ', '.join(provider_names()) or 'none'
        raise StructuralProviderError(f"unknown structural provider '{name}' (known: {known})")
    return factory()


def _load_builtin() -> None:
    if 'graphify' not in _REGISTRY:
        from .graphify import GraphifyProvider
        register_provider('graphify', GraphifyProvider)
