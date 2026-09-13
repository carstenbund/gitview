"""Optional structural evidence for GitView.

Historical evidence (``gitview.graph``) is always available. Structural
evidence — file dependencies and clusters as observed by an external
analyser at one commit — is optional and arrives through a provider seam.
GitView stores the neutral model with provenance and never depends on any
particular analyser.
"""

from .models import (
    STRUCTURAL_SCHEMA_VERSION,
    StructuralEdge,
    StructuralNode,
    StructuralObservation,
    StructuralSnapshot,
)
from .provider import (
    StructuralProvider,
    StructuralProviderError,
    get_provider,
    provider_names,
    register_provider,
)

__all__ = [
    'STRUCTURAL_SCHEMA_VERSION',
    'StructuralEdge',
    'StructuralNode',
    'StructuralObservation',
    'StructuralProvider',
    'StructuralProviderError',
    'StructuralSnapshot',
    'get_provider',
    'provider_names',
    'register_provider',
]
