"""SecurePath evidence replay primitives.

The core package has no network or secret requirements. Live dependencies are
loaded only by explicit integration commands.
"""

from .models import Claim, ConnectorResult, EvidencePacket, EvidenceState, Source
from .service import ResearchService

__all__ = [
    "Claim",
    "ConnectorResult",
    "EvidencePacket",
    "EvidenceState",
    "ResearchService",
    "Source",
]

__version__ = "0.2.0"
