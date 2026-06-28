from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProviderGenerationResult:
    """
    Normalize provider response returned by provider-specific clients.
    """
    
    content: str
    raw_response: dict[str, Any] | None = None
    reasoning_details: Any | None = None
    usage: dict[str, Any] | None = None
    response_metadata: dict[str, Any] | None = None
    
