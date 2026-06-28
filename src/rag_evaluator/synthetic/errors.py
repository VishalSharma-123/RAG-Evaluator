from __future__ import annotations


class SyntheticError(Exception):
    """Base exception for synthetic generation."""


class SyntheticGenerationError(SyntheticError):
    """Raised when generation fails."""


class SyntheticParsingError(SyntheticError):
    """Raised when model output cannot be parsed."""


class SyntheticValidationError(SyntheticError):
    """Raised when parsed synthetic samples fail validation."""


class SyntheticProviderError(SyntheticError):
    """Raised when provider communication fails."""
