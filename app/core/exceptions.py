class RagError(Exception):
    """Base exception for expected RAG errors."""


class EmptyIndexError(RagError):
    """Raised when a question is submitted before ingestion."""


class UnsupportedDocumentError(RagError):
    """Raised when a document type is not supported."""
