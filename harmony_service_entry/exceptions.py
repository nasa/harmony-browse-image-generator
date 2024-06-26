"""Module defining harmony service errors raised by HyBIG service."""

from harmony.util import HarmonyException

SERVICE_NAME = 'harmony-browse-image-generator'


class HyBIGServiceError(HarmonyException):
    """Base service exception."""

    def __init__(self, message=None):
        """All service errors are assocated with SERVICE_NAME."""
        super().__init__(message=message, category=SERVICE_NAME)


class HyBIGInvalidMessageError(HyBIGServiceError):
    """Input Harmony Message could not be used as presented."""
