"""Module defining custom exceptions."""


class HyBIGError(Exception):
    """Base error class for exceptions rasied by HyBIG library."""

    def __init__(self, message=None):
        """All HyBIG errors have a message field."""
        self.message = message


class HyBIGNoColorInformation(HyBIGError):
    """Used to describe missing color information."""


class HyBIGValueError(HyBIGError):
    """Input was incorrect for the routine."""
