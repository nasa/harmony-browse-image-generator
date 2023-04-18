""" Module defining custom exceptions, designed to return user-friendly error
    messaging to the end-user.

"""
from harmony.util import HarmonyException


class HyBIGException(HarmonyException):
    """ Base service exception. """
    def __init__(self, message=None):
        super().__init__(message, 'sds/harmony-browse-image-generator')
