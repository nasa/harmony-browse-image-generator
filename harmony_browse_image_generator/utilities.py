""" Module containing utility functionality. """
from mimetypes import guess_type as guess_mime_type
from os.path import splitext
from pathlib import Path

KNOWN_MIME_TYPES = {
    '.nc4': 'application/x-netcdf4',
    '.h5': 'application/x-hdf5',
    '.wld': 'text/plain',
    '.jgw': 'text/plain',
    '.pgw': 'text/plain'
}


def get_file_mime_type(file_name: Path | str) -> str | None:
    """ This function tries to infer the MIME type of a file string. If the
        `mimetypes.guess_type` function cannot guess the MIME type of the
        granule, a dictionary of known file types is checked using the file
        extension. That dictionary only contains keys for MIME types that
        `mimetypes.guess_type` cannot resolve.

    """
    mime_type = guess_mime_type(file_name, False)

    if not mime_type or mime_type[0] is None:
        mime_type = (KNOWN_MIME_TYPES.get(splitext(file_name)[1]), None)

    return mime_type[0]
