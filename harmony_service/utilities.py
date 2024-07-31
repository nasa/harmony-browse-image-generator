"""Module containing service utility functionality."""

import re
from mimetypes import guess_type as guess_mime_type
from os.path import splitext
from pathlib import Path

KNOWN_MIME_TYPES = {
    '.nc4': 'application/x-netcdf4',
    '.h5': 'application/x-hdf5',
    '.wld': 'text/plain',
    '.jgw': 'text/plain',
    '.pgw': 'text/plain',
}


def get_tiled_file_extension(file_name: Path) -> str:
    """Return the correct extension to add to a staged file.

    Harmony's generate output filename can drop an extension incorrectly, so we
    generate the correct one to pass in.

    """
    ext_pattern = r"(\.r\d+c\d+)?\.(png|jpg|pgw|jgw|txt)(.aux.xml)?"
    match = re.search(ext_pattern, file_name.name)
    return match.group()


def get_asset_name(name: str, url: str) -> str:
    """Return the name of the asset.

    For tiled assets, we need to create a unique name beyond just the moniker
    of data, metadata, or auxiliary in order to store each item in the asset
    dictionary.

    """
    tiled_pattern = r"\.(r\d+c\d+)\."
    tile_id = re.search(tiled_pattern, url)
    if tile_id is not None:
        name = f'{name}_{tile_id.groups()[0]}'
    return name


def get_file_mime_type(file_name: Path | str) -> str | None:
    """This function tries to infer the MIME type of a file string. If the
    `mimetypes.guess_type` function cannot guess the MIME type of the
    granule, a dictionary of known file types is checked using the file
    extension. That dictionary only contains keys for MIME types that
    `mimetypes.guess_type` cannot resolve.

    """
    mime_type = guess_mime_type(file_name, False)

    if not mime_type or mime_type[0] is None:
        mime_type = (KNOWN_MIME_TYPES.get(splitext(file_name)[1]), None)

    return mime_type[0]
