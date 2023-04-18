""" Module containing core functionality for browse image generation. """
from os.path import splitext
from pathlib import Path
from typing import Tuple


def create_browse_imagery(input_file_path: str) -> Tuple[str]:
    """ Take input browse image and return a 2-element tuple for the file paths
        of the output browse image and its associated ESRI world file.

        Note: this function is currently a dummy function, returning empty
        files for both outputs.

    """
    input_non_extension = splitext(input_file_path)[0]

    browse_image = Path(f'{input_non_extension}.png')
    browse_image.touch()

    world_file = Path(f'{input_non_extension}.wld')
    world_file.touch()

    return (str(browse_image), str(world_file))
