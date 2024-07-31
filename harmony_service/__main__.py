"""Run the Harmony Browse Image Generator Adapter via the Harmony CLI."""

from argparse import ArgumentParser
from sys import argv

from harmony import is_harmony_cli, run_cli, setup_cli

from .adapter import BrowseImageGeneratorAdapter
from .exceptions import SERVICE_NAME


def main(arguments: list[str]):
    """Parse command line arguments and invoke the appropriate method."""
    parser = ArgumentParser(
        prog=SERVICE_NAME, description='Run Harmony Browse Image Generator.'
    )

    setup_cli(parser)
    harmony_arguments, _ = parser.parse_known_args(arguments[1:])

    if is_harmony_cli(harmony_arguments):
        run_cli(parser, harmony_arguments, BrowseImageGeneratorAdapter)
    else:
        parser.error('Only --harmony CLIs are supported')


if __name__ == '__main__':
    main(argv)
