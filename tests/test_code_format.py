"""Ensure code formatting."""

from itertools import chain
from pathlib import Path
from unittest import TestCase

from pycodestyle import StyleGuide


class TestCodeFormat(TestCase):
    """This test class should ensure all Harmony service Python code adheres
    to standard Python code styling.

    Ignored errors and warning:

    * E501: Line length, which defaults to 80 characters. This is a
            preferred feature of the code, but not always easily achieved.
    * W503: Break before binary operator. Have to ignore one of W503 or
            W504 to allow for breaking of some long lines. PEP8 suggests
            breaking the line before a binary operator is more "Pythonic".
    * E203, E701: This repository uses black code formatting, which deviates
                  from PEP8 for these errors.
    """

    def test_pycodestyle_adherence(self):
        """Check files for PEP8 compliance."""
        python_files = chain(
            Path('hybig').rglob('*.py'),
            Path('harmony_service').rglob('*.py'),
            Path('tests').rglob('*.py'),
        )

        style_guide = StyleGuide(ignore=['E501', 'W503', 'E203', 'E701'])
        results = style_guide.check_files(python_files)
        self.assertEqual(results.total_errors, 0, 'Found code style issues.')
