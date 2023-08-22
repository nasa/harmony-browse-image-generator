from pathlib import Path
from unittest import TestCase

from harmony_browse_image_generator.utilities import (
    get_asset_name,
    get_file_mime_type,
    get_tiled_file_extension,
)


class TestUtilities(TestCase):
    """A class testing the harmony_browse_image_generator.utilities module."""

    def test_get_file_mime_type(self):
        """Ensure a MIME type can be retrieved from an input file path."""
        with self.subTest('File with MIME type known by Python.'):
            self.assertEqual(get_file_mime_type('file.nc'), 'application/x-netcdf')

        with self.subTest('File with MIME type retrieved from dictionary.'):
            self.assertEqual(get_file_mime_type('file.nc4'), 'application/x-netcdf4')

        with self.subTest('ESRI world file retrieves "text/plain"'):
            self.assertEqual(get_file_mime_type('file.wld'), 'text/plain')

        with self.subTest('File with entirely unknown MIME type.'):
            self.assertIsNone(get_file_mime_type('file.xyzzyx'))

    def test_get_tiled_file_extension(self):
        """ensure correct extensions are extracted"""

        test_params = [
            (Path('/tmp/tmp4w/14316c44a.r00c02.png.aux.xml'), '.r00c02.png.aux.xml'),
            (Path('/tmp/tmp4w/14316c44a.png.aux.xml'), '.png.aux.xml'),
            (Path('/tmp/tmp4w/14316c44a.r00c02.png'), '.r00c02.png'),
            (Path('/tmp/tmp4w/14316c44a.png'), '.png'),
            (Path('/tmp/tmp4w/14316c44a.pgw'), '.pgw'),
            (Path('/tmp/tmp4w/14316c44a.r100c02.jpg.aux.xml'), '.r100c02.jpg.aux.xml'),
            (Path('/tmp/tmp4w/14316c44a.jpg.aux.xml'), '.jpg.aux.xml'),
            (Path('/tmp/tmp4w/14316c44a.r00c02.jpg'), '.r00c02.jpg'),
            (Path('/tmp/tmp4w/14316c44a.jpg'), '.jpg'),
            (Path('/tmp/tmp4w/14316c44a.jgw'), '.jgw'),
        ]
        for test_path, expected_extension in test_params:
            actual_extension = get_tiled_file_extension(test_path)
            self.assertEqual(expected_extension, actual_extension)

    def test_get_asset_name(self):
        """ensure correct asset names are generated"""

        test_params = [
            (
                ('name', 'https://tmp_bucket/tmp4w/14316c44a.r00c02.png.aux.xml'),
                'name_r00c02',
            ),
            (('name', 'https://tmp_bucket/tmp4w/14316c44a.png.aux.xml'), 'name'),
            (('name', 'https://tmp_bucket/tmp4w/14316c44a.r00c02.png'), 'name_r00c02'),
            (('name', 'https://tmp_bucket/tmp4w/14316c44a.png'), 'name'),
            (('name', 'https://tmp_bucket/tmp4w/14316c44a.pgw'), 'name'),
            (
                ('name', 'https://tmp_bucket/tmp4w/14316c44a.r100c02.jpg.aux.xml'),
                'name_r100c02',
            ),
            (('name', 'https://tmp_bucket/tmp4w/14316c44a.jpg.aux.xml'), 'name'),
            (
                ('name', 'https://tmp_bucket/tmp4w/14316c44a.r00c202.jpg'),
                'name_r00c202',
            ),
            (('name', 'https://tmp_bucket/tmp4w/14316c44a.jpg'), 'name'),
            (('name', 'https://tmp_bucket/tmp4w/14316c44a.jgw'), 'name'),
        ]
        for test_params, expected_name in test_params:
            actual_name = get_asset_name(*list(test_params))
            self.assertEqual(expected_name, actual_name, f'params: {test_params}')
