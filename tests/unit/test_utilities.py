from pathlib import Path
from unittest import TestCase

from harmony_service.utilities import (
    asset_matches_variable,
    get_asset_name,
    get_file_mime_type,
    get_tiled_file_extension,
)


class TestUtilities(TestCase):
    """A class testing the hybig.utilities module."""

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
        """Ensure correct extensions are extracted"""
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
            # Per-dimension outputs (>4 band inputs) carry a .zNN identifier, which
            # may appear alone or alongside a tile locator.
            (Path('/tmp/tmp4w/14316c44a.z03.png'), '.z03.png'),
            (Path('/tmp/tmp4w/14316c44a.z03.pgw'), '.z03.pgw'),
            (Path('/tmp/tmp4w/14316c44a.z03.png.aux.xml'), '.z03.png.aux.xml'),
            (Path('/tmp/tmp4w/14316c44a.z03.r00c02.png'), '.z03.r00c02.png'),
            (
                Path('/tmp/tmp4w/14316c44a.z03.r00c02.png.aux.xml'),
                '.z03.r00c02.png.aux.xml',
            ),
        ]
        for test_path, expected_extension in test_params:
            actual_extension = get_tiled_file_extension(test_path)
            self.assertEqual(expected_extension, actual_extension)

    def test_get_asset_name(self):
        """Ensure correct asset names are generated"""
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
            # Per-dimension outputs get a dimension (z) identifier so each band's
            # assets remain distinct, optionally combined with a tile locator.
            (
                ('name', 'https://tmp_bucket/tmp4w/14316c44a.z03.png'),
                'name_z03',
            ),
            (
                ('name', 'https://tmp_bucket/tmp4w/14316c44a.z03.png.aux.xml'),
                'name_z03',
            ),
            (
                ('name', 'https://tmp_bucket/tmp4w/14316c44a.z03.r00c02.png'),
                'name_z03_r00c02',
            ),
        ]
        for test_params, expected_name in test_params:
            actual_name = get_asset_name(*list(test_params))
            self.assertEqual(expected_name, actual_name, f'params: {test_params}')

    def test_asset_matches_variable(self):
        """Ensure an asset is matched to a variable by the end of its file name."""
        granule_prefix = 'https://example.com/ECOv003_L2T_LSTE_42783_027_22MBB_03'
        test_params = [
            ((f'{granule_prefix}_LST.tif', 'LST'), True),
            ((f'{granule_prefix}_view_zenith.tif', 'view_zenith'), True),
            ((f'{granule_prefix}_view_zenith.tif', 'LST'), False),
            ((f'{granule_prefix}_LST_err.tif', 'LST'), False),
            ((f'{granule_prefix}_LST.tif', 'LST_err'), False),
            ((f'{granule_prefix}_QC.tif', 'qc'), False),
            ((f'{granule_prefix}_cloud.tif?token=abc_LST', 'LST'), False),
            (('s3://bucket/path/LST.tif', 'LST'), True),
            (('s3://bucket/path/granuleLST.tif', 'LST'), False),
            (('/local/path/granule_B01_WTR.tif', 'WTR'), True),
            (('/local/path/granule_B01_WTR.tif', 'B01_WTR'), True),
            (('/local/path/granule_B02_BWTR.tif', 'WTR'), False),
        ]
        for test_params, expected in test_params:
            actual = asset_matches_variable(*test_params)
            self.assertEqual(expected, actual, f'params: {test_params}')
