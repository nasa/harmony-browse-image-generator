""" Unit tests for browse module. """

from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp
from unittest import TestCase
from unittest.mock import MagicMock, Mock, patch

import numpy as np
from numpy.testing import assert_array_equal
from osgeo_utils.auxiliary.color_palette import ColorPalette
from rasterio.io import DatasetReader

from harmony_browse_image_generator.browse import (
    convert_colormap_to_palette, convert_mulitband_to_raster,
    convert_singleband_to_raster, get_color_palette, output_image_file,
    output_world_file, palette_from_remote_colortable)
from harmony_browse_image_generator.exceptions import HyBIGException


class TestBrowse(TestCase):
    """A class testing the harmony_browse_image_generator.browse module."""

    @classmethod
    def setUpClass(cls):
        cls.fixtures = mkdtemp()

        cls.data = np.array(
            [
                [100, 200, 300, 400],
                [100, 200, 300, 400],
                [100, 200, 300, 400],
                [100, 200, 300, 400],
            ]
        ).astype('uint16')

        cls.levels = [100, 200, 300, 400]

        # R, G, B, A tuples
        red = (255, 0, 0, 255)
        yellow = (255, 255, 0, 255)
        green = (0, 255, 0, 255)
        blue = (0, 0, 255, 255)

        cls.colors = [red, yellow, green, blue]
        cls.colormap = {100: red, 200: yellow, 300: green, 400: blue}

    @classmethod
    def tearDownClass(cls):
        rmtree(cls.fixtures, ignore_errors=True)

    def test_convert_singleband_to_raster(self):
        with self.subTest('Test single band raster without colortable'):
            ds = Mock(DatasetReader)
            ds.read.return_value = self.data
            ds.colormap = MagicMock(side_effect=ValueError)

            expected_raster = np.array(
                [
                    [
                        [0, 104, 198, 255],
                        [0, 104, 198, 255],
                        [0, 104, 198, 255],
                        [0, 104, 198, 255],
                    ],
                    [
                        [0, 104, 198, 255],
                        [0, 104, 198, 255],
                        [0, 104, 198, 255],
                        [0, 104, 198, 255],
                    ],
                    [
                        [0, 104, 198, 255],
                        [0, 104, 198, 255],
                        [0, 104, 198, 255],
                        [0, 104, 198, 255],
                    ],
                    [
                        [255, 255, 255, 255],
                        [255, 255, 255, 255],
                        [255, 255, 255, 255],
                        [255, 255, 255, 255],
                    ],
                ],
                dtype='uint8',
            )

            actual_raster = convert_singleband_to_raster(ds)
            assert_array_equal(expected_raster, actual_raster)

        with self.subTest('Test 1band image with associated colormap.'):
            ds = Mock(DatasetReader)
            ds.read.return_value = self.data
            ds.colormap.return_value = self.colormap

            expected_raster = np.array(
                [
                    [  # red
                        [255, 255, 0, 0],
                        [255, 255, 0, 0],
                        [255, 255, 0, 0],
                        [255, 255, 0, 0],
                    ],
                    [  # green
                        [0, 255, 255, 0],
                        [0, 255, 255, 0],
                        [0, 255, 255, 0],
                        [0, 255, 255, 0],
                    ],
                    [  # blue
                        [0, 0, 0, 255],
                        [0, 0, 0, 255],
                        [0, 0, 0, 255],
                        [0, 0, 0, 255],
                    ],
                    [  # alpha
                        [255, 255, 255, 255],
                        [255, 255, 255, 255],
                        [255, 255, 255, 255],
                        [255, 255, 255, 255],
                    ],
                ],
                dtype='uint8',
            )
            # Read down: red, yellow, green, blue

            actual_raster = convert_singleband_to_raster(ds)
            assert_array_equal(expected_raster, actual_raster)

    def test_convert_multiband_to_raster(self):
        with self.subTest('Convert a 3 band image to raster.'):
            ds = Mock(DatasetReader)
            ds.count = 3
            ds.read.return_value = np.stack([self.data, self.data, self.data])
            ds.colormap.side_effect = ValueError

            expected_raster = np.array(
                [
                    [
                        [0, 85, 170, 255],
                        [0, 85, 170, 255],
                        [0, 85, 170, 255],
                        [0, 85, 170, 255],
                    ],
                    [
                        [0, 85, 170, 255],
                        [0, 85, 170, 255],
                        [0, 85, 170, 255],
                        [0, 85, 170, 255],
                    ],
                    [
                        [0, 85, 170, 255],
                        [0, 85, 170, 255],
                        [0, 85, 170, 255],
                        [0, 85, 170, 255],
                    ],
                ],
                dtype='uint8',
            )

            actual_raster = convert_mulitband_to_raster(ds)
            assert_array_equal(expected_raster, actual_raster.data)

        with self.subTest('Convert a 4 band image to raster'):
            ds = Mock(DatasetReader)
            alpha = np.ones_like(self.data) * 255
            alpha[0, 0] = 1
            ds.count = 4
            ds.read.return_value = np.stack([self.data, self.data, self.data, alpha])
            expected_raster = np.array(
                [
                    [
                        [0, 85, 170, 255],
                        [0, 85, 170, 255],
                        [0, 85, 170, 255],
                        [0, 85, 170, 255],
                    ],
                    [
                        [0, 85, 170, 255],
                        [0, 85, 170, 255],
                        [0, 85, 170, 255],
                        [0, 85, 170, 255],
                    ],
                    [
                        [0, 85, 170, 255],
                        [0, 85, 170, 255],
                        [0, 85, 170, 255],
                        [0, 85, 170, 255],
                    ],
                    [
                        [1, 255, 255, 255],
                        [255, 255, 255, 255],
                        [255, 255, 255, 255],
                        [255, 255, 255, 255],
                    ],
                ],
                dtype='uint8',
            )

            actual_raster = convert_mulitband_to_raster(ds)
            assert_array_equal(expected_raster, actual_raster.data)

        with self.subTest('fails on too many bands'):
            ds = Mock(DatasetReader)
            ds.count = 5
            ds.read.return_value = np.stack(
                [self.data, self.data, self.data, self.data, self.data]
            )

            with self.assertRaises(HyBIGException) as excepted:
                convert_mulitband_to_raster(ds)

            self.assertEqual(
                excepted.exception.message, 'Cannot create image from 5 band image'
            )

    def test_get_color_palette(self):
        with self.subTest('colormap exists on geotiff'):
            ds = Mock(DatasetReader)
            ds.colormap.return_value = self.colormap

            lines = [
                "100 255 0 0 255",
                "200 255 255 0 255",
                "300 0 255 0 255",
                "400 0 0 255 255",
            ]

            expected_palette = ColorPalette()
            expected_palette.read_file_txt(lines=lines)

            actual_palette = get_color_palette(ds)

            self.assertEqual(expected_palette, actual_palette)

        with self.subTest('A dataset without a colormap returns None'):
            ds = Mock(DatasetReader)
            ds.colormap.side_effect = ValueError
            expected_palette = None

            actual_palette = get_color_palette(ds)

            self.assertEqual(expected_palette, actual_palette)

    def test_output_image_file(self):
        input_filename = Path('/path/to/some.tiff')

        with self.subTest('with default driver [PNG]'):
            expected_image_file = Path('/path/to/some.png')

            actual_image_file = output_image_file(input_filename)
            self.assertEqual(expected_image_file, actual_image_file)

        with self.subTest('With JPEG driver'):
            expected_image_file = Path('/path/to/some.jpg')

            actual_image_file = output_image_file(input_filename, driver='JPEG')
            self.assertEqual(expected_image_file, actual_image_file)

    def test_output_world_file(self):
        input_filename = Path('/path/to/some.tiff')

        with self.subTest('default driver [PNG]'):
            expected_world_file = Path('/path/to/some.pgw')
            actual_world_file = output_world_file(input_filename)
            self.assertEqual(expected_world_file, actual_world_file)

        with self.subTest('JPEG driver'):
            expected_world_file = Path('/path/to/some.jgw')
            actual_world_file = output_world_file(input_filename, driver='JPEG')
            self.assertEqual(expected_world_file, actual_world_file)

    def test_convert_colormap_to_palette(self):
        """A paletted geotiff has a colormap like a dictionary."""

        def encode_color(r, g, b, a=255):
            """How an rgb[a] triplet is coded for a palette."""
            return (((((int(a) << 8) + int(r)) << 8) + int(g)) << 8) + int(b)

        with self.subTest('Test and demonstrate 3 band'):
            input_colormap = {
                5: (255, 0, 0),  # red
                100: (0, 255, 0),  # blue
                30: (0, 0, 255),  # green
                50: (10, 100, 201),
            }

        # 3 color palettes are stores as coded 4-bypte int values
        # Alpha/Red/Green/Blue
        # where Alpha is always 255
        # so (1,1,1) is stored as '0b11111111000000010000000100000001' or 4278255873
        # (255, 0, 0), as: 0b11111111 11111111 00000000 00000000

        red = 0b11111111111111110000000000000000
        green = 0b11111111000000001111111100000000
        blue = 0b11111111000000000000000011111111
        last = encode_color(10, 100, 201)

        actual_palette = convert_colormap_to_palette(input_colormap)

        self.assertEqual(actual_palette.get_color(5), red)
        self.assertEqual(actual_palette.get_color(100), green)
        self.assertEqual(actual_palette.get_color(30), blue)
        self.assertEqual(actual_palette.get_color(50), last)

        with self.subTest('Test and demonstrate 4 band'):
            input_colormap = {
                5: (255, 0, 0, 100),  # red
                100: (0, 255, 0, 200),  # blue
                30: (0, 0, 255, 255),  # green
                50: (10, 100, 201, 255),  # other
            }

            actual_palette = convert_colormap_to_palette(input_colormap)

            for color_level, rgba_tuple in input_colormap.items():
                self.assertEqual(
                    actual_palette.get_color(color_level), encode_color(*rgba_tuple)
                )

    @patch('harmony_browse_image_generator.browse.requests.get')
    def test_palette_from_remote_colortable(self, mock_get):
        with self.subTest('successful retrieval of colortable'):
            returned_colortable = (
                'nv 0 0 0 0\n-9999 43 0 26\n273.15 45 0 28\n273.30 48 0 31'
            )
            url = 'http://colortable.org'
            mock_get.return_value.ok = True
            mock_get.return_value.text = returned_colortable

            expected_palette = ColorPalette()
            expected_palette.read_file_txt(
                lines=[
                    'nv 0 0 0 0',
                    '-9999 43 0 26',
                    '273.15 45 0 28',
                    '273.30 48 0 31',
                ]
            )

            actual_palette = palette_from_remote_colortable(url)
            self.assertEqual(expected_palette, actual_palette)

        with self.subTest('failed retrieval of colortable'):
            url = 'http://this-domain-does-not-exist.com/bad-url'
            mock_get.return_value.ok = False

            with self.assertRaises(HyBIGException) as excepted:
                palette_from_remote_colortable(url)

            self.assertEqual(
                excepted.exception.message,
                (
                    'Could not read remote colortable at'
                    ' http://this-domain-does-not-exist.com/bad-url'
                ),
            )
