""" Unit tests for browse module. """

from logging import getLogger
from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp
from unittest import TestCase
from unittest.mock import MagicMock, Mock, call, patch

import numpy as np
from harmony.message import Message as HarmonyMessage
from numpy.testing import assert_array_equal
from osgeo_utils.auxiliary.color_palette import ColorPalette
from PIL import Image
from rasterio import Affine
from rasterio.crs import CRS
from rasterio.io import DatasetReader, DatasetWriter
from rasterio.warp import Resampling
from rasterio.plot import reshape_as_image

from harmony_browse_image_generator.browse import (
    convert_colormap_to_palette,
    convert_mulitband_to_raster,
    convert_singleband_to_raster,
    create_browse_imagery,
    get_color_map_from_image,
    get_color_palette,
    output_image_file,
    output_world_file,
    palette_from_remote_colortable,
    palettize_raster,
    prepare_raster_for_writing,
    validate_file_type,
)
from harmony_browse_image_generator.exceptions import HyBIGException
from tests.unit.utility import rasterio_test_file


def encode_color(r, g, b, a=255):
    """How an rgb[a] triplet is coded for a palette."""
    return (((((int(a) << 8) + int(r)) << 8) + int(g)) << 8) + int(b)


class TestBrowse(TestCase):
    """A class testing the harmony_browse_image_generator.browse module."""

    @classmethod
    def setUpClass(cls):
        cls.logger = getLogger()

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
        cls.random = np.random.default_rng()

    def test_create_browse_imagery_with_bad_raster(self):
        """Check that preferred metadata for global projection is found."""
        two_dimensional_raster = np.array(
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
            ],
            dtype='uint8',
        )
        message = HarmonyMessage({'format': {'mime': 'JPEG'}})

        with rasterio_test_file(
            raster_data=two_dimensional_raster,
            height=two_dimensional_raster.shape[1],
            width=two_dimensional_raster.shape[2],
            count=2,
        ) as test_tif_filename:
            with self.assertRaisesRegex(
                HyBIGException, 'incorrect number of bands for image: 2'
            ):
                create_browse_imagery(message, test_tif_filename, self.logger)

    def test_create_browse_imagery_with_single_band_raster(self):
        """Check that preferred metadata for global projection is found."""
        two_dimensional_raster = np.array(
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
            ],
            dtype='uint8',
        )
        message = HarmonyMessage({'format': {'mime': 'JPEG'}})

        with rasterio_test_file(
            raster_data=two_dimensional_raster,
            height=two_dimensional_raster.shape[1],
            width=two_dimensional_raster.shape[2],
            count=2,
        ) as test_tif_filename:
            with self.assertRaisesRegex(
                HyBIGException, 'incorrect number of bands for image: 2'
            ):
                create_browse_imagery(message, test_tif_filename, None)

    @patch('harmony_browse_image_generator.browse.reproject')
    @patch('rasterio.open')
    def test_create_browse_imagery_with_mocks(self, rasterio_open_mock, reproject_mock):
        ds_mock = Mock(DatasetReader)
        dest_write_mock = Mock(DatasetWriter)
        ds_mock.read.return_value = self.data
        ds_mock.driver = 'GTiff'
        ds_mock.height = 4
        ds_mock.width = 4
        ds_mock.transform = Affine.identity()
        ds_mock.crs = CRS.from_string('EPSG:4326')
        ds_mock.count = 1
        ds_mock.colormap = MagicMock(side_effect=ValueError)

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

        rasterio_open_mock.return_value.__enter__.side_effect = [
            ds_mock,
            dest_write_mock,
        ]

        message = HarmonyMessage({'format': {'mime': 'JPEG'}})

        # Act to run the test
        actual_image, actual_world = create_browse_imagery(
            message, './input_file_path', self.logger
        )

        target_transform = Affine(90.0, 0.0, -180.0, 0.0, -45.0, 90.0)
        dest = np.zeros((ds_mock.height, ds_mock.width), dtype='uint8')

        self.assertEqual(reproject_mock.call_count, 3)

        expected_calls = [
            call(
                source=expected_raster[0, :, :],
                destination=dest,
                src_transform=Affine.identity(),
                src_crs=ds_mock.crs,
                dst_transform=target_transform,
                dst_crs=CRS.from_string('EPSG:4326'),
                dst_nodata=255,
                resampling=Resampling.nearest,
            ),
            call(
                source=expected_raster[1, :, :],
                destination=dest,
                src_transform=Affine.identity(),
                src_crs=ds_mock.crs,
                dst_transform=target_transform,
                dst_crs=CRS.from_string('EPSG:4326'),
                dst_nodata=255,
                resampling=Resampling.nearest,
            ),
            call(
                source=expected_raster[2, :, :],
                destination=dest,
                src_transform=Affine.identity(),
                src_crs=ds_mock.crs,
                dst_transform=target_transform,
                dst_crs=CRS.from_string('EPSG:4326'),
                dst_nodata=255,
                resampling=Resampling.nearest,
            ),
        ]

        for actual_call, expected_call in zip(
            reproject_mock.call_args_list, expected_calls
        ):
            np.testing.assert_array_equal(
                actual_call.kwargs['source'], expected_call.kwargs['source']
            )
            np.testing.assert_array_equal(
                actual_call.kwargs['destination'], expected_call.kwargs['destination']
            )
            self.assertEqual(
                actual_call.kwargs['src_transform'],
                expected_call.kwargs['src_transform'],
            )
            self.assertEqual(
                actual_call.kwargs['src_crs'], expected_call.kwargs['src_crs']
            )
            self.assertEqual(
                actual_call.kwargs['dst_transform'],
                expected_call.kwargs['dst_transform'],
            )
            self.assertEqual(
                actual_call.kwargs['dst_crs'], expected_call.kwargs['dst_crs']
            )
            self.assertEqual(
                actual_call.kwargs['dst_nodata'], expected_call.kwargs['dst_nodata']
            )
            self.assertEqual(
                actual_call.kwargs['resampling'], expected_call.kwargs['resampling']
            )

        self.assertEqual(
            Path('./input_file_path.jpg').resolve(), actual_image.resolve()
        )
        self.assertEqual(
            Path('./input_file_path.jgw').resolve(), actual_world.resolve()
        )

    def test_convert_singleband_to_raster_without_colortable(self):
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

    def test_convert_singleband_to_raster_with_colormap(self):
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

    def test_convert_3_multiband_to_raster(self):
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

    def test_convert_4_multiband_to_raster(self):
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

    def test_convert_5_multiband_to_raster(self):
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

    def test_prepare_raster_for_writing_jpeg_3band(self):
        raster = self.random.integers(255, size=(3, 5, 6))
        driver = 'JPEG'
        expected_raster = np.copy(raster)
        expected_color_map = None

        actual_raster, actual_color_map = prepare_raster_for_writing(raster, driver)
        self.assertEqual(expected_color_map, actual_color_map)
        np.testing.assert_array_equal(expected_raster, actual_raster)

    def test_prepare_raster_for_writing_jpeg_4band(self):
        raster = self.random.integers(255, size=(4, 7, 8))
        driver = 'JPEG'
        expected_raster = np.copy(raster[0:3, :, :])
        expected_color_map = None
        actual_raster, actual_color_map = prepare_raster_for_writing(raster, driver)
        self.assertEqual(expected_color_map, actual_color_map)
        np.testing.assert_array_equal(expected_raster, actual_raster)

    @patch('harmony_browse_image_generator.browse.palettize_raster')
    def test_prepare_raster_for_writing_png_4band(self, palettize_mock):
        raster = self.random.integers(255, size=(4, 7, 8))
        driver = 'PNG'

        prepare_raster_for_writing(raster, driver)

        palettize_mock.assert_called_once_with(raster)

    @patch('harmony_browse_image_generator.browse.quantize_pil_image')
    @patch('harmony_browse_image_generator.browse.get_color_map_from_image')
    def test_palettize_raster(self, get_color_map_mock, quantize_mock):
        """Test that the quantize function is called with a correct image."""
        raster = self.random.integers(255, dtype='uint8', size=(4, 10, 11))
        multiband_image = Image.fromarray(reshape_as_image(raster))
        quantized_output = Image.fromarray(
            self.random.integers(254, size=(10, 11), dtype='uint8')
        )
        quantize_mock.return_value = quantized_output

        expected_out_raster = np.array(quantized_output.getdata()).reshape(1, 10, 11)
        print(expected_out_raster.shape)

        out_raster, out_map = palettize_raster(raster)

        quantize_mock.assert_called_once_with(multiband_image, max_colors=255)
        get_color_map_mock.assert_called_once_with(quantized_output)
        np.testing.assert_array_equal(expected_out_raster, out_raster)

    def test_get_color_map_from_image(self):
        """PIL Image yields a color_map

        A palette from an PIL Image is correctly turned into a colormap
        writable by rasterio.

        """
        # random image with values of 0 to 4.
        image_data = self.random.integers(5, size=(5, 6), dtype='uint8')
        palette_sequence = [
            255, 0, 0, 255,
            0, 255, 0, 255,
            0, 0, 255, 255,
            225, 100, 25, 25,
            0, 0, 0, 0
        ]
        test_image = Image.fromarray(image_data)
        test_image.putpalette(palette_sequence, rawmode='RGBA')

        expected_color_map = {
            0: (255, 0, 0, 255),
            1: (0, 255, 0, 255),
            2: (0, 0, 255, 255),
            3: (225, 100, 25, 25),
            4: (0, 0, 0, 0)
        }

        actual_color_map = get_color_map_from_image(test_image)
        self.assertDictEqual(expected_color_map, actual_color_map)

    def test_get_color_palette_map_exists(self):
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

    def test_get_color_palette_map_does_not_exist(self):
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

    def test_validate_file_type_valid(self):
        """validation should not raise exception."""
        ds = Mock(DatasetReader)
        ds.driver = 'GTiff'
        try:
            validate_file_type(ds)
        except Exception:
            self.fail('Valid file type threw unexpected exception.')

    def test_validate_file_type_invalid(self):
        """Only GTiff drivers work."""
        ds = Mock(DatasetReader)
        ds.driver = 'NetCDF4'
        with self.assertRaisesRegex(
            HyBIGException, 'Input file type not supported: NetCDF4'
        ):
            validate_file_type(ds)

    def test_convert_colormap_to_palette_3bands(self):
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

    def test_convert_colormap_to_palette_4bands(self):
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
