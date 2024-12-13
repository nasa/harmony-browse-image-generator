"""Unit tests for browse module."""

import shutil
import tempfile
from logging import Logger, getLogger
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, Mock, call, patch

import numpy as np
from harmony_service_lib.message import Message as HarmonyMessage
from harmony_service_lib.message import Source as HarmonySource
from numpy.testing import assert_array_equal
from osgeo_utils.auxiliary.color_palette import ColorPalette
from PIL import Image
from rasterio import Affine
from rasterio.crs import CRS
from rasterio.io import DatasetReader, DatasetWriter
from rasterio.transform import array_bounds
from rasterio.warp import Resampling
from xarray import DataArray

from hybig.browse import (
    convert_mulitband_to_raster,
    convert_singleband_to_raster,
    create_browse,
    create_browse_imagery,
    get_color_map_from_image,
    get_tiled_filename,
    output_image_file,
    output_world_file,
    palettize_raster,
    prepare_raster_for_writing,
    validate_file_crs,
    validate_file_type,
)
from hybig.color_utility import (
    OPAQUE,
    TRANSPARENT,
    convert_colormap_to_palette,
    get_color_palette,
    palette_from_remote_colortable,
)
from hybig.exceptions import HyBIGError
from tests.unit.utility import rasterio_test_file


class TestBrowse(TestCase):
    """A class testing the hybig.browse module."""

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

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

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
                HyBIGError, 'incorrect number of bands for image: 2'
            ):
                create_browse_imagery(
                    message, test_tif_filename, HarmonySource({}), None, self.logger
                )

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
                HyBIGError, 'incorrect number of bands for image: 2'
            ):
                create_browse_imagery(
                    message, test_tif_filename, HarmonySource({}), None, None
                )

    @patch('hybig.browse.reproject')
    @patch('rasterio.open')
    @patch('hybig.browse.open_rasterio')
    def test_create_browse_imagery_with_mocks(
        self, rioxarray_open_mock, rasterio_open_mock, reproject_mock
    ):
        file_transform = Affine(90.0, 0.0, -180.0, 0.0, -45.0, 90.0)
        da_mock = MagicMock(DataArray)
        in_dataset_mock = Mock(DatasetReader)
        da_mock.rio._manager.acquire.return_value = in_dataset_mock

        dest_write_mock = Mock(DatasetWriter)

        da_mock.__getitem__.return_value = self.data
        in_dataset_mock.driver = 'GTiff'
        da_mock.rio.height = 4
        da_mock.rio.width = 4
        da_mock.rio.transform.return_value = file_transform
        da_mock.rio.crs = CRS.from_string('EPSG:4326')
        da_mock.rio.count = 1
        in_dataset_mock.colormap = Mock(side_effect=ValueError)

        da_mock.rio.transform_bounds.return_value = array_bounds(4, 4, file_transform)

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

        rioxarray_open_mock.return_value.__enter__.side_effect = [
            da_mock,
        ]
        rasterio_open_mock.return_value.__enter__.side_effect = [
            dest_write_mock,
        ]

        message = HarmonyMessage({'format': {'mime': 'JPEG'}})

        # Act to run the test
        out_file_list = create_browse_imagery(
            message,
            self.tmp_dir / 'input_file_path',
            HarmonySource({}),
            None,
            self.logger,
        )

        # Ensure tiling logic was not called:
        self.assertEqual(len(out_file_list), 1)

        actual_image, actual_world, actual_aux = out_file_list[0]

        target_transform = Affine(90.0, 0.0, -180.0, 0.0, -45.0, 90.0)
        dest = np.zeros((da_mock.rio.height, da_mock.rio.width), dtype='uint8')

        self.assertEqual(reproject_mock.call_count, 3)

        expected_calls = [
            call(
                source=expected_raster[0, :, :],
                destination=dest,
                src_transform=file_transform,
                src_crs=da_mock.rio.crs,
                dst_transform=target_transform,
                dst_crs=CRS.from_string('EPSG:4326'),
                dst_nodata=255,
                resampling=Resampling.nearest,
            ),
            call(
                source=expected_raster[1, :, :],
                destination=dest,
                src_transform=file_transform,
                src_crs=da_mock.rio.crs,
                dst_transform=target_transform,
                dst_crs=CRS.from_string('EPSG:4326'),
                dst_nodata=255,
                resampling=Resampling.nearest,
            ),
            call(
                source=expected_raster[2, :, :],
                destination=dest,
                src_transform=file_transform,
                src_crs=da_mock.rio.crs,
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
                actual_call.kwargs['source'],
                expected_call.kwargs['source'],
                strict=True,
            )
            np.testing.assert_array_equal(
                actual_call.kwargs['destination'],
                expected_call.kwargs['destination'],
                strict=True,
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
            (self.tmp_dir / 'input_file_path.jpg').resolve(), actual_image.resolve()
        )
        self.assertEqual(
            (self.tmp_dir / 'input_file_path.jgw').resolve(), actual_world.resolve()
        )
        self.assertEqual(
            (self.tmp_dir / 'input_file_path.jpg.aux.xml').resolve(),
            actual_aux.resolve(),
        )

    def test_convert_singleband_to_raster_without_colortable(self):
        """Tests convert_gray_1band_to_raster."""
        return_data = np.copy(self.data).astype('float64')
        return_data[0][1] = np.nan
        ds = DataArray(return_data).expand_dims('band')

        expected_raster = np.array(
            [
                [
                    [0, 0, 198, 255],
                    [0, 104, 198, 255],
                    [0, 104, 198, 255],
                    [0, 104, 198, 255],
                ],
                [
                    [0, 0, 198, 255],
                    [0, 104, 198, 255],
                    [0, 104, 198, 255],
                    [0, 104, 198, 255],
                ],
                [
                    [0, 0, 198, 255],
                    [0, 104, 198, 255],
                    [0, 104, 198, 255],
                    [0, 104, 198, 255],
                ],
                [
                    [255, 0, 255, 255],
                    [255, 255, 255, 255],
                    [255, 255, 255, 255],
                    [255, 255, 255, 255],
                ],
            ],
            dtype='uint8',
        )
        actual_raster = convert_singleband_to_raster(ds, None)
        assert_array_equal(expected_raster, actual_raster, strict=True)

    def test_convert_singleband_to_raster_with_colormap(self):
        ds = DataArray(self.data).expand_dims('band')

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
        image_palette = convert_colormap_to_palette(self.colormap)
        actual_raster = convert_singleband_to_raster(ds, image_palette)
        assert_array_equal(expected_raster, actual_raster, strict=True)

    def test_convert_singleband_to_raster_with_colormap_and_bad_data(self):
        data_array = np.array(self.data, dtype='float')
        data_array[0, 0] = np.nan
        ds = DataArray(data_array).expand_dims('band')
        nv_color = (10, 20, 30, 40)

        # Read the image down: red, yellow, green, blue
        expected_raster = np.array(
            [
                [  # red
                    [nv_color[0], 255, 0, 0],
                    [255, 255, 0, 0],
                    [255, 255, 0, 0],
                    [255, 255, 0, 0],
                ],
                [  # green
                    [nv_color[1], 255, 255, 0],
                    [0, 255, 255, 0],
                    [0, 255, 255, 0],
                    [0, 255, 255, 0],
                ],
                [  # blue
                    [nv_color[2], 0, 0, 255],
                    [0, 0, 0, 255],
                    [0, 0, 0, 255],
                    [0, 0, 0, 255],
                ],
                [  # alpha
                    [nv_color[3], 255, 255, 255],
                    [255, 255, 255, 255],
                    [255, 255, 255, 255],
                    [255, 255, 255, 255],
                ],
            ],
            dtype='uint8',
        )

        colormap = {**self.colormap, 'nv': nv_color}

        image_palette = convert_colormap_to_palette(colormap)
        actual_raster = convert_singleband_to_raster(ds, image_palette)
        assert_array_equal(expected_raster, actual_raster, strict=True)

    def test_convert_uint16_3_multiband_to_raster(self):
        """Test that uint16 input scales the output."""
        bad_data = np.copy(self.data).astype('float64')
        bad_data[1][1] = np.nan
        bad_data[1][2] = np.nan
        ds = DataArray(
            np.stack([self.data, bad_data, self.data]),
            dims=('band', 'y', 'x'),
        )
        ds.encoding = {'dtype': 'uint16'}

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
                    [0, 0, 0, 255],
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
                    [OPAQUE, OPAQUE, OPAQUE, OPAQUE],
                    [OPAQUE, TRANSPARENT, TRANSPARENT, OPAQUE],
                    [OPAQUE, OPAQUE, OPAQUE, OPAQUE],
                    [OPAQUE, OPAQUE, OPAQUE, OPAQUE],
                ],
            ],
            dtype='uint8',
        )

        actual_raster = convert_mulitband_to_raster(ds)
        assert_array_equal(expected_raster, actual_raster.data, strict=True)

    def test_convert_uint8_3_multiband_to_raster(self):
        """Ensure valid data is unchanged when input is uint8."""
        scale_data = np.array(
            [
                [10, 200, 30, 40],
                [10, np.nan, np.nan, 40],
                [10, 200, 30, 40],
                [10, 200, 30, 40],
            ]
        ).astype('float32')

        ds = DataArray(
            np.stack([scale_data, scale_data, scale_data]),
            dims=('band', 'y', 'x'),
        )
        ds.encoding = {'dtype': 'uint8'}

        expected_data = scale_data.copy()
        expected_data[1][1] = 0
        expected_data[1][2] = 0

        expected_raster = np.array(
            [
                expected_data,
                expected_data,
                expected_data,
                [
                    [OPAQUE, OPAQUE, OPAQUE, OPAQUE],
                    [OPAQUE, TRANSPARENT, TRANSPARENT, OPAQUE],
                    [OPAQUE, OPAQUE, OPAQUE, OPAQUE],
                    [OPAQUE, OPAQUE, OPAQUE, OPAQUE],
                ],
            ],
            dtype='uint8',
        )

        actual_raster = convert_mulitband_to_raster(ds)
        assert_array_equal(expected_raster, actual_raster.data, strict=True)

    def test_convert_4_multiband_uint8_to_raster(self):
        """4-band 'uint8' images are returned unchanged."""
        ds = Mock(DataArray)
        ds.rio.count = 4

        r_data = np.array(
            [
                [10, 200, 30, 40],
                [10, 200, 30, 40],
                [10, 200, 30, 40],
                [10, 200, 30, 40],
            ]
        ).astype('uint8')

        g_data = r_data.copy()
        b_data = r_data.copy()

        a_data = np.ones_like(r_data) * 255
        a_data[0, 0] = 0

        to_numpy_result = np.stack([r_data, g_data, b_data, a_data])

        ds.to_numpy.return_value = to_numpy_result

        expected_raster = to_numpy_result

        actual_raster = convert_mulitband_to_raster(ds)
        assert_array_equal(expected_raster, actual_raster.data, strict=True)

    def test_convert_4_multiband_uint16_to_raster(self):
        """4-band 'uint16' images are scaled if their range exceeds 255."""
        ds = Mock(DataArray)
        ds.rio.count = 4

        r_data = np.array(
            [
                [10, 200, 300, 400],
                [10, 200, 300, 400],
                [10, 200, 300, 400],
                [10, 200, 300, 400],
            ]
        ).astype('uint16')
        g_data = r_data.copy()
        b_data = r_data.copy()

        a_data = np.ones_like(self.data) * OPAQUE
        a_data[0, 0] = TRANSPARENT

        to_numpy_result = np.stack([r_data, g_data, b_data, a_data])

        ds.to_numpy.return_value = to_numpy_result

        # expect the input data to have the data values from 0 to 400 to be
        # scaled into the range 0 to 255.
        expected_raster = np.around(
            np.interp(to_numpy_result, (0, 400), (0.0, 1.0)) * 255.0
        ).astype('uint8')

        actual_raster = convert_mulitband_to_raster(ds)
        assert_array_equal(expected_raster, actual_raster.data, strict=True)

    def test_convert_4_multiband_masked_to_raster(self):
        """4-band images are returned with nan -> 0"""
        ds = Mock(DataArray)
        ds.rio.count = 4
        nan = np.nan
        input_array = np.array(
            [
                [
                    [nan, nan, nan, 234.0],
                    [nan, nan, nan, 225.0],
                    [nan, nan, 255.0, 215.0],
                    [nan, nan, 217.0, 255.0],
                ],
                [
                    [nan, nan, nan, 255.0],
                    [nan, nan, nan, 255.0],
                    [nan, nan, 255.0, 255.0],
                    [nan, nan, 255.0, 255.0],
                ],
                [
                    [nan, nan, nan, 234.0],
                    [nan, nan, nan, 225.0],
                    [nan, nan, 255.0, 215.0],
                    [nan, nan, 217.0, 255.0],
                ],
                [
                    [0.0, 0.0, 0.0, 255.0],
                    [0.0, 0.0, 0.0, 255.0],
                    [0.0, 0.0, 255.0, 255.0],
                    [0.0, 0.0, 255.0, 255.0],
                ],
            ],
            dtype=np.float32,
        )
        ds.to_numpy.return_value = input_array

        expected_raster = np.array(
            [
                [
                    [0, 0, 0, 234],
                    [0, 0, 0, 225],
                    [0, 0, 255, 215],
                    [0, 0, 217, 255],
                ],
                [
                    [0, 0, 0, 255],
                    [0, 0, 0, 255],
                    [0, 0, 255, 255],
                    [0, 0, 255, 255],
                ],
                [
                    [0, 0, 0, 234],
                    [0, 0, 0, 225],
                    [0, 0, 255, 215],
                    [0, 0, 217, 255],
                ],
                [
                    [0, 0, 0, 255],
                    [0, 0, 0, 255],
                    [0, 0, 255, 255],
                    [0, 0, 255, 255],
                ],
            ],
            dtype=np.uint8,
        )

        actual_raster = convert_mulitband_to_raster(ds)
        assert_array_equal(expected_raster.data, actual_raster.data, strict=True)

    def test_convert_5_multiband_to_raster(self):
        ds = Mock(DataArray)
        ds.rio.count = 5
        ds.to_numpy.return_value = np.stack(
            [self.data, self.data, self.data, self.data, self.data]
        )

        with self.assertRaises(HyBIGError) as excepted:
            convert_mulitband_to_raster(ds)

        self.assertEqual(
            excepted.exception.message,
            'Cannot create image from 5 band image. Expecting 3 or 4 bands.',
        )

    def test_prepare_raster_for_writing_jpeg_3band(self):
        raster = self.random.integers(255, size=(3, 5, 6))
        count = 'irrelevant'
        driver = 'JPEG'
        expected_raster = np.copy(raster)
        expected_color_map = None

        actual_raster, actual_color_map = prepare_raster_for_writing(
            raster, driver, count
        )
        self.assertEqual(expected_color_map, actual_color_map)
        np.testing.assert_array_equal(expected_raster, actual_raster, strict=True)

    def test_prepare_raster_for_writing_jpeg_4band(self):
        raster = self.random.integers(255, size=(4, 7, 8))
        driver = 'JPEG'
        count = 'irrelevant'
        expected_raster = np.copy(raster[0:3, :, :])
        expected_color_map = None
        actual_raster, actual_color_map = prepare_raster_for_writing(
            raster, driver, count
        )
        self.assertEqual(expected_color_map, actual_color_map)
        np.testing.assert_array_equal(expected_raster, actual_raster, strict=True)

    @patch('hybig.browse.palettize_raster')
    def test_prepare_raster_for_writing_png_4band(self, palettize_mock):
        raster = self.random.integers(255, size=(4, 7, 8))
        driver = 'PNG'
        count = 'not 1'

        expected, _ = prepare_raster_for_writing(raster, driver, count)
        np.testing.assert_array_equal(raster, expected, strict=True)
        palettize_mock.assert_not_called()

    @patch('hybig.browse.palettize_raster')
    def test_prepare_raster_for_writing_png_3band(self, palettize_mock):
        raster = self.random.integers(255, size=(3, 7, 8))
        driver = 'PNG'
        count = 'not 1'

        expected, _ = prepare_raster_for_writing(raster, driver, count)
        np.testing.assert_array_equal(raster, expected, strict=True)
        palettize_mock.assert_not_called()

    @patch('hybig.browse.palettize_raster')
    def test_prepare_1band_raster_for_writing_png(self, palettize_mock):
        raster = self.random.integers(255, size=(1, 7, 8))
        driver = 'PNG'
        count = 1
        palettize_mock.return_value = (None, None)
        expected, _ = prepare_raster_for_writing(raster, driver, count)
        palettize_mock.assert_called_with(raster)

    @patch('hybig.browse.Image')
    @patch('hybig.browse.get_color_map_from_image')
    def test_palettize_raster_no_alpha_layer(self, get_color_map_mock, image_mock):
        """Test that the quantize function is called by a correct image."""
        raster = self.random.integers(255, dtype='uint8', size=(3, 10, 11))

        quantized_output = Image.fromarray(
            self.random.integers(254, size=(10, 11), dtype='uint8')
        )
        multiband_image_mock = Mock()
        image_mock.fromarray.return_value = multiband_image_mock
        multiband_image_mock.quantize.return_value = quantized_output

        expected_out_raster = np.array(quantized_output).reshape(1, 10, 11)

        out_raster, out_map = palettize_raster(raster)

        multiband_image_mock.quantize.assert_called_once_with(colors=254)
        get_color_map_mock.assert_called_once_with(quantized_output)

        np.testing.assert_array_equal(expected_out_raster, out_raster, strict=True)

    @patch('hybig.browse.Image')
    @patch('hybig.browse.get_color_map_from_image')
    def test_palettize_raster_with_alpha_layer(self, get_color_map_mock, image_mock):
        """Test that the quantize function is called by a correct image."""
        raster = self.random.integers(255, dtype='uint8', size=(4, 10, 11))
        # No transparent pixels
        raster[3, :, :] = 255

        # corner transparent:
        raster[3, 0:3, 0:3] = 0

        quantized_output = Image.fromarray(
            self.random.integers(254, size=(10, 11), dtype='uint8')
        )
        multiband_image_mock = Mock()
        image_mock.fromarray.return_value = multiband_image_mock
        multiband_image_mock.quantize.return_value = quantized_output

        expected_out_raster = np.array(quantized_output).reshape(1, 10, 11)
        expected_out_raster[0, 0:3, 0:3] = 254

        out_raster, out_map = palettize_raster(raster)

        multiband_image_mock.quantize.assert_called_once_with(colors=254)
        get_color_map_mock.assert_called_once_with(quantized_output)

        np.testing.assert_array_equal(expected_out_raster, out_raster, strict=True)

    def test_get_color_map_from_image(self):
        """PIL Image yields a color_map

        A palette from an PIL Image is correctly turned into a colormap
        writable by rasterio.

        """
        # random image with values of 0 to 4.
        image_data = self.random.integers(5, size=(5, 6), dtype='uint8')
        # fmt: off
        palette_sequence = [
            255, 0, 0, 255,
            0, 255, 0, 255,
            0, 0, 255, 255,
            225, 100, 25, 25,
            0, 0, 0, 0
        ]
        # fmt: on
        test_image = Image.fromarray(image_data)
        test_image.putpalette(palette_sequence, rawmode='RGBA')

        expected_color_map = {
            **{
                0: (255, 0, 0, 255),
                1: (0, 255, 0, 255),
                2: (0, 0, 255, 255),
                3: (225, 100, 25, 25),
                4: (0, 0, 0, 0),
            },
            **{idx: (0, 0, 0, 255) for idx in range(5, 256)},
        }

        actual_color_map = get_color_map_from_image(test_image)
        self.assertDictEqual(expected_color_map, actual_color_map)

    def test_get_color_palette_map_exists_source_does_not(self):
        ds = Mock(DatasetReader)
        ds.colormap.return_value = self.colormap

        lines = [
            '100 255 0 0 255',
            '200 255 255 0 255',
            '300 0 255 0 255',
            '400 0 0 255 255',
        ]

        expected_palette = ColorPalette()
        expected_palette.read_file_txt(lines=lines)

        actual_palette = get_color_palette(ds, HarmonySource({}))

        self.assertEqual(expected_palette, actual_palette)

    def test_get_color_palette_source_and_map_do_not_exist(self):
        """get_color_palette returns None

        when source, Item and geotiff have no color information.
        """
        ds = Mock(DatasetReader)
        ds.colormap.side_effect = ValueError

        self.assertIsNone(get_color_palette(ds, HarmonySource({}), None))

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

    def test_get_tiled_filename(self):
        filename = Path('/path/to/some/location.png')

        with self.subTest('no locator'):
            actual_filename = get_tiled_filename(filename)
            self.assertEqual(filename, actual_filename)

        with self.subTest('with locator'):
            locator = {'row': 1, 'col': 10}
            expected_filename = Path('/path/to/some/location.r01c10.png')
            actual_filename = get_tiled_filename(filename, locator)
            self.assertEqual(expected_filename, actual_filename)

    def test_validate_file_crs_valid(self):
        """Valid file should return None."""
        da = Mock(DataArray)
        da.rio.crs = CRS.from_epsg(4326)
        try:
            validate_file_crs(da)
        except Exception:
            self.fail('Valid file threw unexpected exception.')

    def test_validate_file_crs_missing(self):
        """Invalid file should raise exception."""
        da = Mock(DataArray)
        da.rio.crs = None
        with self.assertRaisesRegex(HyBIGError, 'Input geotiff must have defined CRS.'):
            validate_file_crs(da)

    def test_validate_file_type_valid(self):
        """Validation should not raise exception."""
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
            HyBIGError, 'Input file type not supported: NetCDF4'
        ):
            validate_file_type(ds)

    @patch('hybig.color_utility.requests.get')
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

            with self.assertRaises(HyBIGError) as excepted:
                palette_from_remote_colortable(url)

            self.assertEqual(
                excepted.exception.message,
                (
                    'Failed to retrieve color table at'
                    ' http://this-domain-does-not-exist.com/bad-url'
                ),
            )


class TestCreateBrowse(TestCase):
    """A class testing the create_browse function call.

    Ensure library calls the `create_browse_imagery` function the same as the
    service.

    """

    @patch('hybig.browse.create_browse_imagery')
    def test_calls_create_browse_with_correct_params(self, mock_create_browse_imagery):
        """Ensure correct harmony message is created from inputs."""
        source_tiff = '/Path/to/source.tiff'
        params = {
            'mime': 'image/png',
            'crs': {'epsg': 'EPSG:4326'},
            'scale_extent': {
                'x': {'min': -180, 'max': 180},
                'y': {'min': -90, 'max': 90},
            },
            'scale_size': {'x': 10, 'y': 10},
        }
        mock_logger = MagicMock(spec=Logger)
        mock_palette = MagicMock(spec=ColorPalette)

        create_browse(source_tiff, params, mock_palette, mock_logger)

        mock_create_browse_imagery.assert_called_once()
        call_args = mock_create_browse_imagery.call_args[0]
        self.assertIsInstance(call_args[0], HarmonyMessage)
        self.assertEqual(call_args[1], source_tiff)
        self.assertIsInstance(call_args[2], HarmonySource)
        self.assertEqual(call_args[3], mock_palette)
        self.assertEqual(call_args[4], mock_logger)

        # verify message params.
        harmony_message = call_args[0]
        harmony_format = harmony_message.format

        # HarmonyMessage.Format does not have a json representation to compare
        # to so compare the pieces individually.
        self.assertEqual(harmony_format.mime, 'image/png')
        self.assertEqual(harmony_format['crs'], {'epsg': 'EPSG:4326'})
        self.assertEqual(harmony_format['srs'], {'epsg': 'EPSG:4326'})
        self.assertEqual(
            harmony_format['scaleExtent'],
            {
                'x': {'min': -180, 'max': 180},
                'y': {'min': -90, 'max': 90},
            },
        )
        self.assertEqual(harmony_format['scaleSize'], {'x': 10, 'y': 10})
        self.assertIsNone(harmony_message['format']['height'])
        self.assertIsNone(harmony_message['format']['width'])

    @patch('hybig.browse.palette_from_remote_colortable')
    @patch('hybig.browse.create_browse_imagery')
    def test_calls_create_browse_with_remote_palette(
        self, mock_create_browse_imagery, mock_palette_from_remote_color_table
    ):
        """Ensure remote palette is used."""
        mock_palette = MagicMock(sepc=ColorPalette)
        mock_palette_from_remote_color_table.return_value = mock_palette
        remote_color_url = 'https://path/to/colormap.txt'
        source_tiff = '/Path/to/source.tiff'
        mock_logger = MagicMock(spec=Logger)

        # Act
        create_browse(source_tiff, {}, remote_color_url, mock_logger)

        # Assert a remote colortable was fetched.
        mock_palette_from_remote_color_table.assert_called_once_with(remote_color_url)

        mock_create_browse_imagery.assert_called_once()
        (
            call_harmony_message,
            call_source_tiff,
            call_harmony_source,
            call_color_palette,
            call_logger,
        ) = mock_create_browse_imagery.call_args[0]

        # create_browse_imagery called with the color palette returned from
        # palette_from_remote_colortable
        self.assertEqual(call_color_palette, mock_palette)

        self.assertIsInstance(call_harmony_message, HarmonyMessage)
        self.assertIsInstance(call_harmony_source, HarmonySource)
        self.assertEqual(call_source_tiff, source_tiff)
        self.assertEqual(call_logger, mock_logger)
