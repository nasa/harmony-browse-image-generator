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
from numpy.testing import assert_array_equal, assert_equal
from osgeo_utils.auxiliary.color_palette import ColorPalette
from rasterio import Affine
from rasterio.crs import CRS
from rasterio.io import DatasetReader, DatasetWriter
from rasterio.warp import Resampling

from hybig.browse import (
    convert_multiband_to_raster,
    convert_singleband_to_raster,
    create_browse,
    create_browse_imagery,
    get_tiled_filename,
    output_image_file,
    output_world_file,
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

        mock_logger = MagicMock(spec=Logger)

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
                    message, test_tif_filename, HarmonySource({}), None, mock_logger
                )

    @patch('hybig.browse.reproject')
    @patch('rasterio.open')
    def test_create_browse_imagery_with_mocks(self, rasterio_open_mock, reproject_mock):
        file_transform = Affine(90.0, 0.0, -180.0, 0.0, -45.0, 90.0)
        ds = Mock(spec=DatasetReader)

        dest_write_mock = Mock(spec=DatasetWriter)

        ds.read.return_value = self.data[np.newaxis, :, :]
        ds.driver = 'GTiff'
        ds.shape = (4, 4)
        ds.transform = file_transform
        ds.crs = CRS.from_string('EPSG:4326')
        ds.count = 1
        ds.colormap = Mock(side_effect=ValueError)
        ds.bounds = (-180.0, -90.0, 180.0, 90.0)
        ds.window_transform = Mock(return_value=file_transform)
        ds.nodatavals = (255,)
        ds.scales = (1,)
        ds.offsets = (0,)

        expected_raster = np.array(
            [
                [
                    [0, 85, 169, 254],
                    [0, 85, 169, 254],
                    [0, 85, 169, 254],
                    [0, 85, 169, 254],
                ],
            ],
            dtype='uint8',
        )

        rasterio_open_mock.return_value.__enter__.side_effect = [
            ds,
            dest_write_mock,
        ]

        message = HarmonyMessage({'format': {'mime': 'JPEG'}})

        # More detailed traceback for this test since it's end-to-end
        try:
            out_file_list = create_browse_imagery(
                message,
                str(self.tmp_dir / 'input_file_path'),
                HarmonySource({}),
                None,
                self.logger,
            )
        except HyBIGError as e:
            import traceback

            print('\n=== Full Traceback ===')
            traceback.print_exc()
            print('\n=== Exception Chain ===')
            print(f'HyBIGError: {e}')
            if e.__cause__:
                print(f'Caused by: {type(e.__cause__).__name__}: {e.__cause__}')
            raise

        # Ensure tiling logic was not called:
        self.assertEqual(len(out_file_list), 1)

        actual_image, actual_world, actual_aux = out_file_list[0]

        target_transform = Affine(90.0, 0.0, -180.0, 0.0, -45.0, 90.0)
        dest = np.zeros((ds.shape[0], ds.shape[1]), dtype='uint8')

        # For JPEG output with 1-band input, we convert to RGB, so we reproject 3 bands
        self.assertEqual(reproject_mock.call_count, 3)

        # For RGB output, we expect 3 calls (one per band) with TRANSPARENT nodata
        expected_calls = [
            call(
                source=expected_raster[0, :, :],
                destination=dest,
                src_transform=file_transform,
                src_crs=ds.crs,
                dst_transform=target_transform,
                dst_crs=CRS.from_string('EPSG:4326'),
                dst_nodata=0,  # TRANSPARENT for RGB data
                resampling=Resampling.nearest,
            ),
            call(
                source=expected_raster[0, :, :],
                destination=dest,
                src_transform=file_transform,
                src_crs=ds.crs,
                dst_transform=target_transform,
                dst_crs=CRS.from_string('EPSG:4326'),
                dst_nodata=0,  # TRANSPARENT for RGB data
                resampling=Resampling.nearest,
            ),
            call(
                source=expected_raster[0, :, :],
                destination=dest,
                src_transform=file_transform,
                src_crs=ds.crs,
                dst_transform=target_transform,
                dst_crs=CRS.from_string('EPSG:4326'),
                dst_nodata=0,  # TRANSPARENT for RGB data
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
        """Tests scale_grey_1band."""
        return_data = np.copy(self.data).astype('float64')
        return_data[0][1] = np.nan
        return_data = return_data[np.newaxis, :, :]
        # ds = DataArray(return_data).expand_dims('band')

        expected_raster = np.array(
            [
                [
                    [0, 255, 169, 254],
                    [0, 85, 169, 254],
                    [0, 85, 169, 254],
                    [0, 85, 169, 254],
                ],
            ],
            dtype='uint8',
        )
        actual_raster, _, _ = convert_singleband_to_raster(return_data, None)
        assert_array_equal(expected_raster, actual_raster, strict=True)

    def test_convert_singleband_to_raster_with_colormap(self):
        expected_raster = np.array(
            [
                [  # singleband paletted
                    [0, 1, 2, 3],
                    [0, 1, 2, 3],
                    [0, 1, 2, 3],
                    [0, 1, 2, 3],
                ],
            ],
            dtype='uint8',
        )
        expected_palette = {
            0: (255, 0, 0, 255),  # red
            1: (255, 255, 0, 255),  # yellow
            2: (0, 255, 0, 255),  # green
            3: (0, 0, 255, 255),  # blue
            4: (0, 0, 0, 0),  # alpha
        }
        # Read down: red, yellow, green, blue
        image_palette = convert_colormap_to_palette(self.colormap)
        # functional equivalent of DataArray().expand_dims("bands")
        actual_raster, actual_palette, _ = convert_singleband_to_raster(
            self.data[np.newaxis, :, :], image_palette
        )
        assert_array_equal(expected_raster, actual_raster, strict=True)
        assert_equal(expected_palette, actual_palette)

    def test_convert_singleband_to_raster_with_colormap_and_bad_data(self):
        data_array = np.array(self.data, dtype='float')
        data_array[0, 0] = np.nan
        data_array = data_array[np.newaxis, :, :]
        nv_color = (10, 20, 30, 40)

        # Read the image down: red, yellow, green, blue
        expected_raster = np.array(
            [
                [  # singleband paletted
                    [0, 2, 3, 4],
                    [1, 2, 3, 4],
                    [1, 2, 3, 4],
                    [1, 2, 3, 4],
                ],
            ],
            dtype='uint8',
        )
        expected_palette = {
            0: (10, 20, 30, 40),  # nv
            1: (255, 0, 0, 255),  # red
            2: (255, 255, 0, 255),  # yellow
            3: (0, 255, 0, 255),  # green
            4: (0, 0, 255, 255),  # blue
        }

        colormap = {**self.colormap, 'nv': nv_color}

        image_palette = convert_colormap_to_palette(colormap)
        actual_raster, actual_palette, _ = convert_singleband_to_raster(
            data_array, image_palette
        )
        assert_array_equal(expected_raster, actual_raster, strict=True)
        assert_equal(expected_palette, actual_palette)

    def test_convert_uint16_3_multiband_to_raster(self):
        """Test that uint16 input scales the output."""
        bad_data = np.copy(self.data).astype('float64')
        bad_data[1][1] = np.nan
        bad_data[1][2] = np.nan
        data_array = np.stack([self.data, bad_data, self.data]).astype('float64')

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

        actual_raster = convert_multiband_to_raster(data_array)
        assert_array_equal(expected_raster, actual_raster, strict=True)

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

        data_array = np.stack([scale_data, scale_data, scale_data]).astype('float64')

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

        actual_raster = convert_multiband_to_raster(data_array)
        assert_array_equal(expected_raster, actual_raster, strict=True)

    def test_convert_4_multiband_uint8_to_raster(self):
        """4-band 'uint8' images are returned unchanged."""
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

        data_array = np.stack([r_data, g_data, b_data, a_data])

        expected_raster = data_array

        actual_raster = convert_multiband_to_raster(data_array)
        assert_array_equal(expected_raster, actual_raster, strict=True)

    def test_convert_4_multiband_uint16_to_raster(self):
        """4-band 'uint16' images are scaled if their range exceeds 255."""
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

        data_array = np.stack([r_data, g_data, b_data, a_data])

        # expect the input data to have the data values from 0 to 400 to be
        # scaled into the range 0 to 255.
        expected_raster = np.around(
            np.interp(data_array, (0, 400), (0.0, 1.0)) * 255.0
        ).astype('uint8')

        actual_raster = convert_multiband_to_raster(data_array)
        assert_array_equal(expected_raster, actual_raster, strict=True)

    def test_convert_4_multiband_masked_to_raster(self):
        """4-band images are returned with nan -> 0"""
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

        actual_raster = convert_multiband_to_raster(input_array)
        assert_array_equal(expected_raster, actual_raster, strict=True)

    def test_convert_5_multiband_to_raster(self):
        data_array = np.stack([self.data, self.data, self.data, self.data, self.data])

        with self.assertRaises(HyBIGError) as excepted:
            convert_multiband_to_raster(data_array)

        self.assertEqual(
            excepted.exception.message,
            'Cannot create image from 5 band image. Expecting 3 or 4 bands.',
        )

    def test_get_color_palette_map_exists_source_does_not(self):
        ds = Mock(DatasetReader)
        ds.colormap.return_value = self.colormap
        ds.get_nodatavals.return_value = ()

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
        ds = Mock(DatasetReader)
        ds.crs = CRS.from_epsg(4326)
        try:
            validate_file_crs(ds)
        except Exception:
            self.fail('Valid file threw unexpected exception.')

    def test_validate_file_crs_missing(self):
        """Invalid file should raise exception."""
        ds = Mock(DatasetReader)
        ds.crs = None
        with self.assertRaisesRegex(HyBIGError, 'Input geotiff must have defined CRS.'):
            validate_file_crs(ds)

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

    def test_scale_paletted_1band_clips_underflow_values(self):
        """Test that values below the palette min are clipped to lowest color."""
        from hybig.browse import scale_paletted_1band

        # Create test data with values below the palette minimum
        # Palette covers 100-400, but data includes -50 and 50
        data_with_underflow = np.array(
            [
                [-50, 50, 100, 200],
                [100, 200, 300, 400],
                [50, 100, 200, 300],
                [-100, 0, 150, 250],
            ]
        ).astype('float64')
        # functional equivalent of DataArray().expand_dims("bands")
        data_with_underflow = data_with_underflow[np.newaxis, :, :]

        # Expected: underflow values (-50, 50, 0, -100) should map to index 0
        # which is the lowest color (red at value 100)
        expected_raster = np.array(
            [
                [
                    [0, 0, 0, 1],  # -50, 50 -> 0 (red), 100->0, 200->1
                    [0, 1, 2, 3],  # 100->0, 200->1, 300->2, 400->3
                    [0, 0, 1, 2],  # 50, 100 -> 0, 200->1, 300->2
                    [0, 0, 0, 1],  # -100, 0 -> 0, 150->0, 250->1
                ],
            ],
            dtype='uint8',
        )

        image_palette = convert_colormap_to_palette(self.colormap)
        actual_raster, _, _ = scale_paletted_1band(data_with_underflow, image_palette)
        assert_array_equal(expected_raster, actual_raster, strict=True)

    def test_scale_paletted_1band_clips_overflow_values(self):
        """Test that values above the palette max are clipped to highest color."""
        from hybig.browse import scale_paletted_1band

        # Create test data with values above the palette maximum
        # Palette covers 100-400, but data includes 500 and 1000
        data_with_overflow = np.array(
            [
                [100, 200, 300, 400],
                [400, 500, 600, 1000],
                [200, 300, 400, 500],
                [300, 350, 400, 800],
            ]
        ).astype('float64')
        # functional equivalent of DataArray().expand_dims("bands")
        data_with_overflow = data_with_overflow[np.newaxis, :, :]

        # Expected: overflow values (500, 600, 1000, 800) should map to index 3
        # which is the highest color (blue at value 400)
        expected_raster = np.array(
            [
                [
                    [0, 1, 2, 3],  # 100->0, 200->1, 300->2, 400->3
                    [3, 3, 3, 3],  # 400->3, 500->3, 600->3, 1000->3
                    [1, 2, 3, 3],  # 200->1, 300->2, 400->3, 500->3
                    [2, 2, 3, 3],  # 300->2, 350->2, 400->3, 800->3
                ],
            ],
            dtype='uint8',
        )

        image_palette = convert_colormap_to_palette(self.colormap)
        actual_raster, _, _ = scale_paletted_1band(data_with_overflow, image_palette)
        assert_array_equal(expected_raster, actual_raster, strict=True)

    def test_scale_paletted_1band_with_nan_and_clipping(self):
        """Test that NaN values map to nodata while clipping still works."""
        from hybig.browse import scale_paletted_1band

        # Create test data with NaN, underflow, and overflow values
        data_mixed = np.array(
            [
                [np.nan, -50, 100, 500],
                [50, 200, 300, 1000],
                [100, np.nan, 400, 600],
                [-100, 250, np.nan, 800],
            ]
        ).astype('float64')
        # functional equivalent of DataArray().expand_dims("bands")
        data_mixed = data_mixed[np.newaxis, :, :]

        # Expected: NaN -> 4 (nodata), underflow -> 0, overflow -> 3
        expected_raster = np.array(
            [
                [
                    [4, 0, 0, 3],  # NaN->4, -50->0, 100->0, 500->3
                    [0, 1, 2, 3],  # 50->0, 200->1, 300->2, 1000->3
                    [0, 4, 3, 3],  # 100->0, NaN->4, 400->3, 600->3
                    [0, 1, 4, 3],  # -100->0, 250->1, NaN->4, 800->3
                ],
            ],
            dtype='uint8',
        )

        image_palette = convert_colormap_to_palette(self.colormap)
        actual_raster, actual_palette, _ = scale_paletted_1band(
            data_mixed, image_palette
        )
        assert_array_equal(expected_raster, actual_raster, strict=True)

        # Verify nodata color is transparent
        expected_nodata_color = (0, 0, 0, 0)
        self.assertEqual(actual_palette[np.uint8(4)], expected_nodata_color)

    def test_scale_paletted_1band_to_rgb_clips_underflow_values(self):
        """Test RGB output clips values below palette min to lowest color."""
        from hybig.browse import scale_paletted_1band_to_rgb

        # Create test data with values below the palette minimum
        data_with_underflow = np.array(
            [
                [-50, 50, 100, 200],
                [100, 200, 300, 400],
            ]
        ).astype('float64')
        # functional equivalent of DataArray().expand_dims("bands")
        data_with_underflow = data_with_underflow[np.newaxis, :, :]

        image_palette = convert_colormap_to_palette(self.colormap)
        actual_rgb, _ = scale_paletted_1band_to_rgb(data_with_underflow, image_palette)

        # Values -50 and 50 should get red color (255, 0, 0)
        # which is the lowest color in the palette
        self.assertEqual(actual_rgb[0, 0, 0], 255)  # Red channel for -50
        self.assertEqual(actual_rgb[1, 0, 0], 0)  # Green channel for -50
        self.assertEqual(actual_rgb[2, 0, 0], 0)  # Blue channel for -50

        self.assertEqual(actual_rgb[0, 0, 1], 255)  # Red channel for 50
        self.assertEqual(actual_rgb[1, 0, 1], 0)  # Green channel for 50
        self.assertEqual(actual_rgb[2, 0, 1], 0)  # Blue channel for 50

    def test_scale_paletted_1band_to_rgb_clips_overflow_values(self):
        """Test RGB output clips values above palette max to highest color."""
        from hybig.browse import scale_paletted_1band_to_rgb

        # Create test data with values above the palette maximum
        data_with_overflow = np.array(
            [
                [400, 500, 600, 1000],
                [300, 400, 800, 1500],
            ]
        ).astype('float64')
        # functional equivalent of DataArray().expand_dims("bands")
        data_with_overflow = data_with_overflow[np.newaxis, :, :]

        image_palette = convert_colormap_to_palette(self.colormap)
        actual_rgb, _ = scale_paletted_1band_to_rgb(data_with_overflow, image_palette)

        # Values 500, 600, 1000, 800, 1500 should get blue color (0, 0, 255)
        # which is the highest color in the palette
        for col in [1, 2, 3]:  # columns 1, 2, 3 in row 0
            self.assertEqual(actual_rgb[0, 0, col], 0)  # Red channel
            self.assertEqual(actual_rgb[1, 0, col], 0)  # Green channel
            self.assertEqual(actual_rgb[2, 0, col], 255)  # Blue channel

        for col in [2, 3]:  # columns 2, 3 in row 1
            self.assertEqual(actual_rgb[0, 1, col], 0)  # Red channel
            self.assertEqual(actual_rgb[1, 1, col], 0)  # Green channel
            self.assertEqual(actual_rgb[2, 1, col], 255)  # Blue channel

    def test_scale_paletted_1band_to_rgb_with_nan_and_clipping(self):
        """Test RGB output with NaN mapped to nodata and clipping working."""
        from hybig.browse import scale_paletted_1band_to_rgb

        # Create test data with NaN, underflow, and overflow values
        data_mixed = np.array(
            [
                [np.nan, -50, 100, 500],
                [50, 200, np.nan, 1000],
            ]
        ).astype('float64')
        # functional equivalent of DataArray().expand_dims("bands")
        data_mixed = data_mixed[np.newaxis, :, :]

        image_palette = convert_colormap_to_palette(self.colormap)
        actual_rgb, _ = scale_paletted_1band_to_rgb(data_mixed, image_palette)

        # NaN should map to nodata color (0, 0, 0)
        self.assertEqual(actual_rgb[0, 0, 0], 0)  # Red for NaN at (0,0)
        self.assertEqual(actual_rgb[1, 0, 0], 0)  # Green for NaN at (0,0)
        self.assertEqual(actual_rgb[2, 0, 0], 0)  # Blue for NaN at (0,0)

        self.assertEqual(actual_rgb[0, 1, 2], 0)  # Red for NaN at (1,2)
        self.assertEqual(actual_rgb[1, 1, 2], 0)  # Green for NaN at (1,2)
        self.assertEqual(actual_rgb[2, 1, 2], 0)  # Blue for NaN at (1,2)

        # -50 and 50 should clip to red (255, 0, 0)
        self.assertEqual(actual_rgb[0, 0, 1], 255)  # Red for -50
        self.assertEqual(actual_rgb[0, 1, 0], 255)  # Red for 50

        # 500 and 1000 should clip to blue (0, 0, 255)
        self.assertEqual(actual_rgb[2, 0, 3], 255)  # Blue for 500
        self.assertEqual(actual_rgb[2, 1, 3], 255)  # Blue for 1000


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
