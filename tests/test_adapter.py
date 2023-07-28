""" End-to-end tests of the Harmony Browse Image Generator (HyBIG). """
from pathlib import Path
from shutil import copy, rmtree
from tempfile import mkdtemp
from unittest import TestCase
from unittest.mock import call, patch

import numpy as np
from harmony.message import Message
from harmony.util import config
from pystac import Catalog
from rasterio import open as rasterio_open
from rasterio.transform import from_bounds
from rasterio.warp import Resampling

from harmony_browse_image_generator.adapter import BrowseImageGeneratorAdapter
from harmony_browse_image_generator.browse import convert_mulitband_to_raster
from tests.utilities import Granule, create_stac


class TestAdapter(TestCase):
    """ A class testing the harmony_browse_image_generator.adapter module. """
    @classmethod
    def setUpClass(cls):
        """ Define test fixtures that can be shared between tests. """
        cls.access_token = 'fake-token'
        cls.granule_url = 'https://www.example.com/input.tiff'
        cls.input_stac = create_stac(Granule(cls.granule_url,
                                             'image/tiff',
                                             ['data']))
        cls.staging_location = 's3://example-bucket'
        cls.fixtures = Path(__file__).resolve().parent / 'fixtures'
        cls.red_tif_fixture = cls.fixtures / 'red.tif'
        cls.user = 'blightyear'

    def setUp(self):
        """ Define test fixtures that are not shared between tests. """
        self.temp_dir = Path(mkdtemp())
        self.config = config(validate=False)

    def tearDown(self):
        if self.temp_dir.exists():
            rmtree(self.temp_dir)

    def assert_expected_output_catalog(self, catalog: Catalog,
                                       expected_browse_href: str,
                                       expected_browse_title: str,
                                       expected_browse_media_type: str,
                                       expected_world_href: str,
                                       expected_world_title: str,
                                       expected_world_media_type: str,
                                       expected_aux_href: str,
                                       expected_aux_title: str,
                                       expected_aux_media_type: str,
                                       ):
        """ Check the contents of the Harmony output STAC. It should have a
            single data item. The URL, title and media type for this asset will
            be compared to supplied values.

        """
        items = list(catalog.get_items())
        self.assertEqual(len(items), 1)
        self.assertListEqual(list(items[0].assets.keys()),
                             ['data', 'metadata', 'auxiliary'])
        self.assertDictEqual(
            items[0].assets['data'].to_dict(),
            {'href': expected_browse_href,
             'title': expected_browse_title,
             'type': expected_browse_media_type,
             'roles': ['data']}
        )

        self.assertDictEqual(
            items[0].assets['metadata'].to_dict(),
            {'href': expected_world_href,
             'title': expected_world_title,
             'type': expected_world_media_type,
             'roles': ['metadata']}
        )

        self.assertDictEqual(
            items[0].assets['auxiliary'].to_dict(), {
                'href': expected_aux_href,
                'title': expected_aux_title,
                'type': expected_aux_media_type,
                'roles': ['metadata']
            })

    @patch('harmony_browse_image_generator.browse.reproject')
    @patch('harmony_browse_image_generator.adapter.rmtree')
    @patch('harmony_browse_image_generator.adapter.mkdtemp')
    @patch('harmony_browse_image_generator.adapter.download')
    @patch('harmony_browse_image_generator.adapter.stage')
    def test_valid_request(self, mock_stage, mock_download, mock_mkdtemp,
                           mock_rmtree, mock_reproject):
        """ Ensure a request with a correctly formatted message is fully
            processed.

            This test will need updating when the service functions fully.

        """
        expected_downloaded_file = self.temp_dir / 'input.tiff'

        expected_browse_basename = 'input.png'
        expected_browse_full_path = self.temp_dir / 'input.png'

        expected_aux_basename = 'input.png.aux.xml'
        expected_aux_full_path = self.temp_dir / 'input.png.aux.xml'

        expected_world_basename = 'input.pgw'
        expected_world_full_path = self.temp_dir / 'input.pgw'

        expected_browse_url = f'{self.staging_location}/{expected_browse_basename}'
        expected_world_url = f'{self.staging_location}/{expected_world_basename}'
        expected_aux_url = f'{self.staging_location}/{expected_aux_basename}'

        expected_browse_mime = 'image/png'
        expected_world_mime = 'text/plain'
        expected_aux_mime = 'application/xml'

        mock_mkdtemp.return_value = self.temp_dir

        def move_tif(*args, **kwargs):
            """copy fixture tiff to download location. """
            copy(self.red_tif_fixture, expected_downloaded_file)
            return expected_downloaded_file
        mock_download.side_effect = move_tif
        mock_stage.side_effect = [
            expected_browse_url, expected_aux_url, expected_world_url
        ]

        message = Message({
            'accessToken': self.access_token,
            'callback': 'https://example.com/',
            'sources': [{'collection': 'C1234-EEDTEST', 'shortName': 'test'}],
            'stagingLocation': self.staging_location,
            'user': self.user,
            'format': {'mime': 'image/png'},
        })

        hybig = BrowseImageGeneratorAdapter(message, config=self.config,
                                            catalog=self.input_stac)

        _, output_catalog = hybig.invoke()

        # Ensure the output catalog contains the single, expected item:
        self.assert_expected_output_catalog(output_catalog,
                                            expected_browse_url,
                                            expected_browse_basename,
                                            expected_browse_mime,
                                            expected_world_url,
                                            expected_world_basename,
                                            expected_world_mime,
                                            expected_aux_url,
                                            expected_aux_basename,
                                            expected_aux_mime,
                                            )

        # Ensure a download was requested via harmony-service-lib:
        mock_download.assert_called_once_with(self.granule_url, self.temp_dir,
                                              logger=hybig.logger,
                                              cfg=hybig.config,
                                              access_token=self.access_token)

        # Set up for testing reprojection validation calls.
        # All defaults will be computed
        # input CRS is preferred 4326,
        # Scale Extent from icd
        # dimensions from input data
        in_dataset = rasterio_open(self.red_tif_fixture)
        icd_scale_extent = {'xmin': -180.0, 'ymin': -90.0, 'xmax': 180.0, 'ymax': 90.0}
        expected_transform = from_bounds(
            icd_scale_extent['xmin'],
            icd_scale_extent['ymin'],
            icd_scale_extent['xmax'],
            icd_scale_extent['ymax'],
            in_dataset.width,
            in_dataset.height,
        )

        expected_params = {'width': in_dataset.width,
                           'height': in_dataset.height,
                           'crs': in_dataset.crs,
                           'transform': expected_transform,
                           'driver': 'PNG',
                           'dtype': 'uint8',
                           'count': 3}
        raster = convert_mulitband_to_raster(in_dataset)

        dest = np.full((expected_params['height'], expected_params['width']),
                       dtype='uint8', fill_value=0)

        expected_reproject_calls = [
            call(source=raster[0, :, :],
                 destination=dest,
                 src_transform=in_dataset.transform,
                 src_crs=in_dataset.crs,
                 dst_transform=expected_params['transform'],
                 dst_crs=expected_params['crs'],
                 resampling=Resampling.nearest),
            call(source=raster[1, :, :],
                 destination=dest,
                 src_transform=in_dataset.transform,
                 src_crs=in_dataset.crs,
                 dst_transform=expected_params['transform'],
                 dst_crs=expected_params['crs'],
                 resampling=Resampling.nearest),
            call(source=raster[2, :, :],
                 destination=dest,
                 src_transform=in_dataset.transform,
                 src_crs=in_dataset.crs,
                 dst_transform=expected_params['transform'],
                 dst_crs=expected_params['crs'],
                 resampling=Resampling.nearest)
        ]


        self.assertEqual(mock_reproject.call_count, 3)
        for actual_call, expected_call in zip(mock_reproject.call_args_list,
                                              expected_reproject_calls):
            np.testing.assert_array_equal(actual_call.kwargs['source'],
                                          expected_call.kwargs['source'])
            np.testing.assert_array_equal(actual_call.kwargs['destination'],
                                          expected_call.kwargs['destination'])
            self.assertEqual(actual_call.kwargs['src_transform'],
                             expected_call.kwargs['src_transform'])
            self.assertEqual(actual_call.kwargs['src_crs'],
                             expected_call.kwargs['src_crs'])
            self.assertEqual(actual_call.kwargs['dst_transform'],
                             expected_call.kwargs['dst_transform'])
            self.assertEqual(actual_call.kwargs['dst_crs'],
                             expected_call.kwargs['dst_crs'])
            self.assertEqual(actual_call.kwargs['resampling'],
                             expected_call.kwargs['resampling'])

        # Ensure the browse image and ESRI world file were staged as expected:
        # TODO: "expected_downloaded_files" arguments will need updating when
        # the service processes anything.
        mock_stage.assert_has_calls([
            call(expected_browse_full_path,
                 expected_browse_basename,
                 expected_browse_mime,
                 logger=hybig.logger,
                 location=self.staging_location,
                 cfg=self.config),
            call(expected_aux_full_path,
                 expected_aux_basename,
                 expected_aux_mime,
                 logger=hybig.logger,
                 location=self.staging_location,
                 cfg=self.config),
            call(expected_world_full_path,
                 expected_world_basename,
                 expected_world_mime,
                 logger=hybig.logger,
                 location=self.staging_location,
                 cfg=self.config)
        ])

        # Ensure container clean-up was requested:
        mock_rmtree.assert_called_once_with(self.temp_dir)
