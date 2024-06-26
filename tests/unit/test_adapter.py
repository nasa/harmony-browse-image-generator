from datetime import datetime
from unittest import TestCase
from unittest.mock import Mock

from harmony.message import Message
from harmony.util import config
from pystac import Asset, Item

from harmony_service_entry.adapter import BrowseImageGeneratorAdapter
from harmony_service_entry.exceptions import HyBIGInvalidMessageError
from tests.utilities import Granule, create_stac


class TestAdapter(TestCase):
    """A class testing the harmony_service_entry.adapter module."""

    @classmethod
    def setUpClass(cls):
        """Define test fixtures that can be shared between tests."""
        cls.access_token = 'is_it_secret_is_it_safe?'
        cls.callback = 'callback'
        cls.config = config(validate=False)
        cls.staging_location = 'staging_location'
        cls.user = 'mmcfly'
        cls.input_stac = create_stac(
            Granule('www.example.com/file.nc4', 'application/x-netcdf4', ['data'])
        )

    def test_validate_message_scale_extent_no_crs(self):
        """Ensure only messages with expected content will be processed."""
        message = Message(
            {
                'format': {
                    'scaleExtent': {
                        'x': {'min': 3, 'max': 10},
                        'y': {'min': 0, 'max': 10},
                    }
                }
            }
        )
        adapter = BrowseImageGeneratorAdapter(
            message, config=self.config, catalog=self.input_stac
        )
        with self.assertRaisesRegex(
            HyBIGInvalidMessageError,
            ('Harmony message must include a crs with scaleExtent or scaleSizes.'),
        ):
            adapter.validate_message()

    def test_validate_message_scale_size_no_crs(self):
        """Ensure only messages with expected content will be processed."""
        message = Message(
            {
                'format': {
                    'scaleSize': {
                        'x': 5,
                        'y': 5,
                    }
                }
            }
        )
        adapter = BrowseImageGeneratorAdapter(
            message, config=self.config, catalog=self.input_stac
        )
        with self.assertRaisesRegex(
            HyBIGInvalidMessageError,
            ('Harmony message must include a crs with scaleExtent or scaleSizes.'),
        ):
            adapter.validate_message()

    def test_validate_message_has_crs(self):
        """Ensure only messages with expected content will be processed."""
        message = Message(
            {
                'format': {
                    'crs': 'epsg:4326',
                    'scaleExtent': {
                        'x': {'min': 3, 'max': 10},
                        'y': {'min': 0, 'max': 10},
                    },
                }
            }
        )
        try:
            adapter = BrowseImageGeneratorAdapter(
                message, config=self.config, catalog=self.input_stac
            )
            adapter.validate_message()
        except Exception:
            self.fail('valid message threw exception')

    def test_validate_message_empty(self):
        """Ensure only messages with expected content will be processed."""
        message = Message({})
        try:
            adapter = BrowseImageGeneratorAdapter(
                message, config=self.config, catalog=self.input_stac
            )
            adapter.validate_message()
        except Exception:
            self.fail('valid message threw exception')

    def test_create_output_stac_items(self):
        """Ensure a STAC item is created with Assets for both the browse image
        and ESRI world file.

        """
        input_stac_item = next(self.input_stac.get_items())
        message = Message(
            {
                'accessToken': self.access_token,
                'callback': self.callback,
                'sources': [{'collection': 'C1234-EEEDTEST', 'shortName': 'test'}],
                'stagingLocation': self.staging_location,
                'user': self.user,
            }
        )
        adapter = BrowseImageGeneratorAdapter(
            message, config=self.config, catalog=self.input_stac
        )

        browse_image_url = f'{self.staging_location}/browse.png'
        esri_url = f'{self.staging_location}/browse.pgw'
        aux_url = f'{self.staging_location}/browse.png.aux.xml'

        output_stac_item = adapter.create_output_stac_item(
            input_stac_item,
            [
                ('data', browse_image_url, 'data'),
                ('metadata', esri_url, 'metadata'),
                ('auxiliary', aux_url, 'metadata'),
            ],
        )

        # Check item has expected assets:
        self.assertListEqual(
            list(output_stac_item.assets.keys()), ['data', 'metadata', 'auxiliary']
        )

        # Check the browse image asset
        self.assertEqual(output_stac_item.assets['data'].href, browse_image_url)
        self.assertEqual(output_stac_item.assets['data'].media_type, 'image/png')
        self.assertEqual(output_stac_item.assets['data'].title, 'browse.png')

        # Check the world file asset
        self.assertEqual(output_stac_item.assets['metadata'].href, esri_url)
        self.assertEqual(output_stac_item.assets['metadata'].media_type, 'text/plain')
        self.assertEqual(output_stac_item.assets['metadata'].title, 'browse.pgw')

        # Check the Aux file asset
        self.assertEqual(output_stac_item.assets['auxiliary'].href, aux_url)
        self.assertEqual(
            output_stac_item.assets['auxiliary'].media_type, 'application/xml'
        )
        self.assertEqual(
            output_stac_item.assets['auxiliary'].title, 'browse.png.aux.xml'
        )


class TestAdapterAssetFromItem(TestCase):
    """A class testing get_asset_from_item function."""

    def setUp(self):
        self.adapter = BrowseImageGeneratorAdapter({}, {})
        self.visual_asset = Asset(Mock(), roles=['visual'])
        self.data_asset = Asset(Mock(), roles=['data'])
        self.none_asset = Asset(Mock(), roles=[])
        self.other_asset = Asset(Mock(), roles=['other'])

    def item_fixture(self, assets: dict) -> Item:
        item = Item(Mock(), None, None, datetime.now(), {})
        item.assets = assets
        return item

    def test_get_asset_from_item_with_visual_role(self):
        with self.subTest('data asset first'):
            item = self.item_fixture(
                {'data': self.data_asset, 'visual': self.visual_asset}
            )
            expected = self.visual_asset

            actual = self.adapter.get_asset_from_item(item)

            self.assertEqual(expected, actual)

        with self.subTest('visual asset first'):
            item = self.item_fixture(
                {'visual': self.visual_asset, 'data': self.data_asset}
            )
            expected = self.visual_asset

            actual = self.adapter.get_asset_from_item(item)

            self.assertEqual(expected, actual)

    def test_get_asset_from_item_with_data_role(self):
        item = self.item_fixture({'data': self.data_asset, 'other': self.other_asset})
        expected = self.data_asset

        actual = self.adapter.get_asset_from_item(item)

        self.assertEqual(expected, actual)

    def test_get_asset_from_item_no_roles(self):
        item = self.item_fixture({'none': self.none_asset})
        with self.assertRaises(StopIteration):
            self.adapter.get_asset_from_item(item)

    def test_get_asset_from_item_no_matching_roles(self):
        item = self.item_fixture(
            {'first': self.other_asset, 'second': self.other_asset}
        )
        with self.assertRaises(StopIteration):
            self.adapter.get_asset_from_item(item)
