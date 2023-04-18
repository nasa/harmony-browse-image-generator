from unittest import skip, TestCase

from harmony.message import Message
from harmony.util import config

from harmony_browse_image_generator.adapter import BrowseImageGeneratorAdapter
from tests.utilities import create_stac, Granule


class TestAdapter(TestCase):
    """ A class testing the harmony_browse_image_generator.adapter module. """
    @classmethod
    def setUpClass(cls):
        """ Define test fixtures that can be shared between tests. """
        cls.access_token = 'is_it_secret_is_it_safe?'
        cls.callback = 'callback'
        cls.config = config(validate=False)
        cls.staging_location = 'staging_location'
        cls.user = 'mmcfly'
        cls.input_stac = create_stac(Granule('www.example.com/file.nc4',
                                             'application/x-netcdf4',
                                             ['data']))

    @skip('Method not yet implemented')
    def test_validate_message(self):
        """ Ensure only messages with expected content will be processed. """

    def test_create_output_stac_item(self):
        """ Ensure a STAC item is created with Assets for both the browse image
            and ESRI world file.

        """
        input_stac_item = next(self.input_stac.get_items())
        message = Message({
            'accessToken': self.access_token,
            'callback': self.callback,
            'sources': [{'collection': 'C1234-EEEDTEST', 'shortName': 'test'}],
            'stagingLocation': self.staging_location,
            'user': self.user
        })
        adapter = BrowseImageGeneratorAdapter(message, config=self.config,
                                              catalog=self.input_stac)

        browse_image_url = f'{self.staging_location}/browse.png'
        esri_url = f'{self.staging_location}/browse.wld'
        output_stac_item = adapter.create_output_stac_item(
            input_stac_item, browse_image_url, 'image/tiff', esri_url,
            'text/plain'
        )

        # Check item has expected assets:
        self.assertListEqual(list(output_stac_item.assets.keys()),
                             ['data', 'metadata'])

        # Check the browse image asset
        self.assertEqual(output_stac_item.assets['data'].href,
                         browse_image_url)
        self.assertEqual(output_stac_item.assets['data'].media_type,
                         'image/tiff')
        self.assertEqual(output_stac_item.assets['data'].title,
                         'browse.png')

        # Check the world file asset
        self.assertEqual(output_stac_item.assets['metadata'].href,
                         esri_url)
        self.assertEqual(output_stac_item.assets['metadata'].media_type,
                         'text/plain')
        self.assertEqual(output_stac_item.assets['metadata'].title,
                         'browse.wld')
