from datetime import datetime
from unittest import TestCase
from unittest.mock import Mock, patch

from harmony.message import Source as HarmonySource
from osgeo_utils.auxiliary.color_palette import ColorPalette
from pystac import Asset, Item
from rasterio import DatasetReader
from requests import Response

from hybig.color_utility import (
    convert_colormap_to_palette,
    get_color_palette,
    get_color_palette_from_item,
    get_remote_palette_from_source,
)
from hybig.exceptions import (
    HyBIGError,
    HyBIGNoColorInformation,
)


def encode_color(r, g, b, a=255):
    """How an rgb[a] triplet is coded for a palette."""
    return (((((int(a) << 8) + int(r)) << 8) + int(g)) << 8) + int(b)


class TestColorUtility(TestCase):
    """Test color utility routies."""

    @classmethod
    def setUpClass(cls):
        red = (255, 0, 0, 255)
        yellow = (255, 255, 0, 255)
        green = (0, 255, 0, 255)
        blue = (0, 0, 255, 255)

        cls.colormap = {100: red, 200: yellow, 300: green, 400: blue}

    def setUp(self):
        geometry = {}
        bbox = []
        date = datetime.now()
        props = {}
        self.item = Item('id', geometry, bbox, date, props)

    @patch('hybig.color_utility.palette_from_remote_colortable')
    def test_get_color_palette_from_item_with_no_assets(
        self, palette_from_remote_colortable_mock
    ):
        actual = get_color_palette_from_item(self.item)
        self.assertIsNone(actual)
        palette_from_remote_colortable_mock.assert_not_called()

    @patch('hybig.color_utility.palette_from_remote_colortable')
    def test_get_color_palette_from_item_no_palette_asset(
        self, palette_from_remote_colortable_mock
    ):
        asset = Asset('data href', roles=['data'])
        self.item.add_asset('data', asset)

        actual = get_color_palette_from_item(self.item)
        self.assertIsNone(actual)
        palette_from_remote_colortable_mock.assert_not_called()

    @patch('hybig.color_utility.palette_from_remote_colortable')
    def test_get_color_palette_from_item_palette_asset(self, palette_from_remote_mock):
        asset = Asset('data href', roles=['data'])
        palette_asset = Asset('palette href', roles=['palette'])

        self.item.add_asset('data', asset)
        self.item.add_asset('palette', palette_asset)

        expected_palette = ColorPalette()
        palette_from_remote_mock.return_value = expected_palette

        actual = get_color_palette_from_item(self.item)

        palette_from_remote_mock.assert_called_once_with('palette href')
        self.assertEqual(expected_palette, actual)

    @patch('hybig.color_utility.requests.get')
    def test_get_color_palette_from_item_palette_asset_fails(self, get_mock):
        """Raise exception if there is a colortable, but it cannot be retrieved."""
        asset = Asset('data href', roles=['data'])
        palette_asset = Asset('palette href', roles=['palette'])

        self.item.add_asset('data', asset)
        self.item.add_asset('palette', palette_asset)

        failed_response = Mock(Response)
        failed_response.ok = False

        get_mock.return_value = failed_response

        with self.assertRaisesRegex(
            HyBIGError, 'Failed to retrieve color table at palette href'
        ):
            get_color_palette_from_item(self.item)

    @patch('hybig.color_utility.palette_from_remote_colortable')
    def test_get_remote_palette_from_source(self, palette_from_remote_mock):
        with self.subTest('No variables in source'):
            test_source = HarmonySource({})
            with self.assertRaisesRegex(HyBIGNoColorInformation, 'No color in source'):
                get_remote_palette_from_source(test_source)
            palette_from_remote_mock.assert_not_called()

        with self.subTest('No relatedUrls in variable'):
            test_source = HarmonySource({'variables': [{'id': 'fake_id'}]})
            with self.assertRaisesRegex(HyBIGNoColorInformation, 'No color in source'):
                get_remote_palette_from_source(test_source)
            palette_from_remote_mock.assert_not_called()

        with self.subTest('relatedUrls in variable without correct types'):
            test_source = HarmonySource(
                {
                    'variables': [
                        {
                            'id': 'fake_id',
                            'relatedUrls': [
                                {
                                    'url': 'unrelated url',
                                    'urlContentType': 'WRONG Url Content Type',
                                    'type': 'Color Map',
                                }
                            ],
                        }
                    ]
                }
            )
            with self.assertRaisesRegex(HyBIGNoColorInformation, 'No color in source'):
                get_remote_palette_from_source(test_source)
            palette_from_remote_mock.assert_not_called()

        with self.subTest('has relatedUrl in variable with correct types'):
            test_source = HarmonySource(
                {
                    'variables': [
                        {
                            'id': 'fake_id',
                            'relatedUrls': [
                                {
                                    'url': 'correct url to retrieve',
                                    'urlContentType': 'VisualizationURL',
                                    'type': 'Color Map',
                                }
                            ],
                        }
                    ]
                }
            )

            get_remote_palette_from_source(test_source)
            palette_from_remote_mock.assert_called_once_with('correct url to retrieve')
            palette_from_remote_mock.reset_mock()

        with self.subTest('multiple relatedUrls in variable one with correct types'):
            test_source = HarmonySource(
                {
                    'variables': [
                        {
                            'id': 'fake_id',
                            'relatedUrls': [
                                {
                                    'url': 'unrelated to color url',
                                    'type': 'Browse Image',
                                },
                                {
                                    'url': 'correct url of colortable',
                                    'urlContentType': 'VisualizationURL',
                                    'type': 'Color Map',
                                },
                            ],
                        }
                    ]
                }
            )
            get_remote_palette_from_source(test_source)
            palette_from_remote_mock.assert_called_once_with(
                'correct url of colortable'
            )
            palette_from_remote_mock.reset_mock()

        with self.subTest('source contains multiple variables.'):
            test_source = HarmonySource(
                {
                    'variables': [
                        {
                            'id': 'fake_id',
                            'relatedUrls': [
                                {
                                    'url': 'correct url of colortable',
                                    'urlContentType': 'VisualizationURL',
                                    'type': 'Color Map',
                                },
                            ],
                        },
                        {'id': 'fake_id2'},
                    ]
                }
            )
            with self.assertRaisesRegex(HyBIGNoColorInformation, 'No color in source'):
                get_remote_palette_from_source(test_source)
            palette_from_remote_mock.reset_mock()

    @patch('hybig.color_utility.get_remote_palette_from_source')
    def test_get_color_palette_with_item_palette(
        self, get_remote_palette_from_source_mock
    ):
        ds = Mock(DatasetReader)
        ds.colormap = Mock()
        item_palette = convert_colormap_to_palette(self.colormap)
        expected_palette = item_palette

        actual_palette = get_color_palette({}, {}, item_palette)
        self.assertEqual(expected_palette, actual_palette)
        get_remote_palette_from_source_mock.assert_not_called()
        ds.colormap.assert_not_called()

    @patch('hybig.color_utility.requests.get')
    def test_get_color_palette_request_fails(self, get_mock):
        failed_response = Mock(Response)
        failed_response.ok = False

        get_mock.return_value = failed_response
        ds = Mock(DatasetReader)
        ds.colormap = Mock()

        source = HarmonySource(
            {
                'variables': [
                    {
                        'relatedUrls': [
                            {
                                'url': 'url:of:colortable',
                                'urlContentType': 'VisualizationURL',
                                'type': 'Color Map',
                            }
                        ],
                    }
                ],
            }
        )
        with self.assertRaisesRegex(
            HyBIGError, 'Failed to retrieve color table at url:of:colortable'
        ):
            get_color_palette(ds, source, None)
        ds.colormap.assert_not_called()

    @patch('hybig.color_utility.palette_from_remote_colortable')
    def test_get_color_palette_finds_no_url(self, palette_from_remote_mock):
        palette_from_remote_mock.side_effect = HyBIGError('mocked exception')
        ds = Mock(DatasetReader)
        ds.colormap.return_value = self.colormap

        source = HarmonySource(
            {
                'variables': [
                    {
                        'relatedUrls': [
                            {
                                'url': 'url:of:colortable',
                                'urlContentType': 'VisualizationURL',
                                'type': 'Color Map',
                            }
                        ],
                    }
                ],
            }
        )
        with self.assertRaisesRegex(
            HyBIGError, 'Failed to retrieve color table at url:of:colortable'
        ):
            get_color_palette(ds, source, None)
        palette_from_remote_mock.assert_called_once_with('url:of:colortable')

    @patch('hybig.color_utility.palette_from_remote_colortable')
    def test_get_color_palette_source_remote_exists(self, palette_from_remote_mock):
        ds = Mock(DatasetReader)
        ds.colormap.return_value = self.colormap
        palette_fake = {'palette': 'fake'}
        palette_from_remote_mock.return_value = palette_fake

        source = HarmonySource(
            {
                'collection': 'C1238621141-POCLOUD',
                'shortName': 'MUR-JPL-L4-GLOB-v4.1',
                'versionId': '4.1',
                'variables': [
                    {
                        'id': 'V1241047546-POCLOUD',
                        'name': 'analysed_sst',
                        'fullPath': 'analysed_sst',
                        'type': None,
                        'subtype': None,
                        'relatedUrls': [
                            {
                                'url': 'https://gibs.earthdata.nasa.gov/colormaps/txt/GHRSST_Sea_Surface_Temperature.txt',
                                'urlContentType': 'VisualizationURL',
                                'type': 'Color Map',
                            }
                        ],
                    }
                ],
            }
        )

        actual_palette = get_color_palette(ds, source, None)
        palette_from_remote_mock.assert_called_once_with(
            'https://gibs.earthdata.nasa.gov/colormaps/txt/GHRSST_Sea_Surface_Temperature.txt'
        )
        self.assertEqual(palette_fake, actual_palette)

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
