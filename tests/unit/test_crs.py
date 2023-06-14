"""Tests exercising the crs module"""

from contextlib import contextmanager
from pathlib import Path
from shutil import rmtree
from tempfile import NamedTemporaryFile, TemporaryDirectory
from textwrap import dedent
from unittest import TestCase
from unittest.mock import patch

import rasterio
from affine import Affine
from harmony.message import SRS
from rasterio.crs import CRS

from harmony_browse_image_generator.crs import (
    PREFERRED_CRS,
    choose_best_crs_from_metadata,
    choose_target_crs,
)
from harmony_browse_image_generator.exceptions import HyBIGInvalidMessage

## Test constants
WKT_EPSG_3031 = (
    'PROJCS["WGS 84 / Antarctic Polar Stereographic",'
    'GEOGCS["WGS 84",DATUM["WGS_1984",'
    'SPHEROID["WGS 84",6378137,298.257223563,'
    'AUTHORITY["EPSG","7030"]],'
    'AUTHORITY["EPSG","6326"]],'
    'PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],'
    'UNIT["degree",0.0174532925199433,'
    'AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],'
    'PROJECTION["Polar_Stereographic"],'
    'PARAMETER["latitude_of_origin",-71],'
    'PARAMETER["central_meridian",0],'
    'PARAMETER["false_easting",0],'
    'PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],'
    'AUTHORITY["EPSG","3031"]]'
)

# This wkt describes 3411, but when converted to CRS, DOES NOT get updated to EPSG:3413
WKT_EPSG_3411 = (
    'PROJCS["unknown",GEOGCS["unknown",'
    'DATUM["unknown",SPHEROID["unknown",6378273,298.279411123064]],'
    'PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],'
    'UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]],'
    'PROJECTION["Polar_Stereographic"],'
    'PARAMETER["latitude_of_origin",70],'
    'PARAMETER["central_meridian",-45],'
    'PARAMETER["false_easting",0],'
    'PARAMETER["false_northing",0],'
    'UNIT["metre",1,AUTHORITY["EPSG","9001"]],'
    'AXIS["Easting",SOUTH],AXIS["Northing",SOUTH]]'
)

WKT_EPSG_6050 = dedent(
    '''
    PROJCS["GR96 / EPSG Arctic zone 1-25",
        GEOGCS["GR96",
            DATUM["Greenland_1996",
                SPHEROID["GRS 1980",6378137,298.257222101],
                TOWGS84[0,0,0,0,0,0,0]],
            PRIMEM["Greenwich",0,
                AUTHORITY["EPSG","8901"]],
            UNIT["degree",0.0174532925199433,
                AUTHORITY["EPSG","9122"]],
            AUTHORITY["EPSG","4747"]],
        PROJECTION["Lambert_Conformal_Conic_2SP"],
        PARAMETER["latitude_of_origin",85.4371183333333],
        PARAMETER["central_meridian",-30],
        PARAMETER["standard_parallel_1",87],
        PARAMETER["standard_parallel_2",83.6666666666667],
        PARAMETER["false_easting",25500000],
        PARAMETER["false_northing",1500000],
        UNIT["metre",1,
            AUTHORITY["EPSG","9001"]],
        AXIS["Easting",EAST],
        AXIS["Northing",NORTH],
        AUTHORITY["EPSG","6050"]]
    '''
)


class TestCrs(TestCase):
    """A class that tests the crs module."""

    @classmethod
    def setUp(self):
        self.temp_dir = Path(TemporaryDirectory().name)

    @classmethod
    def tearDown(self):
        if self.temp_dir.exists():
            rmtree(self.temp_dir)

    @classmethod
    @contextmanager
    def test_rastero_file(self, **options):
        """Helper function to create a test geotiff file.

        rasterio.DatasetReader is best instantiated by opening an existing file.  This
        function creates a fake temporary file with default and optional
        metadata, and then it yields the name of the file to the caller.

        This file can be opened and examined and when the context exits it
        cleans itself up.

        """
        default_options = {
            'count': 3,
            'height': 1000,
            'width': 2000,
            'crs': CRS.from_string('EPSG:4326'),
            'transform': Affine.scale(100, 200),
            'dtype': 'uint8',
        }

        with NamedTemporaryFile(suffix='.tif') as tmp_file:
            with rasterio.open(tmp_file.name, 'w', **default_options | options):
                pass
            yield tmp_file.name

    def test_choose_target_crs_with_epsg_from_harmony_message(self):
        """Test SRS has an epsg code."""
        expected_CRS = CRS.from_epsg(6932)
        test_srs = SRS({'epsg': 'EPSG:6932'})
        actual_CRS = choose_target_crs(test_srs, None)
        self.assertEqual(expected_CRS, actual_CRS)

    def test_choose_target_crs_with_wkt_from_harmony_message(self):
        """Test SRS has wkt string."""
        expected_CRS = CRS.from_epsg(6050)
        test_srs = SRS({'wkt': WKT_EPSG_6050})
        actual_CRS = choose_target_crs(test_srs, None)
        self.assertEqual(expected_CRS, actual_CRS)

    def test_choose_target_crs_with_proj4_from_harmony_message(self):
        """Test SRS has proj4 string."""
        expected_CRS = CRS.from_epsg(5938)
        test_srs = SRS(
            {
                'proj4': (
                    '+proj=stere +lat_0=90 +lon_0=-33 +k=0.994'
                    ' +x_0=2000000 +y_0=2000000 +datum=WGS84 +units=m +no_defs +type=crs'
                )
            }
        )
        actual_CRS = choose_target_crs(test_srs, None)
        self.assertEqual(expected_CRS, actual_CRS)

    def test_choose_target_crs_with_invalid_SRS_from_harmony_message(self):
        """Test SRS does not have epsg, wkt or proj4 string."""
        test_srs_is_json = {'how': 'did this happen?'}
        with self.assertRaisesRegex(HyBIGInvalidMessage, 'Bad input SRS'):
            choose_target_crs(test_srs_is_json, None)

    @patch('harmony_browse_image_generator.crs.choose_crs_from_metadata')
    def test_choose_target_harmony_message_has_crs_but_no_srs(self, mock_choose_fxn):
        """Explicitly show we do not support format.crs only.

        We do not handle the case of a format.crs without a format.srs and
        expect to try to guess the SRS from the input data.

        """
        test_srs = None
        in_dataset = {'test': 'object'}

        choose_target_crs(test_srs, in_dataset)
        mock_choose_fxn.assert_called_once_with(in_dataset)

    def test_choose_target_crs_with_preferred_metadata_north(self):
        """Check that preferred metadata for northern projection is found."""
        expected_CRS = PREFERRED_CRS['north']
        with self.test_rastero_file(
            height=448,
            width=304,
            crs=CRS.from_epsg(3413),
            transform=Affine(25000.0, 0.0, -3850000.0, 0.0, -25000.0, 5850000.0),
            dtype='uint16',
        ) as tmp_file:
            with rasterio.open(tmp_file) as in_dataset:
                actual_CRS = choose_target_crs(None, in_dataset)

                self.assertEqual(expected_CRS, actual_CRS)

    def test_choose_target_crs_with_preferred_metadata_south(self):
        """Check that preferred metadata for southern projection is found."""
        expected_CRS = PREFERRED_CRS['south']

        with self.test_rastero_file(
            crs=WKT_EPSG_3031,
            transform=Affine.scale(500, 300),
            dtype='uint16',
        ) as tmp_file:
            with rasterio.open(tmp_file) as in_dataset:
                actual_CRS = choose_target_crs(None, in_dataset)

                self.assertEqual(expected_CRS, actual_CRS)

    def test_choose_target_crs_with_preferred_metadata_global(self):
        """Check that preferred metadata for global projection is found."""
        expected_CRS = PREFERRED_CRS['global']
        with self.test_rastero_file(
            count=3,
            crs=CRS.from_proj4('+proj=longlat +datum=WGS84 +no_defs +type=crs'),
        ) as tmp_file:
            with rasterio.open(tmp_file) as in_dataset:
                actual_CRS = choose_target_crs(None, in_dataset)

                self.assertEqual(expected_CRS, actual_CRS)

    def test_choose_target_crs_with_non_preferred_metadata_north(self):
        """Check that NSIDCs 3411 as WKT is correctly tansformed to epsg:3413."""
        expected_CRS = PREFERRED_CRS['north']

        input_CRS = CRS.from_wkt(WKT_EPSG_3411)

        with self.test_rastero_file(crs=input_CRS) as tmp_file:
            with rasterio.open(tmp_file) as in_dataset:
                actual_CRS = choose_target_crs(None, in_dataset)
                self.assertEqual(expected_CRS, actual_CRS)

    def test_choose_target_crs_from_metadata_south(self):
        """Test EASE-2 South is correctly transformed to epsg:3031."""
        expected_CRS = PREFERRED_CRS['south']
        ease_grid_2_south = 'EPSG:6932'
        input_CRS = CRS.from_string(ease_grid_2_south)

        with self.test_rastero_file(crs=input_CRS) as tmp_file:
            with rasterio.open(tmp_file) as in_dataset:
                actual_CRS = choose_best_crs_from_metadata(in_dataset.crs)
                self.assertEqual(expected_CRS, actual_CRS)

    def test_choose_target_crs_from_metadata_global(self):
        """Check EASE-2 Global is correctly transformed to epsg:4326."""
        expected_CRS = PREFERRED_CRS['global']
        ease_grid_2_global = 'EPSG:6933'
        input_CRS = CRS.from_string(ease_grid_2_global)

        with self.test_rastero_file(crs=input_CRS) as tmp_file:
            with rasterio.open(tmp_file) as in_dataset:
                actual_CRS = choose_best_crs_from_metadata(in_dataset.crs)
                self.assertEqual(expected_CRS, actual_CRS)

    def test_multiple_crs_from_metadata(self):
        """Test curated results.

        This set of tests tests EPSG codes against expected
        preferred CRS outputs.

        These epsg codes were pulled form NSDIC's metadata database and each
        one checked against the espg.io website by hand to pick the most
        appropriate preferred CRS to select.

        Each tuple is comprised of an input EPSG code, the expected preferred
        hemisphere, and part of a name to help the reviewer.

        """
        epsg_test_codes = [
            ('EPSG:3031', 'south', 'WGS 84 / Antarctic Polar Stereographic'),
            ('EPSG:3408', 'north', 'WGS 84 / NSIDC EASE-Grid 2.0 North'),
            ('EPSG:3409', 'south', 'WGS 84 / NSIDC EASE-Grid 2.0 South'),
            ('EPSG:3411', 'north', 'WGS 84 / NSIDC Sea Ice Polar Stereographic North'),
            ('EPSG:3412', 'south', 'WGS 84 / NSIDC Sea Ice Polar Stereographic South'),
            ('EPSG:3995', 'north', 'WGS 84 / Arctic Polar Stereographic'),
            ('EPSG:3413', 'north', 'WGS 84 / NSIDC Sea Ice Polar Stereographic North'),
            ('EPSG:6931', 'north', 'WGS 84 / NSIDC EASE-Grid 2.0 North'),
            ('EPSG:3976', 'south', 'WGS 84 / NSIDC Sea Ice Polar Stereographic South'),
            ('EPSG:6932', 'south', 'WGS 84 / NSIDC EASE-Grid 2.0 South'),
            ('EPSG:6933', 'global', 'WGS 84 / NSIDC EASE-Grid 2.0 Global'),
            ('EPSG:32612', 'global', 'WGS 84 / UTM zone 12N'),
            ('EPSG:4326', 'global', 'WGS 84 / DATUM["WGS_1984",SPHEROID["WGS 84"]]'),
            ('EPSG:5332', 'global', 'ITRF2008 / International_Terrestrial_Reference'),
            ('EPSG:32613', 'global', 'WGS 84 / UTM zone 13N'),
            ('EPSG:3034', 'global', 'ETRS89-extended / LCC Europe"'),
            ('EPSG:3338', 'global', 'NAD83 / Alaska Albers"'),
            ('EPSG:3410', 'global', 'WGS 84 / NSIDC EASE-Grid 2.0 Global'),
            ('EPSG:4047', 'global', 'Unspecified datum 1980 Authalic Sphere'),
            ('EPSG:4087', 'global', 'WGS 84 / World Equidistant Cylindrical'),
            ('EPSG:4269', 'global', 'NAD83,DATUM["North_American_Datum_1983"]'),
            ('EPSG:4978', 'global', 'WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84"]'),
            ('EPSG:6340', 'global', 'NAD83(2011) / UTM zone 11N"'),
            ('EPSG:6341', 'global', 'NAD83(2011) / UTM zone 12N"'),
            ('EPSG:6342', 'global', 'NAD83(2011) / UTM zone 13N"'),
            ('EPSG:26913', 'global', 'NAD83 / UTM zone 13N'),
            ('EPSG:32606', 'global', 'WGS 84 / UTM zone 6N'),
            ('EPSG:32607', 'global', 'WGS 84 / UTM zone 7N'),
            ('EPSG:32608', 'global', 'WGS 84 / UTM zone 8N'),
            ('EPSG:32610', 'global', 'WGS 84 / UTM zone 10N'),
            ('EPSG:32611', 'global', 'WGS 84 / UTM zone 11N'),
            ('EPSG:32616', 'global', 'WGS 84 / UTM zone 16N'),
            ('EPSG:32644', 'global', 'WGS 84 / UTM zone 44N'),
            ('EPSG:32645', 'global', 'WGS 84 / UTM zone 45N'),
        ]
        for epsg_code, expected, name in epsg_test_codes:
            with self.subTest(f'{epsg_code}: {name}'):
                actual_CRS = choose_best_crs_from_metadata(epsg_code)
                self.assertEqual(actual_CRS, PREFERRED_CRS[expected])
