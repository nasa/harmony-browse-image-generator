"""Utility functions shared across test files."""

from contextlib import contextmanager
from tempfile import NamedTemporaryFile

import rasterio
from affine import Affine
from rasterio.crs import CRS


@contextmanager
def rasterio_test_file(raster_data=None, **options):
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
        with rasterio.Env(CHECK_DISK_FREE_SPACE="NO"):
            with rasterio.open(
                tmp_file.name, 'w', **default_options | options
            ) as tmp_rasterio_file:
                if raster_data is not None:
                    tmp_rasterio_file.write(raster_data)

            yield tmp_file.name
