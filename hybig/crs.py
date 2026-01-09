"""Coordinate Reference System module.

This module contains functions related to coordinate reference system (CRS)
selection and manipulation.  A CRS can also be called a spatial reference
system (SRS) and the names can be used interchangeably.

In this module, SRS is used to represent the Harmony Message projection
information as it has a named SRS object that is retrieved from the Harmony Message's
Format object.

"""

import rasterio
from harmony_service_lib.message import SRS
from pyproj import CRS as pyCRS
from rasterio.io import DatasetReader

from hybig.exceptions import HyBIGValueError

# These are the CRSs that GIBS will accept as input. When the user hasn't
# directly specified an output CRS, the code will attempt to choose the best
# one of these.
PREFERRED_CRS = {
    'north': 'EPSG:3413',  # WGS 84 / NSIDC Sea Ice Polar Stereographic North
    'south': 'EPSG:3031',  # WGS 84 / Antarctic Polar Stereographic
    'global': 'EPSG:4326',  # WGS 84 -- WGS84 - World Geodetic System 1984, used in GPS
}


def choose_target_crs(srs: SRS | None, src_ds: DatasetReader) -> rasterio.CRS:
    """Return the target CRS for the output image.

    If a harmony message defines a SRS, we use that as the target ouptut CRS.
    Otherwise, we try determine the best "GIBS compatible" CRS based on the
    input granule metadata.

    """
    if srs is not None:
        return choose_crs_from_srs(srs)
    return choose_crs_from_metadata(src_ds)


def choose_crs_from_srs(srs: SRS):
    """Choose the best CRS based on the Harmony Request's SRS.

    create a CRS from the harmony srs information.
    prefer epsg to wkt
    prefer wkt to proj4

    Raise HyBIGValueError if the harmony SRS cannot be converted to a
    rasterio CRS for any reason.

    """
    try:
        # harmony defines properties for classes in a way that type checkers
        # can't pick up on, so we use type: ignore to suppress it
        if srs.epsg is not None and srs.epsg != '':  # type: ignore
            return rasterio.CRS.from_string(srs.epsg)  # type: ignore
        if srs.wkt is not None and srs.wkt != '':  # type: ignore
            return rasterio.CRS.from_string(srs.wkt)  # type: ignore
        return rasterio.CRS.from_string(srs.proj4)  # type: ignore
    except Exception as exception:
        raise HyBIGValueError(f'Bad input SRS: {str(exception)}') from exception


def is_preferred_crs(crs: rasterio.CRS) -> bool:
    """Returns true if the input rasterio.CRS is preferred by GIBS."""
    if crs.to_string() in PREFERRED_CRS.values():
        return True
    return False


def choose_crs_from_metadata(src_ds: DatasetReader) -> rasterio.CRS | None:
    """Determine the best CRS based on input metadata."""
    if is_preferred_crs(src_ds.crs):
        return src_ds.crs
    return choose_best_crs_from_metadata(src_ds.crs)


def choose_best_crs_from_metadata(crs: rasterio.CRS) -> rasterio.CRS:
    """Determine the best preferred CRS based on the input CRS.

    We are targeting GIBS which has three preferred CRSs a Northern Polar
    Stereo, Southern Polar Stero and Global Projection. Using the information
    from the input granule we want to determine which CRS the dataset will be
    best represented by.

    This routine transforms the input CRS into a dict and examines the CRS
    projection parameters to determine the best option for regridding.

    It is very naive, if the proj is lonlat it uses the preferred global
    projection.

    If the projection latitude of origin is above 80° N -> use the preferred
    northern projection

    If the projection latitude of origin is below -80° N -> use the preferred
    southern projection

    Otherwise it defaults to the global projection.

    """
    projection_params = pyCRS(crs).to_dict()

    if projection_params.get('proj', None) == 'longlat':
        return rasterio.CRS.from_string(PREFERRED_CRS['global'])

    if projection_params.get('lat_0', 0.0) >= 80:
        return rasterio.CRS.from_string(PREFERRED_CRS['north'])

    if projection_params.get('lat_0', 0.0) <= -80:
        return rasterio.CRS.from_string(PREFERRED_CRS['south'])

    return rasterio.CRS.from_string(PREFERRED_CRS['global'])
