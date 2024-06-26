"""Coordinate Reference System module.

This module contains functions related to coordinate reference system (CRS)
selection and manipulation.  A CRS can also be called a spatial reference
system (SRS) and the names can be used interchangeably.

In this module, SRS is used to represent the Harmony Message projection
information as it has a named SRS object that is retrieved from the Harmony Message's
Format object.

"""

from harmony.message import SRS
from pyproj.crs import CRS as pyCRS

# pylint: disable-next=no-name-in-module
from rasterio.crs import CRS
from xarray import DataArray

from hybig.exceptions import HyBIGValueError

# These are the CRSs that GIBS will accept as input. When the user hasn't
# directly specified an output CRS, the code will attempt to choose the best
# one of these.
PREFERRED_CRS = {
    'north': 'EPSG:3413',  # WGS 84 / NSIDC Sea Ice Polar Stereographic North
    'south': 'EPSG:3031',  # WGS 84 / Antarctic Polar Stereographic
    'global': 'EPSG:4326',  # WGS 84 -- WGS84 - World Geodetic System 1984, used in GPS
}


def choose_target_crs(srs: SRS, data_array: DataArray) -> CRS:
    """Return the target CRS for the output image.

    If a harmony message defines a SRS, we use that as the target ouptut CRS.
    Otherwise, we try determine the best "GIBS compatible" CRS based on the
    input granule metadata.

    """
    if srs is not None:
        return choose_crs_from_srs(srs)
    return choose_crs_from_metadata(data_array)


def choose_crs_from_srs(srs: SRS):
    """Choose the best CRS based on the Harmony Request's SRS.

    create a CRS from the harmony srs information.
    prefer epsg to wkt
    prefer wkt to proj4

    Raise HyBIGValueError if the harmony SRS cannot be converted to a
    rasterio CRS for any reason.

    """
    try:
        if srs.epsg is not None and srs.epsg != "":
            return CRS.from_string(srs.epsg)
        if srs.wkt is not None and srs.wkt != "":
            return CRS.from_string(srs.wkt)
        return CRS.from_string(srs.proj4)
    except Exception as exception:
        raise HyBIGValueError(f'Bad input SRS: {str(exception)}') from exception


def is_preferred_crs(crs: CRS) -> bool:
    """Returns true if the input CRS is preferred by GIBS."""
    if crs.to_string() in PREFERRED_CRS.values():
        return True
    return False


def choose_crs_from_metadata(data_array: DataArray) -> CRS | None:
    """Determine the best CRS based on input metadata."""
    if is_preferred_crs(data_array.rio.crs):
        return data_array.rio.crs
    return choose_best_crs_from_metadata(data_array.rio.crs)


def choose_best_crs_from_metadata(crs: CRS) -> CRS:
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
        return CRS.from_string(PREFERRED_CRS['global'])

    if projection_params.get('lat_0', 0.0) >= 80:
        return CRS.from_string(PREFERRED_CRS['north'])

    if projection_params.get('lat_0', 0.0) <= -80:
        return CRS.from_string(PREFERRED_CRS['south'])

    return CRS.from_string(PREFERRED_CRS['global'])
