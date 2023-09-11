"""Utility routines to aid in determining target size parameters.

This module handles determiniation of the output image size parameters.  These
are geared towards satisfying GIBS requirements, but allows a user to override
most options from the input Harmony Message.

There's a number of "rules" from the GIBS ICD that are codified in this module.

"""
from collections import namedtuple
from typing import TypedDict

import numpy as np
from numpy import ndarray
from affine import Affine
from harmony.message import Message
from pyproj import Transformer
from pyproj.crs import CRS as pyCRS

# pylint: disable-next=no-name-in-module
from rasterio.crs import CRS
from rasterio.io import DatasetReader
from rasterio.transform import (
    AffineTransformer,
    from_bounds,
    from_origin,
)

from harmony_browse_image_generator.crs import (
    PREFERRED_CRS,
    choose_best_crs_from_metadata,
    choose_target_crs,
    is_preferred_crs,
)
from harmony_browse_image_generator.exceptions import HyBIGValueError
from harmony_browse_image_generator.message_utility import (
    has_dimensions,
    has_scale_extents,
    has_scale_sizes,
)


class GridParams(TypedDict):
    """Convenience to describe a grid parameters dictionary."""
    height: int
    width: int
    crs: CRS
    transform: Affine


class ScaleExtent(TypedDict):
    """Convenience to describe a scale extent dictionary."""
    xmin: float
    ymin: float
    xmax: float
    ymax: float


class ScaleSize(TypedDict):
    """Convenience to describe a scale size dictionary."""
    x: float
    y: float


class Dimensions(TypedDict):
    """Convenience to describe a scale a dimension dictionary."""
    width: int
    height: int


ResolutionInfo = namedtuple(
    'resolutionInfo',
    [
        'resolution',
        'pixel_size',
        'height',
        'width',
        'overview_levels',
        'overview_scale',
    ],
)

# GIBS output resolution information from 423-ICD-009, Revision A Effecitve
# Date: November 2016

# This is Table 4.1.8-1 WGS84 (EPSG: 4326 Resolutions)
epsg_4326_resolutions = [
    ResolutionInfo("2km", 0.017578125, 10240, 20480, 6, 2),
    ResolutionInfo("1km", 0.0087890625, 20480, 40960, 7, 2),
    ResolutionInfo("500m", 0.00439453125, 40960, 81920, 8, 2),
    ResolutionInfo("250m", 0.002197265625, 81920, 163840, 9, 2),
    ResolutionInfo("125m", 0.0010986328125, 163840, 327680, 10, 2),
    ResolutionInfo("62.5m", 0.00054931640625, 327680, 655360, 11, 2),
    ResolutionInfo("31.25m", 0.000274658203125, 655360, 1310720, 12, 2),
    ResolutionInfo("15.625m", 0.0001373291015625, 1310720, 2521440, 13, 2),
]

# This is Table 4.1.8-2 NSIDC Sea Ice Polar Stereographic Extent (EPSG:3413) Resolutions
epsg_3413_resolutions = [
    ResolutionInfo("2km", 2048, 4096, 4096, 3, 2),
    ResolutionInfo("1km", 1024, 8192, 8192, 4, 2),
    ResolutionInfo("500m", 512, 16384, 16384, 5, 2),
    ResolutionInfo("250m", 256, 32768, 32768, 6, 2),
    ResolutionInfo("125m", 128, 65536, 65536, 7, 2),
    ResolutionInfo("62.5m", 64, 131072, 131072, 8, 2),
    ResolutionInfo("31.25m", 32, 252144, 252144, 9, 2),
    ResolutionInfo("15.625m", 16, 524288, 524288, 10, 2),
]

# The Antarctic resolutions match the northern resolutions precisely.
# Table 4.1.8-3. Antarctic Polar Stereographic (EPSG: 3031) Resolutions
epsg_3031_resolutions = epsg_3413_resolutions


def get_target_grid_parameters(message: Message,
                               dataset: DatasetReader) -> GridParams:
    """Get the output image parameters.

    This computes the target grid of the ouptut image. The grid is defined by
    of the extent of the output: ScaleExtents, and either the grid cell sizes:
    ScaleSizes or the scaleDimensions: width and height.

    - User submitted parameters take precedence over computed
    parameters.

    - Computed parameters should generate GIBS suitable images.

    """
    target_crs = choose_target_crs(message.format.srs, dataset)
    target_scale_extent = choose_scale_extent(message, target_crs)
    target_dimensions = choose_target_dimensions(
        message, dataset, target_scale_extent, target_crs
    )
    return get_rasterio_parameters(target_crs, target_scale_extent, target_dimensions)


def choose_scale_extent(message: Message, target_crs: CRS) -> ScaleExtent:
    """Return the scaleExtent of the target image.

    Check the message for a defined scale extent and returns that or returns
    the best alternative based on the target_crs either returning it from a
    lookup based on the ICD, or computed with pyproj.area_of_use

    """
    if has_scale_extents(message):
        scale_extent = {
            'xmin': message.format.scaleExtent.x.min,
            'ymin': message.format.scaleExtent.y.min,
            'xmax': message.format.scaleExtent.x.max,
            'ymax': message.format.scaleExtent.y.max,
        }
    elif is_preferred_crs(target_crs):
        scale_extent = icd_defined_extent_from_crs(target_crs)
    else:
        # compute a best guess area based on the CRS's region of interest
        scale_extent = best_guess_scale_extent(target_crs)
    return scale_extent


def choose_target_dimensions(message: Message, dataset: DatasetReader,
                             scale_extent: ScaleExtent,
                             target_crs: CRS) -> Dimensions:
    """This selects or computes the target Dimensions.

    This routine finalizes the output grid.  The target dimensions are
    returned.  These along with the CRS and scaleExent will be used to
    selecting the correct output transformation

    To determine the target dimensions the following must occur.

    The input harmony message format is searched for height and width or scaleSizes.
    If scalesizes are found, they are converted to height and width and returned.

    If no scaleSizes or grid sizes are input. The input dataset metadata
    scaleSizes are checked against the GIBS preferred resolutions and the best
    fit resolution with the ScaleExtent is used to compute a target dimension
    to return.

    """
    if has_dimensions(message):
        dimensions = {'height': message.format.height, 'width': message.format.width}
    elif has_scale_sizes(message):
        scale_size = message.format.scaleSize
        width = round((scale_extent['xmax'] - scale_extent['xmin']) / scale_size.x)
        height = round((scale_extent['ymax'] - scale_extent['ymin']) / scale_size.y)
        dimensions = {'height': height, 'width': width}
    else:
        dimensions = best_guess_target_dimensions(dataset, scale_extent, target_crs)

    return dimensions


def get_rasterio_parameters(crs: CRS, scale_extent: ScaleExtent,
                            dimensions: Dimensions) -> GridParams:
    """Convert the grid into rasterio consumable format.

    Returns a Dictionary of keyword params suitable for rasterio to use in
    writing an output image.

    To write images, rasterio wants parameters: 'width', 'height', 'crs', and
    'transform'

    """
    transform = from_bounds(
        scale_extent['xmin'],
        scale_extent['ymin'],
        scale_extent['xmax'],
        scale_extent['ymax'],
        dimensions['width'],
        dimensions['height'],
    )
    return {
        'width': dimensions['width'],
        'height': dimensions['height'],
        'crs': crs,
        'transform': transform,
    }


def create_tiled_output_parameters(
        grid_parameters: GridParams
) -> (list[GridParams], list[dict] | None):
    """Split the output grid if necessary.

    When the grid resolution is less than 1km (.0087890625 degrees)
    (and the scale extents are full width.? TODO [MHS, 08/08/2023] )

    returns a list of GridParams object that completely cover the input scale
    extent.

    """
    if not needs_tiling(grid_parameters):
        return [grid_parameters], [None]

    crs = grid_parameters['crs']
    full_width = grid_parameters['width']
    full_height = grid_parameters['height']
    transform = grid_parameters['transform']
    resolution_x = transform.a
    resolution_y = np.abs(transform.e)

    cells_per_tile_width = compute_cells_per_tile(resolution_x, crs)
    cells_per_tile_height = compute_cells_per_tile(resolution_y, crs)

    width_origins = compute_tile_boundaries(cells_per_tile_width, full_width)
    height_origins = compute_tile_boundaries(cells_per_tile_height, full_height)
    width_dimensions = compute_tile_dimensions(width_origins)
    height_dimensions = compute_tile_dimensions(height_origins)

    transformer = AffineTransformer(transform)

    grid_parameter_list = []
    tile_locator = []
    for h_idx, row in enumerate(height_origins):
        for w_idx, col in enumerate(width_origins):
            height = height_dimensions[h_idx]
            width = width_dimensions[w_idx]
            x_loc, y_loc = transformer.xy(row, col, offset='ul')
            tile_grid_transform = from_origin(x_loc, y_loc, resolution_x, resolution_y)
            if height > 0 and width > 0:
                grid_parameter_list.append({
                    'width': width,
                    'height': height,
                    'crs': crs,
                    'transform': tile_grid_transform,
                })
                tile_locator.append({'row': h_idx, 'col': w_idx})

    return grid_parameter_list, tile_locator


def compute_tile_dimensions(origins: ndarray) -> ndarray:
    """return a list of tile dimensions.

    From a list of origin locations, return the dimension for each tile.

    """
    return list(np.diff(origins, append=origins[-1]).astype('int'))


def compute_tile_boundaries(target_size: int, full_size: int) -> list[float]:
    """returns a list of boundary cells.

    The returned boundary cells are the column [or row] values for each of the
    output tiles. They should always start at 0, and end at the full_size
    input, stepping by target size until the last step, which can be any size
    including 0.

    """
    n_boundaries = int(full_size // target_size) + 1
    boundaries = [target_size * index for index in range(n_boundaries)]

    if boundaries[-1] != full_size:
        boundaries.append(full_size)

    return boundaries


def compute_cells_per_tile(resolution: float, crs: CRS) -> int:
    """optimum cells per tile."""

    # constants determined with GIBS.
    degrees_per_tile = 10.
    meters_per_tile = 1000000.

    if crs.is_projected:
        cells_per_tile = int(np.round(meters_per_tile / resolution))
    else:
        cells_per_tile = int(np.round(degrees_per_tile / resolution))

    return cells_per_tile


def needs_tiling(grid_parameters: GridParams) -> bool:
    """Returns true if the grid is too large for GIBS.

    For a lat/lon unprojected grid, this means resolution of 1km or ~.01
    degrees/grid cell

    For a projected grid, I think 1km at full resoution is too coarse to chop
    and we should choose 250m or 500m as the default TODO [MHS, 08/09/2023]

    """
    if (
        grid_parameters['crs'].is_projected
        and grid_parameters['transform'].a <= 500.
    ):
        should_tile = True
    elif (
        not grid_parameters['crs'].is_projected
        and grid_parameters['transform'].a <= 0.01
    ):
        should_tile = True
    else:
        should_tile = False

    return should_tile


def icd_defined_extent_from_crs(crs: CRS) -> ScaleExtent:
    """return the predefined scaleExtent for a GIBS image.

    looks up which projetion is being used and returns the scaleExtent.

    """
    if crs.to_string() == PREFERRED_CRS['global']:
        scale_extent = {'xmin': -180.0, 'ymin': -90.0, 'xmax': 180.0, 'ymax': 90.0}
    elif crs.to_string() in [PREFERRED_CRS['north'], PREFERRED_CRS['south']]:
        # both north and south preferred CRSs have same extents.
        scale_extent = {
            'xmin': -4194304.0,
            'ymin': -4194304.0,
            'xmax': 4194304.0,
            'ymax': 4194304.0,
        }
    else:
        raise HyBIGValueError(f'Invalid input CRS: {crs.to_string()}')

    return scale_extent


def best_guess_scale_extent(in_crs: CRS) -> ScaleExtent:
    """Guess the best scale extent.

    This routine will try to guess what a user intended if they did not include
    a scaleExtent and also used a non-preferred CRS. We convert the CRS into a
    pyproj crs check for an area of use.  If this exists we return the bounds,
    projecting them if the crs is a projected crs.

    if no area_of_use exists, we return the ICD defined scale extent that
    relates to the closest prefered CRS.

    """
    crs = pyCRS(in_crs.to_wkt(version='WKT2'))
    if crs.area_of_use is None:
        best_crs = choose_best_crs_from_metadata(crs)
        scale_extent = icd_defined_extent_from_crs(best_crs)
    elif crs.is_projected:
        transformer = Transformer.from_crs(crs.geodetic_crs, crs, always_xy=True)
        projected_bounds = transformer.transform_bounds(*crs.area_of_use.bounds)
        scale_extent = {
            'xmin': projected_bounds[0],
            'ymin': projected_bounds[1],
            'xmax': projected_bounds[2],
            'ymax': projected_bounds[3],
        }
    else:
        scale_extent = {
            'xmin': crs.area_of_use.bounds[0],
            'ymin': crs.area_of_use.bounds[1],
            'xmax': crs.area_of_use.bounds[2],
            'ymax': crs.area_of_use.bounds[3],
        }

    return scale_extent


def best_guess_target_dimensions(dataset: DatasetReader,
                                 scale_extent: ScaleExtent,
                                 crs: CRS) -> Dimensions:
    """Return best guess for output image dimensions.

    Using the information from the scaleExtent and the input dataset metadata,
    compute the height and width appropriate for the output image.

    North and South have matching resolutions so we just use the 3413 resolutions.

    """
    resolution_list = None
    if not crs.is_projected:
        resolution_list = epsg_4326_resolutions
    else:
        resolution_list = epsg_3413_resolutions

    return guess_dimension(dataset, scale_extent, resolution_list)


def guess_dimension(dataset: DatasetReader, scale_extent: ScaleExtent,
                    icd_resolution_list: list[ResolutionInfo]) -> Dimensions:
    """Guess Dimensions given the input information and ICD resolutions."""
    coarsest_resolution = icd_resolution_list[0].pixel_size
    xres = (scale_extent['xmax'] - scale_extent['xmin']) / dataset.width
    yres = (scale_extent['ymax'] - scale_extent['ymin']) / dataset.height
    if xres > coarsest_resolution and yres > coarsest_resolution:
        dimensions = {'height': dataset.height, 'width': dataset.width}
    else:
        target_res = find_closest_resolution(
            list({xres, yres}), icd_resolution_list
        )
        dimensions = compute_target_dimensions(scale_extent, target_res.pixel_size)
    return dimensions


def compute_target_dimensions(scale_extent: ScaleExtent,
                              res: float) -> Dimensions:
    """compute dimensions from inputs.

    We round the computed dimensions because we need to have integer values of
    height and width. Because of the rounding, this function could return a
    height/width that in combination with the ScaleExtent would be slightly
    inconsistent with the input res (ScaleSize).  This is ok, because for
    computed values of scaleExtent, we are already self consistent.  If there's
    a discrepancy, the user has described a scaleExtent in the input
    HarmonyMessage and has *not* specified a scaleSize. So we honor the
    scaleExtent.

    """
    return {
        'height': round((scale_extent['ymax'] - scale_extent['ymin']) / res),
        'width': round((scale_extent['xmax'] - scale_extent['xmin']) / res),
    }


def find_closest_resolution(resolutions: list[float], resolution_info:
                            list[ResolutionInfo]) -> ResolutionInfo:
    """Return closest match to GIBS preferred Resolution Info.

    Cycle through all input resolutions and return the resolution_info that has
    the smallest absolute difference to any of the input resolutions.

    """
    best_info = None
    smallest_diff = np.Infinity
    for res in resolutions:
        for info in resolution_info:
            resolution_diff = np.abs(res - info.pixel_size)
            if resolution_diff < smallest_diff:
                smallest_diff = resolution_diff
                best_info = info

    return best_info
