"""Utility routines to aid in determining target size parameters.

This module handles determiniation of the output image size parameters.  These
are geared towards satisfying GIBS requirements, but allows a user to override
most options from the input Harmony Message.

There's a number of "rules" from the GIBS ICD that are codified in this module.

"""

from collections import namedtuple
from typing import TypedDict

import numpy as np
from affine import Affine
from harmony.message import Message
from harmony.message_utility import has_dimensions, has_scale_extents, has_scale_sizes

# pylint: disable-next=no-name-in-module
from rasterio.crs import CRS
from rasterio.transform import AffineTransformer, from_bounds, from_origin
from xarray import DataArray

from hybig.crs import (
    choose_target_crs,
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
    'ResolutionInfo',
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

# Conversion used above: resolution / pixel_size
METERS_PER_DEGREE = 113777.77777777778

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


def get_target_grid_parameters(message: Message, data_array: DataArray) -> GridParams:
    """Get the output image parameters.

    This computes the target grid of the ouptut image. The grid is defined by
    of the extent of the output: ScaleExtents, and either the grid cell sizes:
    ScaleSizes or the dimensions: width and height.

    - User submitted parameters take precedence over computed
    parameters.

    - Computed parameters attempt to generate GIBS suitable images.

    """
    target_crs = choose_target_crs(message.format.srs, data_array)
    target_scale_extent = choose_scale_extent(message, target_crs, data_array)
    target_dimensions = choose_target_dimensions(
        message, data_array, target_scale_extent, target_crs
    )
    return get_rasterio_parameters(target_crs, target_scale_extent, target_dimensions)


def choose_scale_extent(
    message: Message, target_crs: CRS, data_array: DataArray
) -> ScaleExtent:
    """Return the scaleExtent for the target image.

    Returns a scale extent found in the input Message.

    Otherwise, computes a bounding box in the target CRS based on the input
    granule extent.

    """
    if has_scale_extents(message):
        # These values must be in the target_crs projection.
        scale_extent = ScaleExtent(
            {
                'xmin': message.format.scaleExtent.x.min,
                'ymin': message.format.scaleExtent.y.min,
                'xmax': message.format.scaleExtent.x.max,
                'ymax': message.format.scaleExtent.y.max,
            }
        )
    else:
        left, bottom, right, top = data_array.rio.transform_bounds(target_crs)
        scale_extent = ScaleExtent(
            {'xmin': left, 'ymin': bottom, 'xmax': right, 'ymax': top}
        )
    return scale_extent


def choose_target_dimensions(
    message: Message, data_array: DataArray, scale_extent: ScaleExtent, target_crs: CRS
) -> Dimensions:
    """This selects or computes the target Dimensions.

    This routine finalizes the output grid.  The target dimensions are
    returned.  These along with the CRS and scaleExent will be used to
    selecting the correct output transformation for the final image.

    To determine the target dimensions the following occurs.

    The input harmony message format is searched for dimensions or scaleSizes:
    If dimensions (height and width) are found, they are returned.
    If scaleSize is found, it is converted to height and width and returned.

    If no scaleSize or dimensions are input. The input dataset metadata
    resolution is checked against the GIBS preferred resolutions and the best
    fit resolution along with the scaleExtent is used to compute the target
    dimensions to return.

    """
    if has_dimensions(message):
        dimensions = Dimensions(
            {'height': message.format.height, 'width': message.format.width}
        )
    elif has_scale_sizes(message):
        dimensions = compute_target_dimensions(
            scale_extent, message.format.scaleSize.x, message.format.scaleSize.y
        )
    else:
        dimensions = best_guess_target_dimensions(data_array, scale_extent, target_crs)

    return dimensions


def get_rasterio_parameters(
    crs: CRS, scale_extent: ScaleExtent, dimensions: Dimensions
) -> GridParams:
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
    grid_parameters: GridParams,
) -> tuple[list[GridParams], list[dict] | list[None]]:
    """Split the output grid if necessary.

    When the number of grid cells exceeds 8192x8192, we tile the output
    to a reasonable size: 4096x4096

    returns a list of GridParams object that completely cover the input scale
    extent tiled as needed.

    """
    if not needs_tiling(grid_parameters):
        return [grid_parameters], [None]

    crs = grid_parameters['crs']
    full_width = grid_parameters['width']
    full_height = grid_parameters['height']
    transform = grid_parameters['transform']
    resolution_x = transform.a
    resolution_y = np.abs(transform.e)

    cells_per_tile_width = get_cells_per_tile()
    cells_per_tile_height = get_cells_per_tile()

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
                grid_parameter_list.append(
                    GridParams(
                        {
                            'width': width,
                            'height': height,
                            'crs': crs,
                            'transform': tile_grid_transform,
                        }
                    )
                )
                tile_locator.append({'row': h_idx, 'col': w_idx})

    return grid_parameter_list, tile_locator


def compute_tile_dimensions(origins: list[int]) -> list[int]:
    """Return a list of tile dimensions.

    From a list of origin locations, return the dimension for each tile.

    """
    return list(np.diff(origins, append=origins[-1]).astype('int'))


def compute_tile_boundaries(target_size: int, full_size: int) -> list[int]:
    """Returns a list of boundary cells.

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


def get_cells_per_tile() -> int:
    """Optimum cells per tile.

    From discussions this is chosen to be 4096, so that any image that is tiled
    will end up with 4096x4096 gridcell tiles.
    """
    return 4096


def needs_tiling(grid_parameters: GridParams) -> bool:
    """Returns true if the grid is too large for GIBS.

    From discussion, this limit is set to 8192*8192 cells.
    """
    MAX_UNTILED_GRIDCELLS = 8192 * 8192
    return grid_parameters['height'] * grid_parameters['width'] > MAX_UNTILED_GRIDCELLS


def best_guess_target_dimensions(
    data_array: DataArray, scale_extent: ScaleExtent, target_crs: CRS
) -> Dimensions:
    """Return best guess for output image dimensions.

    Using the information from the scaleExtent and the input dataset metadata,
    compute the height and width appropriate for the output image.

    North and South projections have matching resolutions so we just use the
    3413 resolutions for all projected data.
    """
    resolution_list = None
    if not target_crs.is_projected:
        resolution_list = epsg_4326_resolutions
    else:
        resolution_list = epsg_3413_resolutions

    x_res, y_res = resolution_in_target_crs_units(data_array, target_crs)
    return guess_dimensions(x_res, y_res, scale_extent, resolution_list)


def resolution_in_target_crs_units(
    data_array: DataArray, target_crs: CRS
) -> tuple[float, float]:
    """Return the x and y target resolutions

    The input resolution can be determined from the Affine transformation, but
    if the input dataset is projected, and the target CRS is unprojected, we
    need to convert the resolution meters into degrees.

    This routine is only called for best_guess_target_dimensions which means
    the user has not supplied any input parameters and we are trying to
    determine the dimensions for the output image.
    """
    if data_array.rio.crs.is_projected == target_crs.is_projected:
        x_res = data_array.rio.transform().a
        y_res = abs(data_array.rio.transform().e)
    elif target_crs.is_projected:
        # transform from latlon to meters
        x_res = data_array.rio.transform().a * METERS_PER_DEGREE
        y_res = abs(data_array.rio.transform().e) * METERS_PER_DEGREE
    else:
        # transform from meters to lat/lon
        x_res = data_array.rio.transform().a / METERS_PER_DEGREE
        y_res = abs(data_array.rio.transform().e) / METERS_PER_DEGREE

    return x_res, y_res


def guess_dimensions(
    x_res: float,
    y_res: float,
    scale_extent: ScaleExtent,
    icd_resolution_list: list[ResolutionInfo],
) -> Dimensions:
    """Guess Dimensions given the input information and ICD resolutions."""
    coarsest_resolution = icd_resolution_list[0].pixel_size

    if x_res > coarsest_resolution and y_res > coarsest_resolution:
        dimensions = compute_target_dimensions(scale_extent, x_res, y_res)
    else:
        target_res = find_closest_resolution(list({x_res, y_res}), icd_resolution_list)
        dimensions = compute_target_dimensions(
            scale_extent, target_res.pixel_size, target_res.pixel_size
        )
    return dimensions


def compute_target_dimensions(
    scale_extent: ScaleExtent, x_res: float, y_res: float
) -> Dimensions:
    """Compute dimensions from inputs.

    We round the computed dimensions because we need to have integer values of
    height and width. Because of the rounding, this function could return a
    height/width that in combination with the ScaleExtent would be slightly
    inconsistent with the input res (ScaleSize).  This is ok, because for
    computed values of scaleExtent, we are already self consistent.  If there's
    a discrepancy, the user has described a scaleExtent in the input
    HarmonyMessage and has *not* specified a scaleSize. So we honor the
    scaleExtent in a previous function.

    """
    return {
        'height': round((scale_extent['ymax'] - scale_extent['ymin']) / y_res),
        'width': round((scale_extent['xmax'] - scale_extent['xmin']) / x_res),
    }


def find_closest_resolution(
    resolutions: list[float], resolution_info: list[ResolutionInfo]
) -> ResolutionInfo | None:
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
