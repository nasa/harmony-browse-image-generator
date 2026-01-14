"""Module containing core functionality for browse image generation."""

import re
from itertools import zip_longest
from logging import Logger, getLogger
from pathlib import Path

import numpy as np
import rasterio
from affine import dumpsw
from harmony_service_lib.message import Message as HarmonyMessage
from harmony_service_lib.message import Source as HarmonySource
from matplotlib.colors import BoundaryNorm, Normalize
from numpy.typing import NDArray
from osgeo_utils.auxiliary.color_palette import ColorPalette
from rasterio.io import DatasetReader
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import Window

from hybig.browse_utility import get_harmony_message_from_params
from hybig.color_utility import (
    NODATA_IDX,
    OPAQUE,
    TRANSPARENT,
    ColorMap,
    colormap_from_colors,
    get_color_palette,
    greyscale_colormap,
    palette_from_remote_colortable,
)
from hybig.exceptions import HyBIGError
from hybig.sizes import (
    GridParams,
    create_tiled_output_parameters,
    get_target_grid_parameters,
)

DST_NODATA = NODATA_IDX


def create_browse(
    source_tiff: str,
    params: dict | None = None,
    palette: str | ColorPalette | None = None,
    logger: Logger | None = None,
) -> list[tuple[Path, Path, Path]]:
    """Create browse imagery from an input geotiff.

    This is the exposed library function to allow users to create browse images
    from the hybig-py library. It parses the input params and constructs the
    correct Harmony input structure [Message.Format] to call the service's
    entry point create_browse_imagery.

    Output images are created and deposited into the input GeoTIFF's directory.

    Args:
        source_tiff: str, location of the input geotiff to process.

        params: [dict | None], A dictionary with the following keys:

            mime: [str], MIME type of the output image (default: 'image/png').
                  any string that contains 'jpeg' will return a jpeg image,
                  otherwise create a png.

            crs: [dict | None], Target image's Coordinate Reference System.
                 A dictionary with 'epsg', 'proj4' or 'wkt' key.

            scale_extent: [dict | None], Scale Extents for the image. This dictionary
                contains "x" and "y" keys each whose value which is a dictionary
                of "min", "max" values in the same units as the crs.
                e.g.: { "x": { "min": 0.5, "max": 125 },
                        "y": { "min": 52, "max": 75.22 } }

            scale_size: [dict | None], Scale sizes for the image.  The dictionary
                contains "x" and "y" keys with the horizontal and veritcal
                resolution in the same units as the crs.
                e.g.: { "x": 10, "y": 10 }

            height: [int | None], height of the output image in gridcells.

            width: [int | none], width of the output image in gridcells.

        palette: [str | ColorPalette | none], either a URL to a remote color palette
             that is fetched and loaded or a ColorPalette object used to color
             the output browse image. If not provided, a grayscale image is
             generated.

        logger: [Logger | None], a configured Logger object. If None a default
             logger will be used.

    Note:
      if supplied, scale_size, scale_extent, height and width must be
      internally consistent.  To define a valid output grid:
            * Specify scale_extent and 1 of:
              * height and width
              * scale_sizes (in the x and y horizontal spatial dimensions)
            * Specify all three of the above, but ensure values are consistent
              with one another, noting that:
              scale_size.x = (scale_extent.x.max - scale_extent.x.min) / width
              scale_size.y = (scale_extent.y.max - scale_extent.y.min) / height

    Returns:
        List of 3-element tuples. These are the file paths of:
        - The output browse image
        - Its associated ESRI world file (containing georeferencing information)
        - The auxiliary XML file (containing duplicative georeferencing information)


    Example Usage:
        results = create_browse(
            "/path/to/geotiff",
            {
                "mime": "image/png",
                "crs": {"epsg": "EPSG:4326"},
                "scale_extent": {
                    "x": {"min": -180, "max": 180},
                    "y": {"min": -90, "max": 90},
                },
                "scale_size": {"x": 10, "y": 10},
            },
            "https://remote-colortable",
            logger,
        )

    """
    harmony_message = get_harmony_message_from_params(params)

    if logger is None:
        logger = getLogger('hybig-py')

    if isinstance(palette, str):
        color_palette = palette_from_remote_colortable(palette)
    else:
        color_palette = palette

    return create_browse_imagery(
        harmony_message, source_tiff, HarmonySource({}), color_palette, logger
    )


def create_browse_imagery(
    message: HarmonyMessage,
    input_file_path: str,
    source: HarmonySource,
    item_color_palette: ColorPalette | None,
    logger: Logger,
) -> list[tuple[Path, Path, Path]]:
    """Create browse image from input geotiff.

    Take input browse image and return a 3-element tuple for the file paths of
    the output browse image, its associated ESRI world file and the auxilary
    xml file.

    """
    output_driver = image_driver(message.format.mime)  # type: ignore
    out_image_file = output_image_file(Path(input_file_path), driver=output_driver)
    out_world_file = output_world_file(Path(input_file_path), driver=output_driver)

    try:
        with rasterio.open(input_file_path) as src_ds:
            validate_file_type(src_ds)
            validate_file_crs(src_ds)

            band_count = src_ds.count
            color_palette = None

            if band_count == 1:
                color_palette = get_color_palette(src_ds, source, item_color_palette)
            elif band_count not in (3, 4):
                raise HyBIGError(f'incorrect number of bands for image: {src_ds.count}')

            grid_parameters = get_target_grid_parameters(message, src_ds)
            grid_parameter_list, tile_locators = create_tiled_output_parameters(
                grid_parameters
            )

            # A list of (image_path, world_file_path, aux_xml_path)
            processed_files: list[tuple[Path, Path, Path]] = []
            for grid_params, tile_location in zip_longest(
                grid_parameter_list, tile_locators
            ):
                tiled_out_image_file = get_tiled_filename(out_image_file, tile_location)
                tiled_out_world_file = get_tiled_filename(out_world_file, tile_location)
                tiled_out_aux_xml_file = get_aux_xml_filename(tiled_out_image_file)
                logger.info(f'out image file: {tiled_out_image_file}: {tile_location}')

                process_tile(
                    src_ds,
                    grid_params,
                    color_palette,
                    output_driver,
                    tiled_out_image_file,
                    tiled_out_world_file,
                    logger,
                )
                processed_files.append(
                    (tiled_out_image_file, tiled_out_world_file, tiled_out_aux_xml_file)
                )

    except Exception as exception:
        raise HyBIGError(str(exception)) from exception

    return processed_files


def process_tile(
    src_ds: DatasetReader,
    grid_params: GridParams,
    color_palette: ColorPalette | None,
    output_driver: str,
    out_file_name: Path,
    out_world_name: Path,
    logger: Logger,
) -> None:
    """Read a region from the source dataset, convert raster, and write output."""
    band_count = src_ds.count

    src_window = calculate_source_window(src_ds, grid_params)

    # Tile is outside source bounds
    if src_window is None:
        return

    # Explicitly load a subset of ds
    tile_source = read_window_with_mask_and_scale(src_ds, src_window)
    src_crs = src_ds.crs
    src_transform = src_ds.window_transform(src_window)

    dst_nodata = TRANSPARENT

    if band_count == 1:
        if output_driver == 'JPEG':
            raster, color_map = convert_singleband_to_rgb(tile_source, color_palette)
        else:
            raster, color_map, dst_nodata = convert_singleband_to_raster(
                tile_source, color_palette
            )
    else:
        raster = convert_multiband_to_raster(tile_source)
        color_map = None
        if output_driver == 'JPEG':
            raster = raster[0:3, :, :]

    write_georaster_as_browse(
        raster,
        src_crs,
        src_transform,
        color_map,
        dst_nodata,
        grid_params,
        logger,
        driver=output_driver,
        out_file_name=out_file_name,
        out_world_name=out_world_name,
    )

    # Explicit cleanup
    del raster
    del tile_source


def calculate_source_window(
    src_ds: DatasetReader,
    grid_params: GridParams,
) -> Window | None:
    """Calculate the source window needed to cover the target tile.

    Returns a Window defining which portion of the source to read,
    with some buffer for reprojection edge effects.
    """
    try:
        # Get target tile bounds in target CRS
        dst_height = grid_params['height']
        dst_width = grid_params['width']
        dst_crs = grid_params['crs']
        dst_transform = grid_params['transform']

        # Calculate tile bounds in destination CRS
        dst_left = dst_transform.c
        dst_top = dst_transform.f
        dst_right = dst_left + dst_width * dst_transform.a
        dst_bottom = dst_top + dst_height * dst_transform.e

        dst_bounds = (
            min(dst_left, dst_right),
            min(dst_top, dst_bottom),
            max(dst_left, dst_right),
            max(dst_top, dst_bottom),
        )

        # Transform bounds to source CRS
        src_crs = src_ds.crs
        src_bounds = transform_bounds(dst_crs, src_crs, *dst_bounds)

        # Add buffer for reprojection (10% on each side)
        width = src_bounds[2] - src_bounds[0]
        height = src_bounds[3] - src_bounds[1]
        buffer_x = width * 0.1
        buffer_y = height * 0.1

        buffered_bounds = (
            src_bounds[0] - buffer_x,
            src_bounds[1] - buffer_y,
            src_bounds[2] + buffer_x,
            src_bounds[3] + buffer_y,
        )

        # Convert to window in source pixel coordinates
        src_transform = src_ds.transform
        if len(src_ds.shape) == 3:
            src_height, src_width = src_ds.shape[1], src_ds.shape[2]
        else:
            # Single band
            src_height, src_width = src_ds.shape[0], src_ds.shape[1]

        # Inverse transform to get pixel coordinates from geographic coordinates
        # For a point (x, y), the pixel coordinate is:
        #   col = (x - transform.c) / transform.a
        #   row = (y - transform.f) / transform.e
        # Works like rasterio.windows.from_bounds but also handles positive y pixel size
        # which is an edge case for some datasets like PODAAC's GHRSST MUR.
        left, bottom, right, top = buffered_bounds
        col_left = (left - src_transform.c) / src_transform.a
        col_right = (right - src_transform.c) / src_transform.a

        row_top = (top - src_transform.f) / src_transform.e
        row_bottom = (bottom - src_transform.f) / src_transform.e

        # Handle both positive and negative y scales
        col_min = min(col_left, col_right)
        col_max = max(col_left, col_right)
        row_min = min(row_top, row_bottom)
        row_max = max(row_top, row_bottom)

        # Convert to integer pixel bounds and clip to image extent
        col_off = max(0, int(np.floor(col_min)))
        row_off = max(0, int(np.floor(row_min)))
        col_end = min(src_width, int(np.ceil(col_max)))
        row_end = min(src_height, int(np.ceil(row_max)))

        win_width = col_end - col_off
        win_height = row_end - row_off

        if win_width <= 0 or win_height <= 0:
            return None

        return Window(col_off, row_off, win_width, win_height)  # type: ignore

    except Exception:
        # If calculation fails, return None
        return None


def read_window_with_mask_and_scale(
    src_ds: DatasetReader,
    window: Window,
    bands: list[int] | None = None,
) -> NDArray:
    """Read a window from a rasterio dataset with masking and scaling applied.

    Replicates the behavior of rioxarray's mask_and_scale=True option.
    """
    if bands is None:
        bands = list(range(1, src_ds.count + 1))

    data = src_ds.read(bands, window=window)

    # Convert to float for NaN support
    data = data.astype('float64')

    # Apply masking and scaling per band
    for i, band_idx in enumerate(bands):
        band_data = data[i]  # note that this passes by reference

        # Get nodata value for this band
        nodata = src_ds.nodatavals[band_idx - 1]  # nodatavals is 0-indexed

        # Mask nodata values
        if nodata is not None:
            mask = np.isnan(band_data) if np.isnan(nodata) else (band_data == nodata)
            band_data[mask] = np.nan

        scale = (src_ds.scales or [None])[band_idx - 1] or 1.0
        offset = (src_ds.offsets or [None])[band_idx - 1] or 0.0

        # Apply scale/offset to non-NaN values
        if scale != 1.0 or offset != 0.0:
            valid_mask = ~np.isnan(band_data)
            band_data[valid_mask] = band_data[valid_mask] * scale + offset

    return data


def convert_multiband_to_raster(data_array: NDArray) -> NDArray[np.uint8]:
    """Convert multiband to a raster image.

    Return a 4-band raster, where the alpha layer is presumed to be the missing
    data mask.

    Convert 3-band data into a 4-band raster by generating an alpha layer from
    any missing data in the RGB bands.

    """
    if data_array.shape[0] not in [3, 4]:
        raise HyBIGError(
            f'Cannot create image from {data_array.shape[0]} band image. '
            'Expecting 3 or 4 bands.'
        )

    if data_array.shape[0] == 4:
        return convert_to_uint8(data_array, str(data_array.dtype))

    # Input NaNs in any of the RGB bands are made transparent.
    nan_mask = np.isnan(data_array).any(axis=0)
    nan_alpha = np.where(nan_mask, TRANSPARENT, OPAQUE)

    raster = convert_to_uint8(data_array, str(data_array.dtype))

    return np.concatenate((raster, nan_alpha[None, ...]), axis=0)


def convert_to_uint8(bands: NDArray, dtype: str | None) -> NDArray[np.uint8]:
    """Convert banded data with NaNs (missing) into a uint8 data cube."""
    max_val = np.nanmax(bands)

    # previously this used scaled.filled(0) which only works on masked arrays
    if dtype != 'uint8' and max_val > 255:
        min_val = np.nanmin(bands)
        # Normalize to 0-255 range
        with np.errstate(invalid='ignore'):  # Suppress NaN warnings
            scaled = (bands - min_val) / (max_val - min_val) * 255.0
        return np.nan_to_num(np.around(scaled), nan=0).astype('uint8')

    return np.nan_to_num(bands, nan=0).astype('uint8')


def convert_singleband_to_raster(
    data_array: NDArray,
    color_palette: ColorPalette | None = None,
) -> tuple[NDArray, ColorMap, np.uint8]:
    """Convert input dataset to a 1-band palettized image with colormap.

    Uses a palette if provided otherwise returns a greyscale image.
    """
    if color_palette is None:
        return scale_grey_1band(data_array)
    return scale_paletted_1band(data_array, color_palette)


def scale_grey_1band(data_array: NDArray) -> tuple[NDArray, ColorMap, np.uint8]:
    """Normalize input array and return scaled data with greyscale ColorMap."""
    band = data_array[0, :, :]

    # Scale input data from 0 to 254
    norm = Normalize(vmin=np.nanmin(band), vmax=np.nanmax(band))
    normalized_data = norm(band) * 254.0

    # Set any missing (nan) to palette's NODATA_IDX
    result = np.round(normalized_data)
    result[np.isnan(band)] = NODATA_IDX

    return (
        result.astype('uint8')[np.newaxis, :, :],
        greyscale_colormap(),
        np.uint8(NODATA_IDX),
    )


def convert_singleband_to_rgb(
    data_array: NDArray,
    color_palette: ColorPalette | None = None,
) -> tuple[NDArray, None]:
    """Convert input 1-band dataset to RGB image for JPEG output.

    Uses a palette if provided, otherwise returns a greyscale RGB image.
    Returns a 3-band RGB array and None for colormap (since RGB doesn't need colormap).
    """
    if color_palette is None:
        return scale_grey_1band_to_rgb(data_array)
    return scale_paletted_1band_to_rgb(data_array, color_palette)


def scale_grey_1band_to_rgb(data_array: NDArray) -> tuple[NDArray, None]:
    """Normalize input array and return as 3-band RGB grayscale image."""
    band = data_array[0, :, :]

    # Scale input data from 0 to 254. Note that this means nodata and
    # the valid data min will occupy the same color level
    norm = Normalize(vmin=np.nanmin(band), vmax=np.nanmax(band))
    normalized_data = norm(band) * 254.0

    # Set any missing (nan) to 0 black
    normalized_data[np.isnan(band)] = 0

    grey_data = np.round(normalized_data).astype('uint8')
    return np.stack([grey_data, grey_data, grey_data], axis=0), None


def prepare_palette_colors(
    palette: ColorPalette, with_alpha: bool = True
) -> tuple[list[tuple], tuple, int | None]:
    """Extract colors and nodata handling from a palette.

    Returns:
        Tuple of (colors_list, nodata_color, nodata_index_or_none)
    """
    colors = [
        palette.color_to_color_entry(value, with_alpha=with_alpha)
        for value in palette.pal.values()
    ]

    nodata_color = (0, 0, 0, 0) if with_alpha else (0, 0, 0)
    nodata_index = None

    if palette.ndv is not None:
        nodata_color = palette.color_to_color_entry(palette.ndv, with_alpha=with_alpha)
        if palette.ndv in palette.pal.values():
            nodata_index = list(palette.pal.values()).index(palette.ndv)

    return colors, nodata_color, nodata_index


def scale_paletted_1band_to_rgb(
    data_array: NDArray, palette: ColorPalette
) -> tuple[NDArray, None]:
    """Scale a 1-band image with palette into RGB image for JPEG output."""
    band = data_array[0, :, :]
    levels = list(palette.pal.keys())
    colors, nodata_color, _ = prepare_palette_colors(palette, with_alpha=False)
    colors_array = np.array(colors, dtype='uint8')

    norm = BoundaryNorm(levels, len(levels) - 1)

    # Store NaN mask before normalization
    nan_mask = np.isnan(band)

    # Replace NaN with first level to avoid issues during normalization
    band_clean = np.where(nan_mask, levels[0], band)

    # Get palette indices and clip to valid range
    indexed_band = np.clip(norm(band_clean), 0, len(colors) - 1).astype(int)

    # Vectorized color lookup
    rgb_array = colors_array[indexed_band].transpose(2, 0, 1)

    # Handle nodata (overwrite any color assignment)
    if nan_mask.any():
        rgb_array[0, nan_mask] = nodata_color[0]
        rgb_array[1, nan_mask] = nodata_color[1]
        rgb_array[2, nan_mask] = nodata_color[2]

    return np.ascontiguousarray(rgb_array), None


def scale_paletted_1band(
    data_array: NDArray, palette: ColorPalette
) -> tuple[NDArray, ColorMap, np.uint8]:
    """Scale a 1-band image with palette into modified image and associated color_map.

    Use the palette's levels and values, transform the input data_array into
    the correct levels indexed from 0-255 return the scaled array alongside
    a colormap corresponding to the new levels.

    Values below the minimum palette level are clipped to the lowest color.
    Values above the maximum palette level are clipped to the highest color.
    Only NaN values are mapped to the nodata index.

    Returns:
        Tuple of (raster_data, color_map, nodata_index)
    """
    band = data_array[0, :, :]
    levels = list(palette.pal.keys())
    colors, nodata_color, existing_nodata_idx = prepare_palette_colors(
        palette, with_alpha=True
    )

    # Determine where nodata sits in the final colormap
    if existing_nodata_idx is not None:
        # Don't add nodata_color; it's already in colors
        dst_nodata = np.uint8(existing_nodata_idx)
    elif palette.ndv is not None:
        # Nodata not in palette, add it at the beginning
        dst_nodata = np.uint8(0)
        colors = [nodata_color, *colors]
        # This check is done explicitly since some colormaps may be <256 elements
        if len(colors) > 256:
            colors = colors[:-1]
    else:
        # if there is no ndv, add one to the end of the colormap
        dst_nodata = np.uint8(len(colors))
        colors = [*colors, nodata_color]
        # This check is done explicitly since some colormaps may be <256 elements
        if len(colors) > 256:
            colors = [*colors[:-2], nodata_color]

    norm = BoundaryNorm(levels, len(levels) - 1)

    nan_mask = np.isnan(band)
    if band.flags.writeable:
        band[nan_mask] = levels[0]
        band_clean = band
    else:
        band_clean = np.where(nan_mask, levels[0], band)
    scaled_band = norm(band_clean)

    # Apply offset and clip to valid palette range
    if dst_nodata == 0:
        # boundary norm indexes [0, levels) by default, so if the NODATA index is 0,
        # all the palette indices need to be incremented by 1.
        scaled_band = scaled_band + 1
        np.clip(scaled_band, 1, len(colors) - 1, out=scaled_band)
    else:
        # Palette occupies indices 0 to dst_nodata-1
        np.clip(scaled_band, 0, dst_nodata - 1, out=scaled_band)

    # Only set NaN values to nodata index
    scaled_band[nan_mask] = dst_nodata

    del nan_mask
    del band_clean

    color_map = colormap_from_colors(colors)
    raster = scaled_band.data.astype('uint8')[np.newaxis, :, :]
    return raster, color_map, dst_nodata


def image_driver(mime: str) -> str:
    """Return requested rasterio driver for output image."""
    if re.search('jpeg', mime, re.I) or re.search('jpg', mime, re.I):
        return 'JPEG'
    return 'PNG'


def get_aux_xml_filename(image_filename: Path) -> Path:
    """Get aux.xml filenames."""
    return image_filename.with_suffix(image_filename.suffix + '.aux.xml')


def get_tiled_filename(input_file: Path, locator: dict[str, int] | None = None) -> Path:
    """Add a column, row identifier to the output files.

    Only update if there is a valid locator dict.
    """
    if locator is not None:
        return input_file.with_suffix(
            f'.r{int(locator["row"]):02d}c{int(locator["col"]):02d}{input_file.suffix}'
        )
    return input_file


def output_image_file(input_file_path: Path, driver: str = 'PNG'):
    """Generate the output image name."""
    if driver == 'PNG':
        ext = '.png'
    else:
        ext = '.jpg'
    return input_file_path.with_suffix(ext)


def output_world_file(input_file_path: Path, driver: str = 'PNG'):
    """Generate an output world file name."""
    if driver == 'PNG':
        ext = '.pgw'
    else:
        ext = '.jgw'
    return input_file_path.with_suffix(ext)


def validate_file_crs(src_ds: DatasetReader) -> None:
    """Explicit check for a CRS on the input geotiff.

    Raises HyBIGError if crs is missing.
    """
    if src_ds.crs is None:
        raise HyBIGError('Input geotiff must have defined CRS.')


def validate_file_type(src_ds: DatasetReader) -> None:
    """Ensure we can work with the input data file.

    Raise an exception if this file is unusable by the service.

    """
    if src_ds.driver != 'GTiff':
        raise HyBIGError(f'Input file type not supported: {src_ds.driver}')


def get_destination(grid_parameters: GridParams, n_bands: int) -> NDArray:
    """Initialize an array for writing an output raster."""
    return np.zeros(
        (n_bands, grid_parameters['height'], grid_parameters['width']), dtype='uint8'
    )


def write_georaster_as_browse(
    raster: NDArray,
    src_crs: rasterio.CRS,
    src_transform: rasterio.Affine,
    color_map: dict | None,
    dst_nodata: int | np.uint8,
    grid_parameters: GridParams,
    logger: Logger,
    driver: str = 'PNG',
    out_file_name: str | Path = 'outfile.png',
    out_world_name: str | Path = 'outfile.pgw',
) -> None:
    """Write raster data to output file.

    Writes the raster to an output file using metadata from the original
    source, and over-riding some values based on the input message and derived
    grid_parameters, doing a nearest neighbor resampling to the target grid.

    """
    n_bands = raster.shape[0]

    creation_options = {
        **grid_parameters,
        'driver': driver,
        'dtype': 'uint8',
        'count': n_bands,
    }

    dst_array = get_destination(grid_parameters, n_bands)

    logger.info(f'Create output image with options: {creation_options}')

    with rasterio.open(out_file_name, 'w', **creation_options) as dst_raster:
        for dim in range(0, n_bands):
            reproject(
                source=raster[dim, :, :],
                destination=dst_array[dim, :, :],
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=grid_parameters['transform'],
                dst_crs=grid_parameters['crs'],
                dst_nodata=int(dst_nodata),
                resampling=Resampling.nearest,
            )

        dst_raster.write(dst_array)
        if color_map is not None:
            dst_raster.write_colormap(1, color_map)

    with open(out_world_name, 'w', encoding='UTF-8') as out_wd:
        out_wd.write(dumpsw(creation_options['transform']))
