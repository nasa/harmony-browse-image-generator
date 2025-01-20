"""Module containing core functionality for browse image generation."""

import re
from itertools import zip_longest
from logging import Logger, getLogger
from pathlib import Path

import matplotlib
import numpy as np
import rasterio
from affine import dumpsw
from harmony_service_lib.message import Message as HarmonyMessage
from harmony_service_lib.message import Source as HarmonySource
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from numpy import ndarray, uint8
from osgeo_utils.auxiliary.color_palette import ColorPalette
from PIL import Image
from rasterio.io import DatasetReader
from rasterio.plot import reshape_as_image, reshape_as_raster
from rasterio.warp import Resampling, reproject
from rioxarray import open_rasterio
from xarray import DataArray

from hybig.browse_utility import get_harmony_message_from_params
from hybig.color_utility import (
    NODATA_IDX,
    NODATA_RGBA,
    OPAQUE,
    TRANSPARENT,
    ColorMap,
    all_black_color_map,
    colormap_from_colors,
    get_color_palette,
    greyscale_colormap,
    palette_from_remote_colortable,
    remove_alpha,
)
from hybig.exceptions import HyBIGError
from hybig.sizes import (
    GridParams,
    create_tiled_output_parameters,
    get_target_grid_parameters,
)


def create_browse(
    source_tiff: str,
    params: dict = None,
    palette: str | ColorPalette | None = None,
    logger: Logger = None,
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
    logger.info(message)
    output_driver = image_driver(message.format.mime)
    out_image_file = output_image_file(Path(input_file_path), driver=output_driver)
    out_world_file = output_world_file(Path(input_file_path), driver=output_driver)

    try:
        with open_rasterio(
            input_file_path, mode='r', mask_and_scale=True
        ) as rio_in_array:
            in_dataset = rio_in_array.rio._manager.acquire()
            validate_file_type(in_dataset)
            validate_file_crs(rio_in_array)

            if rio_in_array.rio.count == 1:
                color_palette = get_color_palette(
                    in_dataset, source, item_color_palette
                )
                raster, color_map = convert_singleband_to_raster(
                    rio_in_array, color_palette
                )
            elif rio_in_array.rio.count in (3, 4):
                raster = convert_mulitband_to_raster(rio_in_array)
                color_map = None
                if output_driver == 'JPEG':
                    raster = raster[0:3, :, :]
            else:
                raise HyBIGError(
                    f'incorrect number of bands for image: {rio_in_array.rio.count}'
                )

            grid_parameters = get_target_grid_parameters(message, rio_in_array)
            grid_parameter_list, tile_locators = create_tiled_output_parameters(
                grid_parameters
            )

            processed_files = []
            for grid_parameters, tile_location in zip_longest(
                grid_parameter_list, tile_locators
            ):
                tiled_out_image_file = get_tiled_filename(out_image_file, tile_location)
                tiled_out_world_file = get_tiled_filename(out_world_file, tile_location)
                tiled_out_aux_xml_file = get_aux_xml_filename(tiled_out_image_file)
                logger.info(f'out image file: {tiled_out_image_file}: {tile_location}')

                write_georaster_as_browse(
                    rio_in_array,
                    raster,
                    color_map,
                    grid_parameters,
                    logger=logger,
                    driver=output_driver,
                    out_file_name=tiled_out_image_file,
                    out_world_name=tiled_out_world_file,
                )
                processed_files.append(
                    (tiled_out_image_file, tiled_out_world_file, tiled_out_aux_xml_file)
                )

    except Exception as exception:
        raise HyBIGError(str(exception)) from exception

    return processed_files


def convert_mulitband_to_raster(data_array: DataArray) -> ndarray[uint8]:
    """Convert multiband to a raster image.

    Return a 4-band raster, where the alpha layer is presumed to be the missing
    data mask.

    Convert 3-band data into a 4-band raster by generating an alpha layer from
    any missing data in the RGB bands.

    """
    if data_array.rio.count not in [3, 4]:
        raise HyBIGError(
            f'Cannot create image from {data_array.rio.count} band image. '
            'Expecting 3 or 4 bands.'
        )

    bands = data_array.to_numpy()

    if data_array.rio.count == 4:
        return convert_to_uint8(bands, original_dtype(data_array))

    # Input NaNs in any of the RGB bands are made transparent.
    nan_mask = np.isnan(bands).any(axis=0)
    nan_alpha = np.where(nan_mask, TRANSPARENT, OPAQUE)

    raster = convert_to_uint8(bands, original_dtype(data_array))

    return np.concatenate((raster, nan_alpha[None, ...]), axis=0)


def convert_to_uint8(bands: ndarray, dtype: str | None) -> ndarray[uint8]:
    """Convert Banded data with NaNs (missing) into a uint8 data cube.

    Nearly all of the time this will simply pass through the data coercing it
    back into unsigned ints and setting the missing values to 0 that will be
    masked as transparent in the output png.

    There is a some small non-zero chance that the input RGB image was 16-bit
    and if any of the values exceed 255, we must normalize all of input data to
    the range 0-255.

    """

    if dtype != 'uint8' and np.nanmax(bands) > 255:
        norm = Normalize(vmin=np.nanmin(bands), vmax=np.nanmax(bands))
        scaled = np.around(norm(bands) * 255.0)
        raster = scaled.filled(0).astype('uint8')
    else:
        raster = np.nan_to_num(bands).astype('uint8')

    return raster


def original_dtype(data_array: DataArray) -> str | None:
    """Return the original input data's type.

    rastero_open retains the input dtype in the encoding dictionary and is used
    to understand what kind of casts are safe.

    """
    return data_array.encoding.get('dtype') or data_array.encoding.get('rasterio_dtype')


def convert_singleband_to_raster(
    data_array: DataArray,
    color_palette: ColorPalette | None = None,
) -> tuple[ndarray, ColorMap]:
    """Convert input dataset to a 4 band raster image.

    Use a palette if provided otherwise return a greyscale image.
    """
    if color_palette is None:
        return scale_grey_1band(data_array)
    return scale_paletted_1band(data_array, color_palette)


def convert_gray_1band_to_raster(data_array: DataArray) -> ndarray:
    """Convert a 1-band raster without a color association."""
    band = data_array[0, :, :]
    cmap = matplotlib.colormaps['Greys_r']
    cmap.set_bad(NODATA_RGBA)
    norm = Normalize(vmin=np.nanmin(band), vmax=np.nanmax(band))
    scalar_map = ScalarMappable(cmap=cmap, norm=norm)

    rgba_image = np.zeros((*band.shape, 4), dtype='uint8')
    for row_no in range(band.shape[0]):
        rgba_image_slice = scalar_map.to_rgba(band[row_no, :], bytes=True)
        rgba_image[row_no, :, :] = rgba_image_slice

    return reshape_as_raster(rgba_image)


def scale_grey_1band(data_array: DataArray) -> tuple[ndarray, ColorMap]:
    """Normalize input array and return scaled data with greyscale ColorMap."""
    band = data_array[0, :, :]
    norm = Normalize(vmin=np.nanmin(band), vmax=np.nanmax(band))

    # Scale input data from 0 to 254
    normalized_data = norm(band) * 254.0

    # Set any missing to missing
    normalized_data[np.isnan(normalized_data)] = NODATA_IDX

    grey_colormap = greyscale_colormap()
    raster_data = np.expand_dims(np.round(normalized_data).data, 0)
    return raster_data, grey_colormap


def scale_paletted_1band(
    data_array: DataArray, palette: ColorPalette
) -> tuple[ndarray, ColorMap]:
    """Scale a 1-band image with palette into modified image and associated color_map.

    Use the palette's levels and values, transform the input data_array into
    the correct levels indexed from 0-255 return the scaled array along side of
    a colormap corresponding to the new levels.
    """
    band = data_array[0, :, :]
    levels = list(palette.pal.keys())
    colors = [
        palette.color_to_color_entry(value, with_alpha=True)
        for value in palette.pal.values()
    ]
    norm = matplotlib.colors.BoundaryNorm(levels, len(levels) - 1)

    # handle palette no data value
    nodata_color = (0, 0, 0, 0)
    if palette.ndv is not None:
        nodata_color = palette.color_to_color_entry(palette.ndv, with_alpha=True)

    colors = [*colors, nodata_color]

    scaled_band = norm(band)

    # Set underflow and nan values to nodata
    scaled_band[scaled_band == -1] = len(colors) - 1
    scaled_band[np.isnan(band)] = len(colors) - 1

    color_map = colormap_from_colors(colors)
    return np.expand_dims(scaled_band.data, 0), color_map


def convert_paletted_1band_to_raster(
    data_array: DataArray, palette: ColorPalette
) -> ndarray:
    """Convert a 1 band image with palette into a rgba raster image."""
    band = data_array[0, :, :]
    levels = list(palette.pal.keys())
    colors = [
        palette.color_to_color_entry(value, with_alpha=True)
        for value in palette.pal.values()
    ]
    scaled_colors = [
        (r / 255.0, g / 255.0, b / 255.0, a / 255.0) for r, g, b, a in colors
    ]

    cmap, norm = matplotlib.colors.from_levels_and_colors(
        levels, scaled_colors, extend='max'
    )

    # handle palette no data value
    if palette.ndv is not None:
        nodata_colors = palette.color_to_color_entry(palette.ndv, with_alpha=True)
        cmap.set_bad(
            (
                nodata_colors[0] / 255.0,
                nodata_colors[1] / 255.0,
                nodata_colors[2] / 255.0,
                nodata_colors[3] / 255.0,
            )
        )

    scalar_map = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    rgba_image = np.zeros((*band.shape, 4), dtype='uint8')
    for row_no in range(band.shape[0]):
        rgba_image[row_no, :, :] = scalar_map.to_rgba(
            np.ma.masked_invalid(band[row_no, :]), bytes=True
        )
    return reshape_as_raster(rgba_image)


def image_driver(mime: str) -> str:
    """Return requested rasterio driver for output image."""
    if re.search('jpeg', mime, re.I):
        return 'JPEG'
    return 'PNG'


def palettize_raster(raster: ndarray) -> tuple[ndarray, dict]:
    """Convert an RGB or RGBA image into a 1band image and palette.

    Converts a 3 or 4 band np raster into a PIL image.
    Quantizes the image into a 1band raster with palette

    Transparency is handled by first removing the Alpha layer and creating
    quantized raster from just the RGB layers. Next the Alpha layer values are
    treated as either transparent or opaque and any transparent values are
    written to the final raster as 254 and add the mapped RGBA value to the
    color palette.
    """
    # reserves index 255 for transparent and off grid fill values
    # 0 to 254
    max_colors = 255
    rgb_raster, alpha = remove_alpha(raster)

    multiband_image = Image.fromarray(reshape_as_image(rgb_raster))
    quantized_image = multiband_image.quantize(colors=max_colors)

    color_map = get_color_map_from_image(quantized_image)

    quantized_array, color_map = add_alpha(alpha, np.array(quantized_image), color_map)

    one_band_raster = np.expand_dims(quantized_array, 0)
    return one_band_raster, color_map


def add_alpha(
    alpha: ndarray | None, quantized_array: ndarray, color_map: dict
) -> tuple[ndarray, dict]:
    """If the input data had alpha values, manually set the quantized_image
    index to the transparent index in those places.
    """
    if alpha is not None and np.any(alpha != OPAQUE):
        # Set any alpha to the transparent index value
        quantized_array = np.where(alpha != OPAQUE, NODATA_IDX, quantized_array)
        color_map[NODATA_IDX] = NODATA_RGBA
    return quantized_array, color_map


def get_color_map_from_image(image: Image) -> dict:
    """Get a writable color map

    Read the RGBA palette from a PIL Image and covert into a dictionary
    that can be written by rasterio.

    """
    color_tuples = np.array(image.getpalette(rawmode='RGBA')).reshape(-1, 4)
    color_map = all_black_color_map()
    for idx, color_tuple in enumerate(color_tuples):
        color_map[idx] = tuple(color_tuple)
    return color_map


def get_colormap_from_scaled_colors(scaled_colors) -> dict:
    pass


def get_aux_xml_filename(image_filename: Path) -> Path:
    """Get aux.xml filenames."""
    return image_filename.with_suffix(image_filename.suffix + '.aux.xml')


def get_tiled_filename(input_file: Path, locator: dict | None = None) -> Path:
    """Add a column, row identifier to the output files.

    Only update if there is a valid locator dict.
    """
    if locator is not None:
        return input_file.with_suffix(
            f".r{int(locator['row']):02d}c{int(locator['col']):02d}{input_file.suffix}"
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


def validate_file_crs(data_array: DataArray) -> None:
    """Explicit check for a CRS on the input geotiff.

    Raises HyBIGError if crs is missing.
    """
    if data_array.rio.crs is None:
        raise HyBIGError('Input geotiff must have defined CRS.')


def validate_file_type(dsr: DatasetReader) -> None:
    """Ensure we can work with the input data file.

    Raise an exception if this file is unusable by the service.

    """
    if dsr.driver != 'GTiff':
        raise HyBIGError(f'Input file type not supported: {dsr.driver}')


def get_destination(grid_parameters: GridParams, n_bands: int) -> ndarray:
    """Initialize an array for writing an output raster."""
    return np.zeros(
        (n_bands, grid_parameters['height'], grid_parameters['width']), dtype='uint8'
    )


def write_georaster_as_browse(
    data_array: DataArray,
    raster: ndarray,
    color_map: dict | None,
    grid_parameters: GridParams,
    driver='PNG',
    out_file_name='outfile.png',
    out_world_name='outfile.pgw',
    logger=Logger,
) -> None:
    """Write raster data to output file.

    Writes the raster to an output file using metadata from the original
    source, and over-riding some values based on the input message and derived
    grid_parameters, doing a nearest neighbor resampling to the target grid.

    """
    n_bands = raster.shape[0]

    if color_map is not None:
        dst_nodata = NODATA_IDX
        color_map[dst_nodata] = NODATA_RGBA
    else:
        # for banded data set the each band's destination nodata to zero (TRANSPARENT).
        dst_nodata = TRANSPARENT

    creation_options = {
        **grid_parameters,
        'driver': driver,
        'dtype': 'uint8',
        'count': n_bands,
    }

    dest_array = get_destination(grid_parameters, n_bands)

    logger.info(f'Create output image with options: {creation_options}')

    with rasterio.open(out_file_name, 'w', **creation_options) as dst_raster:
        for dim in range(0, n_bands):
            reproject(
                source=raster[dim, :, :],
                destination=dest_array[dim, :, :],
                src_transform=data_array.rio.transform(),
                src_crs=data_array.rio.crs,
                dst_transform=grid_parameters['transform'],
                dst_crs=grid_parameters['crs'],
                dst_nodata=dst_nodata,
                resampling=Resampling.nearest,
            )

        dst_raster.write(dest_array)
        if color_map is not None:
            dst_raster.write_colormap(1, color_map)

    with open(out_world_name, 'w', encoding='UTF-8') as out_wd:
        out_wd.write(dumpsw(creation_options['transform']))
