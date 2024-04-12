"""Module containing core functionality for browse image generation."""

import re
from itertools import zip_longest
from logging import Logger
from pathlib import Path

import matplotlib
import numpy as np
import rasterio
from affine import dumpsw
from harmony.message import Message as HarmonyMessage
from harmony.message import Source as HarmonySource
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from numpy import around, concatenate, ndarray
from osgeo_utils.auxiliary.color_palette import ColorPalette
from PIL import Image
from rasterio.io import DatasetReader
from rasterio.plot import reshape_as_image, reshape_as_raster
from rasterio.warp import Resampling, reproject
from rioxarray import open_rasterio
from xarray import DataArray

from harmony_browse_image_generator.color_utility import (
    NODATA_IDX,
    NODATA_RGBA,
    TRANSPARENT_IDX,
    TRANSPARENT_RGBA,
    get_color_palette,
)
from harmony_browse_image_generator.exceptions import HyBIGError
from harmony_browse_image_generator.sizes import (
    GridParams,
    create_tiled_output_parameters,
    get_target_grid_parameters,
)


def create_browse_imagery(
    message: HarmonyMessage,
    input_file_path: str,
    source: HarmonySource,
    item_color_palette: ColorPalette | None,
    logger: Logger,
) -> list[tuple[Path, Path, Path]]:
    """Create browse image from input geotiff.

    Take input browse image and return a 2-element tuple for the file paths
    of the output browse image and its associated ESRI world file.

    """
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
                raster = convert_singleband_to_raster(rio_in_array, color_palette)
            elif rio_in_array.rio.count in (3, 4):
                raster = convert_mulitband_to_raster(rio_in_array)
            else:
                raise HyBIGError(
                    f'incorrect number of bands for image: {rio_in_array.rio.count}'
                )

            raster, color_map = prepare_raster_for_writing(raster, output_driver)

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


def convert_mulitband_to_raster(data_array: DataArray) -> ndarray:
    """Convert multiband to a raster image.

    Reads the three/four bands from the file, then normalizes them to the range
    0 to 255. This assumes the input image is already in RGB or RGBA format and
    just ensures that the output is 8bit.

    """
    bands = data_array.to_numpy()

    if data_array.rio.count == 3:
        norm = Normalize(vmin=np.nanmin(bands), vmax=np.nanmax(bands))
        raster = around(norm(bands) * 255.0).astype('uint8')

    elif data_array.rio.count == 4:
        norm = Normalize(
            vmin=np.nanmin(bands[0:3, :, :]), vmax=np.nanmax(bands[0:3, :, :])
        )
        partial_raster = around(norm(bands[0:3, :, :]) * 255.0).astype('uint8')
        raster = concatenate([partial_raster.data, bands[3:4, :, :]])

    else:
        raise HyBIGError(f'Cannot create image from {data_array.rio.count} band image')

    return raster


def convert_singleband_to_raster(
    data_array: DataArray,
    color_palette: ColorPalette | None = None,
) -> ndarray:
    """Convert input dataset to a 4 band raster image.

    Use a palette if provided otherwise return a greyscale image.
    """
    if color_palette is None:
        return convert_gray_1band_to_raster(data_array)
    return convert_paletted_1band_to_raster(data_array, color_palette)


def convert_gray_1band_to_raster(data_array: DataArray) -> ndarray:
    """Convert a 1-band raster without a color association."""
    band = data_array[0, :, :]
    cmap = matplotlib.colormaps['Greys_r']
    cmap.set_bad(TRANSPARENT_RGBA)
    norm = Normalize(vmin=np.nanmin(band), vmax=np.nanmax(band))
    scalar_map = ScalarMappable(cmap=cmap, norm=norm)

    rgba_image = np.zeros((*band.shape, 4), dtype='uint8')
    for row_no in range(band.shape[0]):
        rgba_image_slice = scalar_map.to_rgba(band[row_no, :], bytes=True)
        rgba_image[row_no, :, :] = rgba_image_slice

    return reshape_as_raster(rgba_image)


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


def prepare_raster_for_writing(
    raster: ndarray, driver: str
) -> tuple[ndarray, dict | None]:
    """Remove alpha layer if writing a jpeg."""
    if driver == 'JPEG':
        if raster.shape[0] == 4:
            raster = raster[0:3, :, :]
        return raster, None

    return palettize_raster(raster)


def remove_alpha(raster: ndarray) -> tuple[ndarray, ndarray | None]:
    """Pull raster array off of input if it exists."""
    if raster.shape[0] == 4:
        return raster[0:3, :, :], raster[3, :, :]
    return raster, None


def palettize_raster(raster: ndarray) -> tuple[ndarray, dict]:
    """convert an RGB or RGBA image into a 1band image and palette.

    Converts a 3 or 4 band np raster into a PIL image.
    Quantizes the image into a 1band raster with palette

    Transparency is handled by first removing the Alpha layer and creating
    quantized raster from just the RGB layers. Next the Alpha layer values are
    treated as either transparent or opaque and any transparent values are
    written to the final raster as 254 and add the mapped RGBA value to the
    color palette.
    """
    # reserves 254 for transparent images and 255 for off grid fill values
    # 0 to 253
    max_colors = 254
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
    index to the transparent index in those places."""
    max_alpha = 255
    if alpha is not None and np.any(alpha != max_alpha):
        # Set any alpha to the transparent index value
        quantized_array = np.where(alpha != max_alpha, TRANSPARENT_IDX, quantized_array)
        color_map[TRANSPARENT_IDX] = TRANSPARENT_RGBA
    return quantized_array, color_map


def get_color_map_from_image(image: Image) -> dict:
    """Get a writable color map

    Read the RGBA palette from a PIL Image and covert into a dictionary
    that can be written by rasterio.

    """
    color_tuples = np.array(image.getpalette(rawmode='RGBA')).reshape(-1, 4)
    color_map = {}
    for idx in range(0, color_tuples.shape[0]):
        color_map[idx] = tuple(color_tuples[idx])
    return color_map


def get_aux_xml_filename(image_filename: Path) -> Path:
    """get aux.xml filenames."""
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
    dst_nodata = NODATA_IDX
    if color_map is not None:
        color_map[dst_nodata] = NODATA_RGBA

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
