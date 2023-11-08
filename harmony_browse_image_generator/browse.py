"""Module containing core functionality for browse image generation."""
import re
from itertools import zip_longest
from logging import Logger
from pathlib import Path

import matplotlib
import numpy as np
import rasterio
import requests
from affine import dumpsw
from harmony.message import Message as HarmonyMessage
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from numpy import around, concatenate, ndarray
from osgeo_utils.auxiliary.color_palette import ColorPalette
from PIL import Image
from rasterio.io import DatasetReader
from rasterio.plot import reshape_as_image, reshape_as_raster
from rasterio.warp import Resampling, reproject

from harmony_browse_image_generator.exceptions import HyBIGException
from harmony_browse_image_generator.sizes import (
    create_tiled_output_parameters,
    get_target_grid_parameters,
)


def create_browse_imagery(message: HarmonyMessage, input_file_path: str,
                          logger: Logger) -> list[tuple[Path, Path, Path]]:
    """Create browse image from input geotiff.

    Take input browse image and return a 2-element tuple for the file paths
    of the output browse image and its associated ESRI world file.

    """
    output_driver = image_driver(message.format.mime)
    out_image_file = output_image_file(Path(input_file_path),
                                       driver=output_driver)
    out_world_file = output_world_file(Path(input_file_path),
                                       driver=output_driver)

    try:
        with rasterio.open(input_file_path, mode='r') as in_dataset:
            validate_file_type(in_dataset)

            if in_dataset.count == 1:
                raster = convert_singleband_to_raster(in_dataset)
            elif in_dataset.count in (3, 4):
                raster = convert_mulitband_to_raster(in_dataset)
            else:
                raise HyBIGException(
                    f'incorrect number of bands for image: {in_dataset.count}')

            raster, color_map = prepare_raster_for_writing(raster, output_driver)

            grid_parameters = get_target_grid_parameters(message, in_dataset)
            grid_parameter_list, tile_locators = create_tiled_output_parameters(
                grid_parameters
            )

            processed_files = []
            for grid_parameters, tile_location in zip_longest(grid_parameter_list,
                                                              tile_locators):
                tiled_out_image_file = get_tiled_filename(out_image_file, tile_location)
                tiled_out_world_file = get_tiled_filename(out_world_file, tile_location)
                tiled_out_aux_xml_file = get_aux_xml_filename(tiled_out_image_file)
                logger.info(f'out image file: {tiled_out_image_file}: {tile_location}')

                write_georaster_as_browse(
                    in_dataset,
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
        raise HyBIGException(str(exception)) from exception

    return processed_files


def convert_mulitband_to_raster(dataset: DatasetReader) -> ndarray:
    """Convert multiband to a raster image.

    Reads the three/four bands from the file, then normalizes them to the range
    0 to 255. This assumes the input image is already in RGB or RGBA format and
    just ensures that the output is 8bit.

    """
    bands = dataset.read()
    norm = Normalize()

    if dataset.count == 3:
        norm.autoscale(bands)
        raster = around(norm(bands) * 255.0).astype('uint8')

    elif dataset.count == 4:
        norm.autoscale(bands[0:3, :, :])
        partial_raster = around(norm(bands[0:3, :, :]) * 255.0).astype('uint8')
        raster = concatenate([partial_raster.data, bands[3:4, :, :]])

    else:
        raise HyBIGException(
            f'Cannot create image from {dataset.count} band image')

    return raster


def convert_singleband_to_raster(dataset: DatasetReader) -> ndarray:
    """Convert input dataset to a 4 band raster image.

    If the image is paletted, read the palette and use that for the image
    otherwise, use a default grayscale colormap for the image.

    """
    color_palette = get_color_palette(dataset)
    # Can add Message and visicurl to above later
    if color_palette:
        return convert_paletted_1band_to_raster(dataset, color_palette)
    return convert_gray_1band_to_raster(dataset)


def convert_gray_1band_to_raster(dataset):
    """Convert a 1-band raster without a color association."""
    band = dataset.read(1)
    cmap = matplotlib.colormaps['Greys_r']
    norm = Normalize()
    norm.autoscale(band)
    scalar_map = ScalarMappable(cmap=cmap, norm=norm)

    rgba_image = np.zeros((*band.shape, 4), dtype='uint8')
    for row_no in range(band.shape[0]):
        rgba_image_slice = scalar_map.to_rgba(band[row_no, :], bytes=True)
        rgba_image[row_no, :, :] = rgba_image_slice
    return reshape_as_raster(rgba_image[..., 0:3])


def convert_paletted_1band_to_raster(dataset: DatasetReader,
                                     palette: ColorPalette) -> ndarray:
    """Convert a 1 band image with palette into a rgba raster image."""
    band = dataset.read(1)
    levels = list(palette.pal.keys())
    colors = [
        palette.color_to_color_entry(value, with_alpha=True)
        for value in palette.pal.values()
    ]
    scaled_colors = [
        (r / 255.0, g / 255.0, b / 255.0, a / 255.0) for r, g, b, a in colors
    ]
    cmap, norm = matplotlib.colors.from_levels_and_colors(levels,
                                                          scaled_colors,
                                                          extend='max')
    scalar_map = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    scaled_raster = scalar_map.to_rgba(band, bytes=True)
    return reshape_as_raster(scaled_raster)


def get_color_palette(dataset) -> ColorPalette | None:
    """Get a color palette if color information is provided."""
    try:
        colormap = dataset.colormap(1)
        palette = convert_colormap_to_palette(colormap)
    except ValueError:
        palette = None

    return palette


def convert_colormap_to_palette(colormap: dict) -> ColorPalette:
    """Convert a GeoTIFF palette to GDAL ColorPalette.

    Reformats a palette as dictionary and loads it into a ColorPalette.

    a GeoTIFF's colormap looks like a dictionary of tiff's value to (r,g,b,[a])
    {0: (9, 60, 112, 255),
     1: (9, 65, 122, 255),
     2: (9, 80, 132, 255),
     3: (9, 100, 142, 255),...}

    """
    list_of_key_rgba = [list((key, *rgba)) for key, rgba in colormap.items()]
    strings_of_key_r_g_b_a = [
        ' '.join([str(item) for item in line]) for line in list_of_key_rgba
    ]
    palette = ColorPalette()
    palette.read_file_txt(lines=strings_of_key_r_g_b_a)
    return palette


def image_driver(mime: str) -> str:
    """Return requested rasterio driver for output image."""
    if re.search('jpeg', mime, re.I):
        return 'JPEG'
    return 'PNG'


def palette_from_remote_colortable(url: str) -> ColorPalette:
    """Return a gdal ColorPalette from a remote colortable."""
    response = requests.get(url, timeout=10)
    if not response.ok:
        raise HyBIGException(f'Could not read remote colortable at {url}')

    palette = ColorPalette()
    palette.read_file_txt(lines=response.text.split('\n'))
    return palette


def prepare_raster_for_writing(raster: ndarray, driver: str) -> (ndarray, dict | None):
    """Remove alpha layer if writing a jpeg."""
    if driver == 'JPEG':
        if raster.shape[0] == 4:
            raster = raster[0:3, :, :]
        return raster, None

    return palettize_raster(raster)


def palettize_raster(raster: ndarray) -> (ndarray, dict):
    """convert an RGB or RGBA image into a 1band image and palette.

    Converts a 3 or 4 band np raster into a PIL image.
    Quantizes the image into a 1band raster with palette

    Transparency is handled by first removing the Alpha layer and creating
    quantized raster from just the RGB layers. Next the Alpha layer values are
    treated as either transparent or opaque and any transparent values are
    written to the final raster as 254 and add the mapped RGBA value to the
    color palette.

    """
    # 0 to 253;
    # reserve 254 for transparent images
    # reserve 255 for off grid fill values
    max_colors = 254

    # TODO [MHS, 11/01/2023] DAS-2020 raster, alpha = remove_alpha(raster)
    multiband_image = Image.fromarray(reshape_as_image(raster))
    quantized_image = multiband_image.quantize(colors=max_colors)

    color_map = get_color_map_from_image(quantized_image)
    # TODO [MHS, 11/01/2023] DAS-2020 if transparency replace values in
    # quantized image and color_map

    one_band_raster = np.expand_dims(np.array(quantized_image), 0)
    return one_band_raster, color_map


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
    return image_filename.with_suffix(
        image_filename.suffix + '.aux.xml'
    )


def get_tiled_filename(input_file: Path, locator: dict | None = None) -> Path:
    """Add a column, row identifier to the output files.

    Only update if there is a valid locator dict.
    """
    if locator is not None:
        print(f'locator: {locator}')
        return input_file.with_suffix(
            f".r{int(locator['row']):02d}c{int(locator['col']):02d}{input_file.suffix}")
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


def validate_file_type(dsr: DatasetReader) -> None:
    """Ensure we can work with the input data file.

    Raise an exception if this file is unusable by the service.

    """
    if dsr.driver != 'GTiff':
        raise HyBIGException(f'Input file type not supported: {dsr.driver}')


def get_destination(grid_parameters: dict, n_bands: int) -> ndarray:
    """Initialize an array for writing an output raster."""
    return np.zeros(
        (n_bands, grid_parameters['height'], grid_parameters['width']),
        dtype='uint8')


def write_georaster_as_browse(dataset: DatasetReader,
                              raster: ndarray,
                              color_map: ndarray | None,
                              grid_parameters: dict,
                              driver='PNG',
                              out_file_name='outfile.png',
                              out_world_name='outfile.pgw',
                              logger=Logger) -> None:
    """Write raster data to output file.

    Writes the raster to an output file using metadata from the original
    source, and over-riding some values based on the input message and derived
    grid_parameters, doing a nearest neighbor resampling to the target grid.

    """
    n_bands = raster.shape[0]
    dst_nodata = 255
    if color_map is not None:
        color_map[dst_nodata] = (0, 0, 0, 0)

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
            reproject(source=raster[dim, :, :],
                      destination=dest_array[dim, :, :],
                      src_transform=dataset.transform,
                      src_crs=dataset.crs,
                      dst_transform=grid_parameters['transform'],
                      dst_crs=grid_parameters['crs'],
                      dst_nodata=dst_nodata,
                      resampling=Resampling.nearest)

        dst_raster.write(dest_array)
        if color_map is not None:
            dst_raster.write_colormap(1, color_map)

    with open(out_world_name, 'w', encoding='UTF-8') as out_wd:
        out_wd.write(dumpsw(creation_options['transform']))
