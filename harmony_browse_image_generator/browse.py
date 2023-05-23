""" Module containing core functionality for browse image generation. """
import re
from pathlib import Path

import matplotlib
import rasterio
import requests
from affine import dumpsw
from harmony.message import Message as HarmonyMessage
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from numpy import around, concatenate, ndarray
from osgeo_utils.auxiliary.color_palette import ColorPalette
from rasterio.io import DatasetReader
from rasterio.plot import reshape_as_raster

from harmony_browse_image_generator.exceptions import HyBIGException


def create_browse_imagery(message: HarmonyMessage,
                          input_file_path: str) -> tuple[Path, Path]:
    """Take input browse image and return a 2-element tuple for the file paths
    of the output browse image and its associated ESRI world file

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

            raster = prepare_raster_for_writing(raster, output_driver)

            write_georaster_as_browse(
                in_dataset,
                raster,
                driver=output_driver,
                out_file_name=out_image_file,
                out_world_name=out_world_file,
            )

    except Exception as exception:
        raise HyBIGException(str(exception)) from exception

    return (out_image_file, out_world_file)


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
    """convert input dataset to a 4 band raster image.

    If the image is paletted, read the palette and use that for the image
    otherwise, use a default grayscale colormap for the image.

    """
    color_palette = get_color_palette(dataset)
    # Can add Message and visicurl to above later
    if color_palette:
        return convert_paletted_1band_to_raster(dataset, color_palette)
    return convert_colormaped_1band_to_raster(dataset)


def convert_colormaped_1band_to_raster(dataset):
    """Converts a 1-band raster without a color association."""
    band = dataset.read(1)
    cmap = matplotlib.colormaps['Greys_r']
    norm = Normalize()
    norm.autoscale(band)
    scalar_map = ScalarMappable(cmap=cmap, norm=norm)
    rgba_image = around(scalar_map.to_rgba(band) * 255.0).astype('uint8')
    return reshape_as_raster(rgba_image)


def convert_paletted_1band_to_raster(dataset: DatasetReader,
                                     palette: ColorPalette) -> ndarray:
    """Converts a 1 band image with palette into a rgba raster image."""
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
    scaled_raster = scalar_map.to_rgba(band) * 255.0
    return reshape_as_raster(around(scaled_raster).astype('uint8'))


def get_color_palette(dataset) -> ColorPalette | None:
    """Get a color palette if color information is provided."""
    try:
        colormap = dataset.colormap(1)
        palette = convert_colormap_to_palette(colormap)
    except ValueError:
        palette = None

    return palette


def convert_colormap_to_palette(colormap: dict) -> ColorPalette:
    """Converts a GeoTIFF palette to GDAL ColorPalette.

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
    """Return requested rasterio driver for output image.

    """
    if re.search('jpeg', mime, re.I):
        return 'JPEG'
    return 'PNG'


def palette_from_remote_colortable(url: str) -> ColorPalette:
    """Returns a gdal ColorPalette from a remote colortable."""
    response = requests.get(url, timeout=10)
    if not response.ok:
        raise HyBIGException(f'Could not read remote colortable at {url}')

    palette = ColorPalette()
    palette.read_file_txt(lines=response.text.split('\n'))
    return palette


def prepare_raster_for_writing(raster: ndarray, driver: str) -> ndarray:
    """remove alpha layer if writing a jpeg."""
    if driver == 'JPEG' and raster.shape[0] == 4:
        raster = raster[0:3, :, :]
    return raster


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


def write_georaster_as_browse(dataset: DatasetReader,
                              raster: ndarray,
                              driver='PNG',
                              out_file_name='outfile.png',
                              out_world_name='outfile.pgw') -> None:
    """Write raster data to output file.

    Writes the raster to an output file using metadata from the original
    source, but over-riding some important values. The input meta contains
    information about the source's width, height, crs and affine transformation
    that are all used in the output png.

    """
    options = {
        **dataset.meta,
        'driver': driver,
        'dtype': 'uint8',
        'photometric': 'RGB',
        'count': raster.shape[0],
        'compress': 'lzw',
    }

    with rasterio.open(out_file_name, 'w', **options) as dest_raster:
        dest_raster.write(raster)

    with open(out_world_name, 'w', encoding='UTF-8') as out_wd:
        out_wd.write(dumpsw(dataset.meta['transform']))
