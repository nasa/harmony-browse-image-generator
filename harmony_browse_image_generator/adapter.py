""" `HarmonyAdapter` for Harmony Browse Image Generator (HyBIG).

    The class in this file is the top level of abstraction for a service that
    will accept a GeoTIFF input and create a browse image (PNG/JPEG) and
    accompanying ESRI world file. By default, this service will aim to create
    Global Imagery Browse Services (GIBS) compatible browse imagery.

"""
from os.path import basename
from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp

from harmony import BaseHarmonyAdapter
from harmony.message import Source as HarmonySource
from harmony.util import bbox_to_geometry, download, generate_output_filename, stage
from pystac import Asset, Catalog, Item

from harmony_browse_image_generator.browse import create_browse_imagery
from harmony_browse_image_generator.exceptions import HyBIGInvalidMessage
from harmony_browse_image_generator.message_utility import (
    has_crs,
    has_scale_extents,
    has_scale_sizes,
)
from harmony_browse_image_generator.utilities import (
    get_asset_name,
    get_file_mime_type,
    get_tiled_file_extension,
)


class BrowseImageGeneratorAdapter(BaseHarmonyAdapter):
    """ This class extends the BaseHarmonyAdapter class from the
        harmony-service-lib package to implement HyBIG operations.

    """

    def invoke(self) -> Catalog:
        """ Adds validation to process_item based invocations. """
        self.validate_message()
        return super().invoke()

    def validate_message(self):
        """Validates that the contents of the Harmony message provides all
            necessary parameters.

        Currently imposed rules:
        1. if scale extent and scale size must each by accompanied by crs.

        """

        if has_scale_extents(self.message) or has_scale_sizes(self.message):
            if not has_crs(self.message):
                raise HyBIGInvalidMessage(
                    'Harmony message must include a crs '
                    'with scaleExtent or scaleSizes.'
                )

    def process_item(self, item: Item, source: HarmonySource) -> Item:
        """ Processes a single input STAC item. """
        try:
            working_directory = mkdtemp()
            results = item.clone()
            results.assets = {}

            asset = next(item_asset for item_asset in item.assets.values()
                         if 'data' in (item_asset.roles or []))

            # Download the input:
            input_data_filename = download(
                asset.href,
                working_directory,
                logger=self.logger,
                cfg=self.config,
                access_token=self.message.accessToken)

            # Create browse images.
            image_file_list = create_browse_imagery(
                self.message,
                input_data_filename,
                logger=self.logger
            )

            # image_file_list is a list of tuples (image, world, auxiliary)
            # we need to stage them each individually, and then add their final
            # locations to a list before creating the stac item.
            item_assets = []

            for browse_image_name, world_file_name, aux_xml_file_name in image_file_list:
                # Stage the images:
                browse_image_url = self.stage_output(browse_image_name,
                                                     asset.href)
                browse_aux_url = self.stage_output(aux_xml_file_name,
                                                   asset.href)
                world_file_url = self.stage_output(world_file_name,
                                                   asset.href)
                item_assets.append(('data', browse_image_url, 'data'))
                item_assets.append(('metadata', world_file_url, 'metadata'))
                item_assets.append(('auxiliary', browse_aux_url, 'metadata'))

            manifest_url = self.stage_manifest(image_file_list, asset.href)
            item_assets.insert(0, ('data', manifest_url, 'metadata'))

            return self.create_output_stac_item(item, item_assets)

        except Exception as exception:
            self.logger.exception(exception)
            raise exception
        finally:
            rmtree(working_directory)

    def stage_output(self, transformed_file: Path, input_file: str) -> str:
        """ Generate an output file name based on the input asset URL and the
            operations performed to produce the output. Use this name to stage
            the output in the S3 location specified in the input Harmony
            message.

        """

        ext = get_tiled_file_extension(transformed_file)
        output_file_name = generate_output_filename(
            input_file, ext=ext
        )

        return stage(transformed_file,
                     output_file_name,
                     get_file_mime_type(transformed_file),
                     location=self.message.stagingLocation,
                     logger=self.logger,
                     cfg=self.config)

    def create_output_stac_item(self, input_stac_item: Item,
                                item_assets: list[tuple[str, str, str]]) -> Item:
        """Create an output STAC item used to access the browse imagery and
            ESRI world file as staged in S3.

        asset_items are an array of tuples where the tuples should be: (name,
        url, role)

        """
        output_stac_item = input_stac_item.clone()
        output_stac_item.assets = {}
        # The output bounding box will vary by projection, so the following
        # line will need to be updated when the service is either supplied or
        # determines a GIBS-compatible projection.
        output_stac_item.bbox = input_stac_item.bbox
        output_stac_item.geometry = bbox_to_geometry(output_stac_item.bbox)

        for name, url, role in item_assets:
            asset_name = get_asset_name(name, url)

            output_stac_item.assets[asset_name] = Asset(
                url, title=basename(url),
                media_type=get_file_mime_type(url), roles=[role]
            )

        return output_stac_item

    def stage_manifest(self, image_file_list: list[tuple[Path, Path, Path]],
                       asset_href: str) -> str:
        """Write a manifest file of the output images.

        Write a file that will serve as the 'data' key for tiled output.  At
        some point we will have to find out why this is necessary, but this is
        a clever work around to that necessity.

        """
        manifest_fn = Path(image_file_list[0][0]).parent / 'manifest.txt'

        with open(manifest_fn, 'w', encoding='UTF-8') as file_pointer:
            file_pointer.writelines(
                f'{img}, {wld}, {aux}\n' for img, wld, aux in image_file_list)

        return self.stage_output(manifest_fn, asset_href)
