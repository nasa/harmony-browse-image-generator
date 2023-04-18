""" `HarmonyAdapter` for Harmony Browse Image Generator (HyBIG).

    The class in this file is the top level of abstraction for a service that
    will accept a GeoTIFF input and create a browse image (PNG/JPEG) and
    accompanying ESRI world file. By default, this service will aim to create
    Global Imagery Browse Services (GIBS) compatible browse imagery.

"""
from os.path import basename, splitext
from shutil import rmtree
from tempfile import mkdtemp
from typing import Optional

from harmony import BaseHarmonyAdapter
from harmony.message import Source as HarmonySource
from harmony.util import (bbox_to_geometry, download, generate_output_filename,
                          stage)
from pystac import Asset, Catalog, Item

from harmony_browse_image_generator.browse import create_browse_imagery
from harmony_browse_image_generator.utilities import get_file_mime_type


class BrowseImageGeneratorAdapter(BaseHarmonyAdapter):
    """ This class extends the BaseHarmonyAdapter class from the
        harmony-service-lib package to implement HyBIG operations.

    """
    def invoke(self) -> Catalog:
        """ Adds validation to process_item based invocations. """
        self.validate_message()
        return super().invoke()

    def validate_message(self):
        """ Validates that the contents of the Harmony message provides all
            necessary parameters.

            Validation rules will be added as part of DAS-1829.

        """

    def process_item(self, item: Item, source: HarmonySource) -> Item:
        """ Processes a single input STAC item. """
        try:
            working_directory = mkdtemp()
            results = item.clone()
            results.assets = {}

            asset = next((item_asset for item_asset in item.assets.values()
                          if 'data' in (item_asset.roles or [])))

            # Download the input:
            input_data = download(asset.href, working_directory,
                                  logger=self.logger, cfg=self.config,
                                  access_token=self.message.accessToken)

            # The following line would be replaced by invoking service logic
            # The assumption is output will contain a 2-element tuple with a
            # browse image and an ESRI world file:
            # (Note until the service logic is created, the input file will be
            # entirely spurious)
            browse_image_name, world_file_name = create_browse_imagery(
                input_data
            )

            # Stage the browse image:
            browse_image_mime_type = get_file_mime_type(browse_image_name)
            browse_image_url = self.stage_output(browse_image_name,
                                                 asset.href,
                                                 browse_image_mime_type)

            # Stage the world file:
            world_file_mime_type = get_file_mime_type(world_file_name)
            world_file_url = self.stage_output(world_file_name, asset.href,
                                               world_file_mime_type)

            return self.create_output_stac_item(item, browse_image_url,
                                                browse_image_mime_type,
                                                world_file_url,
                                                world_file_mime_type)
        except Exception as exception:
            self.logger.exception(exception)
            raise exception
        finally:
            rmtree(working_directory)

    def stage_output(self, transformed_file: str, input_file: str,
                     transformed_mime_type: Optional[str]) -> str:
        """ Generate an output file name based on the input asset URL and the
            operations performed to produce the output. Use this name to stage
            the output in the S3 location specified in the input Harmony
            message.

        """
        output_file_name = generate_output_filename(
            input_file, ext=splitext(transformed_file)[1]
        )

        return stage(transformed_file, output_file_name, transformed_mime_type,
                     location=self.message.stagingLocation,
                     logger=self.logger, cfg=self.config)

    def create_output_stac_item(self, input_stac_item: Item,
                                browse_image_url: str,
                                browse_image_mime_type: str,
                                world_file_url: str,
                                world_file_mime_type: str) -> Item:
        """ Create an output STAC item used to access the browse imagery and
            ESRI world file as staged in S3.

        """
        output_stac_item = input_stac_item.clone()
        output_stac_item.assets = {}
        # The output bounding box will vary by projection, so the following
        # line will need to be updated when the service is either supplied or
        # determines a GIBS-compatible projection.
        output_stac_item.bbox = input_stac_item.bbox
        output_stac_item.geometry = bbox_to_geometry(output_stac_item.bbox)

        output_stac_item.assets['data'] = Asset(
            browse_image_url, title=basename(browse_image_url),
            media_type=browse_image_mime_type, roles=['data']
        )

        output_stac_item.assets['metadata'] = Asset(
            world_file_url, title=basename(world_file_url),
            media_type=world_file_mime_type, roles=['metadata']
        )

        return output_stac_item
