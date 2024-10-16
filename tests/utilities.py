"""Utilities used to extend unittest capabilities."""

from collections import namedtuple
from datetime import datetime

from harmony_service_lib.util import bbox_to_geometry
from pystac import Asset, Catalog, Item

Granule = namedtuple('Granule', ['url', 'media_type', 'roles'])


def create_stac(granule: Granule) -> Catalog:
    """Create a SpatioTemporal Asset Catalog (STAC). These are used as inputs
    for Harmony requests, containing the URL and other information for
    input granules.

    For simplicity the geometric and temporal properties of each item are
    set to default values.

    """
    catalog = Catalog(id='input catalog', description='test input')

    item = Item(
        id='input granule',
        bbox=[-180, -90, 180, 90],
        geometry=bbox_to_geometry([-180, -90, 180, 90]),
        datetime=datetime(2020, 1, 1),
        properties=None,
    )

    item.add_asset(
        'input data',
        Asset(granule.url, media_type=granule.media_type, roles=granule.roles),
    )
    catalog.add_item(item)

    return catalog
