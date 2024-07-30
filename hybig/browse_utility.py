"""Module containing utility functionality for browse generation."""

from harmony.message import Message as HarmonyMessage


def get_harmony_message_from_params(params: dict | None) -> HarmonyMessage:
    """Constructs a harmony message from the input parms.

    We have to create a harmony message to pass to the create_browse_imagery
    function so that both the library and service calls are identical.

    """
    if params is None:
        params = {}
    mime = params.get('mime', 'image/png')
    crs = params.get('crs', None)
    scale_extent = params.get('scale_extent', None)
    scale_size = params.get('scale_size', None)
    height = params.get('height', None)
    width = params.get('width', None)

    return HarmonyMessage(
        {
            "format": {
                "mime": mime,
                "crs": crs,
                "srs": crs,
                "scaleExtent": scale_extent,
                "scaleSize": scale_size,
                "height": height,
                "width": width,
            },
        }
    )
