"""Vendored ColorPalette class.

Minimal replacement for osgeo_utils.auxiliary.color_palette.ColorPalette,
covering only the interface used by this codebase.

Color values are stored as packed 32-bit integers in ARGB byte order:
    (alpha << 24) | (red << 16) | (green << 8) | blue
"""

import re
from collections import OrderedDict


def _pack_color(cc: str) -> int:
    """Pack a space-separated 'r g b [a]' string into an ARGB int."""
    parts = re.findall(r'\d+', cc)
    try:
        if len(parts) == 3:
            r, g, b = int(parts[0]), int(parts[1]), int(parts[2])
            return (255 << 24) | (r << 16) | (g << 8) | b
        if len(parts) == 4:
            r, g, b, a = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            return (a << 24) | (r << 16) | (g << 8) | b
        if len(parts) == 1:
            return int(parts[0])
    except (ValueError, IndexError):
        pass
    return 0


def _parse_key(s: str):
    """Convert a string key to int or float."""
    try:
        return int(s)
    except ValueError:
        return float(s)


_NDV_KEYS = (None, 'nv', 'ndv')


class ColorPalette:
    """Palette mapping data values to packed ARGB colors.

    Text format (GDAL colour-relief): each line is::

        <value>  <r> <g> <b> [<a>]

    where <value> may be an int, a float, or the special token ``nv``
    (no-data value).  Lines with fewer than two whitespace-separated
    fields are silently skipped.
    """

    def __init__(self):
        self.pal: OrderedDict = OrderedDict()
        self.ndv: int | None = None

    def __repr__(self):
        return str(self.pal)

    def __eq__(self, other):
        if not isinstance(other, ColorPalette):
            return NotImplemented
        return self.pal == other.pal and self.ndv == other.ndv

    def get_color(self, key):
        """Return the packed color for *key*, or ``ndv`` for no-data keys."""
        return self.ndv if key in _NDV_KEYS else self.pal[key]

    def read_file_txt(
        self,
        filename=None,
        lines=None,
    ):
        """Parse a GDAL text color-relief file into this palette."""
        if filename is not None:
            with open(filename, encoding='utf-8') as fh:
                lines = fh.readlines()

        self.pal.clear()
        for line in lines:
            parts = line.strip().split(' ', 1)
            if len(parts) < 2:
                continue
            key_str = parts[0].strip()
            color = _pack_color(parts[1])
            if key_str.lower() in ('nv', 'ndv'):
                self.ndv = color
            else:
                try:
                    self.pal[_parse_key(key_str)] = color
                except ValueError:
                    pass

    @staticmethod
    def color_to_color_entry(color: int, with_alpha: bool | None = None):
        """Unpack a packed ARGB int to an (r, g, b[, a]) tuple.

        Returns a 4-tuple when *with_alpha* is True, or when *with_alpha* is
        None and the stored alpha is less than 255.  Returns a 3-tuple when
        *with_alpha* is False, or when *with_alpha* is None and alpha == 255.
        """
        b = color & 0xFF
        g = (color >> 8) & 0xFF
        r = (color >> 16) & 0xFF
        a = (color >> 24) & 0xFF
        if with_alpha or (with_alpha is None and a < 255):
            return r, g, b, a
        return r, g, b
