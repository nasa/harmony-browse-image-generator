# Changelog

HyBIG follows semantic versioning. All notable changes to this project will be
documented in this file. The format is based on [Keep a
Changelog](http://keepachangelog.com/en/1.0.0/).

## [vX.Y.Z] - Unreleased

### Changed

* GitHub release notes for HyBIG will now include the commit history for that
  release.

## [v2.5.0] - Unreleased

### Changed

* Correctly handle clipping behavior for values outside the colormap range

## [v2.4.2] - 2025-10-28

### Changed

* Dynamically determine nodata index based on where it is supplied in the remote/embedded color table for a product.

## [v2.4.1] - 2025-06-23

### Changed

* Fix bug with JPEG driver on single-banded input granules. Since we are now palettizing where possible since 2.4.0, this creates an issue when trying to output in JPEG since color palettes (and transparency) are not supported.


## [v2.4.0] - 2025-04-28

### Changed

* Fix rasterization issues with palettized granules. Source images now retain their palette in a scaled form, rather than reinterpreting the palette. [[#50](https://github.com/nasa/harmony-browse-image-generator/pull/50)]
* Minor bugfixes and type formatting improvements.

## [v2.3.0] - 2025-02-26

### Changed

* Fix images that cross the antimeridian. Target extents are corrected when the bounding box crosses the dateline. [[#48](https://github.com/nasa/harmony-browse-image-generator/pull/48)]

## [v2.2.0] - 2024-12-19

### Changed

* NODATA and TRANSPARENT values are merged. [[#41](https://github.com/nasa/harmony-browse-image-generator/pull/41)]
  - User visible change: paletted PNG output images will have up to 254 color
    values and a 255th value that is transparent.
  - Internal code changes: removes `TRANSPARENT_IDX` (254) and uses
    `NODATA_IDX` (255) in its stead.  A color of (0,0,0,0) was previously set to
    both the indexes (254 and 255) in the output PNGs and now only 255 will have
    this value. This change ensures the round-trip from single band to RGBA to
    Paletted PNG is consistent.

## [v2.1.0] - 2024-12-13

### Changed

* Input GeoTIFF RGB[A] images are **no longer palettized** when converted to a PNG. The new resulting output browse images are now 3 or 4 band PNG retaining the color information of the input image.[[#39](https://github.com/nasa/harmony-browse-image-generator/pull/39)]
* Changed pre-commit configuration to remove `black-jupyter` dependency [[#38](https://github.com/nasa/harmony-browse-image-generator/pull/38)]
* Updates service image's python to 3.12 [[#38](https://github.com/nasa/harmony-browse-image-generator/pull/38)]
* Simplifies test scripts to run with pytest and pytest plugins [[#38](https://github.com/nasa/harmony-browse-image-generator/pull/38)]

### Removed

* Removes `test_code_format.py` in favor of `ruff` pre-commit configuration [[#38](https://github.com/nasa/harmony-browse-image-generator/pull/38)]


## [v2.0.2] - 2024-10-15

### Fixed

**DAS-2259**
- Corrects bug with RGBA input tifs.

## [v2.0.1] - 2024-10-06

### Changed

* Updates service image to be built on AMD64.
* Updates internal libraries


## [v2.0.0] - 2024-07-19

**DAS-2180** - Adds pip installable library.

This release is a refactor that extracts browse image generation logic from the
harmony service code. There are no user visible changes to the existing
functionality.  The new library,
[hybig-py](https://pypi.org/project/hybig-py/), provides the `create_browse`
function to generate browse images, see the README.md for details.

## [v1.2.2] - 2024-06-18

### Changed
Removes internal dependency on conda.

## [v1.2.1] - 2024-06-10

### Changed
Updated internal library dependencies.

## [v1.2.0] - 2024-05-28

### Added
Adds functionality to retrieve '`visual`' asset for multi-file
granules. Harmony creates this type of asset when the UMM-G is correctly
configured with a "BROWSE IMAGE SOURCE" subtype.

## [v1.1.0] - 2024-04-30

### Changed
Changes the computation for an output image's default scale extent. Previously
we considered ICD preferred ScaleExtents as covering the entire globe or pole.
This change now takes just the input image bounds and converts them to the target crs
and uses that transformed boundry as the default region to make a scale extent from.

Upgraded harmony-service-lib to v1.0.26

## [v1.0.2] - 2024-04-05

This version of HyBIG correctly handles missing/bad input data marked by _FillValue or NoData.
Anytime a bad value occurs in the input raster, the output png image will set to transparent.

## [v1.0.1] - 2024-04-05

This version of HyBIG updates the repository to use `black` code formatting
throughout. There should be no functional change to the service.

## [v1.0.0] - 2024-01-22
This version of the Harmony Browse Image Generator (HyBIG) contains all
functionality previously released internally to EOSDIS as
sds/harmony-browse-image-generator:0.0.11.

Additional contents to the repository include updated documentation and files
outlined by the NASA open-source guidelines.

For more information on internal releases prior to NASA open-source approval,
see legacy-CHANGELOG.md.

[unreleased]: https://github.com/nasa/harmony-browse-image-generator/
[v2.4.1]: https://github.com/nasa/harmony-browse-image-generator/releases/tag/2.4.1
[v2.4.0]: https://github.com/nasa/harmony-browse-image-generator/releases/tag/2.4.0
[v2.3.0]: https://github.com/nasa/harmony-browse-image-generator/releases/tag/2.3.0
[v2.2.0]: https://github.com/nasa/harmony-browse-image-generator/releases/tag/2.2.0
[v2.1.0]: https://github.com/nasa/harmony-browse-image-generator/releases/tag/2.1.0
[v2.0.2]: https://github.com/nasa/harmony-browse-image-generator/releases/tag/2.0.2
[v2.0.1]: https://github.com/nasa/harmony-browse-image-generator/releases/tag/2.0.1
[v2.0.0]: https://github.com/nasa/harmony-browse-image-generator/releases/tag/2.0.0
[v1.2.2]: https://github.com/nasa/harmony-browse-image-generator/releases/tag/1.2.2
[v1.2.1]: https://github.com/nasa/harmony-browse-image-generator/releases/tag/1.2.1
[v1.2.0]: https://github.com/nasa/harmony-browse-image-generator/releases/tag/1.2.0
[v1.1.0]: https://github.com/nasa/harmony-browse-image-generator/releases/tag/1.1.0
[v1.0.2]: https://github.com/nasa/harmony-browse-image-generator/releases/tag/1.0.2
[v1.0.1]: https://github.com/nasa/harmony-browse-image-generator/releases/tag/1.0.1
[v1.0.0]: https://github.com/nasa/harmony-browse-image-generator/releases/tag/1.0.0
