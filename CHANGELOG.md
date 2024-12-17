# Changelog

HyBIG follows semantic versioning. All notable changes to this project will be
documented in this file. The format is based on [Keep a
Changelog](http://keepachangelog.com/en/1.0.0/).

## [v2.1.0] - 2024-12-13

### Changed

* Input GeoTIFF RGB[A] images are **no longer palettized** when converted to a PNG. The new resulting output browse images are now 3 or 4 band PNG retaining the color information of the input image.[#39](https://github.com/nasa/harmony-browse-image-generator/pull/39)
* Changed pre-commit configuration to remove `black-jupyter` dependency [#38](https://github.com/nasa/harmony-browse-image-generator/pull/38)
* Updates service image's python to 3.12 [#38](https://github.com/nasa/harmony-browse-image-generator/pull/38)
* Simplifies test scripts to run with pytest and pytest plugins [#38](https://github.com/nasa/harmony-browse-image-generator/pull/38)

### Removed

* Removes `test_code_format.py` in favor of `ruff` pre-commit configuration [#38](https://github.com/nasa/harmony-browse-image-generator/pull/38)


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
