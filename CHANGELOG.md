## v1.1.0 - 2024-04-30

### Changed
Changes the computation for an output image's default scale extent. Previously
we considered ICD preferred ScaleExtents as covering the entire globe or pole.
This change now takes just the input image bounds and converts them to the target crs
and uses that transformed boundry as the default region to make a scale extent from.

Upgraded harmony-service-lib to v1.0.26

## v1.0.2 - 2024-04-05

This version of HyBIG correctly handles missing/bad input data marked by _FillValue or NoData.
Anytime a bad value occurs in the input raster, the output png image will set to transparent.

## v1.0.1 - 2024-04-05

This version of HyBIG updates the repository to use `black` code formatting
throughout. There should be no functional change to the service.

## v1.0.0 - 2024-01-22
This version of the Harmony Browse Image Generator (HyBIG) contains all
functionality previously released internally to EOSDIS as
sds/harmony-browse-image-generator:0.0.11.

Additional contents to the repository include updated documentation and files
outlined by the NASA open-source guidelines.

For more information on internal releases prior to NASA open-source approval,
see legacy-CHANGELOG.md.
