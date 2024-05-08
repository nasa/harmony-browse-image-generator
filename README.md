# Harmony Browse Image Generator (HyBIG).

This Harmony backend service is designed to produce browse imagery, with
default behaviour to produce browse imagery that is compatible with the NASA
Global Image Browse Services ([GIBS](https://www.earthdata.nasa.gov/eosdis/science-system-description/eosdis-components/gibs)).

This means that defaults for images are selected to match the visualization
generation requirements and recommendations put forth in the GIBS Interface
Control Document (ICD).

HyBIG creates paletted PNG images and associated metadata from GeoTIFF input
images. Scientific parameter raster data as well as RGB[A] raster images can
be converted to browse PNGs.  These browse images undergo transformation by
reprojection, tiling and coloring to seamlessly integrate with GIBS.

### Reprojection

GIBS expects to recieve images in one of three Coordinate Reference System (CRS) projections.

|Region     |Code      |Name                                                     |
|---        |---       |---                                                      |
|north polar| EPSG:3413|WGS 84 / NSIDC Sea Ice Polar Stereographic North         |
|south polar| EPSG:3031|WGS 84 / Antarctic Polar Stereographic                   |
|global     |EPSG:4326 |WGS 84 -- WGS84 - World Geodetic System 1984, used in GPS|

HyBIG processing will attempt to choose a GIBS suitable target CRS from the
input image information or read it from the inputs.  Reprojection is done by
resampling via nearest neighbor. This is a reminder that these are not science
data, but browse imagery.


### Tiling

By agreement with GIBS, large output images are tiled to a smaller,
easier-to-handle size.  The largest untiled image HyBIG will create is 67108864
total cells (8192 x 8192). When the output image would exceed this threshold,
HyBIG will automatically tile the output into multiple images
4096&nbsp;x&nbsp;4096 cells in size. The edge tiles are truncated.


### Coloring

HyBIG images are colored in a number of different ways. A palette can be
included in the input [STAC
Item](https://github.com/radiantearth/stac-spec/blob/master/item-spec/item-spec.md).
If an Item's asset includes a value with a role `palette`, it is assumed to be a
reference to a remote colortable and the colortable is fetched from the asset's
`href`and parsed as a GDAL color table.

If the STAC Item does not contain color information, then the Harmony message
source is searched for a related URL with a "content type" of
`VisualizationURL` and a "type" `Color Map`, and a remote color table is fetched from that location.

If no remote color information is provided, the input image is searched for a colormap and that is used.

Finally, if no color information can be found grayscale is used.


## Repository structure:

```
|- 📂 bin
|- 📂 docker
|- 📂 docs
|- 📂 harmony_browse_image_generator
|- 📂 tests
|- CHANGELOG.md
|- CONTRIBUTING.md
|- LICENSE
|- README.md
|- conda_requirements.txt
|- dev-requirements.txt
|- legacy-CHANGELOG.md
|- pip_requirements.txt
```

* `bin` - A directory containing utility scripts to build the service and test
  images. A script to extract the release notes for the most recent service
  version, as contained in `CHANGELOG.md` is also in this directory.

* `docker` - A directory containing the Dockerfiles for the service and test
  images. It also contains `service_version.txt`, which contains the semantic
  version number of the service image. Any time an update is made that should
  have an accompanying service image release, this file should be updated.

* `docs` - A directory with example usage notebooks.

* `tests` - A directory containing the service unit test suite.

* `harmony_browse_image_generator` - A directory containing Python source code
  for the HyBIG. `adapter.py` contains the `BrowseImageGeneratorAdapter`
  class that is invoked by calls to the service.

* `CHANGELOG.md` - This file contains a record of changes applied to each new
  release of a service Docker image. Any release of a new service version
  should have a record of what was changed in this file.

* `CONTRIBUTING.md` - This file contains guidance for making contributions to
  HyBIG, including recommended git best practices.

* `README.md` - This file, containing guidance on developing the service.

* `conda_requirements.txt` - A list of service dependencies, such as GDAL, that
  cannot be installed via Pip.

* `dev-requirements.txt` - list of packages required for service development.

* `legacy-CHANGELOG.md` - Notes for each version that was previously released
  internally to EOSDIS, prior to open-source publication of the code and Docker
  image.

* `pip_requirements.txt` - A list of service Python package dependencies.


## Local development:

Local testing of service functionality is best achieved via a local instance of
[Harmony](https://github.com/nasa/harmony). Please see instructions there
regarding creation of a local Harmony instance.

If testing small functions locally that do not require inputs from the main
Harmony application, it is recommended that you create a Python virtual
environment via conda, and then install the necessary dependencies for the
service within that environment via conda and pip then install the pre-commit hooks.

```
> conda create -name hybig-env python==3.11
> conda install --file conda_requirements.txt
> pip install -r pip_requirements.txt
> pip install -r dev-requirements.txt

> pre-commit install
```


## Tests:

This service utilises the Python `unittest` package to perform unit tests on
classes and functions in the service. After local development is complete, and
test have been updated, they can be run in Docker via:

```bash
$ ./bin/build-image
$ ./bin/build-test
$ ./bin/run-test
```

The `tests/run_tests.sh` script will also generate a coverage report, rendered
in HTML, and scan the code with `pylint`.

Currently, the `unittest` suite is run automatically within a GitHub workflow
as part of a CI/CD pipeline. These tests are run for all changes made in a PR
against the `main` branch. The tests must pass in order to merge the PR.

The unit tests are also run prior to publication of a new Docker image, when
commits including changes to `docker/service_version.txt` are merged into the
`main` branch. If these unit tests fail, the new version of the Docker image
will not be published.

## Versioning:

Service Docker images for HyBIG adhere to semantic version numbers:
major.minor.patch.

* Major increments: These are non-backwards compatible API changes.
* Minor increments: These are backwards compatible API changes.
* Patch increments: These updates do not affect the API to the service.

When publishing a new Docker image for the service, two files need to be
updated:

* `CHANGELOG.md` - Notes should be added to capture the changes to the service.
* `docker/service_version.txt` - The semantic version number should be updated.

## CI/CD:

The CI/CD for HyBIG is contained in GitHub workflows in the
`.github/workflows` directory:

* `run_tests.yml` - A reusable workflow that builds the service and test Docker
  images, then runs the Python unit test suite in an instance of the test
  Docker container.
* `run_tests_on_pull_requests.yml` - Triggered for all PRs against the `main`
  branch. It runs the workflow in `run_tests.yml` to ensure all tests pass for
  the new code.
* `publish_docker_image.yml` - Triggered either manually or for commits to the
  `main` branch that contain changes to the `docker/service_version.txt` file.

The `publish_docker_image.yml` workflow will:

* Run the full unit test suite, to prevent publication of broken code.
* Extract the semantic version number from `docker/service_version.txt`.
* Extract the released notes for the most recent version from `CHANGELOG.md`.
* Create a GitHub release that will also tag the related git commit with the
  semantic version number.
* Build and deploy a this service's docker image to `ghcr.io`.

Before triggering a release, ensure both the `docker/service_version.txt` and
`CHANGELOG.md` files are updated. The `CHANGELOG.md` file requires a specific
format for a new release, as it looks for the following string to define the
newest release of the code (starting at the top of the file).

```
## vX.Y.Z - YYYY-MM-DD
```

### pre-commit hooks:

This repository uses [pre-commit](https://pre-commit.com/) to enable pre-commit
checking the repository for some coding standard best practices. These include:

* Removing trailing whitespaces.
* Removing blank lines at the end of a file.
* JSON files have valid formats.
* [ruff](https://github.com/astral-sh/ruff) Python linting checks.
* [black](https://black.readthedocs.io/en/stable/index.html) Python code
  formatting checks.

To enable these checks locally:

```bash
# Install pre-commit Python package as part of test requirements:
pip install -r tests/pip_test_requirements.txt

# Install the git hook scripts:
pre-commit install

# (Optional) Run against all files:
pre-commit run --all-files
```

When you try to make a new commit locally, `pre-commit` will automatically run.
If any of the hooks detect non-compliance (e.g., trailing whitespace), that
hook will state it failed, and also try to fix the issue. You will need to
review and `git add` the changes before you can make a commit.

It is planned to implement additional hooks, possibly including tools such as
`mypy`.

[pre-commit.ci](pre-commit.ci) is configured such that these same hooks will be
automatically run for every pull request.

## Releasing a new version of the service:

Once a new Docker image has been published with a new semantic version tag,
that service version can be released to a Harmony environment by following the
directions in the [Harmony Managing Existing Services
Guide](https://github.com/nasa/harmony/blob/main/docs/guides/managing-existing-services.md).

## Get in touch:

You can reach out to the maintainers of this repository via email:

* david.p.auty@nasa.gov
* matthew.savoie@colorado.edu
* owen.m.littlejohns@nasa.gov
