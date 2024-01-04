# Harmony Browse Image Generator (HyBIG) backend service.

This Harmony backend service is designed to produce browse imagery, with
default behaviour to produce browse imagery that is compatible with Global
Image Browse Services (GIBS).

## Repository structure:

```
|- .pre-commit-config.yaml
|- CHANGELOG.md
|- README.md
|- bin
|- conda_requirements.txt
|- dev-requirements.txt
|- docker
|- docs
|- harmony_browse_image_generator
|- pip_requirements.txt
|- tests
```

* `.pre-commit-config` - a pre-commit configuration file describing functions to
  be run on every git commit.
* `CHANGELOG.md` - This file contains a record of changes applied to each new
  release of a service Docker image. Any release of a new service version
  should have a record of what was changed in this file.
* `README.md` - This file, containing guidance on developing the service.
* `bin` - A directory containing utility scripts to build the service and test
  images. This also includes scripts that Bamboo uses to deploy new service
  images to AWS ECR.
* `conda_requirements.txt` - A list of service dependencies, such as GDAL, that
  cannot be installed via Pip.
* `dev-requirements.txt` - list of packages required for service development.
* `docker` - A directory containing the Dockerfiles for the service and test
  images. It also contains `service_version.txt`, which contains the semantic
  version number of the service image. Any time an update is made that should
  have an accompanying service image release, this file should be updated.
* `docs` - directory with example usage notebooks.
* `harmony_browse_image_generator` - The directory containing Python source code
  for the HyBIG. `adapter.py` contains the `BrowseImageGeneratorAdapter`
  class that is invoked by calls to the service.
* `pip_requirements.txt` - A list of service Python package dependencies.
* `tests` - A directory containing the service unit test suite.

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
test have been updated, they can be run via:

```bash
$ ./bin/build-image
$ ./bin/build-test
$ ./bin/run-test
```

The `tests/run_tests.sh` script will also generate a coverage report, rendered
in HTML, and scan the code with `pylint`.

Currently, the `unittest` suite is run automatically within Bamboo as part of a
CI/CD pipeline. In future, this project will be migrated from Bitbucket to
GitHub, at which point the CI/CD will be migrated to workflows that use GitHub
Actions.

## Versioning:

Service Docker images for HyBIG adhere to semantic version numbers:
major.minor.patch.

* Major increments: These are non-backwards compatible API changes.
* Minor increments: These are backwards compatible API changes.
* Patch increments: These updates do not affect the API to the service.

When publishing a new Docker image for the service, two files need to be
updated:

* CHANGELOG.md - Notes should be added to capture the changes to the service.
* docker/service_version.txt - The semantic version number should be updated.

## Docker image publication:

Initially service Docker images will be hosted in AWS Elastic Container
Registry (ECR). When this repository is migrated to the NASA GitHub
organisation, service images will be published to ghcr.io, instead.

## Releasing a new version of the service:

Once a new Docker image has been published with a new semantic version tag,
that service version can be released to a Harmony environment by updating the
main Harmony Bamboo deployment project. Find the environment you wish to
release the service version to and update the associated environment variable
to update the semantic version tag at the end of the full Docker image name.
