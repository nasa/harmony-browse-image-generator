#!/bin/sh
###############################################################################
#
# A script invoked by the test Dockerfile to run the Python `unittest` suite
# for the Harmony Browse Image Generator. The script first runs the test suite,
# then it checks for linting errors.
#
# 2020-05-07: Adapted from SwotRepr project.
# 2022-01-03: Removed safety checks, as these are now run in Snyk.
# 2023-04-04: Updated for use with the Harmony Browse Image Generator (HyBIG).
# 2024-07-30: Changes coverage to use pytest and output a unified
# result. xmlrunner was unable to handle finding the tests in the separate
# locations. Also use a relative path to the html output so that coverages can be
# run outside of docker.
#
###############################################################################

# Exit status used to report back to caller
STATUS=0

export HDF5_DISABLE_VERSION_CHECK=1


# Exit status used to report back to caller
STATUS=0

# Run the standard set of unit tests, producing JUnit compatible output
pytest --cov=hybig --cov=harmony_service \
       --cov-report=html:reports/coverage \
       --cov-report term \
       --junitxml=reports/test-reports/test-results-"$(date +'%Y%m%d%H%M%S')".xml || STATUS=1

# Run pylint
pylint hybig harmony_service
RESULT=$((3 & $?))

if [ "$RESULT" -ne "0" ]; then
    STATUS=1
    echo "ERROR: pylint generated errors"
fi

exit $STATUS
