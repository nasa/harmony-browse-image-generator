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
#
###############################################################################

# Exit status used to report back to caller
STATUS=0

export HDF5_DISABLE_VERSION_CHECK=1

# Run the standard set of unit tests, producing JUnit compatible output
coverage run -m xmlrunner discover tests -o tests/reports
RESULT=$?

if [ "$RESULT" -ne "0" ]; then
    STATUS=1
    echo "ERROR: unittest generated errors"
fi

echo "\n"
echo "Test Coverage Estimates"
coverage report --omit="tests/*"
coverage html --omit="tests/*" -d tests/coverage

# Run pylint
# Ignored errors/warnings:
# W1203 - use of f-strings in log statements. This warning is leftover from
#         using ''.format() vs % notation. For more information, see:
#     	  https://github.com/PyCQA/pylint/issues/2354#issuecomment-414526879
pylint hybig harmony_service --disable=W1203
RESULT=$?
RESULT=$((3 & $RESULT))

if [ "$RESULT" -ne "0" ]; then
    STATUS=1
    echo "ERROR: pylint generated errors"
fi

exit $STATUS
