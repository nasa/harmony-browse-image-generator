###############################################################################
#
# Test image for the Harmony Browse Image Generator (HyBIG). This test image
# uses the main service image, ghcr.io/nasa/harmony-browse-image-generator, as
# a base layer for the tests. This ensures that the contents of the service
# image are tested, preventing discrepancies between the service and test
# environments.
#
# 2023-04-04: Updated for HyBIG.
# 2023-04-17: Added --no-cache-dir to keep Docker images slim.
# 2024-01-22: Updated to use new open-source service image name.
#
###############################################################################
FROM ghcr.io/nasa/harmony-browse-image-generator

# Install additional Pip requirements (for testing)
COPY tests/pip_test_requirements.txt .
RUN conda run --name hybig pip install --no-input --no-cache-dir \
	-r pip_test_requirements.txt

# Copy test directory containing Python unittest suite, test data and utilities
COPY ./tests tests

# Configure a container to be executable via the `docker run` command.
ENTRYPOINT ["/home/tests/run_tests.sh"]
