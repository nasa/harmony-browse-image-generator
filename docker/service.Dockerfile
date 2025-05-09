###############################################################################
#
# Service image for ghcr.io/nasa/harmony-browse-image-generator, HyBIG, a
# Harmony backend service that creates JPEG or PNG browse images from input
# GeoTIFF imagery.
#
# This image installs dependencies via Pip. The service code is then copied
# into the Docker image.
#
# 2023-01-26: Updated for the Harmony Regridding Service.
# 2023-04-04: Updated for HyBIG.
# 2023-04-23: Updated conda clean and pip install to keep Docker image slim.
# 2024-06-18: Updates to remove conda dependency.
# 2024-07-30: Updates to handle separate service an science code directories
# and updates the entrypoint of the new service container
#
###############################################################################
FROM python:3.12-slim

WORKDIR "/home"

RUN apt-get update && apt-get install -y \
    build-essential \
    libgdal-dev \
    gdal-bin \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN export GDAL_VERSION=$(gdal-config --version) && \
    echo "GDAL version: $GDAL_VERSION"

ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

RUN pip install GDAL==$(gdal-config --version)

# Install Pip dependencies
COPY pip_requirements.txt /home/

RUN pip install --no-input --no-cache-dir \
    -r pip_requirements.txt

# Copy service code.
COPY ./harmony_service harmony_service
COPY ./hybig hybig

# Set GDAL related environment variables.
ENV CPL_ZIP_ENCODING=UTF-8

# Configure a container to be executable via the `docker run` command.
ENTRYPOINT ["python", "-m", "harmony_service"]
