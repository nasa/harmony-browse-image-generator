###############################################################################
#
# Service image for sds/harmony-browse-image-generator, HyBIG, a Harmony
# backend service that creates JPEG or PNG browse images from input GeoTIFF
# imagery.
#
# This image installs dependencies via Pip. The service code is then copied
# into the Docker image.
#
# 2023-01-26: Updated for the Harmony Regridding Service.
# 2023-04-04: Updated for HyBIG.
# 2023-04-23: Updated conda clean and pip install to keep Docker image slim.
#
###############################################################################
FROM continuumio/miniconda3

WORKDIR "/home"

# Create Conda environment:
COPY conda_requirements.txt conda_requirements.txt
RUN conda create --yes --name hybig --file conda_requirements.txt \
	python=3.10 --channel conda-forge --channel defaults -q && \
	conda clean --all --force-pkgs-dirs --yes

# Install additional Pip dependencies
COPY pip_requirements.txt pip_requirements.txt
RUN conda run --name hybig pip install --no-input --no-cache-dir \
	-r pip_requirements.txt

# Copy service code.
COPY ./harmony_browse_image_generator harmony_browse_image_generator

# Set conda environment for HyBIG, as `conda run` will not stream logging.
# Setting these environment variables is the equivalent of `conda activate`.
ENV _CE_CONDA='' \
	_CE_M='' \
	CONDA_DEFAULT_ENV=hybig \
	CONDA_EXE=/opt/conda/bin/conda \
	CONDA_PREFIX=/opt/conda/envs/hybig \
	CONDA_PREFIX_1=/opt/conda \
	CONDA_PROMPT_MODIFIER=(hybig) \
	CONDA_PYTHON_EXE=/opt/conda/hybig/python \
	CONDA_ROOT=/opt/conda \
	CONDA_SHLVL=2 \
	PATH="/opt/conda/envs/hybig/bin:${PATH}" \
	SHLVL=1

# Set GDAL related environment variables.
ENV CPL_ZIP_ENCODING=UTF-8 \
	GDAL_DATA=/opt/conda/envs/hybig/share/gdal \
	GSETTINGS_SCHEMA_DIR=/opt/conda/envs/hybig/share/glib-2.0/schemas \
	GSETTINGS_SCHEMA_DIR_CONDA_BACKUP='' \
	PROJ_LIB=/opt/conda/envs/hybig/share/proj

# Configure a container to be executable via the `docker run` command.
ENTRYPOINT ["python", "-m", "harmony_browse_image_generator"]
