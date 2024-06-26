# Use the condaforge/mambaforge image as the base image
FROM nvidia/cuda:11.8.0-base-ubuntu22.04

#install miniforge copied from https://github.com/conda-forge/miniforge-images/blob/master/ubuntu/Dockerfile
ARG MINIFORGE_NAME=Miniforge3
ARG MINIFORGE_VERSION=23.3.1-1
ARG TARGETPLATFORM

ENV CONDA_DIR=/opt/conda
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH=${CONDA_DIR}/bin:${PATH}
RUN apt-get update > /dev/null && \
    apt-get install --no-install-recommends --yes \
        wget bzip2 ca-certificates \
        git \
        tini \
        > /dev/null && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    wget --no-hsts --quiet https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/${MINIFORGE_NAME}-${MINIFORGE_VERSION}-Linux-$(uname -m).sh -O /tmp/miniforge.sh && \
    /bin/bash /tmp/miniforge.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniforge.sh && \
    conda clean --tarballs --index-cache --packages --yes && \
    find ${CONDA_DIR} -follow -type f -name '*.a' -delete && \
    find ${CONDA_DIR} -follow -type f -name '*.pyc' -delete && \
    conda clean --force-pkgs-dirs --all --yes  && \
    echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate base" >> /etc/skel/.bashrc && \
    echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate base" >> ~/.bashrc


WORKDIR /opt/pfp
COPY polyfingerprints /tmp/packages/polyfingerprints

# Add /opt/pfp to the PYTHONPATH
ENV PYTHONPATH="/tmp/packages:${PYTHONPATH}"


# Copy the script folder to the /opt/pfp directory
COPY scripts /opt/pfp/scripts

#change the permissions of the script folder to make it readable and executable by all users, but not writable
RUN chmod -R 755 /opt/pfp/scripts

RUN apt-get update && apt-get update -y
RUN mamba update -n base -c defaults conda

# Install logrotate
RUN apt-get install -y logrotate

# Copy the logrotate configuration file
COPY /docker/conf/logrotate.conf /etc/logrotate.conf

COPY env_gpu.yaml env_gpu.yaml

# Create a new Conda environment called "pfp"
RUN mamba create --name pfp python=3.12 -y

RUN mamba env update -n pfp -f env_gpu.yaml

# copy the examples folder to the /opt/pfp directory
COPY /examples /opt/pfp/examples




# make /bin/bash the default shell
SHELL ["/bin/bash", "-c"]

CMD ["/bin/bash","/opt/pfp/scripts/startup.sh"]