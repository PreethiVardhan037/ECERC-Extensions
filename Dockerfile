# Reproducible environment for the ECERC project -- the SOP's stack: Python 3.9 + PyTorch 2.0.1 (pins in requirements.txt).
#
#   docker build -t ecerc .                                     # GPU-capable PyTorch wheel (CUDA 11.7 runtime bundled by PyPI), ~6 GB
#   docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu -t ecerc-cpu .    # small CPU-only image
#   bash docker_check.sh                                        # build + run the unit tests inside the container (no data needed)
#
#   docker run --rm -it --gpus all -v "$PWD/data":/workspace/data -v "$PWD/tuning_results":/workspace/tuning_results ecerc
#   # inside:  python download_data.py && python auto_tune.py MELD       (--gpus all needs the NVIDIA container toolkit)
FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /workspace

ARG TORCH_INDEX=""
COPY requirements.txt /tmp/requirements.txt
RUN if [ -n "$TORCH_INDEX" ]; then pip install "torch==2.0.1" --index-url "$TORCH_INDEX"; fi \
 && pip install -r /tmp/requirements.txt

COPY . /workspace
CMD ["bash"]
