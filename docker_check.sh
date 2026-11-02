#!/bin/bash
# Builds the image and runs the unit tests (and optionally the end-to-end smoke test) INSIDE the container.
#   bash docker_check.sh            # unit tests only (about 1 minute)
#   bash docker_check.sh --smoke    # + the end-to-end smoke test on synthetic data (a few minutes)
# Optional: TORCH_INDEX=https://download.pytorch.org/whl/cpu bash docker_check.sh      (CPU-only image)
set -euo pipefail
cd "$(dirname "$0")"
command -v docker >/dev/null || { echo "docker is not installed on this machine"; exit 1; }
BUILD=(docker build -t ecerc-check .)
[ -n "${TORCH_INDEX:-}" ] && BUILD=(docker build --build-arg TORCH_INDEX="$TORCH_INDEX" -t ecerc-check .)
"${BUILD[@]}"
docker run --rm ecerc-check python --version
docker run --rm ecerc-check python -c "import torch, numpy, sklearn; print('torch', torch.__version__, '| numpy', numpy.__version__, '| sklearn', sklearn.__version__)"
docker run --rm ecerc-check python tests/run_tests.py
if [ "${1:-}" = "--smoke" ]; then docker run --rm ecerc-check python tests/smoke_test.py; fi
echo "DOCKER CHECK PASSED"
