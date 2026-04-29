#!/usr/bin/env bash

set -euo pipefail

BOLD='\033[1m'
GREEN='\033[32m'
BLUE='\033[34m'
NO_COLOR='\033[0m'

function info() {
  echo -e "[${BLUE}${BOLD}INFO${NO_COLOR}] $*"
}

function ok() {
  echo -e "[${GREEN}${BOLD} OK ${NO_COLOR}] $*"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
IMAGE_NAME="${IMAGE_NAME:-vision-gpu-build:local}"
DOCKERFILE_PATH="${DOCKERFILE_PATH:-${REPO_ROOT}/docker/Dockerfile.cuda}"
WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace/vision}"
DEFAULT_BASE_IMAGE="nvidia/cuda:12.4.1-devel-ubuntu22.04"
LOCAL_BASE_IMAGE="registry.cn-hangzhou.aliyuncs.com/wheelos/apollo:dev-x86_64-22.04-gpu"
if docker image inspect "${LOCAL_BASE_IMAGE}" >/dev/null 2>&1; then
  BASE_IMAGE="${BASE_IMAGE:-${LOCAL_BASE_IMAGE}}"
else
  BASE_IMAGE="${BASE_IMAGE:-${DEFAULT_BASE_IMAGE}}"
fi
GPU_CAPABILITY="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '.')"
CUDA_ARCHS="${CUDA_ARCHS:-compute_${GPU_CAPABILITY}:sm_${GPU_CAPABILITY}}"

if [[ $# -gt 0 ]]; then
  BAZEL_COMMAND="$*"
else
  BAZEL_COMMAND="bazel test --config=gpu --@rules_cuda//cuda:archs=${CUDA_ARCHS} //vision/lidar:all"
fi

RUNTIME_IMAGE="${IMAGE_NAME}"
if docker run --rm "${BASE_IMAGE}" bash -lc 'command -v bazel >/dev/null 2>&1' \
    >/dev/null 2>&1; then
  info "Using base image ${BASE_IMAGE} directly"
  RUNTIME_IMAGE="${BASE_IMAGE}"
else
  info "Building ${IMAGE_NAME} from ${DOCKERFILE_PATH} using ${BASE_IMAGE}"
  docker build \
    --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
    -f "${DOCKERFILE_PATH}" \
    -t "${IMAGE_NAME}" \
    "${REPO_ROOT}"
fi

info "Running GPU build command: ${BAZEL_COMMAND}"
docker run --rm \
  --gpus all \
  --user "$(id -u):$(id -g)" \
  -e "USER=$(id -un)" \
  -e "HOME=${HOME}" \
  -v "${REPO_ROOT}:${WORKSPACE_DIR}" \
  -w "${WORKSPACE_DIR}" \
  "${RUNTIME_IMAGE}" \
  bash -lc "${BAZEL_COMMAND}"

ok "Container build completed"
