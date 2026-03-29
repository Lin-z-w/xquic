#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${ROOT_DIR}/build"

SSL_TYPE_STR="boringssl"
SSL_PATH_STR="${ROOT_DIR}/third_party/boringssl"

ONNX_RUNTIME_DIR="/usr/local/onnxruntime-linux-x64-1.24.4"
ONNX_MODEL_DIR="${ROOT_DIR}/third_party/state_predict/onnx_export"
STATE_MODEL_PATH="${ONNX_MODEL_DIR}/state_prediction_no_validation.onnx"
QUEUE_MODEL_PATH="${ONNX_MODEL_DIR}/qu_queue_depth/qu_queue_depth_regressor.onnx"

if [ ! -f "${STATE_MODEL_PATH}" ]; then
    echo "state model not found: ${STATE_MODEL_PATH}"
    exit 1
fi

if [ ! -f "${QUEUE_MODEL_PATH}" ]; then
    echo "queue model not found: ${QUEUE_MODEL_PATH}"
    exit 1
fi

rm -rf "${BUILD_DIR}"

git -C "${ROOT_DIR}" submodule update --init --recursive

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake -DGCOV=off \
    -DCMAKE_BUILD_TYPE=Debug \
    -DXQC_ENABLE_TESTING=1 \
    -DXQC_SUPPORT_SENDMMSG_BUILD=1 \
    -DXQC_ENABLE_EVENT_LOG=0 \
    -DXQC_ONLY_ERROR_LOG=1 \
    -DXQC_ENABLE_BBR2=1 \
    -DXQC_ENABLE_RENO=1 \
    -DXQC_ENABLE_COPA=1 \
    -DXQC_ENABLE_ML_CC=1 \
    -DXQC_ENABLE_UNLIMITED=1 \
    -DXQC_PRINT_SECRET=1 \
    -DSSL_TYPE="${SSL_TYPE_STR}" \
    -DSSL_PATH="${SSL_PATH_STR}" \
    -DONNX_RUNTIME_DIR="${ONNX_RUNTIME_DIR}" \
    -DONNX_MODEL_DIR="${ONNX_MODEL_DIR}" \
    "${ROOT_DIR}"

make -j"$(nproc)"
