#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENDOR_DIR="${ROOT_DIR}/.vendor/ProtoMotions"
PROTOMOTIONS_URL="https://github.com/NVlabs/ProtoMotions.git"
PROTOMOTIONS_COMMIT="b93d29ce731812af7d0ab29c744fa1396e26a8f9"

if [[ ! -d "${VENDOR_DIR}/.git" ]]; then
    mkdir -p "$(dirname "${VENDOR_DIR}")"
    git clone "${PROTOMOTIONS_URL}" "${VENDOR_DIR}"
fi

git -C "${VENDOR_DIR}" fetch origin
git -C "${VENDOR_DIR}" checkout --detach "${PROTOMOTIONS_COMMIT}"
cp -a "${ROOT_DIR}/overlay/." "${VENDOR_DIR}/"

echo "Prepared ProtoMotions at: ${VENDOR_DIR}"
echo "Activate env_isaaclab before running ./commands.sh."

