#!/usr/bin/env bash
# TensorBoard for rengu-flow runs. Always use the parent output/ dir so run names appear in the sidebar.
#   bash scripts/tensorboard.sh
#   bash scripts/tensorboard.sh /path/to/output
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=lib/tensorboard_common.sh
source "${REPO_ROOT}/scripts/lib/tensorboard_common.sh"

LOGDIR="${1:-${REPO_ROOT}/output}"
cd "${REPO_ROOT}"
tb_launch "${LOGDIR}" "${PORT:-6006}"
