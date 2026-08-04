#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUPERPROJECT="$(cd "$ROOT/.." && pwd)"
DEPS="$ROOT/dependencies"

if [ ! -f "$SUPERPROJECT/.gitmodules" ]; then
  echo "ERROR: expected $SUPERPROJECT/.gitmodules but it was not found." >&2
  exit 1
fi

git -C "$SUPERPROJECT" submodule sync --recursive >/dev/null 2>&1 || true
git -C "$SUPERPROJECT" submodule update --init --recursive -- \
  gui/dependencies/OpenXR-SDK \
  gui/dependencies/args \
  gui/dependencies/dlss \
  gui/dependencies/fmt \
  gui/dependencies/glfw \
  gui/dependencies/imgui \
  gui/dependencies/pybind11 \
  gui/dependencies/tinylogger \
  gui/dependencies/zlib

echo "GUI dependencies are ready in $DEPS"
