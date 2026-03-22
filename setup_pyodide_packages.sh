#!/usr/bin/env bash
# Download additional Pyodide packages (sklearn, scipy, etc.) into the local
# Deno/Pyodide cache so they're available inside the RLM sandbox.
#
# DSPy's RLM runs Python in a Pyodide/WASM sandbox via Deno. The npm pyodide
# package ships only ~15 core wheels. Pyodide's loadPackagesFromImports() can
# load more from the local cache, but the wheels must be present on disk.
#
# This script downloads the missing wheels from the Pyodide CDN.
# Run once after `uv sync` / fresh Deno cache.

set -euo pipefail

PYODIDE_VERSION="0.29.2"
CDN_BASE="https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full"

# Find the Deno cache directory for pyodide
CACHE_DIR="$HOME/Library/Caches/deno/npm/registry.npmjs.org/pyodide/${PYODIDE_VERSION}"
if [ ! -d "$CACHE_DIR" ]; then
    # Linux / other OS
    CACHE_DIR="$HOME/.cache/deno/npm/registry.npmjs.org/pyodide/${PYODIDE_VERSION}"
fi

if [ ! -d "$CACHE_DIR" ]; then
    echo "Error: Pyodide cache not found. Run 'uv run python -c \"import dspy\"' first to populate it."
    exit 1
fi

echo "Pyodide cache: $CACHE_DIR"
echo "Pyodide version: $PYODIDE_VERSION"
echo ""

# Packages needed for sklearn + scipy + micropip
PACKAGES=(
    "micropip-0.11.0-py3-none-any.whl"
    "scikit_learn-1.7.0-cp313-cp313-pyodide_2025_0_wasm32.whl"
    "scipy-1.14.1-cp313-cp313-pyodide_2025_0_wasm32.whl"
    "libopenblas-0.3.26.zip"
    "joblib-1.4.2-py3-none-any.whl"
    "threadpoolctl-3.5.0-py3-none-any.whl"
)

for pkg in "${PACKAGES[@]}"; do
    if [ -f "$CACHE_DIR/$pkg" ]; then
        echo "  [ok] $pkg"
    else
        echo "  [dl] $pkg"
        curl -sL "$CDN_BASE/$pkg" -o "$CACHE_DIR/$pkg"
    fi
done

echo ""
echo "Done. Verify with:"
echo "  uv run python -c \"from dspy.primitives.python_interpreter import PythonInterpreter; i=PythonInterpreter(); i.start(); print(i.execute('import sklearn; print(sklearn.__version__)')); i.shutdown()\""
