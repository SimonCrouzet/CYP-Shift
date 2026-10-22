#!/usr/bin/env bash
# Build the cypshift environment and run the preflight check.
#
# Creates (or updates) the conda env from environment.yml, installs the package with every
# runtime extra, then runs scripts/check_env.py. Idempotent: re-running updates in place.
#
#   ./setup.sh                      # env name "cypshift" via conda/mamba on PATH
#   CONDA_EXE=micromamba ./setup.sh # force a particular conda frontend
#   ENV_PREFIX=/mnt/wsl/ext/envs/cypshift ./setup.sh   # install into a prefix, not a named env
#   EXTRAS=nn,tfm ./setup.sh        # install a subset of extras instead of the full dev set
#   ./setup.sh --fetch              # also download the core challenge data into data/raw
#
# Flags after `setup.sh` are forwarded to check_env.py (e.g. --fetch, --data-dir).

set -euo pipefail
cd "$(dirname "$0")"

ENV_NAME="${ENV_NAME:-cypshift}"
ENV_PREFIX="${ENV_PREFIX:-}"
EXTRAS="${EXTRAS:-dev}"

# Pick a conda frontend: explicit CONDA_EXE, else mamba, micromamba, or conda on PATH.
pick_conda() {
    if [[ -n "${CONDA_EXE:-}" ]] && command -v "${CONDA_EXE}" >/dev/null 2>&1; then
        echo "${CONDA_EXE}"; return
    fi
    for candidate in mamba micromamba conda; do
        if command -v "${candidate}" >/dev/null 2>&1; then echo "${candidate}"; return; fi
    done
    echo ""
}

CONDA="$(pick_conda)"
if [[ -z "${CONDA}" ]]; then
    echo "ERROR: no conda/mamba/micromamba on PATH. Install one, or set CONDA_EXE." >&2
    exit 1
fi
echo ">>> conda frontend: ${CONDA}"

# Target the env by prefix (-p) or by name (-n). A prefix lets the cluster place the env on
# fast local storage instead of the home directory.
if [[ -n "${ENV_PREFIX}" ]]; then
    TARGET=(-p "${ENV_PREFIX}")
    PY="${ENV_PREFIX}/bin/python"
    echo ">>> env prefix: ${ENV_PREFIX}"
else
    TARGET=(-n "${ENV_NAME}")
    PY=""   # resolved after creation via `conda run`
    echo ">>> env name: ${ENV_NAME}"
fi

# Create the env if absent, else update it in place from environment.yml.
if "${CONDA}" env list | grep -qE "(^|[[:space:]/])${ENV_PREFIX:-${ENV_NAME}}([[:space:]]|$)"; then
    echo ">>> env exists; updating from environment.yml"
    "${CONDA}" env update "${TARGET[@]}" -f environment.yml --prune
else
    echo ">>> creating env from environment.yml"
    "${CONDA}" env create "${TARGET[@]}" -f environment.yml
fi

# Resolve the env's python for a named env (prefix path is already known).
if [[ -z "${PY}" ]]; then
    PY="$("${CONDA}" run "${TARGET[@]}" python -c 'import sys; print(sys.executable)')"
fi
echo ">>> env python: ${PY}"

# environment.yml installs `-e .[dev]` via its pip section, but reinstall the requested extras
# explicitly so EXTRAS=nn,tfm (a lighter cluster install) is honoured on an update.
echo ">>> installing cypshift[${EXTRAS}]"
"${PY}" -m pip install -e ".[${EXTRAS}]"

echo ">>> preflight"
"${PY}" scripts/check_env.py "$@"
