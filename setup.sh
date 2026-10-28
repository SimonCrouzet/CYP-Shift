#!/usr/bin/env bash
# Prepare the cypshift environment for a run, checking first and installing only what is missing.
#
# Launch it and follow the output: it finds a conda frontend, ensures the env exists, installs
# the package only when an import probe shows it missing, then runs the visual preflight
# (scripts/check_env.py) which ends in READY / NOT READY and the launch command.
#
#   ./setup.sh                       # env "cypshift" via conda/mamba on PATH
#   ./setup.sh --fetch               # also download the core challenge data into data/raw
#   ENV_PREFIX=/scratch/envs/cypshift ./setup.sh   # install into a prefix on local disk
#   EXTRAS=nn,tfm ./setup.sh         # install a subset of extras instead of the full dev set
#   FORCE_UPDATE=1 ./setup.sh        # rebuild the env from environment.yml even if it exists
#   FORCE_INSTALL=1 ./setup.sh       # reinstall the package even if the import probe passes
#
# Flags after `setup.sh` are forwarded to check_env.py (e.g. --fetch, --data-dir).

set -euo pipefail
cd "$(dirname "$0")"

ENV_NAME="${ENV_NAME:-cypshift}"
ENV_PREFIX="${ENV_PREFIX:-}"
EXTRAS="${EXTRAS:-dev}"

# Visual marks, on only for an interactive terminal with NO_COLOR unset.
if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
    BOLD=$'\e[1m'; GREEN=$'\e[32m'; YELLOW=$'\e[33m'; RED=$'\e[31m'; RESET=$'\e[0m'
else
    BOLD=; GREEN=; YELLOW=; RED=; RESET=
fi
ok()   { printf '  %s✓%s %s\n' "${GREEN}" "${RESET}" "$1"; }
warn() { printf '  %s⚠%s %s\n' "${YELLOW}" "${RESET}" "$1"; }
err()  { printf '  %s✗%s %s\n' "${RED}" "${RESET}" "$1"; }
step() { printf '\n%s%s%s\n' "${BOLD}" "$1" "${RESET}"; }

# --- conda frontend ---------------------------------------------------------
step "conda frontend"
CONDA=""
if [[ -n "${CONDA_EXE:-}" ]] && command -v "${CONDA_EXE}" >/dev/null 2>&1; then
    CONDA="${CONDA_EXE}"
else
    for candidate in mamba micromamba conda; do
        command -v "${candidate}" >/dev/null 2>&1 && { CONDA="${candidate}"; break; }
    done
fi
if [[ -z "${CONDA}" ]]; then
    err "no conda/mamba/micromamba on PATH -- install one, or set CONDA_EXE"
    exit 1
fi
ok "using ${CONDA}"

if [[ -n "${ENV_PREFIX}" ]]; then
    TARGET=(-p "${ENV_PREFIX}")
    PY="${ENV_PREFIX}/bin/python"
else
    TARGET=(-n "${ENV_NAME}")
    PY=""
fi

env_exists() {
    if [[ -n "${ENV_PREFIX}" ]]; then
        [[ -x "${ENV_PREFIX}/bin/python" ]]
    else
        "${CONDA}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"
    fi
}

# --- environment ------------------------------------------------------------
step "environment (${ENV_PREFIX:-${ENV_NAME}})"
if env_exists && [[ -z "${FORCE_UPDATE:-}" ]]; then
    ok "env present (FORCE_UPDATE=1 to rebuild from environment.yml)"
elif env_exists; then
    warn "FORCE_UPDATE set -- updating env from environment.yml"
    "${CONDA}" env update "${TARGET[@]}" -f environment.yml --prune
    ok "env updated"
else
    warn "env absent -- creating from environment.yml"
    "${CONDA}" env create "${TARGET[@]}" -f environment.yml
    ok "env created"
fi

# Resolve the env's python for a named env (prefix path is already known).
if [[ -z "${PY}" ]]; then
    PY="$("${CONDA}" run "${TARGET[@]}" python -c 'import sys; print(sys.executable)')"
fi
ok "python: ${PY}"

# --- package ----------------------------------------------------------------
# Probe the modules the requested extras must provide for a run; install only if any are
# missing, so a satisfied env is not reinstalled on every launch.
step "package cypshift[${EXTRAS}]"
PROBE=(cypshift)
case ",${EXTRAS}," in *,dev,*|*,nn,*) PROBE+=(torch chemprop);; esac
case ",${EXTRAS}," in *,dev,*|*,tfm,*) PROBE+=(tabpfn);; esac

if [[ -z "${FORCE_INSTALL:-}" ]] && "${PY}" -c \
    "import importlib,sys; [importlib.import_module(m) for m in sys.argv[1:]]" "${PROBE[@]}" \
    >/dev/null 2>&1; then
    ok "already installed (${PROBE[*]}); skipping pip (FORCE_INSTALL=1 to reinstall)"
else
    warn "installing -e .[${EXTRAS}]"
    "${PY}" -m pip install -e ".[${EXTRAS}]"
    ok "installed"
fi

# --- preflight --------------------------------------------------------------
step "preflight"
"${PY}" scripts/check_env.py "$@"
