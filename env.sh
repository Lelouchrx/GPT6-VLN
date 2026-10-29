#!/usr/bin/env bash
_agent_vln_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export VLN_WORKSPACE="$_agent_vln_root/.runtime"
export HABITAT_LAB_PATH="$HOME/VLN/third_party/habitat-lab"
export NAV_MODEL_PYTHON="$_agent_vln_root/.venv/bin/python"
export NAV_CODEX_BIN="$HOME/.local/bin/codex"
export PATH="$_agent_vln_root/.venv/bin:$HOME/.local/bin:$PATH"
export PYTHONPATH="$_agent_vln_root:$HABITAT_LAB_PATH/habitat-lab:$HABITAT_LAB_PATH/habitat-baselines${PYTHONPATH:+:$PYTHONPATH}"
export MAGNUM_LOG="${MAGNUM_LOG:-quiet}"
export HABITAT_SIM_LOG="${HABITAT_SIM_LOG:-quiet}"
unset _agent_vln_root
