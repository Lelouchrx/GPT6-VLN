#!/usr/bin/env bash
set -euo pipefail
agent_vln_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$agent_vln_root/env.sh"
cd "$agent_vln_root"
exec "$agent_vln_root/.venv/bin/python" eval.py "$@"
