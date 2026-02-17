#!/bin/bash -l

set -euo pipefail

if ! command -v module >/dev/null 2>&1; then
  echo "module command is not available in this shell."
  echo "Run this on the HPC login node (or an interactive job shell)."
  exit 1
fi

if [ "$#" -gt 0 ]; then
  TARGETS=("$@")
else
  TARGETS=(Python GCCcore uv ollama dspy dspy-ai)
fi

echo "Node: $(hostname)"
echo "Checking module availability for: ${TARGETS[*]}"
echo

for target in "${TARGETS[@]}"; do
  echo "=== ${target} ==="
  mapfile -t matches < <(
    module -t spider "$target" 2>&1 \
      | awk '/^[[:alnum:]_.+-]+\/[[:alnum:]_.+-]+$/ { print }'
  )
  if [ "${#matches[@]}" -eq 0 ]; then
    echo "No module match found for '${target}'."
  else
    printf '%s\n' "${matches[@]}"
  fi
  echo
done

cat <<'EOF'
Notes:
- You usually get Python via modules (e.g. GCCcore + Python).
- dspy/dspy-ai is typically a Python package, not a cluster module.
- If no dspy module appears, install it into a venv with:
    python -m pip install "dspy-ai>=3.0.4"
EOF
