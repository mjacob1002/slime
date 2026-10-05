#!/usr/bin/env bash
# Re-apply slime's SGLang DEBUG_METRICS extensions inside a docker container.
# Use after `docker compose up --build` since edits to /sgl-workspace/sglang/
# are baked into the image and don't survive a rebuild.
#
# Usage: scripts/sglang_patches/apply.sh <container_name>
# Both injection scripts are idempotent (no-op if already applied).

set -euo pipefail

CONTAINER="${1:-slime-dev-mjacob1002-2.0}"
DIR="$(cd "$(dirname "$0")" && pwd)"

for script in inject_jsonl_writer.py inject_cli_flags.py; do
  docker cp "$DIR/$script" "$CONTAINER":/tmp/
  docker exec -u root "$CONTAINER" python3 "/tmp/$script"
done
