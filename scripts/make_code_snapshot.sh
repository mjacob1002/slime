#!/usr/bin/env bash
# Freeze the code a run will execute, so later edits to the live tree cannot change a
# queued or in-flight run.
#
#   bash scripts/make_code_snapshot.sh <name> "<one-line note>"
#
# Writes .snapshots/<name>/ with the same layout as the earlier hand-made snapshots
# (fair_v2_20260930, fair_v3_20260930): slime/ examples/ scripts/ slime_plugins/ tools/
# train.py train_streaming.py pyproject.toml, plus SNAPSHOT_NOTE.txt, SNAPSHOT_GIT_HEAD.txt,
# SNAPSHOT_GIT_DIFF.patch (tracked changes) and SNAPSHOT_SHA256.txt. A run uses it through
# BENCH_CODE_ROOT=/workspace/slime/.snapshots/<name> in its run.sh.
#
# Refuses to overwrite an existing snapshot: a snapshot a run has already used must not change.
set -euo pipefail
NAME=${1:?usage: make_code_snapshot.sh <name> "<note>"}
NOTE=${2:?usage: make_code_snapshot.sh <name> "<note>"}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DEST=$REPO/.snapshots/$NAME
[ -e "$DEST" ] && { echo "refusing to overwrite existing snapshot $DEST" >&2; exit 1; }
mkdir -p "$DEST/logs/sglang_metrics"
cd "$REPO"
for entry in slime examples scripts slime_plugins tools train.py train_streaming.py pyproject.toml; do
  rsync -a --exclude '__pycache__' --exclude '*.pyc' --exclude '*.bak*' "$entry" "$DEST/"
done
printf '%s\n' "$NOTE" > "$DEST/SNAPSHOT_NOTE.txt"
git rev-parse HEAD > "$DEST/SNAPSHOT_GIT_HEAD.txt"
git diff HEAD > "$DEST/SNAPSHOT_GIT_DIFF.patch"
( cd "$DEST" && find . -type f ! -name 'SNAPSHOT_SHA256.txt' -print0 | sort -z | xargs -0 sha256sum > SNAPSHOT_SHA256.txt )
echo "snapshot: $DEST ($(du -sh "$DEST" | cut -f1), $(wc -l < "$DEST/SNAPSHOT_SHA256.txt") files, HEAD $(cut -c1-8 "$DEST/SNAPSHOT_GIT_HEAD.txt"))"
