#!/usr/bin/env bash
# Creates the absolute paths expected by the original scripts.
# Usage: scripts/link_paths.sh [DATA_DIR]   (DATA_DIR defaults to ./data)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${1:-$REPO/data}"
mkdir -p /home/user "$DATA"
[ -e /home/user/paper ] && [ ! -L /home/user/paper ] && { echo "/home/user/paper exists and is not a link; aborting"; exit 1; }
ln -sfn "$REPO" /home/user/paper
ln -sfn "$(cd "$DATA" && pwd)" /home/user/data
echo "/home/user/paper -> $REPO"; echo "/home/user/data  -> $DATA"
