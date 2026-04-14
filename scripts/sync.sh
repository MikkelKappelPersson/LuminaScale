#!/bin/bash
set -e

# Bidirectional sync with HPC
# Usage: ./scripts/sync.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE="aicloud:~/projects/LuminaScale"

echo "🔄 Bidirectional sync with HPC"
echo "📍 Local:  $REPO_ROOT"
echo "📍 Remote: $REMOTE"
echo "⏱️  Started at $(date '+%Y-%m-%d %H:%M:%S')"

rclone bisync "$REPO_ROOT" "$REMOTE" \
  --filter-from "$REPO_ROOT/.rcloneignore" \
  --verbose \
  --stats-one-line \
  --transfers=8 \
  --checkers=8

echo "✅ Sync complete at $(date '+%Y-%m-%d %H:%M:%S')"
