#!/bin/bash

# Watch local changes and auto-sync bidirectionally with HPC
# Usage: ./scripts/watch-and-sync.sh
# Ctrl+C to stop

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WATCH_DIRS=("$REPO_ROOT/src" "$REPO_ROOT/configs")

echo "👁️  Watching for changes in: ${WATCH_DIRS[*]}"
echo "Press Ctrl+C to stop"
echo ""

# Track last sync time
LAST_SYNC=$(date +%s)

while true; do
  CURRENT=$(date +%s)
  
  # Check if any files were modified since last sync
  HAS_CHANGES=0
  for dir in "${WATCH_DIRS[@]}"; do
    if [ -d "$dir" ]; then
      if find "$dir" -type f -newermt "$(date -d @$LAST_SYNC '+%Y-%m-%d %H:%M:%S')" 2>/dev/null | grep -q .; then
        HAS_CHANGES=1
        break
      fi
    fi
  done
  
  if [ $HAS_CHANGES -eq 1 ]; then
    echo ""
    echo "🔄 Changes detected at $(date '+%H:%M:%S'), syncing..."
    "$REPO_ROOT/scripts/sync.sh" 2>&1 | grep -E "(✅|error|conflict)" || true
    LAST_SYNC=$CURRENT
    sleep 2  # Debounce: wait 2s before checking again
  fi
  
  sleep 5  # Check every 5 seconds
done
