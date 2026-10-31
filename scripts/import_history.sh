#!/usr/bin/env bash
# Daily historical import within the free API-Football quota.
# Safe to run every day (cron): finished jobs are skipped, the run stops when
# the budget is spent and resumes the next day.
#
#   scripts/import_history.sh 2015-2025 80
set -euo pipefail
SEASONS="${1:-2015-2025}"
BUDGET="${2:-80}"

football-ai db-upgrade
# 1) Free source first (no API quota): results, match stats, closing odds since the 1990s
football-ai import-csv --seasons "$SEASONS" || echo "CSV import incomplete (see output)"
# 2) API-Football, priority order fixtures > teams > standings > injuries > details
football-ai import-history --seasons "$SEASONS" --max-requests "$BUDGET"
football-ai jobs
football-ai db-stats
