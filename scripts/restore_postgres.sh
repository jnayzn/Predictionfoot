#!/usr/bin/env bash
# Restore a dump produced by backup_postgres.sh INTO the database of DATABASE_URL.
# Destructive (existing objects are dropped): requires --yes.
#   DATABASE_URL=... scripts/restore_postgres.sh var/backups/football_ai_XXXX.dump --yes
set -euo pipefail
: "${DATABASE_URL:?DATABASE_URL is required}"
FILE="${1:?dump file required}"
if [[ "${2:-}" != "--yes" ]]; then
  echo "This will overwrite the database in DATABASE_URL. Re-run with --yes to confirm." >&2
  exit 2
fi
URL="${DATABASE_URL/postgresql+psycopg:/postgresql:}"
pg_restore --clean --if-exists --no-owner --exit-on-error --dbname="$URL" "$FILE"
echo "restored $FILE"
