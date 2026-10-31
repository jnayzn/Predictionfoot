#!/usr/bin/env bash
# PostgreSQL backup: compressed custom-format dump + integrity check + retention.
#   DATABASE_URL=postgresql+psycopg://user:pass@host:5432/db scripts/backup_postgres.sh [dir] [days]
set -euo pipefail
: "${DATABASE_URL:?DATABASE_URL is required}"
DIR="${1:-var/backups}"
KEEP_DAYS="${2:-14}"
URL="${DATABASE_URL/postgresql+psycopg:/postgresql:}"   # libpq URL
mkdir -p "$DIR"
FILE="$DIR/football_ai_$(date -u +%Y%m%dT%H%M%SZ).dump"
pg_dump --format=custom --compress=6 --no-owner --dbname="$URL" --file="$FILE.partial"
pg_restore --list "$FILE.partial" > /dev/null          # the archive is readable
mv "$FILE.partial" "$FILE"
find "$DIR" -name 'football_ai_*.dump' -mtime +"$KEEP_DAYS" -delete
echo "$FILE"
