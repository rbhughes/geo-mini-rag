#!/usr/bin/env bash
# Does appraisal help retrieval, or only save money?
#
# Builds two indexes from the same code and the same root directory:
#   raw        ingest walks every file under the root, appraisal ignored
#   appraised  ingest takes only what the manifest admits
# then scores both against the same question set and prints them side by side.
#
# Usage: evals/appraisal_ab.sh [ROOT] [QUESTIONS] [OUTDIR]
set -euo pipefail

ROOT="${1:-data/raw}"
QUESTIONS="${2:-evals/subset.jsonl}"
OUT="${3:-evals/runs/$(date +%Y%m%d-%H%M%S)}"
RAW_DB="$OUT/raw.duckdb"
APP_DB="$OUT/appraised.duckdb"

mkdir -p "$OUT"
echo "root=$ROOT questions=$QUESTIONS out=$OUT"
{
  echo "root: $ROOT"
  echo "questions: $QUESTIONS"
  echo "commit: $(git rev-parse --short HEAD 2>/dev/null || echo none)"
  echo "started: $(date -u +%FT%TZ)"
} > "$OUT/run.txt"

# Appraisal writes to the appraised index only; the raw index never sees it.
echo "== appraise"
uv run geo-mini-rag appraise --root "$ROOT" --db "$APP_DB" | tee "$OUT/appraise.txt"

echo "== ingest: appraised (manifest-gated)"
/usr/bin/time -p uv run geo-mini-rag ingest --root "$ROOT" --db "$APP_DB" \
  --manifest latest 2>&1 | tee "$OUT/ingest_appraised.txt"

echo "== ingest: raw (every file under the root)"
/usr/bin/time -p uv run geo-mini-rag ingest --root "$ROOT" --db "$RAW_DB" 2>&1 \
  | tee "$OUT/ingest_raw.txt"

for name in raw appraised; do
  db="$OUT/$name.duckdb"
  echo "== eval: $name"
  uv run geo-mini-rag eval "$QUESTIONS" --db "$db" -v | tee "$OUT/eval_$name.txt"
  uv run geo-mini-rag stats --db "$db" | tee "$OUT/stats_$name.txt"
done

echo "finished: $(date -u +%FT%TZ)" >> "$OUT/run.txt"
echo "results in $OUT"
