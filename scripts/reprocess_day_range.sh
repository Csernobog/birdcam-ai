#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./reprocess_day_range.sh 20260215 083000 090000
#
# Env overrides:
#   BIRDCAM_BASE=/mnt/nas_ha/birdcam
#   AI_URL=http://127.0.0.1:8000
#   MIN_CONF=0.20
#   APPEND_RESULT=1   (default 1)
#   DRAW_DEBUG=1      (default 1)

DATE="${1:-}"
START="${2:-}"
END="${3:-}"

if [[ -z "$DATE" || -z "$START" || -z "$END" ]]; then
  echo "Usage: $0 YYYYMMDD START_HHMMSS END_HHMMSS"
  exit 2
fi

BASE="${BIRDCAM_BASE:-/mnt/nas_ha/birdcam}"
AI_URL="${AI_URL:-http://127.0.0.1:8000}"
MIN_CONF="${MIN_CONF:-0.20}"
APPEND_RESULT="${APPEND_RESULT:-1}"
DRAW_DEBUG="${DRAW_DEBUG:-1}"

DAY_DIR="${BASE}/${DATE}"
if [[ ! -d "$DAY_DIR" ]]; then
  echo "Day dir not found: ${DAY_DIR}"
  exit 1
fi

OUTDIR="${BASE}/_reports"
mkdir -p "$OUTDIR"

SUMMARY_TSV="${OUTDIR}/summary_${DATE}_${START}-${END}.tsv"
SUMMARY_CSV="${OUTDIR}/summary_${DATE}_${START}-${END}.csv"

printf "date\ttime\thas_bird\tbest_conf\tbest_frame\tdetected_box\n" > "$SUMMARY_TSV"

# run dirs look like: run_HHMMSS inside DAY_DIR
mapfile -t RUN_DIRS < <(
  find "$DAY_DIR" -maxdepth 1 -type d -name "run_*" -printf "%f\n" \
    | sort \
    | awk -v s="$START" -v e="$END" -F_ '{t=$2; if (t>=s && t<=e) print $0}'
)

if [[ ${#RUN_DIRS[@]} -eq 0 ]]; then
  echo "No run dirs found in ${DAY_DIR} between ${START}-${END}"
  exit 0
fi

echo "Processing ${#RUN_DIRS[@]} runs in ${DAY_DIR} for ${DATE} ${START}-${END}"
echo "AI_URL=${AI_URL} MIN_CONF=${MIN_CONF} (ROI forced OFF)"
echo "DRAW_DEBUG=${DRAW_DEBUG} APPEND_RESULT=${APPEND_RESULT}"
echo

for run in "${RUN_DIRS[@]}"; do
  time_part="$(echo "$run" | awk -F_ '{print $2}')"
  run_path="${DAY_DIR}/${run}"

  # Soft warning if crop inputs missing
  missing_crop=0
  for frame in 1 2 3 4; do
    if [[ ! -f "${run_path}/cam_${frame}_crop.jpg" ]]; then
      missing_crop=1
    fi
  done
  if [[ "$missing_crop" -eq 1 ]]; then
    echo "WARN: ${DATE}/${run}: missing one or more cam_x_crop.jpg"
  fi

  # 1) debug_draw on frames 1..4 (ROI OFF)
  if [[ "$DRAW_DEBUG" == "1" ]]; then
    for frame in 1 2 3 4; do
      if ! curl -fsS "${AI_URL}/debug_draw?date=${DATE}&time=${time_part}&frame=${frame}&min_conf=${MIN_CONF}&use_roi=0" >/dev/null; then
        echo "WARN: debug_draw failed for ${DATE}/${run} frame=${frame}"
      fi
    done
  fi

  # 2) classify (ROI OFF)
  resp="$(curl -fsS "${AI_URL}/classify?date=${DATE}&time=${time_part}&min_conf=${MIN_CONF}&use_roi=0&save=0")" || {
    echo "WARN: classify failed for ${DATE}/${run}"
    continue
  }

  has_bird="$(echo "$resp" | jq -r '.has_bird // false')"
  best_conf="$(echo "$resp" | jq -r '.best_conf // 0')"
  best_frame="$(echo "$resp" | jq -r '.best_frame // null')"
  bbox="$(echo "$resp" | jq -c '(.best_bbox // .best.bbox // null)')"

  printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$DATE" "$time_part" "$has_bird" "$best_conf" "$best_frame" "$bbox" \
    >> "$SUMMARY_TSV"

  # 3) Append to result.txt (keep old content, add batch line)
  if [[ "$APPEND_RESULT" == "1" ]]; then
    printf "BATCH %s %s has_bird=%s best_conf=%s best_frame=%s bbox=%s min_conf=%s use_roi=0\n" \
      "$DATE" "$time_part" "$has_bird" "$best_conf" "$best_frame" "$bbox" "$MIN_CONF" \
      >> "${run_path}/result.txt"
  fi
done

# CSV export
awk -F'\t' 'BEGIN{OFS=","} {gsub(/"/, "\"\"", $6); print $1,$2,$3,$4,$5,"\"" $6 "\""}' "$SUMMARY_TSV" > "$SUMMARY_CSV"

echo
echo "Done."
echo "TSV: $SUMMARY_TSV"
echo "CSV: $SUMMARY_CSV"
echo

echo "Summary:"
total_runs="$(($(wc -l < "$SUMMARY_TSV") - 1))"
hits="$(awk -F'\t' 'NR>1 && $3=="true"{c++} END{print c+0}' "$SUMMARY_TSV")"
max_conf="$(awk -F'\t' 'NR>1{if($4+0>m)m=$4+0} END{print m+0}' "$SUMMARY_TSV")"
avg_conf_hits="$(awk -F'\t' 'NR>1 && $3=="true"{s+=$4; c++} END{ if(c>0) printf "%.6f\n", s/c; else print 0 }' "$SUMMARY_TSV")"

echo "  runs:           ${total_runs}"
echo "  hits:           ${hits}"
echo "  hit_rate:       $(awk -v h="$hits" -v t="$total_runs" 'BEGIN{if(t>0) printf "%.2f%%\n", 100*h/t; else print "0.00%"}')"
echo "  max_conf:       ${max_conf}"
echo "  avg_conf(hit):  ${avg_conf_hits}"
echo

echo "Top 10 detections by best_conf:"
{
  echo -e "date\ttime\thas_bird\tbest_conf\tbest_frame\tdetected_box"
  awk -F'\t' 'NR>1 && $3=="true"{print $0}' "$SUMMARY_TSV" | sort -t$'\t' -k4,4nr | head -10
} | column -t -s $'\t'
