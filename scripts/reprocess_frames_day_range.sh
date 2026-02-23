#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./reprocess_frames_day_range.sh 20260215 083000 090000
#
# Env overrides:
#   BIRDCAM_BASE=/mnt/nas_ha/birdcam
#   AI_URL=http://127.0.0.1:8000
#   MIN_CONF=0.20          # bird_present threshold (raw_bird.score)
#   USE_ROI=0              # pass-through to endpoint
#   SAVE_DIR=debug_frames  # under DAY folder
#   CSV_NAME=frames.csv    # output csv name under SAVE_DIR

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
USE_ROI="${USE_ROI:-0}"
SAVE_DIR="${SAVE_DIR:-debug_frames}"
CSV_NAME="${CSV_NAME:-frames.csv}"

DAY_DIR="${BASE}/${DATE}"
if [[ ! -d "$DAY_DIR" ]]; then
  echo "Day dir not found: ${DAY_DIR}"
  exit 1
fi

OUTDIR="${DAY_DIR}/${SAVE_DIR}"
mkdir -p "$OUTDIR"
CSV_PATH="${OUTDIR}/${CSV_NAME}"

# CSV header (includes empty manual labels)
printf "%s\n" \
"day,run_id,ts,cam,bird_present,bird_conf,bird_xmin,bird_ymin,bird_xmax,bird_ymax,top_class_id,top_conf,top_xmin,top_ymin,top_xmax,top_ymax,image_path,debug_path,OK,FALS_P15,FALS_N15,TOP_IS_BIRD" \
> "$CSV_PATH"

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
echo "AI_URL=${AI_URL} MIN_CONF=${MIN_CONF} USE_ROI=${USE_ROI}"
echo "OUTDIR=${OUTDIR}"
echo

for run in "${RUN_DIRS[@]}"; do
  time_part="$(echo "$run" | awk -F_ '{print $2}')"
  run_path="${DAY_DIR}/${run}"

  for cam in 1 2 3 4; do
    img_path="${run_path}/cam_${cam}_crop.jpg"
    if [[ ! -f "$img_path" ]]; then
      echo "WARN: missing ${DATE}/${run}/cam_${cam}_crop.jpg"
      continue
    fi

    save_name="debug_${DATE}_${time_part}_cam${cam}.jpg"

    # Call debug_draw; it will write image into DATE/SAVE_DIR and return JSON containing raw_top/raw_bird
    resp="$(curl -fsS \
      "${AI_URL}/debug_draw?date=${DATE}&time=${time_part}&frame=${cam}&min_conf=${MIN_CONF}&use_roi=${USE_ROI}&save_dir=${SAVE_DIR}&save_name=${save_name}&draw_top10=0" \
    )" || {
      echo "WARN: debug_draw failed for ${DATE}/${run} cam=${cam}"
      continue
    }

    debug_path="$(echo "$resp" | jq -r '.out_path // empty')"

    # raw bird fields (may be null)
    bird_conf="$(echo "$resp" | jq -r '.raw_bird.score // ""')"
    bird_xmin="$(echo "$resp" | jq -r '.raw_bird.bbox.xmin // ""')"
    bird_ymin="$(echo "$resp" | jq -r '.raw_bird.bbox.ymin // ""')"
    bird_xmax="$(echo "$resp" | jq -r '.raw_bird.bbox.xmax // ""')"
    bird_ymax="$(echo "$resp" | jq -r '.raw_bird.bbox.ymax // ""')"

    # bird_present based on raw bird conf >= MIN_CONF
    bird_present="0"
    if [[ -n "$bird_conf" ]]; then
      bird_present="$(awk -v c="$bird_conf" -v t="$MIN_CONF" 'BEGIN{print (c+0>=t+0) ? 1 : 0}')"
    fi

    # raw top fields
    top_class_id="$(echo "$resp" | jq -r '.raw_top.class // ""')"
    top_conf="$(echo "$resp" | jq -r '.raw_top.score // ""')"
    top_xmin="$(echo "$resp" | jq -r '.raw_top.bbox.xmin // ""')"
    top_ymin="$(echo "$resp" | jq -r '.raw_top.bbox.ymin // ""')"
    top_xmax="$(echo "$resp" | jq -r '.raw_top.bbox.xmax // ""')"
    top_ymax="$(echo "$resp" | jq -r '.raw_top.bbox.ymax // ""')"

    # Write CSV row
    # Note: keep ts as HHMMSS, run_id as run_HHMMSS to match folder naming
    printf "%s\n" \
"${DATE},${run},${time_part},${cam},${bird_present},${bird_conf},${bird_xmin},${bird_ymin},${bird_xmax},${bird_ymax},${top_class_id},${top_conf},${top_xmin},${top_ymin},${top_xmax},${top_ymax},${img_path},${debug_path},,,," \
>> "$CSV_PATH"
  done
done

echo
echo "Done."
echo "CSV: ${CSV_PATH}"
echo "Images: ${OUTDIR}/debug_${DATE}_*_cam*.jpg"
