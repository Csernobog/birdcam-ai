#!/usr/bin/env bash
set -euo pipefail

NAME="bird_ai"
IMAGE="birdcam_ai:latest"
APP_DIR="/home/mano/birdcam_ai"
NAS_DIR="/mnt/nas_ha"

echo "[+] Stopping old container if exists..."
docker rm -f "$NAME" 2>/dev/null || true

echo "[+] Building image..."
docker build --network=host  -t "$IMAGE" "$APP_DIR"

echo "[+] Starting container..."

docker run -d \
  --name "$NAME" \
  --restart unless-stopped \
  --network host \
  -v "$NAS_DIR/birdcam:/nas/birdcam" \
  -v "$APP_DIR/models/extracted:/app/models/extracted:ro" \
  -e BIRDCAM_BASE=/nas/birdcam \
  -e BIRDCAM_MODEL=/app/models/extracted/detect.tflite \
  -e BIRDCAM_MIN_CONF=0.35 \
  -e BIRDCAM_THREADS=4 \
  -e BIRDCAM_ENABLE_SPATIAL_FILTER=0 \
  -e BIRDCAM_REJECT_YCENTER_GT=0.90 \
  -e BIRDCAM_USE_ROI=0 \
  -e BIRDCAM_ORIENTATION=auto \
  -e BIRDCAM_ROI_ORIENTATION=auto \
  -e BIRDCAM_ROI_LANDSCAPE="80,0,520,260" \
  -e BIRDCAM_ROI_PORTRAIT="80,0,520,260" \
  -e BIRDCAM_EXCLUDE_ASPECT_MIN=2.0 \
  -e BIRDCAM_EXCLUDE_AREA_MIN=0.05 \
  -e BIRDCAM_EXCLUDE_AREA_MAX=0.25 \
  -e BIRDCAM_EXCLUDE_IOU=0.20 \
  -e BIRDCAM_EXCLUDE_BOXES_NORM="\
    0,0,0.4049,0.1610;\
    0.2860,0.2331,0.3250,0.3749;\
    0.1666,0.4638,0.4188,0.5648;\
    0.2390,0.5456,0.3671,0.9216;\
    0.8970,0.4230,1.0389,0.5900"\
  -e BIRDCAM_BIG_BBOX_PENALTY=1 \
  -e BIRDCAM_BIG_BBOX_AREA=0.12 \
  -e BIRDCAM_BIG_BBOX_LOW_SCORE=0.30 \
  -e BIRDCAM_NEAR_BEST_SMALL_WINS=1 \
  -e BIRDCAM_NEAR_BEST_SMALL_AREA=0.10 \
  -e BIRDCAM_NEAR_BEST_DELTA_SCORE=0.12 \
  -e BIRDCAM_NEAR_BEST_TRIGGER_AREA=0.15 \
  "$IMAGE"

echo
docker ps --filter "name=^/${NAME}$"

