#!/usr/bin/env python3
import os
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import cv2
from fastapi import FastAPI, HTTPException, Query
from tflite_runtime.interpreter import Interpreter


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_digits(s: str, name: str) -> str:
    if not s.isdigit():
        raise ValueError(f"{name} must be digits only")
    return s


def parse_roi(s: str) -> Tuple[int, int, int, int]:
    """Parse 'x1,y1,x2,y2' into ints. Allows spaces."""
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI must be 'x1,y1,x2,y2'")
    x1, y1, x2, y2 = (int(p) for p in parts)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("ROI must have x2>x1 and y2>y1")
    return x1, y1, x2, y2


def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v




def safe_simple_name(s: str, name: str) -> str:
    """
    Only allow simple filenames/dirnames to avoid path traversal.
    Allowed: letters, digits, underscore, dash, dot.
    """
    if not s:
        raise ValueError(f"{name} must be non-empty")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", s):
        raise ValueError(f"{name} contains invalid characters: {s}")
    return s

def bbox_area(b: Dict[str, float]) -> float:
    return max(0.0, b["xmax"] - b["xmin"]) * max(0.0, b["ymax"] - b["ymin"])


def bbox_aspect_hw(b: Dict[str, float]) -> float:
    w = max(1e-9, b["xmax"] - b["xmin"])
    h = max(1e-9, b["ymax"] - b["ymin"])
    return h / w


def bbox_iou(a: Dict[str, float], b: Dict[str, float]) -> float:
    ix1 = max(a["xmin"], b["xmin"])
    iy1 = max(a["ymin"], b["ymin"])
    ix2 = min(a["xmax"], b["xmax"])
    iy2 = min(a["ymax"], b["ymax"])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = bbox_area(a) + bbox_area(b) - inter
    return inter / ua if ua > 0 else 0.0


def parse_exclude_boxes_norm(s: str) -> List[Dict[str, float]]:
    # "xmin,ymin,xmax,ymax; xmin,ymin,xmax,ymax"
    out: List[Dict[str, float]] = []
    if not s:
        return out
    for part in s.split(";"):
        part = part.strip()
        if not part:
            continue
        vals = [float(x.strip()) for x in part.split(",")]
        if len(vals) != 4:
            continue
        x1, y1, x2, y2 = vals
        out.append({"xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2})
    return out

def preprocess_for_uint8_ssd(img_bgr: np.ndarray, in_h: int, in_w: int) -> np.ndarray:
    """Typical SSD expects uint8 RGB [1,H,W,3]."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, (in_w, in_h), interpolation=cv2.INTER_AREA)
    return resized.astype(np.uint8)


class SsdDetector:
    """SSD + TFLite_Detection_PostProcess outputs."""

    def __init__(
        self,
        model_path: str,
        num_threads: int = 4,
        exclude_boxes_norm: Optional[List[Dict[str, float]]] = None,
        exclude_iou: float = 0.25,
        exclude_aspect_min: float = 0.0,
        exclude_area_min: float = 0.0,
        exclude_area_max: float = 1.0,
        class0_as_bird_min_conf: float = 0.50,
    ):
        self.model_path = model_path
        # Exclusion filters (used to skip common false positives like the feeder)
        self.exclude_boxes_norm = exclude_boxes_norm or []
        self.exclude_iou = float(exclude_iou)
        self.exclude_aspect_min = float(exclude_aspect_min)
        self.exclude_area_min = float(exclude_area_min)
        self.exclude_area_max = float(exclude_area_max)
        self.interpreter = Interpreter(model_path=model_path, num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.class0_as_bird_min_conf = float(class0_as_bird_min_conf)   
        in_shape = self.input_details[0]["shape"]
        self.in_h, self.in_w = int(in_shape[1]), int(in_shape[2])
        self.in_index = self.input_details[0]["index"]

        # (meta, tensor_index)
        self._out_meta = [(o, o["index"]) for o in self.output_details]
    def _effective_class(self, c_raw: int, score: float, target_class: Optional[int]) -> int:
        """
        Promote class 0 to target_class if score >= threshold.
        If target_class is None, we cannot promote meaningfully, so return raw.
        """
        if target_class is None:
            return c_raw
        if c_raw == 0 and score >= self.class0_as_bird_min_conf:
            return int(target_class)
        return c_raw
    def run_ssd(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        inp = preprocess_for_uint8_ssd(img_bgr, self.in_h, self.in_w)
        inp = np.expand_dims(inp, axis=0)

        self.interpreter.set_tensor(self.in_index, inp)
        self.interpreter.invoke()

        outs = [self.interpreter.get_tensor(idx) for _, idx in self._out_meta]

        boxes = classes = scores = None
        for o, arr in zip(self.output_details, outs):
            shape = arr.shape
            name = (o.get("name") or "").lower()

            # Typical SSD postprocess outputs:
            # boxes:  [1, N, 4] (ymin,xmin,ymax,xmax) normalized
            # classes:[1, N]
            # scores: [1, N]
            if arr.ndim == 3 and shape[-1] == 4:
                boxes = arr
            elif arr.ndim == 2 and shape[-1] > 1:
                if "score" in name:
                    scores = arr
                elif "class" in name:
                    classes = arr
                else:
                    # heuristic
                    if arr.dtype in (np.float32, np.float16) and float(np.max(arr)) <= 1.0:
                        scores = arr
                    else:
                        classes = arr

        # Fallback positional assumption (common order): boxes, classes, scores, count
        if boxes is None and len(outs) >= 1:
            boxes = outs[0]
        if classes is None and len(outs) >= 2:
            classes = outs[1]
        if scores is None and len(outs) >= 3:
            scores = outs[2]

        boxes = np.squeeze(boxes, axis=0) if boxes is not None else np.zeros((0, 4), dtype=np.float32)
        classes = np.squeeze(classes, axis=0) if classes is not None else np.zeros((0,), dtype=np.float32)
        scores = np.squeeze(scores, axis=0) if scores is not None else np.zeros((0,), dtype=np.float32)

        classes = classes.astype(np.int32, copy=False)
        scores = scores.astype(np.float32, copy=False)
        boxes = boxes.astype(np.float32, copy=False)

        return boxes, classes, scores

    def best_detection_any(
        self,
        img_path: str,
        min_conf: float,
        use_roi: bool,
        roi: Optional[Tuple[int, int, int, int]],
        target_class: Optional[int],
    ) -> Dict[str, Any]:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            return {"ok": False, "error": f"cannot_read: {img_path}"}

        full_h, full_w = img.shape[:2]

        crop = img
        roi_box = None
        if use_roi and roi is not None:
            x1, y1, x2, y2 = roi
            # clamp ROI to image bounds
            x1 = int(clamp(x1, 0, full_w - 1))
            x2 = int(clamp(x2, 1, full_w))
            y1 = int(clamp(y1, 0, full_h - 1))
            y2 = int(clamp(y2, 1, full_h))
            if x2 > x1 and y2 > y1:
                crop = img[y1:y2, x1:x2].copy()
                roi_box = (x1, y1, x2, y2)

        boxes, classes, scores = self.run_ssd(crop)

        # Collect all candidates above min_conf (and matching target_class if provided),
        # map bboxes to FULL image normalized coords if ROI used, then pick the best
        # candidate that is NOT excluded by feeder/shape filters.
        candidates: List[Dict[str, Any]] = []
        n = min(len(scores), len(classes), len(boxes))
        tgt = int(target_class) if target_class is not None else None

        for i in range(n):
            s = float(scores[i])
            if s < min_conf:
                continue

            c_raw = int(classes[i])
            c_eff = self._effective_class(c_raw, s, tgt)

            # NOTE: target_class filtering must use effective class (after promotion)
            if tgt is not None and c_eff != tgt:
                continue

            ymin, xmin, ymax, xmax = [float(x) for x in boxes[i]]

            # map bbox back to FULL image normalized coords if ROI used
            if roi_box is not None:
                x1, y1, x2, y2 = roi_box
                roi_w = float(x2 - x1)
                roi_h = float(y2 - y1)
                ymin_f = (y1 + ymin * roi_h) / float(full_h)
                ymax_f = (y1 + ymax * roi_h) / float(full_h)
                xmin_f = (x1 + xmin * roi_w) / float(full_w)
                xmax_f = (x1 + xmax * roi_w) / float(full_w)
                ymin, xmin, ymax, xmax = ymin_f, xmin_f, ymax_f, xmax_f

            bbox = {"ymin": ymin, "xmin": xmin, "ymax": ymax, "xmax": xmax}
            promoted = (c_raw == 0 and tgt is not None and c_eff == tgt)

            candidates.append(
                {
                    "score": s,
                    "class": c_eff,  # effective class (after promotion)
                    "bbox": bbox,
                    "raw_class": c_raw,
                    "promoted_from_class0": promoted,
                }
            )

        candidates.sort(key=lambda d: d["score"], reverse=True)

        def is_excluded(bbox: Dict[str, float]) -> bool:
            # A) Shape/area filter: useful to reject the "whole feeder" false positive
            if self.exclude_aspect_min > 0.0:
                asp = bbox_aspect_hw(bbox)
                if asp >= self.exclude_aspect_min:
                    a = bbox_area(bbox)
                    if a >= self.exclude_area_min and a <= self.exclude_area_max:
                        return True

            # B) Exclusion zones (IoU): useful to reject "cap-only" false positives
            for xb in self.exclude_boxes_norm:
                if bbox_iou(bbox, xb) >= self.exclude_iou:
                    return True
            return False

        kept: List[Dict[str, Any]] = []
        skipped = 0
                
        for cand in candidates:
            if is_excluded(cand["bbox"]):
                skipped += 1
                continue
            kept.append(cand)

        best = kept[0] if kept else None

        out: Dict[str, Any] = {
            "ok": True,
            "best": best,
            "image_shape": [int(full_h), int(full_w), 3],
            "candidates": int(len(candidates)),
            "skipped": int(skipped),
            "kept": kept[:10],
        }
        if roi_box is not None:
            out["roi"] = {"x1": roi_box[0], "y1": roi_box[1], "x2": roi_box[2], "y2": roi_box[3]}
        return out    


def build_run_folder(base: str, date: str, time_: str) -> str:
    d = safe_digits(date, "date")
    t = safe_digits(time_, "time")
    return os.path.join(base, d, f"run_{t}")


def append_result(folder: str, payload: Dict[str, Any]) -> Optional[str]:
    try:
        p = os.path.join(folder, "result.txt")
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return None
    except Exception as e:
        return str(e)


def classify_folder(
    det: SsdDetector,
    folder: str,
    min_conf: float,
    save: bool,
    enable_spatial: bool,
    reject_yc: float,
    use_roi: bool,
    roi: Optional[Tuple[int, int, int, int]],
    target_class: Optional[int],
) -> Dict[str, Any]:
    cams = [os.path.join(folder, f"cam_{i}_crop.jpg") for i in range(1, 5)]
    frames: List[Dict[str, Any]] = []

    best_conf = 0.0
    best_frame = None
    best_bbox = None
    best_class = None
    
    def spatial_ok(bbox: Dict[str, float]) -> bool:
        y_center = (bbox["ymin"] + bbox["ymax"]) / 2.0
        return y_center <= reject_yc
    
    for idx, p in enumerate(cams, start=1):
        r = det.best_detection_any(p, min_conf=min_conf, use_roi=use_roi, roi=roi, target_class=target_class)
        frame: Dict[str, Any] = {"frame": idx, "path": p, "ok": r.get("ok", False)}

        if not r.get("ok"):
            frame["error"] = r.get("error")
            frames.append(frame)
            continue

        if "roi" in r:
            frame["roi"] = r["roi"]

        best = r.get("best")

        if enable_spatial:
            kept = r.get("kept") or []

            best2 = None
            for cand in kept:
                if spatial_ok(cand["bbox"]):
                    best2 = cand
                    break

            if best2 is None:
                if best is not None:
                    b = best["bbox"]
                    y_center = (b["ymin"] + b["ymax"]) / 2.0
                    frame["rejected"] = {"reason": "y_center_gt_all", "y_center": y_center, "thr": reject_yc}
                best = None
            else:
                if best is not None and best2 is not best:
                    b = best["bbox"]
                    y_center = (b["ymin"] + b["ymax"]) / 2.0
                    frame["rejected"] = {"reason": "y_center_gt", "y_center": y_center, "thr": reject_yc, "fallback": True}
                best = best2

        if best:
            frame["det"] = best
            if best["score"] > best_conf:
                best_conf = best["score"]
                best_frame = idx
                best_bbox = best["bbox"]
                best_class = best["class"]

        frames.append(frame)

    out: Dict[str, Any] = {
        "ok": True,
        "ts": int(time.time()),
        "ts_utc": utc_iso(),
        "model": det.model_path,
        "min_conf": float(min_conf),
        "folder": folder,
        "use_roi": bool(use_roi and roi is not None),
        "target_class": int(target_class) if target_class is not None else None,
        "roi": {"x1": roi[0], "y1": roi[1], "x2": roi[2], "y2": roi[3]} if (use_roi and roi is not None) else None,
        "frames": frames,
        "has_bird": best_frame is not None,  # prefilter "hit"
        "best_conf": float(best_conf),
        "best_frame": best_frame,
        "best_bbox": best_bbox,
        "best_class": best_class,
        "note": "Prefilter: has_bird=true if ANY detection >= min_conf after optional ROI + spatial filter.",
    }

    if save:
        err = append_result(folder, out)
        if err:
            out["result_write_error"] = err

    return out


def create_app() -> FastAPI:
    base = os.environ.get("BIRDCAM_BASE", "/nas/birdcam")
    model = os.environ.get("BIRDCAM_MODEL", "/app/models/model.tflite")
    threads = int(os.environ.get("BIRDCAM_THREADS", "4"))
    default_min_conf = float(os.environ.get("BIRDCAM_MIN_CONF", "0.35"))

    # spatial filter (legacy, still useful)
    enable_spatial = os.environ.get("BIRDCAM_ENABLE_SPATIAL_FILTER", "1") == "1"
    reject_yc = float(os.environ.get("BIRDCAM_REJECT_YCENTER_GT", "0.90"))

    # ROI crop
    use_roi = os.environ.get("BIRDCAM_USE_ROI", "0") == "1"
    orientation = os.environ.get("BIRDCAM_ROI_ORIENTATION", os.environ.get("BIRDCAM_ORIENTATION", "auto")).strip().lower()  # auto|landscape|portrait
    roi_landscape_s = os.environ.get("BIRDCAM_ROI_LANDSCAPE", "80,0,520,260")
    roi_portrait_s = os.environ.get("BIRDCAM_ROI_PORTRAIT", "0,0,0,0")  # disabled by default

    roi_landscape = None
    roi_portrait = None
    try:
        roi_landscape = parse_roi(roi_landscape_s)
    except Exception:
        roi_landscape = None
    try:
        # allow portrait to be disabled with 0s
        if roi_portrait_s and roi_portrait_s != "0,0,0,0":
            roi_portrait = parse_roi(roi_portrait_s)
    except Exception:
        roi_portrait = None

    # target class filter (COCO bird = 15)
    default_target_class = int(os.environ.get("BIRDCAM_TARGET_CLASS", "15"))

    # exclusion filters (skip common false positives like the feeder)
    exclude_boxes_norm = parse_exclude_boxes_norm(os.environ.get("BIRDCAM_EXCLUDE_BOXES_NORM", ""))
    exclude_iou = float(os.environ.get("BIRDCAM_EXCLUDE_IOU", "0.25"))
    exclude_aspect_min = float(os.environ.get("BIRDCAM_EXCLUDE_ASPECT_MIN", "0"))
    exclude_area_min = float(os.environ.get("BIRDCAM_EXCLUDE_AREA_MIN", "0"))
    exclude_area_max = float(os.environ.get("BIRDCAM_EXCLUDE_AREA_MAX", "1"))
    # class0 promotion: treat class 0 as "bird" if score >= threshold
    class0_as_bird_min_conf = float(os.environ.get("BIRDCAM_CLASS0_AS_BIRD_MIN_CONF", "0.50"))
    det = SsdDetector(
        model_path=model,
        num_threads=threads,
        exclude_boxes_norm=exclude_boxes_norm,
        exclude_iou=exclude_iou,
        exclude_aspect_min=exclude_aspect_min,
        exclude_area_min=exclude_area_min,
        exclude_area_max=exclude_area_max,
        class0_as_bird_min_conf=class0_as_bird_min_conf,
    )
    app = FastAPI(title="birdcam_local_ai", version="1.0.3")

    def pick_roi_for_image(h: int, w: int, use_roi_req: bool) -> Optional[Tuple[int, int, int, int]]:
        if not use_roi_req:
            return None
        o = orientation
        if o == "auto":
            o = "landscape" if w >= h else "portrait"
        if o == "portrait":
            return roi_portrait
        return roi_landscape

    @app.get("/health")
    def health():
        return {
            "ok": True,
            "ts_utc": utc_iso(),
            "base": base,
            "model": os.path.basename(model),
            "threads": threads,
            "default_min_conf": default_min_conf,
            "target_class": int(default_target_class),
            "spatial_filter": enable_spatial,
            "reject_ycenter_gt": reject_yc,
            "use_roi": use_roi,
            "orientation": orientation,
            "roi_landscape": roi_landscape_s,
            "roi_portrait": roi_portrait_s,
            "exclude": {
                "boxes_norm": exclude_boxes_norm,
                "iou": exclude_iou,
                "aspect_min": exclude_aspect_min,
                "area_min": exclude_area_min,
                "area_max": exclude_area_max,
            "class0_as_bird_min_conf": class0_as_bird_min_conf,
            },
        }

    @app.get("/classify")
    def classify(
        date: str = Query(..., description="YYYYMMDD or YYMMDD, digits only"),
        time_: str = Query(..., alias="time", description="HHMMSS, digits only"),
        min_conf: Optional[float] = Query(None, ge=0.0, le=1.0),
        save: bool = Query(True, description="append result.txt into the run folder"),
        use_roi_q: Optional[bool] = Query(None, alias="use_roi", description="Override ROI for this request"),
    ):
        try:
            folder = build_run_folder(base, date, time_)
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))

        if not os.path.isdir(folder):
            raise HTTPException(status_code=404, detail=f"folder_not_found: {folder}")

        use_roi_req = use_roi if use_roi_q is None else bool(use_roi_q)

        # Determine ROI based on first readable image (so landscape/portrait can be auto)
        roi = None
        if use_roi_req:
            for i in range(1, 5):
                p = os.path.join(folder, f"cam_{i}_crop.jpg")
                img = cv2.imread(p, cv2.IMREAD_COLOR)
                if img is not None:
                    h, w = img.shape[:2]
                    roi = pick_roi_for_image(h, w, use_roi_req)
                    break

        mc = default_min_conf if min_conf is None else float(min_conf)
        return classify_folder(det, folder, mc, save, enable_spatial, reject_yc, use_roi_req, roi, default_target_class)

    @app.get("/debug_one")
    def debug_one(
        date: str = Query(..., description="YYYYMMDD or YYMMDD, digits only"),
        time_: str = Query(..., alias="time", description="HHMMSS, digits only"),
        frame: int = Query(1, ge=1, le=4),
        min_conf: Optional[float] = Query(None, ge=0.0, le=1.0),
        use_roi_q: Optional[bool] = Query(None, alias='use_roi', description='Override ROI for this request'),
    ):
        try:
            folder = build_run_folder(base, date, time_)
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))

        p = os.path.join(folder, f"cam_{frame}_crop.jpg")
        if not os.path.exists(p):
            raise HTTPException(status_code=404, detail=f"missing: {p}")

        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=500, detail=f"cannot_read: {p}")

        h, w = img.shape[:2]
        use_roi_req = use_roi if use_roi_q is None else bool(use_roi_q)
        roi = pick_roi_for_image(h, w, use_roi_req)

        mc = default_min_conf if min_conf is None else float(min_conf)
        r = det.best_detection_any(p, min_conf=mc, use_roi=use_roi_req, roi=roi, target_class=default_target_class)

        # Also dump raw top10 on the (possibly cropped) image for diagnosis
        crop = img
        roi_box = None
        if use_roi_req and roi is not None:
            x1, y1, x2, y2 = roi
            x1 = int(clamp(x1, 0, w - 1)); x2 = int(clamp(x2, 1, w))
            y1 = int(clamp(y1, 0, h - 1)); y2 = int(clamp(y2, 1, h))
            if x2 > x1 and y2 > y1:
                crop = img[y1:y2, x1:x2].copy()
                roi_box = (x1, y1, x2, y2)

        boxes, classes, scores = det.run_ssd(crop)
        idxs = np.argsort(scores)[::-1][:10]
        top10 = []
        for i in idxs:
            b = boxes[i]
            ymin, xmin, ymax, xmax = [float(x) for x in b]
            if roi_box is not None:
                x1, y1, x2, y2 = roi_box
                roi_w = float(x2 - x1); roi_h = float(y2 - y1)
                ymin = (y1 + ymin * roi_h) / float(h)
                ymax = (y1 + ymax * roi_h) / float(h)
                xmin = (x1 + xmin * roi_w) / float(w)
                xmax = (x1 + xmax * roi_w) / float(w)
            top10.append({
                "i": int(i),
                "score": float(scores[i]),
                "class": int(classes[i]),
                "bbox": {"ymin": ymin, "xmin": xmin, "ymax": ymax, "xmax": xmax},
            })

        od = []
        for o in det.output_details:
            od.append(
                {
                    "name": o.get("name"),
                    "index": int(o.get("index")),
                    "shape": [int(x) for x in o.get("shape", [])],
                    "dtype": str(o.get("dtype")),
                    "quantization": list(o.get("quantization", [])),
                    "quantization_parameters": o.get("quantization_parameters", {}),
                }
            )

        return {
            "ok": True,
            "ts_utc": utc_iso(),
            "path": p,
            "image_shape": [int(h), int(w), 3],
            "use_roi": bool(use_roi_req and roi is not None),
            "roi": {"x1": roi[0], "y1": roi[1], "x2": roi[2], "y2": roi[3]} if (use_roi_req and roi is not None) else None,
            "output_details": od,
            "best": r.get("best"),
            "scores_minmax": [float(np.min(scores)) if scores.size else 0.0, float(np.max(scores)) if scores.size else 0.0],
            "classes_minmax": [int(np.min(classes)) if classes.size else 0, int(np.max(classes)) if classes.size else 0],
            "top10": top10,
            "note": "top10 and best bboxes are mapped to FULL image normalized coords.",
        }

    @app.get("/debug_draw")
    def debug_draw(
        date: str = Query(..., description="YYYYMMDD or YYMMDD, digits only"),
        time_: str = Query(..., alias="time", description="HHMMSS, digits only"),
        frame: int = Query(1, ge=1, le=4),
        min_conf: Optional[float] = Query(None, ge=0.0, le=1.0),
        thickness: int = Query(3, ge=1, le=10),
        save_name: Optional[str] = Query(None, description="Optional output filename, default debug_cam_{frame}.jpg"),
        save_dir: Optional[str] = Query(None, description="Optional subdir under the DAY folder (e.g. debug_frames)"),
        use_roi_q: Optional[bool] = Query(None, alias='use_roi', description='Override ROI for this request'),
        draw_excludes: bool = Query(True, description="Draw exclusion zones on debug image"),
        draw_top10: bool = Query(False, description="Draw top10 raw detections in purple"),
    ):
        try:
            folder = build_run_folder(base, date, time_)
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))

        img_path = os.path.join(folder, f"cam_{frame}_crop.jpg")
        if not os.path.exists(img_path):
            raise HTTPException(status_code=404, detail=f"missing: {img_path}")

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=500, detail=f"cannot_read: {img_path}")

        h, w = img.shape[:2]

        use_roi_req = use_roi if use_roi_q is None else bool(use_roi_q)
        roi = pick_roi_for_image(h, w, use_roi_req)
        mc = default_min_conf if min_conf is None else float(min_conf)

        # Existing filtered best (kept for reference/comparison)
        r = det.best_detection_any(
            img_path,
            min_conf=mc,
            use_roi=use_roi_req,
            roi=roi,
            target_class=default_target_class,
        )
        best_filtered = r.get("best")

        # --- RAW diagnostics (independent of min_conf/spatial/excludes) ---
        crop = img
        roi_box: Optional[Tuple[int, int, int, int]] = None
        if use_roi_req and roi is not None:
            x1, y1, x2, y2 = roi
            x1 = int(clamp(x1, 0, w - 1))
            x2 = int(clamp(x2, 1, w))
            y1 = int(clamp(y1, 0, h - 1))
            y2 = int(clamp(y2, 1, h))
            if x2 > x1 and y2 > y1:
                crop = img[y1:y2, x1:x2].copy()
                roi_box = (x1, y1, x2, y2)

        boxes, classes, scores = det.run_ssd(crop)

        def map_bbox_to_full(b: np.ndarray) -> Dict[str, float]:
            ymin, xmin, ymax, xmax = [float(x) for x in b]
            if roi_box is not None:
                x1, y1, x2, y2 = roi_box
                roi_w = float(x2 - x1)
                roi_h = float(y2 - y1)
                ymin = (y1 + ymin * roi_h) / float(h)
                ymax = (y1 + ymax * roi_h) / float(h)
                xmin = (x1 + xmin * roi_w) / float(w)
                xmax = (x1 + xmax * roi_w) / float(w)
            return {"ymin": ymin, "xmin": xmin, "ymax": ymax, "xmax": xmax}

        raw_top: Optional[Dict[str, Any]] = None
        raw_bird: Optional[Dict[str, Any]] = None
        n = min(len(scores), len(classes), len(boxes))
        if n > 0:
            i_top = int(np.argmax(scores[:n]))
            raw_top = {
                "i": i_top,
                "score": float(scores[i_top]),
                "class": int(classes[i_top]),
                "bbox": map_bbox_to_full(boxes[i_top]),
            }

            bird_idxs = np.where(classes[:n] == int(default_target_class))[0]
            if bird_idxs.size > 0:
                bi = int(bird_idxs[np.argmax(scores[bird_idxs])])
                raw_bird = {
                    "i": bi,
                    "score": float(scores[bi]),
                    "class": int(classes[bi]),
                    "bbox": map_bbox_to_full(boxes[bi]),
                }

        out_img = img.copy()

        # Draw ROI rectangle for reference (cyan)
        if roi is not None and use_roi_req:
            x1, y1, x2, y2 = roi
            x1 = int(clamp(x1, 0, w - 1)); x2 = int(clamp(x2, 1, w))
            y1 = int(clamp(y1, 0, h - 1)); y2 = int(clamp(y2, 1, h))
            cv2.rectangle(out_img, (x1, y1), (x2, y2), (255, 255, 0), 2)

        # Draw exclusion zones (red)
        if draw_excludes and getattr(det, "exclude_boxes_norm", None):
            for xb in det.exclude_boxes_norm:
                ex1 = int(clamp(float(xb["xmin"]) * w, 0, w - 1))
                ey1 = int(clamp(float(xb["ymin"]) * h, 0, h - 1))
                ex2 = int(clamp(float(xb["xmax"]) * w, 0, w - 1))
                ey2 = int(clamp(float(xb["ymax"]) * h, 0, h - 1))
                cv2.rectangle(out_img, (ex1, ey1), (ex2, ey2), (0, 0, 255), 2)
                cv2.putText(
                    out_img,
                    "EXCLUDE",
                    (ex1 + 3, max(15, ey1 + 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

        # Existing option: draw raw top10 detections (purple) on FULL image
        # (kept as-is; if ROI is used this reflects full-image inference, not ROI inference)
        if draw_top10:
            boxes10, classes10, scores10 = det.run_ssd(img)
            idxs = np.argsort(scores10)[::-1][:10]
            for i in idxs:
                ymin, xmin, ymax, xmax = [float(x) for x in boxes10[i]]
                x1 = int(clamp(xmin * w, 0, w - 1))
                y1 = int(clamp(ymin * h, 0, h - 1))
                x2 = int(clamp(xmax * w, 0, w - 1))
                y2 = int(clamp(ymax * h, 0, h - 1))
                cv2.rectangle(out_img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                label = f"{int(classes10[i])}:{float(scores10[i]):.3f}"
                cv2.putText(
                    out_img,
                    label,
                    (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 255),
                    2,
                )

        def draw_box(bbox: Dict[str, float], color_bgr: Tuple[int, int, int], label: str):
            x1 = int(clamp(float(bbox["xmin"]) * w, 0, w - 1))
            y1 = int(clamp(float(bbox["ymin"]) * h, 0, h - 1))
            x2 = int(clamp(float(bbox["xmax"]) * w, 0, w - 1))
            y2 = int(clamp(float(bbox["ymax"]) * h, 0, h - 1))
            cv2.rectangle(out_img, (x1, y1), (x2, y2), color_bgr, thickness)
            cv2.putText(out_img, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_bgr, 2)

        # Draw RAW TOP1 (green) and RAW BIRD15 TOP1 (yellow)
        if raw_top and raw_top.get("bbox"):
            draw_box(raw_top["bbox"], (0, 255, 0), f"TOP: c={raw_top['class']} {raw_top['score']:.3f}")
        if raw_bird and raw_bird.get("bbox"):
            draw_box(raw_bird["bbox"], (0, 255, 255), f"BIRD15: {raw_bird['score']:.3f}")

        # Draw filtered best (orange) for comparison
        if best_filtered:
            b = best_filtered["bbox"]
            x1 = int(clamp(b["xmin"] * w, 0, w - 1))
            y1 = int(clamp(b["ymin"] * h, 0, h - 1))
            x2 = int(clamp(b["xmax"] * w, 0, w - 1))
            y2 = int(clamp(b["ymax"] * h, 0, h - 1))

            cv2.rectangle(out_img, (x1, y1), (x2, y2), (0, 165, 255), max(1, thickness - 1))
            label = f"BEST(filtered): c={best_filtered['class']} {best_filtered['score']:.3f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            y_text = max(0, y1 - 8)
            cv2.rectangle(out_img, (x1, max(0, y_text - th - 6)), (x1 + tw + 6, y_text), (0, 165, 255), -1)
            cv2.putText(out_img, label, (x1 + 3, y_text - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        out_name = save_name or f"debug_cam_{frame}.jpg"
        out_name = safe_simple_name(Path(out_name).name, "save_name")

        # Default: save next to run folder; if save_dir is given: save under DAY/save_dir/
        if save_dir:
            sd = safe_simple_name(save_dir, "save_dir")
            day_dir = os.path.join(base, safe_digits(date, "date"))
            out_base = Path(day_dir) / sd
            out_base.mkdir(parents=True, exist_ok=True)
            out_path = str(out_base / out_name)
        else:
            out_path = str(Path(folder) / out_name)

        ok = cv2.imwrite(out_path, out_img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            raise HTTPException(status_code=500, detail=f"failed_to_write: {out_path}")

        return {
            "ok": True,
            "ts_utc": utc_iso(),
            "image_path": img_path,
            "out_path": out_path,
            "min_conf": mc,
            "use_roi": bool(use_roi_req and roi is not None),
            "roi": {"x1": roi[0], "y1": roi[1], "x2": roi[2], "y2": roi[3]} if (use_roi_req and roi is not None) else None,
            "best_filtered": best_filtered,
            "raw_top": raw_top,
            "raw_bird": raw_bird,
            "draw_excludes": bool(draw_excludes),
            "exclude_boxes_norm": det.exclude_boxes_norm if draw_excludes else None,
            "note": "Saved image with ROI(cyan), Excluded(red), RAW TOP(green), RAW BIRD15(yellow), BEST(filtered orange). Bboxes are full-image norm coords.",
        }


    return app


app = create_app()
