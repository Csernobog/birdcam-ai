#!/usr/bin/env python3
import json, sys, time
import numpy as np
import cv2

from tflite_runtime.interpreter import Interpreter

# COCO "bird" class id (a legtöbb COCO SSD mobilenet modellben): 16
COCO_BIRD_ID = 16

def load_interpreter(model_path: str):
    interpreter = Interpreter(model_path=model_path, num_threads=4)
    interpreter.allocate_tensors()
    return interpreter

def get_io_details(interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return input_details, output_details

def preprocess_for_uint8_ssd(img_bgr, in_h, in_w):
    # SSD mobilenet coco tflite tipikusan uint8 inputot vár
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, (in_w, in_h), interpolation=cv2.INTER_AREA)
    return resized.astype(np.uint8)

def run_ssd(interpreter, input_details, output_details, img_bgr):
    in_shape = input_details[0]["shape"]
    in_h, in_w = int(in_shape[1]), int(in_shape[2])

    inp = preprocess_for_uint8_ssd(img_bgr, in_h, in_w)
    inp = np.expand_dims(inp, axis=0)

    interpreter.set_tensor(input_details[0]["index"], inp)
    interpreter.invoke()

    # SSD kimenetek tipikusan:
    # boxes: [1, N, 4], classes: [1, N], scores: [1, N], count: [1]
    # de a sorrend modellfüggő – ezért név/alak alapján próbáljuk megtalálni.
    outs = [interpreter.get_tensor(o["index"]) for o in output_details]

    boxes = classes = scores = count = None

    for o, arr in zip(output_details, outs):
        shape = arr.shape
        name = (o.get("name") or "").lower()

        # Heurisztikák
        if arr.ndim == 3 and shape[-1] == 4:
            boxes = arr
        elif arr.ndim == 2 and shape[-1] > 1:
            # classes vagy scores: mindkettő [1, N]
            # classes általában float32 vagy int, scores float32
            if "score" in name:
                scores = arr
            elif "class" in name:
                classes = arr
            else:
                # fallback: ha float és 0..1 -> scores
                if arr.dtype in (np.float32, np.float16) and np.max(arr) <= 1.0:
                    scores = arr
                else:
                    classes = arr
        elif arr.ndim == 1 and shape[0] == 1:
            count = arr

    # Ha név alapján nem lett meg:
    if boxes is None and len(outs) >= 1: boxes = outs[0]
    if classes is None and len(outs) >= 2: classes = outs[1]
    if scores is None and len(outs) >= 3: scores = outs[2]

    # formázás
    boxes = np.squeeze(boxes, axis=0) if boxes is not None else np.zeros((0,4), dtype=np.float32)
    classes = np.squeeze(classes, axis=0) if classes is not None else np.zeros((0,), dtype=np.float32)
    scores = np.squeeze(scores, axis=0) if scores is not None else np.zeros((0,), dtype=np.float32)

    # néha classes float, néha int
    classes = classes.astype(np.int32, copy=False)

    return boxes, classes, scores

def detect_best_bird(model_path, img_path, min_conf=0.35):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        return {"ok": False, "error": f"cannot_read: {img_path}"}

    interpreter = load_interpreter(model_path)
    input_details, output_details = get_io_details(interpreter)

    boxes, classes, scores = run_ssd(interpreter, input_details, output_details, img)

    best = None
    for i in range(min(len(scores), len(classes), len(boxes))):
        if classes[i] == COCO_BIRD_ID and float(scores[i]) >= min_conf:
            s = float(scores[i])
            if best is None or s > best["score"]:
                # bbox normalized: [ymin, xmin, ymax, xmax]
                ymin, xmin, ymax, xmax = [float(x) for x in boxes[i]]
                best = {
                    "score": s,
                    "bbox": {"ymin": ymin, "xmin": xmin, "ymax": ymax, "xmax": xmax}
                }

    return {"ok": True, "best": best}

def main():
    if len(sys.argv) < 6:
        print("Usage: bird_prefilter_tflite.py <model.tflite> <min_conf> <cam1.jpg> <cam2.jpg> <cam3.jpg> <cam4.jpg>", file=sys.stderr)
        sys.exit(2)

    model_path = sys.argv[1]
    min_conf = float(sys.argv[2])
    imgs = sys.argv[3:7]

    out = {
        "ts": int(time.time()),
        "model": model_path,
        "min_conf": min_conf,
        "frames": [],
        "has_bird": False,
        "best_conf": 0.0,
        "best_frame": None,
        "best_bbox": None,
    }

    for idx, p in enumerate(imgs, start=1):
        r = detect_best_bird(model_path, p, min_conf=min_conf)
        frame = {"frame": idx, "path": p, "ok": r.get("ok", False)}

        if not r.get("ok"):
            frame["error"] = r.get("error")
        else:
            best = r.get("best")
            if best:
                frame["bird"] = best
                if best["score"] > out["best_conf"]:
                    out["best_conf"] = best["score"]
                    out["best_frame"] = idx
                    out["best_bbox"] = best["bbox"]

        out["frames"].append(frame)

    out["has_bird"] = out["best_frame"] is not None
    print(json.dumps(out, ensure_ascii=False))

if __name__ == "__main__":
    main()
