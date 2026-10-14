"""Bước 2: nhãn YOLO -> JSON grounding (thang 0–1000 hoặc pixel) và prompt."""
import json

PROMPT_NORM1000 = (
    "Locate road damage in this image. "
    "Detect only objects of classes: pothole, crack, manhole. "
    "Output only a JSON list of objects with label and bbox_2d coordinates "
    "normalized on a 0-1000 scale: "
    '[{"label": "...", "bbox_2d": [x1, y1, x2, y2]}]. '
    "If no damage is found, output []."
)

PROMPT_PIXEL = (
    "Locate road damage in this image. "
    "Detect only objects of classes: pothole, crack, manhole. "
    "Output only a JSON list of objects with label and bbox_2d coordinates "
    "in absolute pixels of the {w}x{h} image: "
    '[{"label": "...", "bbox_2d": [x1, y1, x2, y2]}]. '
    "If no damage is found, output []."
)


def get_prompt(coord="norm1000", width=640, height=360):
    if coord == "norm1000":
        return PROMPT_NORM1000
    return PROMPT_PIXEL.replace("{w}", str(width)).replace("{h}", str(height))


def _clamp_round(v, hi):
    return int(round(max(0.0, min(float(hi), v))))


def yolo_to_box(label, yolo, coord="norm1000", width=640, height=360):
    """yolo = [xc, yc, w, h] chuẩn hoá [0,1] -> {"label", "bbox_2d": [x1,y1,x2,y2]} số nguyên."""
    xc, yc, w, h = yolo
    sx, sy = (1000, 1000) if coord == "norm1000" else (width, height)
    x1 = _clamp_round((xc - w / 2) * sx, sx)
    y1 = _clamp_round((yc - h / 2) * sy, sy)
    x2 = _clamp_round((xc + w / 2) * sx, sx)
    y2 = _clamp_round((yc + h / 2) * sy, sy)
    if x2 <= x1:  # box quá mảnh sau khi làm tròn: vẫn phải hợp lệ với parse_output
        x1, x2 = (x1, x1 + 1) if x1 < sx else (x1 - 1, x1)
    if y2 <= y1:
        y1, y2 = (y1, y1 + 1) if y1 < sy else (y1 - 1, y1)
    return {"label": label, "bbox_2d": [x1, y1, x2, y2]}


def objects_to_answer(objects, coord="norm1000", width=640, height=360):
    boxes = [yolo_to_box(o["label"], o["yolo"], coord, width, height) for o in objects]
    boxes.sort(key=lambda b: (b["bbox_2d"][1], b["bbox_2d"][0]))  # trên->dưới, trái->phải
    return json.dumps(boxes, separators=(",", ":"))
