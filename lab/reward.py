"""Bước 9 (E5): reward IoU kiểm chứng được cho GRPO."""
from scripts.score import IOU_THR, iou_xyxy, parse_output


def gt_from_objects(objects):
    """objects YOLO của data/splits -> [{"label", "bbox"}] thang 0–1000 (giống cách score.py chấm)."""
    out = []
    for o in objects:
        xc, yc, w, h = o["yolo"]
        out.append({"label": o["label"], "bbox": (
            (xc - w / 2) * 1000, (yc - h / 2) * 1000, (xc + w / 2) * 1000, (yc + h / 2) * 1000)})
    return out


def compute_reward(completion, gt_boxes):
    """0.0 nếu JSON hỏng; 1.0 nếu cả hai rỗng; còn lại 0.2 + 0.8 * ΣIoU_khớp / max(n_gt, n_pred, 1).

    Ghép tham lam theo IoU giảm dần, chỉ cùng lớp, chỉ nhận IoU >= 0.5 (giống score.py).
    Mẫu số max(n_gt, n_pred) phạt box bịa; ngưỡng 0.5 vô hiệu hoá box khổng lồ.
    """
    pred = parse_output(completion)
    if pred is None:
        return 0.0
    if not gt_boxes and not pred:
        return 1.0
    total = 0.0
    for lab in {g["label"] for g in gt_boxes} | {p["label"] for p in pred}:
        g = [x["bbox"] for x in gt_boxes if x["label"] == lab]
        p = [x["bbox"] for x in pred if x["label"] == lab]
        pairs = sorted(((iou_xyxy(a, b), i, j) for i, a in enumerate(g)
                        for j, b in enumerate(p)), reverse=True)
        used_g, used_p = set(), set()
        for v, i, j in pairs:
            if v < IOU_THR:
                break
            if i in used_g or j in used_p:
                continue
            used_g.add(i)
            used_p.add(j)
            total += v
    return 0.2 + 0.8 * total / max(len(gt_boxes), len(pred), 1)
