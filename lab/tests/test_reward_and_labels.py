"""Test cho mã của học viên (chạy: python -m pytest lab/tests -q). Không cần GPU."""
import json

import pytest

from lab.labels import objects_to_answer, yolo_to_box
from lab.reward import compute_reward, gt_from_objects
from scripts.score import _gt_boxes, iou_xyxy, parse_output

GT = [{"label": "pothole", "bbox": (100.0, 200.0, 300.0, 400.0)},
      {"label": "crack", "bbox": (500.0, 500.0, 700.0, 600.0)}]


def _pred(*boxes):
    return json.dumps([{"label": lab, "bbox_2d": list(bb)} for lab, bb in boxes])


# ---------------------------------------------------------------- reward: 5 ca bắt buộc
def test_reward_perfect():
    assert compute_reward(_pred(("pothole", (100, 200, 300, 400)), ("crack", (500, 500, 700, 600))),
                          GT) == pytest.approx(1.0)


def test_reward_broken_json():
    assert compute_reward('[{"label": "pothole", "bbox_2d": [1, 2', GT) == 0.0
    assert compute_reward("Based on the image, there is a pothole.", GT) == 0.0


def test_reward_wrong_label():
    # đúng vị trí nhưng sai lớp: không cặp nào khớp -> chỉ còn 0.2 cho JSON hợp lệ
    assert compute_reward(_pred(("manhole", (100, 200, 300, 400)), ("manhole", (500, 500, 700, 600))),
                          GT) == pytest.approx(0.2)


def test_reward_extra_boxes():
    # 1 GT khớp hoàn hảo + 1 box bịa: 0.2 + 0.8 * 1 / max(1, 2) = 0.6
    gt = GT[:1]
    out = _pred(("pothole", (100, 200, 300, 400)), ("pothole", (800, 800, 900, 900)))
    assert compute_reward(out, gt) == pytest.approx(0.6)


def test_reward_both_empty():
    assert compute_reward("[]", []) == 1.0


# ---------------------------------------------------------------- reward: chống hack
def test_reward_empty_when_gt_nonempty():
    assert compute_reward("[]", GT) == pytest.approx(0.2)


def test_reward_giant_box_gets_no_iou():
    assert compute_reward(_pred(("pothole", (0, 0, 1000, 1000))), GT[:1]) == pytest.approx(0.2)


def test_reward_markdown_fence_ok():
    out = "```json\n" + _pred(("pothole", (100, 200, 300, 400))) + "\n```"
    assert compute_reward(out, GT[:1]) == pytest.approx(1.0)


def test_reward_range():
    for out in ("[]", "xx", _pred(("crack", (500, 500, 690, 600)))):
        assert 0.0 <= compute_reward(out, GT) <= 1.0


# ---------------------------------------------------------------- nhãn YOLO -> 0–1000
def test_yolo_to_box_basic():
    assert yolo_to_box("pothole", [0.5, 0.5, 0.2, 0.1]) == {"label": "pothole",
                                                            "bbox_2d": [400, 450, 600, 550]}


def test_yolo_to_box_clamped_and_valid():
    b = yolo_to_box("crack", [0.999, 0.0, 0.01, 0.0001])["bbox_2d"]
    assert all(0 <= v <= 1000 for v in b) and b[2] > b[0] and b[3] > b[1]


def test_yolo_to_box_pixel():
    assert yolo_to_box("manhole", [0.5, 0.5, 0.25, 0.5], coord="pixel")["bbox_2d"] == [240, 90, 400, 270]


def test_answer_sorted_and_parsable():
    objs = [{"label": "crack", "yolo": [0.8, 0.8, 0.1, 0.1]},
            {"label": "pothole", "yolo": [0.2, 0.3, 0.1, 0.1]},
            {"label": "manhole", "yolo": [0.1, 0.3, 0.1, 0.1]}]
    ans = objects_to_answer(objs)
    parsed = parse_output(ans)
    assert [p["label"] for p in parsed] == ["manhole", "pothole", "crack"]
    assert " " not in ans
    assert objects_to_answer([]) == "[]"


def test_answer_matches_score_gt():
    # Nhãn huấn luyện phải khớp GT mà trọng tài dùng (IoU gần 1 sau khi làm tròn)
    e = {"objects": [{"label": "pothole", "yolo": [0.4123, 0.6781, 0.0532, 0.0411]}]}
    pred = parse_output(objects_to_answer(e["objects"]))
    assert iou_xyxy(pred[0]["bbox"], _gt_boxes(e)[0]["bbox"]) > 0.95
    assert gt_from_objects(e["objects"]) == _gt_boxes(e)
