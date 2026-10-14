# Checklist tiến độ — Lab Day 10 (Nguyễn Sơn Giang)

Môi trường: Kaggle Notebooks, 1 × Tesla T4 16 GB (mỗi thí nghiệm một kernel, `kaggle/push.py`); chấm điểm và phân tích trên máy local.

## Chuẩn bị
- [x] Fork repo, clone về máy, đặt chế độ public
- [x] `python -m pytest -q` trên repo gốc: 68 passed, 1 skipped
- [x] Khoá Kaggle chỉ nằm trong `~/.kaggle/` (không ghi vào notebook, không commit)

## Chặng 1 — Dữ liệu
- [x] `prepare_data.py --kaggle` → `dataset_version` d10-21e4070301
- [x] Số ảnh/box khớp bảng của đề (train 1,422 · val 276 · test 311)
- [x] Không session nào nằm ở hai split (train 13 · val 2 · test 3 session)
- [x] 6 ảnh GT có vẽ box: `report/figures/gt_samples.jpg`

## Chặng 2 — Nhãn và E0
- [x] `yolo_to_box` / `objects_to_answer` (sắp theo y1, x1), prompt thang 0–1000 — `lab/labels.py`
- [x] Kiểm tra khứ hồi: nhãn GT qua bộ chuyển đổi chấm F1 = 1.000
- [x] E0 zero-shot: valid 0.836, F1 0.089 → `report/evidence/e0.json`

## Chặng 3 — Chat template và mask loss
- [x] 1 ảnh 640×360 = 220 token ảnh (image_grid_thw [1, 22, 40])
- [x] Giải mã `labels != -100` = đúng JSON + `<|im_end|>`
- [x] Collate: pad `input_ids/attention_mask/mm_token_type_ids/labels`, nối `pixel_values/image_grid_thw`

## Chặng 4 — E1, E2, E3
- [x] 252 module LoRA (E1, E2), 348 (E3); projector 109,105,152 tham số ở fp32
- [x] ViT/projector **không** bị lượng tử hoá (có `assert`)
- [x] Lưu riêng `projector.pt`, nạp lại trước khi suy luận (E2, E3)
- [x] E1 valid 0.990 ≥ 0.95; F1(E1) − F1(E0) = 0.179 ≥ 0.10
- [x] `e1/e2/e3.json` có `trainable_params`, `peak_vram_gb`, `train_minutes` > 0

## Chặng 5 — E4
- [x] Chỉ đổi `data_size` (0.1 và 1.0), gốc = E3 → `e4_data_0.1.json`, `e4_data_1.0.json`
- [x] Kết luận có nhìn CI95 và bootstrap ghép cặp

## Chặng 6 — E5 (bonus)
- [x] Hàm reward IoU + 14 test (5 ca bắt buộc + chống hack) — `lab/reward.py`, `lab/tests/`
- [x] GRPO 60 bước bằng `trl.GRPOTrainer`, log từng bước: `report/logs/e5_grpo_log.jsonl`
- [x] Phân tích reward hacking và so với SFT kèm CI95 → `report/evidence/e5.json`, REPORT

## Nộp bài
- [x] `REPORT.md`: họ tên, `dataset_version`, bảng tổng hợp khớp evidence, 10 câu hỏi, không còn `<điền`
- [x] ≥ 6 ảnh minh hoạ đúng/sai: `report/figures/e0_examples.jpg`, `e1_examples.jpg`, `e3_examples.jpg`
- [x] Nhật ký VRAM/thời gian: `report/logs/*_run.json`, `*_train_log.jsonl`
- [x] `scripts/`, `tests/` giữ nguyên bản gốc (`git diff` rỗng)
- [x] Không commit `data/`, `work/`, `runs/`, `.env`, `*.pt`, `*.safetensors`
- [x] ≥ 5 commit theo từng chặng
- [x] `python scripts/check_submission.py`: 0 FAIL, 0 WARN
- [x] Push lên GitHub, repo public
