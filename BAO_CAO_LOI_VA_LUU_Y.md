# Báo cáo lỗi và lưu ý kỹ thuật — Lab Day 10

Các lỗi dưới đây đều **gặp thật** khi chạy lab trên Kaggle T4 với transformers 5.19.0, peft 0.21.2, trl 1.9.2, bitsandbytes 0.50.2, torch 2.11.0. Mỗi mục ghi: triệu chứng → nguyên nhân → cách xử lý → cách phát hiện sớm.

## 1. Projector bị lượng tử hoá 4-bit dù đã đặt `llm_int8_skip_modules`

- **Triệu chứng**: smoke test E2 báo số tham số huấn luyện **87.6M** thay vì 33.0M (LoRA) + 109.1M (projector). Lần sửa đầu (`module.float()` + `requires_grad=True`) thì lỗi `RuntimeError: only Tensors of floating point and complex dtype can require gradients` từ `bitsandbytes/nn/modules.py`.
- **Nguyên nhân**: với transformers 5.19, `llm_int8_skip_modules=["visual", "lm_head"]` (như GUIDE) không giữ được toàn bộ `visual.merger` / `visual.deepstack_merger_list` ở fp16 — một phần thành `Linear4bit` với trọng số `Params4bit` (uint8). Tensor nguyên không thể huấn luyện.
- **Xử lý**: liệt kê đủ tên `["visual", "model.visual", "merger", "deepstack_merger_list", "lm_head"]` và thêm `assert` ngay sau khi nạp: không module nào chứa `visual` là `Linear4bit` (`lab/vlm.py::load_base`). Sau đó E2 = 142,135,296 tham số, 24 tensor projector đều fp32.
- **Lưu ý**: lỗi này **im lặng** — nếu chỉ mở `requires_grad` cho phần tham số còn float, E2/E3 vẫn chạy, vẫn PASS `check_submission.py`, nhưng chỉ huấn luyện một phần projector. Luôn đối chiếu số tham số huấn luyện với con số lý thuyết.

## 2. `ValueError: Attempting to unscale FP16 gradients`

- **Nguyên nhân**: `GradScaler` của huấn luyện fp16 hỗn hợp chỉ unscale được gradient fp32; projector để fp16 thì gradient cũng fp16.
- **Xử lý**: projector (và mọi tham số huấn luyện) ở fp32, phép tính dùng `torch.autocast(fp16)`. Trong vòng train có bước kiểm tra một lần: số tham số có grad không phải fp32 = 0.

## 3. `ImportError: Found an incompatible version of torchao` (chỉ E3)

- **Triệu chứng**: E1, E2 chạy được; E3 lỗi ngay khi `get_peft_model`.
- **Nguyên nhân**: Kaggle cài sẵn torchao 0.10; PEFT 0.21.2 thử dispatcher torchao cho các `nn.Linear` thường. Lớp LLM là `Linear4bit` nên được dispatcher bitsandbytes xử lý trước; chỉ LoRA trên ViT (E3, `nn.Linear` fp16) mới rơi tới bước kiểm tra torchao.
- **Xử lý**: `pip uninstall -y torchao` trong kernel (lab không dùng torchao). Mất một lần chạy E3 (~5 phút GPU).

## 4. GRPOTrainer đưa adapter về bf16 — T4 không unscale được

- **Triệu chứng**: `NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BFloat16'` ở bước tối ưu đầu tiên của GRPO.
- **Nguyên nhân**: `trl.GRPOTrainer` (1.9.2) chuyển 504 tensor LoRA sang bf16; T4 không hỗ trợ bf16 tốt và `GradScaler` fp16 không xử lý grad bf16.
- **Xử lý**: sau khi tạo trainer, ép các tham số `requires_grad` về fp32 (`lab/grpo.py`). Ngoài ra `GRPOConfig` của 1.9.2 không còn `warmup_ratio` → lọc kwargs theo `dataclasses.fields(GRPOConfig)` và in các khoá bị bỏ.

## 5. `max_completion_length=128` cắt JSON của ảnh đông box (E5)

- Một box ≈ 22 token, nên ảnh ≥ 6 box bị cắt → JSON hỏng → reward 0 cho cả nhóm → GRPO học cách "khai ít box". Xử lý: chỉ dùng ảnh train ≤ 4 box (1,280/1,422 ảnh) làm prompt GRPO. Dù vậy E5 vẫn giảm số box/ảnh trên test (1.68 → 1.46) — cần theo dõi.

## 6. Tải output Kaggle nhận về trang HTML lỗi 502

- **Triệu chứng**: `score.py` báo `JSONDecodeError: Expecting value: line 1 column 1` với `e5.jsonl` chỉ 332 byte.
- **Nguyên nhân**: `kaggle kernels output` ghi nội dung trang "502 Server Error" vào file thay vì báo lỗi.
- **Xử lý**: tải lại; luôn kiểm tra file dự đoán đủ **311 dòng** và `missing_predictions = 0` trước khi tin vào kết quả.

## 7. Log Kaggle và Unicode trên Windows

- `kaggle kernels output` lỗi `'charmap' codec can't encode` khi ghi log có tiếng Việt → chạy với `PYTHONUTF8=1`; `score.py`/`check_submission.py` cần `PYTHONIOENCODING=utf-8`. Font mặc định của PIL không vẽ được chữ "Đ/Ú" trên ảnh → dùng nhãn ASCII.

## 8. Test của đề mâu thuẫn với `check_submission.py`

- `tests/test_configs_and_docs.py::test_report_template_has_student_markers_and_ten_questions` yêu cầu `REPORT.md` **còn** `<điền` (mẫu trống), trong khi `check_submission.py` yêu cầu **không còn**. Sau khi điền báo cáo, test này luôn FAIL (67 passed, 1 failed, 1 skipped). Không sửa `tests/`; notebook chạy pytest với `--deselect` đúng test này và ghi rõ lý do.

## 9. Lưu ý vận hành khác

- Kaggle free cho tối đa **2 phiên GPU** cùng lúc ("Maximum batch GPU session count of 2 reached") → xếp hàng E2/E3.
- Không tải trọng số về máy (adapter + `projector.pt` ≈ 0.5 GB) khi chỉ cần chấm: `kaggle/fetch.py` chỉ tải `*.jsonl|*.json|*.log`. Adapter/projector để nguyên trong output kernel và gắn trực tiếp làm input cho kernel E5 (`kernel_sources`).
- Thời gian thực tế trên T4 Kaggle: E1 38.75, E2 39.33, E3 51.82 phút train (300 bước) + ~9–10 phút suy luận 311 ảnh — cao hơn ước lượng 20–30 phút của đề.
- `apply_chat_template(..., padding=True)` in cảnh báo "Kwargs passed to processor.__call__ have to be in processor_kwargs"; suy luận theo lô vẫn đúng vì mọi ảnh cùng kích thước và cùng prompt nên độ dài prompt bằng nhau.
