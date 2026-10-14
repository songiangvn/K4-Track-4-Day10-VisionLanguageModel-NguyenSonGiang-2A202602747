"""Bước 3–5 và 7: nạp model 4-bit, encode + mask loss, collate, setup E1/E2/E3, nạp lại adapter."""
import re

import torch

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
PROJECTOR_KEYS = ("visual.merger", "visual.deepstack_merger_list")

# 36 lớp × 7 = 252 module trong LLM; 24 khối × 4 = 96 module trong ViT.
LLM_TARGETS = r".*language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
VIT_TARGETS = r".*visual\.blocks\.\d+\.(attn\.(qkv|proj)|mlp\.linear_fc(1|2))"


def load_base(model_id=MODEL_ID):
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        # ViT + projector + lm_head giữ fp16. Liệt kê cả tên đầy đủ vì cách match tên của
        # transformers 5.x khác bản cũ: chỉ "visual" vẫn để lọt merger thành Linear4bit.
        llm_int8_skip_modules=["visual", "model.visual", "merger", "deepstack_merger_list", "lm_head"],
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, quantization_config=quant, dtype=torch.float16, device_map={"": 0})
    quantized_visual = [n for n, m in model.named_modules()
                        if "visual" in n and type(m).__name__ == "Linear4bit"]
    assert not quantized_visual, f"ViT/projector bị lượng tử hoá 4-bit: {quantized_visual[:8]}"
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"
    return model, processor


def is_projector(name):
    return any(k in name for k in PROJECTOR_KEYS)


def param_report(model):
    """Đếm tham số theo khối (đếm trên số phần tử; tham số 4-bit được nhân 2 vì đóng gói 2 giá trị/byte)."""
    out = {"vit": 0, "projector": 0, "llm": 0, "lm_head": 0, "other": 0}
    for n, p in model.named_parameters():
        k = p.numel() * (2 if p.dtype == torch.uint8 else 1)
        if "lora_" in n:
            continue
        if is_projector(n):
            out["projector"] += k
        elif "visual" in n:
            out["vit"] += k
        elif "lm_head" in n:
            out["lm_head"] += k
        elif "language_model" in n:
            out["llm"] += k
        else:
            out["other"] += k
    return out


def target_regex(setup):
    return f"({LLM_TARGETS}|{VIT_TARGETS})" if setup == "E3" else LLM_TARGETS


def matched_modules(model, setup):
    rx = re.compile(target_regex(setup))
    return [n for n, m in model.named_modules() if rx.fullmatch(n) and hasattr(m, "weight")]


def apply_setup(model, setup, r=16, alpha=32, dropout=0.05):
    from peft import LoraConfig, get_peft_model

    for p in model.parameters():
        p.requires_grad = False
    model.enable_input_require_grads()
    n_match = len(matched_modules(model, setup))
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout,
                     target_modules=target_regex(setup), bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    if setup in ("E2", "E3"):
        for n, m in model.named_modules():
            if n.endswith(PROJECTOR_KEYS):
                m.float()  # fp32: GradScaler không unscale được grad fp16, Adam fp16 dễ tràn
                for p in m.parameters():
                    p.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, {"lora_modules": n_match, "trainable_params": trainable}


def _user_msg(image, prompt):
    return [{"role": "user", "content": [{"type": "image", "image": image},
                                         {"type": "text", "text": prompt}]}]


def encode_example(processor, image, prompt, answer):
    user = _user_msg(image, prompt)
    full = user + [{"role": "assistant", "content": [{"type": "text", "text": answer}]}]
    kw = dict(tokenize=True, return_dict=True, return_tensors="pt")
    p_enc = processor.apply_chat_template(user, add_generation_prompt=True, **kw)
    f_enc = processor.apply_chat_template(full, add_generation_prompt=False, **kw)
    n = p_enc["input_ids"].shape[1]
    assert torch.equal(f_enc["input_ids"][0, :n], p_enc["input_ids"][0]), "prompt không phải tiền tố"
    labels = f_enc["input_ids"].clone()
    labels[:, :n] = -100  # chỉ học phần trả lời của assistant (JSON + <|im_end|>)
    ex = dict(f_enc.items())
    ex["labels"] = labels
    return {k: (v[0] if k not in ("pixel_values", "image_grid_thw") else v) for k, v in ex.items()}


def collate(batch, pad_id):
    out = {}
    seq_keys = [k for k in batch[0] if k not in ("pixel_values", "image_grid_thw")]
    pad_val = {"input_ids": pad_id, "labels": -100}
    for k in seq_keys:
        out[k] = torch.nn.utils.rnn.pad_sequence(
            [b[k] for b in batch], batch_first=True, padding_value=pad_val.get(k, 0))
    out["pixel_values"] = torch.cat([b["pixel_values"] for b in batch], 0)  # nối, KHÔNG pad
    out["image_grid_thw"] = torch.cat([b["image_grid_thw"] for b in batch], 0)
    return out


def projector_state(model):
    return {n: p.detach().float().cpu() for n, p in model.named_parameters()
            if is_projector(n) and "lora_" not in n}


def load_trained(adapter_dir, projector_path=None, model_id=MODEL_ID, trainable=False):
    """Nạp model gốc 4-bit -> adapter LoRA -> projector đã huấn luyện (nếu có)."""
    from peft import PeftModel

    model, processor = load_base(model_id)
    model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=trainable)
    if projector_path:
        sd = torch.load(projector_path, map_location="cpu")
        params = dict(model.named_parameters())
        missing = [k for k in sd if k not in params]
        assert not missing, f"projector key không khớp: {missing[:3]}"
        with torch.no_grad():
            for k, v in sd.items():
                params[k].copy_(v.to(params[k].dtype))
        print(f"đã nạp {len(sd)} tensor projector từ {projector_path}")
    if not trainable:
        model.eval()
    return model, processor
