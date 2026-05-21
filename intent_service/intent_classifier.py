import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

HAS_ML = False
try:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    HAS_ML = True
except ImportError:
    logger.warning("ML dependencies unavailable. Intent service will use fallback rules.")


INTENT_LABELS = [
    "transfer_issue",
    "card_not_received",
    "card_lost",
    "account_blocked",
    "refund_request",
    "fraud_alert",
    "general_inquiry",
]


class IntentClassifier:
    def __init__(self):
        self.checkpoint_path = Path(os.getenv("INTENT_CHECKPOINT_PATH", "checkpoints/final_model"))
        self.base_model_name = os.getenv("INTENT_BASE_MODEL", "unsloth/llama-3-8b-bnb-4bit")
        self.require_cuda = os.getenv("INTENT_REQUIRE_CUDA", "false").lower() == "true"
        self.fallback_mode = True
        self.model = None
        self.tokenizer = None

        if not HAS_ML:
            return
        if self.require_cuda and not torch.cuda.is_available():
            logger.warning("CUDA required but unavailable. Using fallback rules.")
            return

        try:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_path)
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
            )
            self.model = PeftModel.from_pretrained(base_model, self.checkpoint_path)
            self.model.eval()
            self.fallback_mode = False
        except Exception as exc:
            logger.warning("Could not load fine-tuned checkpoint. Using fallback rules. Error: %s", exc)

    def predict(self, message: str):
        if self.fallback_mode:
            return self._predict_fallback(message), 0.65, "fallback keyword rules"

        prompt = (
            "Classify the banking customer message into exactly one intent label.\n"
            f"Allowed labels: {', '.join(INTENT_LABELS)}\n"
            f"Message: {message}\n"
            "Intent:"
        )
        try:
            inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=12,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            input_len = inputs["input_ids"].shape[1]
            decoded = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
            return self._normalize_label(decoded), 0.90, "fine-tuned checkpoint"
        except Exception as exc:
            logger.warning("Fine-tuned inference failed. Error: %s", exc)
            return self._predict_fallback(message), 0.65, "fallback keyword rules after inference error"

    def _normalize_label(self, raw_label: str) -> str:
        text = raw_label.strip().lower()
        for label in INTENT_LABELS:
            if label in text:
                return label
        cleaned = re.sub(r"[^a-z0-9_]+", "_", text).strip("_")
        aliases = {
            "transfer_failure": "transfer_issue",
            "blocked_account": "account_blocked",
            "card_arrival": "card_not_received",
        }
        return aliases.get(cleaned, "general_inquiry")

    def _predict_fallback(self, message: str) -> str:
        msg = message.lower()
        if any(k in msg for k in ["otp", "fraud", "scam", "lừa đảo", "mạo danh", "hacker", "mất tiền"]):
            return "fraud_alert"
        if any(k in msg for k in ["lost card", "stolen card", "mất thẻ", "khóa thẻ", "nuốt thẻ"]):
            return "card_lost"
        if any(k in msg for k in ["not received", "not arrived", "chưa nhận thẻ", "không nhận được thẻ", "card delivery"]):
            return "card_not_received"
        if any(k in msg for k in ["blocked", "bị khóa", "khóa tài khoản", "password", "mật khẩu", "login"]):
            return "account_blocked"
        if any(k in msg for k in ["refund", "reversal", "withdrawal failed", "deducted amount", "hoàn tiền", "pos"]):
            return "refund_request"
        if any(k in msg for k in ["transfer", "chuyển khoản", "chuyển tiền", "bị trừ tiền", "transaction"]):
            return "transfer_issue"
        return "general_inquiry"
