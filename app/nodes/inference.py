import logging
import re
from pathlib import Path
from typing import Dict

try:
    import yaml
except ImportError:
    yaml = None

logger = logging.getLogger(__name__)

HAS_ML = False
try:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    HAS_ML = True
except ImportError:
    logger.warning("ML dependencies are unavailable. IntentClassification will use fallback rules.")


INTENT_LABELS = [
    "transfer_issue",
    "card_not_received",
    "card_lost",
    "account_blocked",
    "refund_request",
    "fraud_alert",
    "general_inquiry",
]


class IntentClassification:
    def __init__(self, model_path):
        self.config_path = Path(model_path)
        self.config: Dict = self._load_config(self.config_path)
        self.checkpoint_path = Path(self.config.get("checkpoint_path", "checkpoints/final_model"))
        self.base_model_name = self.config.get("base_model_name", "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit")
        self.max_new_tokens = int(self.config.get("max_new_tokens", 12))
        self.fallback_mode = True
        self.tokenizer = None
        self.model = None

        if not HAS_ML:
            return

        if self.config.get("require_cuda", False) and not torch.cuda.is_available():
            logger.warning("CUDA is required by config but unavailable. Using fallback rules.")
            return

        try:
            quantization_config = None
            if self.config.get("load_in_4bit", True):
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )

            self.tokenizer = self._load_tokenizer()
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
            logger.warning("Could not load intent checkpoint. Using fallback rules. Error: %s", exc)
            self.fallback_mode = True

    def _load_tokenizer(self):
        try:
            return AutoTokenizer.from_pretrained(self.checkpoint_path, trust_remote_code=True)
        except Exception as exc:
            logger.warning(
                "Could not load tokenizer from adapter checkpoint %s. "
                "Falling back to base model tokenizer. Error: %s",
                self.checkpoint_path,
                exc,
            )
            return AutoTokenizer.from_pretrained(self.base_model_name, trust_remote_code=True)

    def __call__(self, message):
        risk_intent = self._risk_override(message)
        if risk_intent:
            return risk_intent

        if self.fallback_mode:
            return self._predict_fallback(message)

        prompt = self._build_prompt(message)
        try:
            inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            input_len = inputs["input_ids"].shape[1]
            decoded = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
            return self._normalize_label(decoded)
        except Exception as exc:
            logger.warning("Intent inference failed. Using fallback rules. Error: %s", exc)
            return self._predict_fallback(message)

    def _risk_override(self, message: str) -> str:
        msg = message.lower()
        risk_keywords = [
            "otp",
            "one-time password",
            "verification code",
            "pin",
            "password",
            "fraud",
            "scam",
            "phishing",
            "suspicious caller",
            "lừa đảo",
            "mã otp",
            "mã xác thực",
            "mật khẩu",
            "mạo danh",
            "giả danh",
        ]
        if any(keyword in msg for keyword in risk_keywords):
            return "fraud_alert"
        return ""

    def _load_config(self, config_path: Path) -> Dict:
        try:
            with config_path.open("r", encoding="utf-8") as file:
                if yaml is not None:
                    return yaml.safe_load(file) or {}
                return self._read_simple_yaml(file.read())
        except FileNotFoundError:
            logger.warning("Inference config not found at %s. Defaults will be used.", config_path)
            return {}

    def _read_simple_yaml(self, content: str) -> Dict:
        values = {}
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            value = value.strip().strip('"').strip("'")
            if value.lower() in {"true", "false"}:
                values[key.strip()] = value.lower() == "true"
            else:
                values[key.strip()] = value
        return values

    def _build_prompt(self, message: str) -> str:
        labels = ", ".join(INTENT_LABELS)
        return (
            "Classify the banking customer message into exactly one intent label.\n"
            f"Allowed labels: {labels}\n"
            f"Message: {message}\n"
            "Intent:"
        )

    def _normalize_label(self, raw_label: str) -> str:
        text = raw_label.strip().lower()
        for label in INTENT_LABELS:
            if label in text:
                return label

        cleaned = re.sub(r"[^a-z0-9_]+", "_", text).strip("_")
        aliases = {
            "transfer_failure": "transfer_issue",
            "cash_withdrawal": "general_inquiry",
            "blocked_account": "account_blocked",
            "card_arrival": "card_not_received",
        }
        return aliases.get(cleaned, "general_inquiry")

    def _predict_fallback(self, message: str) -> str:
        msg = message.lower()

        if any(k in msg for k in ["otp", "fraud", "scam", "lừa đảo", "mao danh", "mạo danh", "hacker", "mất tiền"]):
            return "fraud_alert"
        if any(k in msg for k in ["lost card", "stolen card", "mất thẻ", "khóa thẻ", "nuốt thẻ"]):
            return "card_lost"
        if any(k in msg for k in ["not received", "not arrived", "chưa nhận thẻ", "không nhận được thẻ", "card delivery"]):
            return "card_not_received"
        if any(k in msg for k in ["blocked", "bị khóa", "khóa tài khoản", "password", "mật khẩu", "login"]):
            return "account_blocked"
        if any(k in msg for k in ["refund", "reversal", "withdrawal failed", "deducted amount", "hoàn tiền", "pos", "trả lại tiền"]):
            return "refund_request"
        if any(k in msg for k in ["transfer", "chuyển khoản", "chuyển tiền", "bị trừ tiền", "transaction"]):
            return "transfer_issue"

        return "general_inquiry"
