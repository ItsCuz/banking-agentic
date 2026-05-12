import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

class IntentClassification:
    def __init__(self, model_path_config: str):
        """
        Khởi tạo mô hình Intent Classification sử dụng Llama-3-8B và LoRA Adapter.
        """
        # 1. Đọc file cấu hình hệ thống
        try:
            with open(model_path_config, 'r') as f:
                cfg = yaml.safe_load(f)
        except FileNotFoundError:
            raise Exception(f"Không tìm thấy file cấu hình tại: {model_path_config}")

        self.adapter_path = cfg['checkpoint_path']
        # Tên mô hình nền (Base Model) đã dùng để tinh chỉnh
        self.base_model_name = "unsloth/llama-3-8b-instruct-bnb-4bit"

        print(f"🔄 [System] Khởi tạo Hybrid Node: Đang nạp mô hình nền {self.base_model_name}...")

        # 2. Cấu hình Quantization (NF4) để tối ưu hóa VRAM
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        # 3. Nạp Tokenizer từ thư mục Adapter
        self.tokenizer = AutoTokenizer.from_pretrained(self.adapter_path)

        # 4. Nạp Base Model với hỗ trợ tăng tốc phần cứng (CUDA)
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

        # 5. Tích hợp LoRA Adapter vào mô hình nền
        print(f"➕ [System] Đang tích hợp LoRA Adapter từ {self.adapter_path}...")
        self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
        
        # Đưa mô hình về chế độ dự đoán (Evaluation)
        self.model.eval()
        print("✅ [System] Intent Classification Node đã sẵn sàng hoạt động tại Local.")

    def __call__(self, message: str) -> str:
        """
        Xử lý tin nhắn đầu vào và trả về nhãn Intent tương ứng.
        """
        # Chuẩn bị input (Sử dụng Template nếu cần thiết)
        inputs = self.tokenizer([message], return_tensors="pt").to("cuda")

        # Thực hiện suy luận không tính đạo hàm để tiết kiệm tài nguyên
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=12,  # Intent thường ngắn nên giới hạn token để tăng tốc
                temperature=0.1,    # Độ ổn định cao, tránh sinh nội dung ngẫu nhiên
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Giải mã và cắt bỏ phần prompt gốc, chỉ lấy phần nhãn vừa sinh ra
        input_len = inputs['input_ids'].shape[1]
        decoded_output = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        
        # Làm sạch kết quả trả về
        predicted_intent = decoded_output.strip().split('\n')[0]
        return predicted_intent