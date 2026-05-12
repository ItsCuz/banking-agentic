import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel  # <--- Thêm dòng này vào đây

class IntentClassification:
    def __init__(self, model_path_config):
        with open(model_path_config, 'r') as f:
            cfg = yaml.safe_load(f)
        
        # Đường dẫn tới thư mục adapter của bạn (checkpoints/final_model)
        adapter_path = cfg['checkpoint_path']
        
        # QUAN TRỌNG: Tên mô hình nền bạn đã dùng để train
        # Bạn có thể tìm thấy tên này trong checkpoints/final_model/adapter_config.json
        base_model_name = "unsloth/llama-3-8b-instruct-bnb-4bit" 
        
        print(f"🔄 Đang nạp mô hình nền: {base_model_name}")
        
        # 1. Cấu hình Quantization cho Tesla P40
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        # 2. Nạp Tokenizer từ thư mục adapter (nó chứa tokenizer_config.json)
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        
        # 3. Nạp Base Model từ HuggingFace (hoặc cache)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 4. "Đắp" Adapter của bạn lên Base Model
        print(f"➕ Đang nạp LoRA Adapter từ: {adapter_path}")
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        
        # Đưa về chế độ inference
        self.model.eval()
        print("✅ Hệ thống đã sẵn sàng!")

    def __call__(self, message):
        # Giữ nguyên logic dự đoán mình đã hướng dẫn ở câu trước
        inputs = self.tokenizer([message], return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=10)
        
        input_len = inputs['input_ids'].shape[1]
        predicted_output = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        return predicted_output.strip()