from unsloth import FastLanguageModel
import yaml
import torch

class IntentClassification:
    def __init__(self, model_path_config):
        # Load file cấu hình để lấy đường dẫn checkpoint [cite: 255, 263]
        with open(model_path_config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Tải mô hình và tokenizer từ checkpoint đã lưu [cite: 255]
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = config['checkpoint_path'],
            max_seq_length = 2048,
            load_in_4bit = True,
            device_map = "cuda",
        )
        FastLanguageModel.for_inference(self.model)

    def __call__(self, message):
        # Nhận tin nhắn và trả về nhãn dự đoán [cite: 256, 260-262]
        inputs = self.tokenizer([message], return_tensors = "pt").to("cuda")
        outputs = self.model.generate(**inputs, max_new_tokens = 64)
        predicted_label = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return predicted_label