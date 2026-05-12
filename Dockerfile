# Sử dụng base image có sẵn CUDA và PyTorch
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Cài đặt Python và các thư viện hệ thống
RUN apt-get update && apt-get install -y python3-pip python3-dev git

# Thiết lập thư mục làm việc
WORKDIR /app

# Copy mã nguồn vào container
COPY . .

# Cài đặt thư viện Python (giống như trên Colab)
RUN pip3 install --no-cache-dir unsloth fastapi uvicorn pyyaml requests nest_asyncio
RUN pip3 install --no-deps xformers trl peft accelerate bitsandbytes

# Mở cổng 8000 cho FastAPI
EXPOSE 8000

# Lệnh khởi chạy server
CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]