# Banking Agentic System - Hybrid Edge-Cloud LLM Architecture

## Tổng quan dự án
Dự án tập trung vào việc xây dựng một trợ lý ngân hàng thông minh (Banking Agent) ứng dụng Multi-modal LLMs. Hệ thống được thiết kế theo kiến trúc **Hybrid (Lai)** nhằm tối ưu hóa tài nguyên phần cứng và tốc độ phản hồi:
- **Local Node**: Chạy trực tiếp trên máy trạm (Workstation), chịu trách nhiệm điều phối (Orchestration) và phân loại ý định (Intent Classification).
- **Cloud Node**: Tận dụng sức mạnh tính toán của Google Colab để vận hành mô hình Drafting LLM quy mô lớn, phục vụ việc sinh phản hồi chi tiết.

## Kiến trúc hệ thống
Hệ thống bao gồm 3 thành phần chính hoạt động phối hợp:
1. **Intent Classification (Local)**: Sử dụng mô hình **Llama-3-8B** được tinh chỉnh qua kỹ thuật **LoRA** và tối ưu hóa bằng **4-bit Quantization**. Việc chạy local giúp nhận diện yêu cầu khách hàng với độ trễ cực thấp (< 0.5s).
2. **Drafting Node (Cloud)**: Kết nối API tới môi trường Google Colab để xử lý các truy vấn phức tạp, đòi hỏi khả năng suy luận mạnh mẽ của các mô hình LLM chuyên dụng.
3. **Orchestrator**: Thành phần trung tâm điều phối dữ liệu giữa Local và Cloud, đảm bảo tính nhất quán của hội thoại.

## Hướng dẫn cài đặt và khởi chạy

### 1. Chuẩn bị môi trường Local
Yêu cầu máy trạm có hỗ trợ GPU NVIDIA (Kiến trúc Pascal trở lên) để đạt hiệu năng tốt nhất.

```bash
# Khởi tạo môi trường ảo
python -m venv venv
source venv/bin/activate  # Hoặc .\venv\Scripts\activate trên Windows

# Cài đặt các thư viện lõi
pip install -r requirements.txt
```

### 2. Cấu hình Cloud API
Đảm bảo đã chạy file Notebook trên Google Colab để mở cổng API. Sau đó cập nhật URL API vào file cấu hình:
configs/inference.yaml -> cloud_api_url: "https://your-colab-link.ngrok-free.app"

### 3. Khởi chạy Backend

```bash
python -m uvicorn app.main:app --reload
```

