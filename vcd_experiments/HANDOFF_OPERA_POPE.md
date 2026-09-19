# Handoff Context & Requirements for OPERA Implementation on POPE

## 1. Bối cảnh dự án
Nhóm dự án đang thực hiện đánh giá các phương pháp giảm thiểu hallucination (Hallucination Mitigation) cho các Large Vision-Language Models (LVLMs) trên máy chủ **1x H100**.
Các mô hình nền tảng đang sử dụng:
- **LLaVA-1.5-7B** (`llava-1.5-7b-hf`)
- **Qwen2-VL-7B-Instruct** (`Qwen2-VL-7B-Instruct`)

Hiện tại, phương pháp **VCD (Visual Contrastive Decoding)** đã được implement và tích hợp thành công vào wrapper của hai model trên. Nhiệm vụ tiếp theo dành cho Agent mới là: **Thực hiện chạy benchmark POPE-COCO (với cả 3 split: random, popular, adversarial) và tích hợp phương pháp OPERA (Over-Trust Penalty and Retrospection).**

Toàn bộ dữ liệu (ảnh COCO, file annotation POPE) và model weights đã có sẵn ở local, đường dẫn sẽ được config qua file YAML, **KHÔNG CẦN VIẾT CODE TẢI DỮ LIỆU**.

## 2. Trạng thái hiện tại (Đã hoàn thành)
- Đã tạo cấu trúc thư mục chung tại `/Users/nguyenmy/GitHub/VCD/vcd_experiments/`.
- Đã viết xong file config quản lý data paths tại `configs/data_paths.yaml`.
- Đã implement wrapper cho model với logic sinh text custom (tách khỏi `generate()` mặc định của HuggingFace) tại:
  - `models/llava_wrapper.py`
  - `models/qwen2vl_wrapper.py`
  *(Lưu ý: Các wrapper này hiện đang được hook logic của VCD, Agent mới có thể tham khảo cách hook vào vòng lặp auto-regressive decoding để làm điều tương tự với OPERA).*

## 3. Yêu cầu cho Agent tiếp theo

### 3.1. Implement OPERA (Over-Trust Penalty and Retrospection)
- Cần nghiên cứu code/paper của phương pháp OPERA và viết logic hook vào quá trình giải mã (decoding) của LLaVA-1.5-7B và Qwen2-VL-7B-Instruct.
- **Gợi ý:** Viết module `opera_core/` tương tự như `vcd_core/` hiện tại, chứa các hàm tính toán Over-trust penalty và Retrospection. Sau đó, copy và tùy chỉnh các wrapper trong thư mục `models/` (ví dụ tạo `llava_opera_wrapper.py`) để tích hợp OPERA vào vòng lặp auto-regressive decoding (vòng lặp `for step in range(max_new_tokens):`).
- Các hyperparameter của OPERA cần được đưa vào file config (như `configs/opera_llava.yaml`) và **bắt buộc phải được log lại** mỗi khi chạy inference.

### 3.2. Implement Pipeline Benchmark cho POPE-COCO
- **Data splits:** Phải chạy đánh giá trên cả 3 file của POPE:
  - `coco_pope_random.json`
  - `coco_pope_popular.json`
  - `coco_pope_adversarial.json`
- **Cách sinh câu trả lời:** Không được dùng chênh lệch logit của 2 token "Yes" và "No" ở bước cuối. Bắt buộc phải **để model sinh ra chuỗi văn bản hoàn chỉnh** (text generation thật) dưới sự can thiệp của OPERA.
- **Parsing:** Viết hàm parse robust (ví dụ: dùng regex, rule-based) để trích xuất câu trả lời "Yes" hoặc "No" từ chuỗi text do model sinh ra.
- **Metrics cần tính (cho từng split riêng biệt):**
  - Accuracy
  - Precision
  - Recall
  - F1-Score
- **Output:** Output kết quả thành file `metrics.json`, đồng thời log toàn bộ output thô (raw generated text, label gốc) vào `raw_outputs.jsonl` và các hyperparameter vào `run_config.json`. Lưu vào thư mục `results/<model>_pope_<timestamp>/`.

## 4. Cấu trúc thư mục hiện tại để tham khảo

```text
vcd_experiments/
├── envs/
├── models/
│   ├── llava_wrapper.py        # Mẫu: Load LLaVA-1.5-7B + hook VCD loop
│   └── qwen2vl_wrapper.py      # Mẫu: Load Qwen2VL + hook VCD loop
├── vcd_core/                   # Mẫu: Logic của VCD
├── configs/
│   └── data_paths.yaml         # Đường dẫn data/models (DÙNG CHUNG)
├── benchmarks/
│   ├── pope/                   # (CẦN CODE TẠI ĐÂY: run_pope.py, eval_pope.py)
│   ├── beaf/
│   └── chair/
└── README.md
```

## 5. Nguyên tắc chung
- Code cần viết rõ ràng, module hóa, parse tham số bằng argparse/yaml (không hardcode path).
- Cần có log tiến trình (ví dụ tqdm) khi chạy trên POPE vì dataset khá lớn.
- Bắt buộc lưu timestamp, seed, param của model vào config log để tiện đối chiếu và audit.
- Chỉ đọc dữ liệu local dựa vào key/value trong `configs/data_paths.yaml`.
