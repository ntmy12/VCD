# Handoff Context & Requirements for VCD Evaluation on POPE

## 1. Bối cảnh dự án
Nhóm dự án đang thực hiện đánh giá phương pháp giảm thiểu hallucination (Hallucination Mitigation) **VCD (Visual Contrastive Decoding)** cho các Large Vision-Language Models (LVLMs) trên máy chủ **1x H100**.
Các mô hình nền tảng đang sử dụng:
- **LLaVA-1.5-7B** (`llava-1.5-7b-hf`)
- **Qwen2-VL-7B-Instruct** (`Qwen2-VL-7B-Instruct`)

Phương pháp VCD đã được implement và tích hợp thành công vào wrapper của hai model trên. **Nhiệm vụ tiếp theo dành cho Agent mới là: Chạy script đánh giá benchmark POPE-COCO (với cả 3 split: random, popular, adversarial) sử dụng logic VCD đã viết sẵn.**

Toàn bộ dữ liệu (ảnh COCO, file annotation POPE) và model weights đã có sẵn ở local, đường dẫn được config qua file YAML, **KHÔNG CẦN VIẾT CODE TẢI DỮ LIỆU**.

## 2. Trạng thái hiện tại (Đã hoàn thành)
- Đã tạo cấu trúc thư mục chung tại `/Users/nguyenmy/GitHub/VCD/vcd_experiments/`.
- Đã viết xong file config quản lý data paths tại `configs/data_paths.yaml`.
- Core logic của VCD (thêm noise diffusion, contrastive decoding penalty với adaptive plausibility constraint) đã được code tại `vcd_core/`.
- Đã implement wrapper cho model với logic sinh text custom để chạy VCD:
  - `models/llava_wrapper.py` (Class `LLaVAVCDWrapper`)
  - `models/qwen2vl_wrapper.py` (Class `Qwen2VLVCDWrapper`)

## 3. Yêu cầu cho Agent tiếp theo

### 3.1. Implement Pipeline Benchmark cho POPE-COCO
- Viết script thực thi việc đọc dữ liệu và đưa vào các wrapper model đã chuẩn bị sẵn để sinh ra câu trả lời. Mã nguồn nên đặt tại:
  - `benchmarks/pope/run_pope.py`
  - `benchmarks/pope/eval_pope.py`
- **Data splits:** Phải chạy đánh giá trên cả 3 file của POPE:
  - `coco_pope_random.json`
  - `coco_pope_popular.json`
  - `coco_pope_adversarial.json`

### 3.2. Điều kiện Generation & Prompt (ĐIỀU KIỆN BẮT BUỘC)
- **Max New Tokens:** Bắt buộc cấu hình `max_new_tokens = 6` (truyền vào `gen_kwargs` hoặc `model.generate()`, không dùng default 128).
- **Decoding Method:** Bắt buộc chạy **Greedy Decoding** (`do_sample=False`, `temperature=0.0`, không sampling ngẫu nhiên).
- **Prompt Suffix cho QwenVL:** Nếu chạy mô hình **Qwen2-VL / QwenVL**, bắt buộc thêm suffix (thêm vào ngay sau câu hỏi): `"Please answer with yes or no."`
  - *Ví dụ:* `prompt = f"{question} Please answer with yes or no."`
- **Cách sinh câu trả lời:** Không được dùng chênh lệch logit của 2 token "Yes" và "No" ở bước cuối. Bắt buộc phải **gọi hàm `generate()` (với tham số `use_vcd=True` hoặc `False` để so sánh) để model sinh ra chuỗi văn bản hoàn chỉnh** (text generation thật) với `max_new_tokens=6` và Greedy Decoding.
- **Parsing:** Viết hàm parse robust (ví dụ: dùng regex, rule-based) để trích xuất câu trả lời "Yes" hoặc "No" từ chuỗi text do model sinh ra.
- **Metrics cần tính (cho từng split riêng biệt):**
  - Accuracy
  - Precision
  - Recall
  - F1-Score
- **Output:** Output kết quả thành file `metrics.json`, đồng thời log toàn bộ output thô (raw generated text, label gốc) vào `raw_outputs.jsonl` và các hyperparameter (như `max_new_tokens=6`, `do_sample=False`, `noise_step`, `cd_alpha`, `cd_beta`, seed) vào `run_config.json`. Lưu tất cả vào thư mục `results/<model>_pope_<timestamp>/`.

## 4. Nguyên tắc chung
- Code cần viết rõ ràng, module hóa, parse tham số bằng argparse/yaml, đọc đường dẫn từ `configs/data_paths.yaml`. Tuyệt đối không hardcode đường dẫn.
- Cần có log tiến trình (ví dụ dùng thư viện `tqdm`) khi chạy vì dataset khá lớn.
- Bắt buộc phải lưu cấu hình, thông số (`max_new_tokens=6`, greedy decoding, hyperparams VCD), seed để tiện đối chiếu sau này.
