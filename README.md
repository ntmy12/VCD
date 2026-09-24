# VCD for Qwen2-VL: CHAIR Benchmark (16-bit, max_new_tokens=128, seed=2027)

This repository is streamlined specifically to evaluate object hallucination using the **CHAIR** benchmark on **Qwen2-VL-7B-Instruct** with **Visual Contrastive Decoding (VCD)**. It runs in **16-bit precision** (`bfloat16` or `float16`), bounded at `max_new_tokens = 128`, and uses the standard reproducible **seed 2027**, optimized for **NVIDIA RTX 6000** GPUs (RTX 6000 Ada Generation, RTX A6000, Turing Quadro RTX 6000).

---

## 📁 Repository Structure

```text
VCD_CHAIR/
├── requirements.txt                         # Dependency requirements for NVIDIA RTX 6000
├── run_chair.sh                             # Root launcher script
├── README.md                                # Project documentation
├── vcd_experiments/
│   ├── run_chair.sh                         # Execution script (Baseline + VCD with batching)
│   ├── wait_and_run_chair.sh                # VRAM monitor (waits for >=15GB free VRAM)
│   ├── benchmarks/
│   │   └── chair/
│   │       ├── run_chair.py                 # Core caption generation and evaluation pipeline
│   │       ├── eval_chair.py                # Standalone CHAIR evaluation (CHAIRs, CHAIRi, Recall)
│   │       ├── chair.py                     # Independent CHAIR metric implementation
│   │       ├── chair.pkl                    # Pre-computed COCO val2014 evaluator cache
│   │       ├── download_chair_images.py     # Fast downloader for the 500 benchmark images
│   │       └── selected_chair_val2014_seed2027.json # 500 standardized sample images (seed 2027)
│   ├── configs/
│   │   └── data_paths.yaml                  # Paths for COCO images and model checkpoints
│   ├── models/
│   │   └── qwen2vl_wrapper.py               # Qwen2-VL 16-bit wrapper with KV-cache VCD & batching
│   ├── vcd_core/
│   │   ├── vcd_add_noise.py                 # Diffusion noise injection for Qwen2-VL patches
│   │   ├── vcd_decoding.py                  # Contrastive decoding logic & APC cutoff
│   │   └── test_vcd_core.py                 # Unit tests for core VCD operations
│   └── results/                             # Output raw captions and metrics.json
```

---

## 🎲 Seed 2027 Guarantee & Reproducibility

This setup satisfies **seed 2027** across all stages:
1. **Sample Selection**: 500 images are loaded directly from `selected_chair_val2014_seed2027.json`, which was deterministically sampled with seed 2027.
2. **Diffusion Noise**: The visual distortion generator (`g_noise`) uses `torch.Generator().manual_seed(2027)` to inject identical Gaussian noise patterns into image patches across all runs.
3. **Sampling Generator**: If sampling is enabled, `g_samp` is seeded deterministically (`seed + 10_000`).
4. **Environment Seeds**: `torch.manual_seed(2027)`, `np.random.seed(2027)`, and `random.seed(2027)` are set before execution.

---

## 🚀 Environment Setup for NVIDIA RTX 6000

Recommended Python version: **3.10** or **3.11** with **PyTorch CUDA 12.1+**:

```bash
# 1. Create and activate a virtual environment
conda create -n qwen2vl_chair python=3.10 -y
conda activate qwen2vl_chair

# 2. Install PyTorch with CUDA 12.1 for RTX 6000
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Install remaining dependencies from requirements.txt
pip install -r requirements.txt

# 4. Download required NLTK tokenizers and word databases
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); nltk.download('omw-1.4')"
```

---

## ⚙️ Dataset Configuration (`configs/data_paths.yaml`)

### 1. Ground-Truth Annotations
- **NO EXTRA DOWNLOAD NEEDED**: The repository includes `benchmarks/chair/chair.pkl` (2.0 MB).
- This cache stores all COCO val2014 ground-truth object categories, bounding box objects, and synonyms. You do not need to download the large COCO annotation files (`instances_val2014.json` / `captions_val2014.json`).

### 2. Image Files (COCO val2014)
CHAIR uses 500 images from COCO val2014 defined in `selected_chair_val2014_seed2027.json`.

- **If your server already has COCO val2014**: Verify or update the path in `configs/data_paths.yaml`:
  ```yaml
  coco_val2014_images: "/home/nvidia-lab/ai4life/phuongnh/vlm-truth/data/coco2014/val2014"
  ```
- **If you are on a new machine without COCO images**: You do not need to download the full 6GB COCO dataset. Simply run the automated downloader to fetch only the 500 required images (~50MB) in less than a minute:
  ```bash
  cd vcd_experiments
  python benchmarks/chair/download_chair_images.py
  ```
  `run_chair.py` automatically detects the downloaded `coco_chair_images/` directory.

---

## 🏃 Running the Benchmark

### 1. Run Both Baseline & VCD (Recommended)
Default configuration uses **batch_size = 4**, **16-bit** (`bf16`), **max_new_tokens = 128**, **seed = 2027**:
```bash
# Run from repository root:
bash run_chair.sh

# Or run with larger batch size on 48GB RTX 6000:
BATCH_SIZE=8 bash run_chair.sh
```

### 2. Run Individual Modes
```bash
# Run Baseline only (no VCD):
bash run_chair.sh baseline

# Run VCD only:
bash run_chair.sh vcd
```

### 3. Automatic VRAM Monitoring
If sharing the GPU with other workloads, wait until >= 15GB VRAM is free:
```bash
cd vcd_experiments
bash wait_and_run_chair.sh
```

### 4. Direct Python Execution
```bash
cd vcd_experiments
python benchmarks/chair/run_chair.py \
    --model qwen2vl \
    --use_vcd \
    --batch_size 4 \
    --max_new_tokens 128 \
    --dtype bf16 \
    --seed 2027
```

---

## 📊 Evaluation Metrics

Upon completion, `run_chair.py` automatically computes and reports:
- **CHAIRs** (Sentence-level Hallucination Rate): Percentage of sentences containing at least one hallucinated object (lower is better).
- **CHAIRi** (Instance-level Hallucination Rate): Ratio of hallucinated object instances to all mentioned objects (lower is better).
- **Recall**: Percentage of ground-truth objects mentioned in generated captions.
- **Caption Length**: Average number of words per generated caption.

Detailed results are stored in `vcd_experiments/results/<run_id>/`:
- `raw_outputs.jsonl`: Generated captions for all 500 images.
- `run_config.json`: Complete execution hyperparameters.
- `metrics.json`: Final evaluated metrics.
- `chair_details.json`: Breakdown of per-image hallucinated words.
