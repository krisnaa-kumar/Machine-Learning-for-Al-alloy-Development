# Machine-Learning-for-Al-alloy-Development

# Alloy PDF → Structured Data
*A lightweight, auditable pipeline that utilises alloy **composition**, **processing parameters**, and **tensile properties** from scientific PDFs using CLIP, OCR, and a **local** multimodal LLM.*

---

## What this repo does

Given a PDF, the pipeline:

1. **Scans image blocks** per page (PyMuPDF), filtering by **aspect ratio** and **grayscale**.  
2. Ranks blocks with **CLIP (ViT-B/32)** against **reference screenshots** of tensile bar charts.  
3. Applies a small **OCR gate** (Tesseract): require tokens like `UTS`, `MPa`, `elongation`; reject `2θ`, `Intensity`, `mm`, `µm` (XRD/SEM).  
4. **Reads** the best **bar-chart** with a local **vision LLM** (Qwen2.5-VL via Ollama) → strict JSON rows `{condition, UTS_MPa, elongation_pct}`.  
   - If no bar chart is found, **fallback** to the *mechanical-properties table* and read rows there.  
5. Locates **“Table 1”** (composition), reads it with the vision LLM, and **normalises** to wt.% with **Al = 100 − Σ(others)**.  
6. Parses **Methods** text (Text-LLM) to extract **processing parameters** (extrusion ratio/rate, solution/aging temps & times, quench temp).  
7. **Validates** and assembles a typed `AlloyRecord`, then writes **one CSV per PDF**.

Deterministic gates (layout/CLIP/OCR) **slash LLM calls** and false positives; schema + unit checks make outputs **analysis-ready**.

---

## Repository layout
├─ main.py # Orchestrates the end-to-end pipeline
├─ extractors/
│ ├─ image_extractor.py # PyMuPDF block scan, CLIP ref-embs, precise PDF cropping
│ └─ table_extractor.py # Find “Table 1” (composition) & mechanical table regions
├─ vision/
│ └─ llm_queries.py # Qwen2.5-VL (Ollama) prompts for charts/tables
├─ processing/
│ └─ params_parser.py # Methods → processing params (text LLM)
├─ validators/
│ └─ normalizer.py # Pydantic schema, unit/range checks
├─ models/
│ └─ alloy_record.py # Typed record for CSV rows
├─ utils/
│ └─ helpers.py # YAML config loader, composition normaliser, label parsers
└─ config.yaml # 


---

## Requirements

- **Python** ≥ 3.10  
- `torch`, `numpy`, `pandas`, `Pillow`, `tqdm`, `PyMuPDF` (`fitz`), `pyyaml`, `pytesseract`, **OpenAI CLIP** (PyTorch)  
- **Tesseract OCR** binary  
- **Ollama** with the **Qwen2.5-VL** model pulled locally

### Install (macOS / Linux)

```bash
# Python env
python -m venv .venv && source .venv/bin/activate
# PyTorch: pick CPU/CUDA wheels appropriate for your system
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas pillow tqdm PyMuPDF pyyaml pytesseract ftfy regex \
            git+https://github.com/openai/CLIP.git

# Tesseract
# macOS:
brew install tesseract
# Ubuntu:
sudo apt-get update && sudo apt-get install -y tesseract-ocr

# Ollama + model
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5-vl



