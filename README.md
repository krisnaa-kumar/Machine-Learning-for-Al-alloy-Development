# Machine-Learning-for-Al-alloy-Development

# Alloy PDF → Structured Data
*An auditable pipeline that utilises alloy **composition**, **processing parameters**, and **tensile properties** from scientific PDFs using CLIP, OCR, and a **local** multimodal LLM.*

---

## About repo

Given a PDF, the pipeline:

1. **Scans image blocks** per page (PyMuPDF), filtering by **aspect ratio** and **grayscale**.  
2. Ranks blocks with **CLIP (ViT-B/32)** against **reference screenshots** of tensile bar charts.  
3. Applies a small **OCR gate** (Tesseract): require tokens like `UTS`, `MPa`, `elongation`; reject `2θ`, `Intensity`, `mm`, `µm` (XRD/SEM).  
4. **Reads** the best **bar-chart** with a local **vision LLM** (Qwen2.5-VL via Ollama) → strict JSON rows `{condition, UTS_MPa, elongation_pct}`.  
   - If no bar chart is found, **fallback** to the *mechanical-properties table* and read rows there.  
5. Locates **“Table 1”** (composition), reads it with the vision LLM, and **normalises** to wt.% with **Al = 100 − Σ(others)**.  
6. Parses **Methods** text (Text-LLM) to extract **processing parameters** (extrusion ratio/rate, solution/aging temps & times, quench temp).  
7. **Validates** and assembles a typed `AlloyRecord`, then writes **one CSV per PDF**.

Deterministic gates (layout/CLIP/OCR) **reduce LLM calls** and false positives, schema + unit checks make outputs **analysis-ready**.

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

```

---

## Usage
```bash
# In one terminal, start Ollama
ollama serve

# In another terminal, run the extractor
python main.py path/to/paper.pdf out/paper.csv --config config.yaml
```

---
## Debugging tips

The pipeline emits helpful artifacts:

- `debug_table.png` — composition table crop  
- `debug_cropped_block.png` — half-panel sent to the LLM  
- `debug_mech_table.png` — fallback mechanical table crop  

If bar charts aren’t found:

- Slightly lower `similarity_threshold` (e.g., `0.26 → 0.24`).  
- Add more/better **reference screenshots** of true tensile bar charts.  
- Ensure **Tesseract** is installed and on `PATH`.

If compositions don’t sum to 100:

- Confirm “Bal./Balance” columns are present—the normaliser drops them, then sets `Al = 100 − Σ(others)`.

---

## Limitations 

- Only tested on 4 papers (mentioned in report) due to time limitations, needs to modified further to hand more literature. 
- Focused on **tensile bar charts** and standard **mechanical tables**; other plot types aren’t handled yet.  
- OCR gate is token-based; unusual captions may slip through.  
- No global batching/caching; one CSV per paper.

---

## ML Models and Bayesian Optimisation

Beyond the LLM extraction pipeline, this repo also includes the multi-objective Bayesian optimisation code (ParEGO + GP acquisition) for inverse alloy design and the ML surrogate models for UTS/elongation predictions (XGBoost, SVR, linear baselines). Reproduction scripts generate the candidate CSVs and Pareto plots used in my report.






