# 💀 PMLM: Per My Last Model

**The Universal Translator for Passive-Aggression and Brainrot.**

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2CA5E0?style=for-the-badge&logo=docker&logoColor=white)
![ONNX](https://img.shields.io/badge/ONNX-005CED?style=for-the-badge&logo=onnx&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue?style=for-the-badge)
![DistilBERT](https://img.shields.io/badge/DistilBERT-Model-blue?style=for-the-badge)
![Gemini](https://img.shields.io/badge/Google%20Gemini-8E75B2?style=for-the-badge&logo=google%20gemini&logoColor=white)
![Claude](https://img.shields.io/badge/Claude-D97757?style=for-the-badge&logo=anthropic&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

## 🧐 What is this?

Ever stared at an email signing off with _"Kind regards"_ and felt a cold shiver down your spine? Or read a text saying _"Great job /s"_ and wondered if you're being praised or roasted?

**PMLM** is a dual-model sentiment analysis engine designed to decode the subtext of modern communication. It doesn't just tell you if text is "positive" or "negative"—it tells you if someone is secretly hating you.

### The Personas

- **💼 The Manager**: Trained on thousands of corporate emails. Detects the subtle toxicity hidden in "per my last email" and "just following up."
- **🧢 The Zoomer**: Fine-tuned on Twitter/Reddit data. Understands that "💀" means funny, "slay" is good, and punctuation is aggressive.

---

## ⚙️ The Pipeline (Technical Deep Dive)

This project is an end-to-end case study in **NLP Distillation**—taking the intelligence of massive LLMs and distilling it into lightweight, specialized models.

### Phase 1: The Synthetic Data Factory 🏭

Real-world passive-aggressive email datasets don't exist (because HR fires people who write them).

- **Engine**: We orchestrated **Gemini 1.5 Flash** and **Claude 3.5 Sonnet** to act as screenwriters.
- **Prompt Engineering**: Created diverse personas ("The Gaslighter", "The Micromanager", "The Martyr") to generate **~9,000 samples** of nuanced workplace toxicity.
- **Classes**: `PASSIVE_AGGRESSIVE` | `NEUTRAL` | `POSITIVE`.

### Phase 2: Model Training ("The Manager") 🧠

We fine-tuned a BERT-based transformer on this synthetic dataset.

- **Result**: A model highly sensitive to corporate-speak and formal toxicity.
- **Problem**: It failed completely on slang (e.g., "This goes hard" was classified as negative).

### Phase 3: Domain Adaptation ("The Zoomer") 🧢

To fix the lack of "street smarts," we performed **Few-Shot Domain Adaptation**.

- **Source**: The `tweet_eval` (irony) dataset.
- **Technique**: We took the "Manager" model and continued pre-training it on ironic tweets.
- **Outcome**: A "Universal" model that understands both corporate insults and Twitter sarcasm.

### Phase 4: Optimization (ONNX) ⚡️

Productionizing a PyTorch model can be heavy. We converted both models to **ONNX (Open Neural Network Exchange)**.

- **Inference Engine**: Switched to `optimum.onnxruntime`.
- **Performance**: **2.5x Speedup** (Inference time dropped from ~15ms to ~6ms on CPU).
- **Files**: Quantized and optimized graph structures.

### Phase 5: Modern Deployment 📦

- **Packaging**: Fully Dockerized with a multi-stage build.
- **Dependency Management**: Switched from `pip` to **`uv`** for deterministic, lightning-fast builds (`uv.lock`).

---

## 🚀 Quick Start

### Option A: Docker (Recommended)

Run the containerized app in seconds.

```bash
# 1. Build the image
docker build -t pmlm-app .

# 2. Run container (mapped to port 7860)
docker run -p 7860:7860 pmlm-app
```

Access at `http://localhost:7860`.

### Option B: Local Development (via `uv`)

```bash
# 1. Install uv
pip install uv

# 2. Sync dependencies
uv sync

# 3. Run the app
uv run streamlit run app.py
```

---

## 📂 Project Structure

```zsh
├── app.py                 # Main application & Homepage
├── utils.py               # Design system & ONNX inference logic
├── data_generation/       # Scripts for Gemini/Claude data synthesis
├── finetuning/            # Training & Domain Adaptation scripts
├── model_corporate/       # ONNX-exported "Manager" Model
├── model_universal/       # ONNX-exported "Zoomer" Model
├── Dockerfile             # Production-ready Docker build
└── pyproject.toml         # Dependency definitions
```

---

### built by a burned-out dev 💀

```python
print("Please fix.")
```
