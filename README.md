<img src="assets/header.svg" width="100%" alt="PMLM: Per My Last Model" />

<p align="center">
  <b>Tells you what that one passive-aggressive coworker actually meant, and what your group chat is really saying.</b><br>
  <sub>Two distilled DistilBERT classifiers · ONNX Runtime + Streamlit · Jan 2026</sub>
</p>

### at a glance

```yaml
what:    classifies text as POSITIVE / NEUTRAL / PASSIVE_AGGRESSIVE, then "translates" the subtext
models:  Manager (corporate email) and Zoomer (adds Twitter/Reddit irony); two separate models
data:    ~9,000 synthetic workplace samples (Gemini + Claude), then tweet_eval irony
speed:   ONNX inference ~15 ms -> ~6 ms on CPU (2.5x)
stack:   [PyTorch, Transformers, ONNX Runtime, Streamlit, Docker, uv]
built:   Jan 2026 · solo side project
status:  code and models work locally; the hosted Space is currently down (see below)
```

### personas

Each persona is its own model, picked from the home page. They are not run together.

- **The Manager** is trained on synthetic corporate emails and picks up the toxicity in "per my last email" and "just following up".
- **The Zoomer** is the Manager continued on ironic tweets, so it handles slang ("this goes hard") and sarcasm as well as corporate-speak.

### pipeline

An end-to-end case study in distillation: a large LLM writes the data, a small model learns it.

1. **Synthetic data.** Real passive-aggressive email datasets don't exist, so Gemini 1.5 Flash and Claude 3.5 Sonnet played personas ("The Gaslighter", "The Micromanager", "The Martyr") to write **~9,000 samples** across `PASSIVE_AGGRESSIVE`, `NEUTRAL` and `POSITIVE`.
2. **The Manager.** A BERT-based transformer fine-tuned on that data. Very sensitive to corporate-speak, but it failed on slang: "This goes hard" came out negative.
3. **The Zoomer.** Few-shot domain adaptation: the Manager is trained further on the `tweet_eval` irony dataset, giving a model that handles both corporate insults and Twitter sarcasm.
4. **ONNX.** Both models exported to ONNX and served with `optimum.onnxruntime`: about **2.5x faster** (~15 ms to ~6 ms per inference on CPU), with quantized, optimized graphs.
5. **Deployment.** Multi-stage Docker build; dependencies managed with `uv` and a lockfile.

### run it

Local, with `uv`:

```bash
uv sync
uv run streamlit run app.py
```

Docker (port 7860):

```bash
docker build -t pmlm-app .
docker run -p 7860:7860 pmlm-app
```

The `.onnx` weights are tracked with Git LFS.

### hosted demo

The Hugging Face Space ([kartikeyaagr/PMLM-app](https://huggingface.co/spaces/kartikeyaagr/PMLM-app)) is in `RUNTIME_ERROR` ("launch timed out, workload was not healthy after 30 min"), so it is not a working live demo. Run it locally as above.

### structure

```
app.py               home page and persona picker
pages/               The Manager and The Zoomer pages
utils.py             styling and ONNX model loading
translator.py        trigger-phrase "real meaning" lookup
data_generation/     Gemini / Claude data synthesis scripts
finetuning/          training and domain adaptation
convertor_utils/     ONNX export and verification
model_corporate/     ONNX Manager model
model_universal/     ONNX Zoomer model
Dockerfile           production build
pyproject.toml       dependencies
```

```python
print("Please fix.")
```

<img src="assets/footer.svg" width="100%" alt="" />
