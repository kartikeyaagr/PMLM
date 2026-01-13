import torch
import numpy as np
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from optimum.onnxruntime import ORTModelForSequenceClassification


def benchmark_inference(
    model_path, is_onnx=False, text="This is a test sentence for benchmarking."
):
    print(f"\nTesting model: {model_path} (ONNX: {is_onnx})")

    try:
        if is_onnx:
            model = ORTModelForSequenceClassification.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model.eval()

        inputs = tokenizer(text, return_tensors="pt")

        # Warmup
        for _ in range(5):
            with torch.no_grad():
                model(**inputs)

        # Benchmark
        start_time = time.time()
        n_iters = 50
        for _ in range(n_iters):
            with torch.no_grad():
                model(**inputs)
        end_time = time.time()

        avg_time = (end_time - start_time) / n_iters * 1000
        print(f"✅ Inference success! Avg time: {avg_time:.2f} ms")

    except Exception as e:
        print(f"❌ Failed: {e}")


if __name__ == "__main__":
    # Test Original vs ONNX for Universal
    benchmark_inference("pmlm_model_universal", is_onnx=False)
    benchmark_inference("pmlm_model_universal_onnx", is_onnx=True)

    # Test ONNX for Base
    benchmark_inference("pmlm_model_onnx", is_onnx=True)
