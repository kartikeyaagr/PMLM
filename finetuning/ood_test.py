import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- CONFIGURATION ---
MODEL_PATH = "./pmlm_model_universal"
BATCH_SIZE = 16


def main():
    # 1. Load Model
    print(f">> Loading model from {MODEL_PATH}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    except OSError:
        print("❌ Error: Could not find model files.")
        return

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f">> Running on device: {device.upper()}")
    model.to(device)
    model.eval()

    # 2. Load Dataset
    print(">> Downloading TweetEval (Irony)...")
    dataset = load_dataset("tweet_eval", "irony", split="test")

    # 3. Inference Loop
    print(">> Running Binary Inference (Toxic vs. Safe)...")

    true_labels = []  # 0 or 1
    binary_preds = []  # 0 or 1

    for i in tqdm(range(0, len(dataset), BATCH_SIZE)):
        batch = dataset[i : i + BATCH_SIZE]
        texts = batch["text"]

        # Ground Truth is already binary: 0 (Not Ironic), 1 (Ironic)
        true_labels.extend(batch["label"])

        # Tokenize
        inputs = tokenizer(
            texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # Get raw class (0, 1, or 2)
            raw_preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

        # --- THE MAPPING LOGIC ---
        # Raw: 0 (Pos), 1 (Neu), 2 (Passive Aggressive)
        # Map: 0 & 1 -> 0 (Safe)
        #      2     -> 1 (Toxic/Ironic)
        for p in raw_preds:
            if p == 2:
                binary_preds.append(1)  # It matches "Ironic"
            else:
                binary_preds.append(0)  # It matches "Not Ironic"

    # 4. Results
    target_names = ["Not Ironic (Safe)", "Ironic (Toxic)"]

    print("\n" + "=" * 40)
    print("   BINARY REALITY CHECK")
    print("=" * 40)

    print(
        classification_report(
            true_labels, binary_preds, target_names=target_names, zero_division=0
        )
    )

    # 5. Visualize Confusion Matrix
    # This helps you see if your model is overly sensitive (False Positives)
    # or oblivious (False Negatives).
    cm = confusion_matrix(true_labels, binary_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)

    # Plotting code
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap="Blues", ax=ax, values_format="d", colorbar=False)
    plt.title("OOD Confusion Matrix")
    plt.show()


if __name__ == "__main__":
    main()
