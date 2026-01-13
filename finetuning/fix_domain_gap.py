import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset, Value
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import shutil

# --- CONFIGURATION ---
BASE_MODEL_PATH = "./pmlm_model"
OUTPUT_DIR = "./pmlm_model_universal"
FEW_SHOT_COUNT = 512


def main():
    print(f">> Loading your corporate model from {BASE_MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL_PATH)

    # 1. Get the "Vaccine" Data
    print(">> Fetching Twitter calibration data...")
    twitter_data = load_dataset("tweet_eval", "irony", split="train")

    # --- THE FIX STARTS HERE ---
    # We force the 'label' column to be a standard Integer (Value),
    # effectively deleting the "ClassLabel" metadata restriction.
    print(">> Casting schema to allow label manipulation...")
    twitter_data = twitter_data.cast_column("label", Value("int64"))
    # --- THE FIX ENDS HERE ---

    # 2. Filter & Map Labels
    def align_labels(example):
        # Now we can safely add 1 without the library screaming
        example["label"] = example["label"] + 1
        return example

    twitter_data = twitter_data.map(align_labels)

    # 3. Create the Few-Shot Dataset
    few_shot_dataset = twitter_data.shuffle(seed=42).select(range(FEW_SHOT_COUNT))

    print(f">> Created a few-shot dataset of {len(few_shot_dataset)} tweets.")

    # 4. Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], truncation=True, padding="max_length", max_length=128
        )

    tokenized_dataset = few_shot_dataset.map(tokenize_function, batched=True)

    # 5. Training Loop
    training_args = TrainingArguments(
        output_dir="./temp_trainer",
        learning_rate=1e-5,
        per_device_train_batch_size=8,
        num_train_epochs=4,
        weight_decay=0.01,
        save_strategy="no",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    print(">> Applying the patch...")
    trainer.train()

    # 6. Save
    print(f">> Saving patched model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    shutil.rmtree("./temp_trainer", ignore_errors=True)
    print(">> Done! Now run ood_test.py pointing to './pmlm_model_universal'")


if __name__ == "__main__":
    main()
