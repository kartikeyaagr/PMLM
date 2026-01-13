import os
from pathlib import Path
from optimum.exporters.onnx import main_export


def convert_model(model_path, output_path):
    """
    Converts a Hugging Face model to ONNX format.
    """
    print(f"🚀 Starting conversion for: {model_path} -> {output_path}")

    # Ensure output directory exists (main_export handles it, but good to be safe)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    try:
        # Export the model
        main_export(
            model_name_or_path=str(model_path),
            output=str(output_path),
            task="text-classification",
            opset=14,  # Standard opset
            device="cpu",
            do_validation=True,
            no_post_process=False,
        )
        print(f"✅ Successfully converted {model_path}")
    except Exception as e:
        print(f"❌ Failed to convert {model_path}: {e}")


if __name__ == "__main__":
    base_dir = Path(".")

    # Define models to convert
    # Format: (Input Directory, Output Directory)
    models_to_convert = [
        ("pmlm_model", "pmlm_model_onnx"),
        ("pmlm_model_universal", "pmlm_model_universal_onnx"),
    ]

    for input_dir, output_dir in models_to_convert:
        input_path = base_dir / input_dir
        output_path = base_dir / output_dir

        if not input_path.exists():
            print(f"⚠️ Warning: Input path {input_path} does not exist. Skipping.")
            continue

        convert_model(input_path, output_path)
