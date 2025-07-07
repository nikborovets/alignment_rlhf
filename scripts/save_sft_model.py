import os

from transformers import AutoModelForCausalLM, AutoTokenizer


def save_sft_model():
    """
    Downloads and saves the SFT model and tokenizer from Hugging Face.
    """
    model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
    output_dir = "models/sft"

    if os.path.exists(output_dir) and os.listdir(output_dir):
        print(f"Model already exists in {output_dir}. Skipping download.")
        return

    print(f"Downloading model {model_name} from Hugging Face...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Saving model and tokenizer to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("SFT model saved successfully.")


if __name__ == "__main__":
    save_sft_model()
