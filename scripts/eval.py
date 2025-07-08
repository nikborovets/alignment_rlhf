import os

import pandas as pd
import torch
from datasets import load_dataset

from src.kolya_rlhf.datasets import PromptDataset
from src.kolya_rlhf.utils import (
    compute_rewards,
    generate_responses,
    get_causal_lm,
    get_reward_model,
    get_tokenizer,
)


def main():
    sft_model_path = "models/sft"
    rlhf_model_path = "models/rlhf"
    rm_model_path = "models/rm"
    output_csv_path = "reports/val_rewards.csv"

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Load tokenizers
    tokenizer_sft = get_tokenizer(sft_model_path)
    tokenizer_rm = get_tokenizer(rm_model_path, padding_side="right")  # RM needs right padding

    # Load models
    sft_model = get_causal_lm(sft_model_path, device)
    rlhf_model = get_causal_lm(rlhf_model_path, device)
    reward_model = get_reward_model(rm_model_path, device)

    # Load dataset
    val_dataset_raw = load_dataset("juyoungml/HelpSteer2-binarized", split="validation")
    prompt_dataset = PromptDataset(val_dataset_raw["prompt"])

    # Generate for SFT model
    sft_prompts, sft_responses = generate_responses(sft_model, tokenizer_sft, prompt_dataset)

    # Generate for RLHF model
    rlhf_prompts, rlhf_responses = generate_responses(rlhf_model, tokenizer_sft, prompt_dataset)

    # Evaluate SFT rewards
    sft_rewards = compute_rewards(reward_model, tokenizer_rm, sft_prompts, sft_responses)

    # Evaluate RLHF rewards
    rlhf_rewards = compute_rewards(reward_model, tokenizer_rm, rlhf_prompts, rlhf_responses)

    # Calculate and print mean rewards
    mean_sft_reward = sft_rewards.mean().item()
    mean_rlhf_reward = rlhf_rewards.mean().item()

    print(f"Mean SFT Reward: {mean_sft_reward:.4f}")
    print(f"Mean RLHF Reward: {mean_rlhf_reward:.4f}")

    # Save results to CSV
    df_sft = pd.DataFrame(
        {
            "prompt": sft_prompts,
            "response": sft_responses,
            "reward": sft_rewards.cpu().numpy(),
            "model": "sft",
        }
    )
    df_rlhf = pd.DataFrame(
        {
            "prompt": rlhf_prompts,
            "response": rlhf_responses,
            "reward": rlhf_rewards.cpu().numpy(),
            "model": "rlhf",
        }
    )
    df_combined = pd.concat([df_sft, df_rlhf], ignore_index=True)

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df_combined.to_csv(output_csv_path, index=False)
    print(f"Results saved to {output_csv_path}")


if __name__ == "__main__":
    main()
