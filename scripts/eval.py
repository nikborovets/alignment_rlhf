import os

import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer


def _prepare_text_pair(prompts, responses):
    return [
        f"<|im_start|>user\n{p.strip()}<|im_end|>\n<|im_start|>assistant\n{r.strip()}<|im_end|>"
        for p, r in zip(prompts, responses, strict=False)
    ]


def generate_responses(model, tokenizer, dataset, batch_size=4, max_new_tokens=50):
    dataloader = DataLoader(dataset, batch_size=batch_size)
    prompts_list = []
    responses_list = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(
            dataloader, desc=f"Generating with {os.path.basename(model.config._name_or_path)}"
        ):
            prompts = batch["prompt"]
            prompts_tokenized = tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True
            ).to(model.device)

            generated_outputs = model.generate(
                **prompts_tokenized,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )

            responses = tokenizer.batch_decode(
                generated_outputs[:, prompts_tokenized.input_ids.shape[1] :],
                skip_special_tokens=True,
            )

            prompts_list.extend(prompts)
            responses_list.extend(responses)

    return prompts_list, responses_list


def evaluate_rewards(reward_model, tokenizer, prompts, responses, batch_size=4):
    rewards_list = []

    reward_model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size), desc="Evaluating rewards"):
            batch_prompts = prompts[i : i + batch_size]
            batch_responses = responses[i : i + batch_size]

            texts = _prepare_text_pair(batch_prompts, batch_responses)
            tokenized = tokenizer(
                texts, padding=True, truncation=True, return_tensors="pt", max_length=512
            ).to(reward_model.device)

            logits = reward_model(**tokenized).logits
            rewards = torch.sigmoid(logits.squeeze(-1)).cpu().numpy().tolist()
            rewards_list.extend(rewards)

    return rewards_list


class PromptDataset(Dataset):
    def __init__(self, prompts):
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx]}


def main():
    sft_model_path = "models/sft"
    rlhf_model_path = "models/rlhf"
    rm_model_path = "models/rm"
    output_csv_path = "reports/val_rewards.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizers (use the same for all)
    tokenizer_sft = AutoTokenizer.from_pretrained(sft_model_path)
    if tokenizer_sft.pad_token is None:
        tokenizer_sft.pad_token = tokenizer_sft.eos_token
    tokenizer_sft.padding_side = "left"

    tokenizer_rm = AutoTokenizer.from_pretrained(rm_model_path)

    # Load models
    sft_model = AutoModelForCausalLM.from_pretrained(sft_model_path).to(device)
    rlhf_model = AutoModelForCausalLM.from_pretrained(rlhf_model_path).to(device)
    reward_model = AutoModelForSequenceClassification.from_pretrained(rm_model_path).to(device)

    # Load dataset
    val_dataset_raw = load_dataset("juyoungml/HelpSteer2-binarized", split="validation")
    prompt_dataset = PromptDataset(val_dataset_raw["prompt"])

    # Generate for SFT model
    sft_prompts, sft_responses = generate_responses(sft_model, tokenizer_sft, prompt_dataset)

    # Generate for RLHF model
    rlhf_prompts, rlhf_responses = generate_responses(rlhf_model, tokenizer_sft, prompt_dataset)

    # Evaluate SFT rewards
    sft_rewards = evaluate_rewards(reward_model, tokenizer_rm, sft_prompts, sft_responses)

    # Evaluate RLHF rewards
    rlhf_rewards = evaluate_rewards(reward_model, tokenizer_rm, rlhf_prompts, rlhf_responses)

    # Calculate and print mean rewards
    mean_sft_reward = sum(sft_rewards) / len(sft_rewards)
    mean_rlhf_reward = sum(rlhf_rewards) / len(rlhf_rewards)

    print(f"Mean SFT Reward: {mean_sft_reward:.4f}")
    print(f"Mean RLHF Reward: {mean_rlhf_reward:.4f}")

    # Save results to CSV
    df_sft = pd.DataFrame(
        {"prompt": sft_prompts, "response": sft_responses, "reward": sft_rewards, "model": "sft"}
    )
    df_rlhf = pd.DataFrame(
        {
            "prompt": rlhf_prompts,
            "response": rlhf_responses,
            "reward": rlhf_rewards,
            "model": "rlhf",
        }
    )
    df_combined = pd.concat([df_sft, df_rlhf], ignore_index=True)

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df_combined.to_csv(output_csv_path, index=False)
    print(f"Results saved to {output_csv_path}")


if __name__ == "__main__":
    main()
