import os

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.kolya_rlhf.datasets import PromptDataset
from src.kolya_rlhf.utils import (
    compute_rewards_with_kl,
    get_causal_lm,
    get_reward_model,
    get_tokenizer,
    prepare_text_pair,
)


class ReinforceTrainer:
    def __init__(self, config, policy_model, ref_policy_model, reward_model, tokenizer, optimizer):
        self.config = config
        self.tokenizer = tokenizer
        self.policy_model = policy_model
        self.ref_policy_model = ref_policy_model
        self.reward_model = reward_model
        self.optimizer = optimizer

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.moving_avg_baseline = torch.tensor(0.0, device=self.device)

    def train(self, dataset):
        dataloader = DataLoader(dataset, batch_size=self.config["batch_size"])
        generation_config = {
            "max_new_tokens": self.config["max_new_tokens"],
            "do_sample": True,
            "top_k": 50,
            "top_p": 0.95,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        for epoch in range(self.config["epochs"]):
            self.optimizer.zero_grad()
            for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}")):
                prompts_text = batch["prompt"]
                prompts_tokenized = self.tokenizer(
                    prompts_text, return_tensors="pt", padding=True, truncation=True
                ).to(self.device)

                # Generate responses
                generated_outputs = self.policy_model.generate(
                    **prompts_tokenized, **generation_config
                )
                responses_text = self.tokenizer.batch_decode(
                    generated_outputs[:, prompts_tokenized.input_ids.shape[1] :],
                    skip_special_tokens=True,
                )

                # Compute rewards and KL penalty
                rewards, kl_penalty, kl_shaped_rewards = compute_rewards_with_kl(
                    self.reward_model,
                    self.policy_model,
                    self.ref_policy_model,
                    self.tokenizer,
                    prompts_text,
                    responses_text,
                    self.config["kl_beta"],
                )

                # Update baseline
                self.moving_avg_baseline = (
                    self.config["baseline_alpha"] * self.moving_avg_baseline
                    + (1 - self.config["baseline_alpha"]) * kl_shaped_rewards.mean().detach()
                )
                advantage = (kl_shaped_rewards - self.moving_avg_baseline).detach()

                # Calculate loss
                all_texts = prepare_text_pair(prompts_text, responses_text)
                all_tokens = self.tokenizer(
                    all_texts, return_tensors="pt", padding=True, truncation=True, max_length=512
                ).to(self.device)

                policy_logits = self.policy_model(**all_tokens).logits
                policy_log_probs = F.log_softmax(policy_logits, dim=-1)

                prompts_len = prompts_tokenized.input_ids.shape[1]
                response_mask = torch.zeros_like(all_tokens.attention_mask)
                response_mask[:, prompts_len:] = 1
                final_response_mask = response_mask * all_tokens.attention_mask

                policy_log_probs_values = policy_log_probs.gather(
                    2, all_tokens.input_ids.unsqueeze(-1)
                ).squeeze(-1)

                response_log_probs_for_loss = (policy_log_probs_values * final_response_mask).sum(
                    dim=1
                )

                loss = -(response_log_probs_for_loss * advantage).mean()
                loss = loss / self.config["gradient_accumulation_steps"]

                loss.backward()

                if (i + 1) % self.config["gradient_accumulation_steps"] == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if i % 10 == 0:
                    tqdm.write(
                        f"Batch {i}: Loss: {loss.item():.4f}, Mean Reward: {rewards.mean().item():.4f}, KL Penalty: {kl_penalty.mean().item():.4f}"
                    )


def main():
    config = {
        "sft_model_path": "models/sft",
        "rm_model_path": "models/rm",
        "rlhf_model_path": "models/rlhf",
        "batch_size": 2,
        "epochs": 1,
        "lr": 2e-6,
        "kl_beta": 0.1,
        "max_new_tokens": 50,
        "baseline_alpha": 0.9,
        "gradient_accumulation_steps": 8,
    }

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    tokenizer = get_tokenizer(config["sft_model_path"])
    policy_model = get_causal_lm(config["sft_model_path"], device)
    ref_policy_model = get_causal_lm(
        config["sft_model_path"],
        device,
    ).eval()
    reward_model = get_reward_model(config["rm_model_path"], device).eval()

    optimizer = AdamW(policy_model.parameters(), lr=config["lr"])

    val_dataset_raw = load_dataset("juyoungml/HelpSteer2-binarized", split="validation")
    prompt_dataset = PromptDataset(val_dataset_raw["prompt"])

    trainer = ReinforceTrainer(
        config, policy_model, ref_policy_model, reward_model, tokenizer, optimizer
    )
    trainer.train(prompt_dataset)

    print(f"Saving RLHF model to {config['rlhf_model_path']}")
    os.makedirs(config["rlhf_model_path"], exist_ok=True)
    policy_model.save_pretrained(config["rlhf_model_path"])
    tokenizer.save_pretrained(config["rlhf_model_path"])


if __name__ == "__main__":
    main()
