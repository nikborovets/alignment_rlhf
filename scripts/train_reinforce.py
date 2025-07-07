import os

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer


class ReinforceTrainer:
    def __init__(self, config, policy_model, ref_policy_model, reward_model, tokenizer, optimizer):
        self.config = config
        self.tokenizer = tokenizer
        self.policy_model = policy_model
        self.ref_policy_model = ref_policy_model
        self.reward_model = reward_model
        self.optimizer = optimizer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.moving_avg_baseline = torch.tensor(0.0, device=self.device)

    def _prepare_text_pair(self, prompts, responses):
        return [
            f"<|im_start|>user\n{p.strip()}<|im_end|>\n<|im_start|>assistant\n{r.strip()}<|im_end|>"
            for p, r in zip(prompts, responses, strict=False)
        ]

    def compute_rewards(self, prompts, responses):
        texts = self._prepare_text_pair(prompts, responses)
        tokenized = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            logits = self.reward_model(**tokenized).logits
            return torch.sigmoid(logits.squeeze(-1))

    def train(self, dataset):
        dataloader = DataLoader(dataset, batch_size=self.config["batch_size"])

        for epoch in range(self.config["epochs"]):
            for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}")):
                self.optimizer.zero_grad()

                prompts_text = batch["prompt"]
                prompts_tokenized = self.tokenizer(
                    prompts_text, return_tensors="pt", padding=True, truncation=True
                ).to(self.device)

                generated_outputs = self.policy_model.generate(
                    **prompts_tokenized,
                    max_new_tokens=self.config["max_new_tokens"],
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

                responses_text = self.tokenizer.batch_decode(
                    generated_outputs[:, prompts_tokenized.input_ids.shape[1] :],
                    skip_special_tokens=True,
                )

                rewards = self.compute_rewards(prompts_text, responses_text)

                all_texts = self._prepare_text_pair(prompts_text, responses_text)
                all_tokens = self.tokenizer(
                    all_texts, return_tensors="pt", padding=True, truncation=True, max_length=512
                ).to(self.device)

                with torch.no_grad():
                    ref_logits = self.ref_policy_model(**all_tokens).logits
                    ref_log_probs = F.log_softmax(ref_logits, dim=-1)

                policy_logits = self.policy_model(**all_tokens).logits
                policy_log_probs = F.log_softmax(policy_logits, dim=-1)

                input_ids = all_tokens.input_ids
                attention_mask = all_tokens.attention_mask

                # More robust log_probs calculation for loss
                shifted_ids = input_ids[:, 1:]
                shifted_log_probs = policy_log_probs[:, :-1]
                log_probs_values = shifted_log_probs.gather(2, shifted_ids.unsqueeze(-1)).squeeze(
                    -1
                )
                mask = attention_mask[:, 1:].float()
                response_log_probs_for_loss = (log_probs_values * mask).sum(dim=1)

                # KL penalty calculation
                ref_log_probs_values = ref_log_probs.gather(2, input_ids.unsqueeze(-1)).squeeze(-1)
                policy_log_probs_values = policy_log_probs.gather(
                    2, input_ids.unsqueeze(-1)
                ).squeeze(-1)

                kl_penalty = (policy_log_probs_values - ref_log_probs_values).sum(dim=1)

                advantage = rewards - self.config["kl_beta"] * kl_penalty - self.moving_avg_baseline
                loss = -(response_log_probs_for_loss * advantage.detach()).mean()

                loss.backward()
                self.optimizer.step()

                self.moving_avg_baseline = self.config[
                    "baseline_alpha"
                ] * self.moving_avg_baseline + (1 - self.config["baseline_alpha"]) * (
                    rewards.mean().item()
                )

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
        "lr": 1e-5,  # Back to conservative LR
        "kl_beta": 0.5,  # Increased KL penalty
        "max_new_tokens": 50,
        "baseline_alpha": 0.9,
    }

    tokenizer = AutoTokenizer.from_pretrained(config["sft_model_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    policy_model = AutoModelForCausalLM.from_pretrained(config["sft_model_path"])
    ref_policy_model = AutoModelForCausalLM.from_pretrained(config["sft_model_path"])
    reward_model = AutoModelForSequenceClassification.from_pretrained(config["rm_model_path"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_model.to(device)
    ref_policy_model.to(device).eval()
    reward_model.to(device).eval()

    optimizer = AdamW(policy_model.parameters(), lr=config["lr"])

    val_dataset_raw = load_dataset("juyoungml/HelpSteer2-binarized", split="validation")
    prompts = val_dataset_raw["prompt"]

    class PromptDataset(Dataset):
        def __init__(self, prompts):
            self.prompts = prompts

        def __len__(self):
            return len(self.prompts)

        def __getitem__(self, idx):
            return {"prompt": self.prompts[idx]}

    prompt_dataset = PromptDataset(prompts)

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
