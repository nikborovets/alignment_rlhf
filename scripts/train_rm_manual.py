import os

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from src.kolya_rlhf.datasets import RewardDataset
from src.kolya_rlhf.utils import get_reward_model, get_tokenizer


def train_reward_model_manual():
    sft_model_path = "models/sft"
    output_dir = "models/rm"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.exists(output_dir) and os.listdir(output_dir):
        print(f"Reward model already exists in {output_dir}. Skipping training.")
        return

    tokenizer = get_tokenizer(sft_model_path, padding_side="right")
    model = get_reward_model(sft_model_path, device, num_labels=1)

    train_dataset_raw = load_dataset("juyoungml/HelpSteer2-binarized", split="train")
    eval_dataset_raw = load_dataset("juyoungml/HelpSteer2-binarized", split="validation")

    train_dataset = RewardDataset(train_dataset_raw, tokenizer)
    eval_dataset = RewardDataset(eval_dataset_raw, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=1)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    scaler = GradScaler()

    num_train_epochs = 1
    num_training_steps = num_train_epochs * len(train_loader)

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    best_eval_accuracy = 0.0
    model.train()
    for epoch in range(num_train_epochs):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for batch in progress_bar:
            optimizer.zero_grad()

            with autocast("cuda"):
                rewards_chosen = model(
                    input_ids=batch["input_ids_chosen"].to(device),
                    attention_mask=batch["attention_mask_chosen"].to(device),
                ).logits

                rewards_rejected = model(
                    input_ids=batch["input_ids_rejected"].to(device),
                    attention_mask=batch["attention_mask_rejected"].to(device),
                ).logits

                loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            lr_scheduler.step()

            progress_bar.set_postfix({"loss": loss.item()})

        # --- Evaluation Step ---
        model.eval()
        total_eval_loss = 0
        total_eval_accuracy = 0
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                with autocast("cuda"):
                    rewards_chosen = model(
                        input_ids=batch["input_ids_chosen"].to(device),
                        attention_mask=batch["attention_mask_chosen"].to(device),
                    ).logits

                    rewards_rejected = model(
                        input_ids=batch["input_ids_rejected"].to(device),
                        attention_mask=batch["attention_mask_rejected"].to(device),
                    ).logits

                    loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()
                    total_eval_loss += loss.item()

                    accuracy = (rewards_chosen > rewards_rejected).float().mean()
                    total_eval_accuracy += accuracy.item()

        avg_eval_loss = total_eval_loss / len(eval_loader)
        avg_eval_accuracy = total_eval_accuracy / len(eval_loader)

        print(f"\n--- Epoch {epoch + 1} Evaluation ---")
        print(f"Eval Loss: {avg_eval_loss:.4f}")
        print(f"Eval Accuracy: {avg_eval_accuracy:.4f}")
        print("--------------------------\n")

        if avg_eval_accuracy > best_eval_accuracy:
            best_eval_accuracy = avg_eval_accuracy
            print(f"New best model found with accuracy: {best_eval_accuracy:.4f}. Saving model...")
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

        model.train()  # Back to training mode

    print("Reward model training complete.")


if __name__ == "__main__":
    train_reward_model_manual()
