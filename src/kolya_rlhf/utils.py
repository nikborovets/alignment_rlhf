import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer


def prepare_text_pair(prompts, responses):
    """Formats prompts and responses into the required chat template."""
    return [
        f"<|im_start|>user\n{p.strip()}<|im_end|>\n<|im_start|>assistant\n{r.strip()}<|im_end|>"
        for p, r in zip(prompts, responses, strict=False)
    ]


def get_tokenizer(path, padding_side="left"):
    """Loads a tokenizer, setting the pad token if it's missing."""
    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side
    return tokenizer


def get_causal_lm(path, device, **kwargs):
    """Loads a causal language model."""
    return AutoModelForCausalLM.from_pretrained(path, **kwargs).to(device)


def get_reward_model(path, device, **kwargs):
    """Loads a sequence classification model for rewards."""
    return AutoModelForSequenceClassification.from_pretrained(path, **kwargs).to(device)


def generate_responses(model, tokenizer, dataset, batch_size=4, generation_config=None):
    """Generates responses for a given dataset of prompts."""
    if generation_config is None:
        generation_config = {"max_new_tokens": 50, "pad_token_id": tokenizer.eos_token_id}

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

            generated_outputs = model.generate(**prompts_tokenized, **generation_config)

            responses = tokenizer.batch_decode(
                generated_outputs[:, prompts_tokenized.input_ids.shape[1] :],
                skip_special_tokens=True,
            )

            prompts_list.extend(prompts)
            responses_list.extend(responses)

    return prompts_list, responses_list


def compute_rewards(reward_model, tokenizer, prompts, responses, batch_size=4):
    """Computes rewards for given prompts and responses, returns a tensor of sigmoid scores."""
    rewards_list = []
    reward_model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size), desc="Evaluating rewards"):
            batch_prompts = prompts[i : i + batch_size]
            batch_responses = responses[i : i + batch_size]

            texts = prepare_text_pair(batch_prompts, batch_responses)
            tokenized = tokenizer(
                texts, padding=True, truncation=True, return_tensors="pt", max_length=512
            ).to(reward_model.device)

            logits = reward_model(**tokenized).logits
            rewards = torch.sigmoid(logits.squeeze(-1))
            rewards_list.append(rewards)

    return torch.cat(rewards_list)


def compute_rewards_with_kl(
    reward_model, policy_model, ref_model, tokenizer, prompts, responses, kl_beta
):
    """
    Computes rewards, KL penalty, and the final KL-shaped rewards.
    Returns tensors for rewards, kl_penalty, and kl_shaped_rewards.
    """
    # 1. Compute base rewards from the reward model
    texts = prepare_text_pair(prompts, responses)
    reward_inputs = tokenizer(
        texts, padding=True, truncation=True, return_tensors="pt", max_length=512
    ).to(reward_model.device)
    with torch.no_grad():
        base_rewards = reward_model(**reward_inputs).logits.squeeze(-1)

    # 2. Compute KL divergence
    prompt_tokens = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(
        policy_model.device
    )
    all_tokens = tokenizer(
        texts, padding=True, truncation=True, return_tensors="pt", max_length=512
    ).to(policy_model.device)

    with torch.no_grad():
        ref_logits = ref_model(**all_tokens).logits
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)

    policy_logits = policy_model(**all_tokens).logits
    policy_log_probs = F.log_softmax(policy_logits, dim=-1)

    # Create a mask for the response part
    prompts_len = prompt_tokens.input_ids.shape[1]
    response_mask = torch.zeros_like(all_tokens.attention_mask)
    response_mask[:, prompts_len:] = 1
    final_response_mask = response_mask * all_tokens.attention_mask

    # ref_log_probs_values = ref_log_probs.gather(
    #     2, all_tokens.input_ids.unsqueeze(-1)
    # ).squeeze(-1)

    # kl_penalty = ((policy_log_probs_values - ref_log_probs_values) * final_response_mask).sum(dim=1)

    # Calculate KL penalty only on the response tokens
    kl_div_per_token = F.kl_div(
        policy_log_probs, ref_log_probs.detach(), reduction="none", log_target=True
    ).sum(dim=-1)
    kl_penalty = (kl_div_per_token * final_response_mask).sum(dim=1)

    # 3. Shape rewards with KL penalty
    kl_shaped_rewards = base_rewards - kl_beta * kl_penalty

    return base_rewards, kl_penalty, kl_shaped_rewards
