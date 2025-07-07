from torch.utils.data import Dataset


class RewardDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=256):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        chosen_text = item["chosen"]
        rejected_text = item["rejected"]

        chosen_tokenized = self.tokenizer(
            chosen_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        rejected_tokenized = self.tokenizer(
            rejected_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids_chosen": chosen_tokenized["input_ids"].squeeze(0),
            "attention_mask_chosen": chosen_tokenized["attention_mask"].squeeze(0),
            "input_ids_rejected": rejected_tokenized["input_ids"].squeeze(0),
            "attention_mask_rejected": rejected_tokenized["attention_mask"].squeeze(0),
        }
