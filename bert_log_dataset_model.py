import torch
from torch.utils.data import Dataset

class LogDataset(Dataset):
  def __init__(self, df, tokenizer, max_length=128):
    self.texts = df["text"].tolist()
    self.labels = df["label_id"].tolist()
    self.tokenizer = tokenizer
    self.max_length = max_length

  def __len__(self):
    return len(self.texts)
  
  def __getitem__(self, idx):
    tokens = self.tokenizer(
      self.texts[idx],
      max_length=self.max_length,
      padding="max_length",
      truncation =True,
      return_tensors="pt",
    )

    return {
      "input_ids": tokens["input_ids"].squeeze(0),
      "attention_mask": tokens["attention_mask"].squeeze(0),
      "label": torch.tensor(self.labels[idx], dtype=torch.long)
    }