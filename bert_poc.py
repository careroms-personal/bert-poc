import json, torch, time
import pandas as pd

from pathlib import Path

from torch.optim import AdamW
from torch.utils.data import DataLoader

from transformers import get_scheduler
from transformers import BertTokenizer
from transformers import BertForSequenceClassification

from bert_poc_models import Datasource
from bert_log_dataset_model import LogDataset

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#Log is from fluent bit
BASE_LOGS_PATH = "C:\\Users\\ASUS\\Projects\\BERT-ecommerce-mocking\\ecommerce-mock\\logs"

logs_path = Path(BASE_LOGS_PATH)
file_names = [f.name for f in logs_path.iterdir() if f.is_file() and f.suffix == ".jsonl"]

datas = []
# 1. Prepare datasource to programable log
FRAMEWORK_MAP = {
  "/api-cart":     "pino",
  "/api-order":    "go",
  "/api-product":  "go",
  "/api-customer": "serilog_compact",
  "/api-payment":  "serilog",
}

for file in file_names:
  file_dir = Path(logs_path / file)

  with open(file_dir, "r") as f:
    lines = f.readlines()
    
  for line in lines:
    branket_start = line.index("[")
    json_array_str = line[branket_start:]

    timestamp, payload = json.loads(json_array_str)

    datas.append(
      Datasource(
        container_id=payload["container_id"],
        container_name=payload["container_name"],
        framework=FRAMEWORK_MAP.get(payload["container_name"], "unknown"),
        timestamp=int(timestamp),
        source=payload["source"],
        log=json.loads(payload["log"])
      )
    )

# 2. Refine data
NOISE_FIELDS_MAP = {
  "pino": ["time", "reqId", "port", "version", "signal", "level"],
  "go": ["ts", "caller", "request_id", "client_ip", "query", "latency"],
  "serilog_compact": ["@t", "@l", "@r", "EventId", "SourceContext", "Meta"],
  "serilog": ["Timestamp", "Level", "EventId", "SourceContext"],
}

def extract_text(data: Datasource) -> str:
  log = data.log.copy()

  if log.get("req"):
    return json.dumps({
      "method": log["req"].get("method"),
      "url": log["req"].get("url"),
    })
  
  if log.get("res"):
    return json.dumps({
      "statusCode": log["res"].get("statusCode"),
      "responseTime": log.get("responseTime")
    })
  
  if data.framework == "serilog" and log.get("Properties"):
    props = log.pop("Properties")
    log.update(props)

  noise = NOISE_FIELDS_MAP.get(data.framework, [])
  filtered = {
    k: v for k, v in log.items()
    if k not in noise and not isinstance(v, dict)
  }

  return json.dumps(filtered)

def extract_category(data: Datasource) -> str | None:
  log = data.log

  if log.get("req"):
    return "HTTP_REQUEST"
  
  if log.get("res"):
    return "HTTP_RESPONSE"
  
  match data.framework:
    case "pino":
      category = log.get("category")
    
    case "go":
      category = log.get("category")

    case "serilog_compact":
      category = log.get("Category")

    case "serilog":
      category = log.get("Properties", {}).get("Category")
    
    case _:
      category = log.get("category") or log.get("Category")

  if category and isinstance(category, str):
    return category.strip()
  
  match data.framework:
    case "serilog_compact":
      if log.get("@mt") or log.get("SourceContext"):
        return "SYSTEM"
    case "serilog":
      props = log.get("Properties", {})
      if props.get("SourceContext"):
        return "SYSTEM"
    case "pino":
      if log.get("msg", "").lower().startswith("server listening"):
        return "SYSTEM"

  return None

records = []

for data in datas:
  category = extract_category(data=data)
  text = extract_text(data=data)

  if category is None or category == "ERROR":
    continue

  records.append({
    "text": text,
    "label": category,
  })

records = [r for r in records if r["label"] is not None]

df = pd.DataFrame(records)

le = LabelEncoder()

def sanitize(text: str) -> str:
  text = text.replace("[CLS]", "")
  text = text.replace("[SEP]", "")
  text = text.replace("[PAD]", "")
  text = text.replace("[MASK]", "")

  return text.strip()

df["label_id"] = le.fit_transform(df["label"])
df["text"] = df["text"].apply(sanitize)

train_df, test_df = train_test_split(
  df,
  test_size=0.2,
  random_state=42,
  stratify=df["label"]
)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

train_dataset = LogDataset(train_df, tokenizer)
test_dataset = LogDataset(test_df, tokenizer)

train_loader = DataLoader(
  train_dataset,
  batch_size=32,
  shuffle=True,
)

test_loader = DataLoader(
  test_dataset,
  batch_size=32,
  shuffle=False
)

model = BertForSequenceClassification.from_pretrained(
  "bert-base-uncased",
  num_labels=6,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = model.to(device)

# Training model
optimizer = AdamW(model.parameters(), lr=2e-5)

num_epochs = 3
num_training_step = num_epochs * len(train_loader)

scheduler = get_scheduler(
  "linear",
  optimizer=optimizer,
  num_warmup_steps=0,
  num_training_steps=num_training_step,
)

# Training Loop
## Print start time
start_time = time.time()
print(f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

for epoch in range(num_epochs):
  model.train()
  total_loss = 0

  for batch in train_loader:
    input_ids = batch["input_ids"].to(device)
    attention_mark = batch["attention_mask"].to(device)
    labels = batch["label"].to(device)

    outputs = model(
      input_ids=input_ids,
      attention_mark=attention_mark,
      labels=labels,
    )

    loss = outputs.loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    total_loss += loss.item()

  avg_loss = total_loss / len(train_loader)

  model.eval()
  all_preds = []
  all_labels = []

  with torch.no_grad():
    for batch in test_loader:
      input_ids = batch["input_ids"].to(device)
      attention_mark = batch["attention_mark"].to(device)
      labels = batch["label"].to(device)

      outputs = model(
        input_ids=input_ids,
        attention_mark=attention_mark,
      )

      preds = torch.argmax(outputs.logits, dim=1)
      all_preds.extend(preds.cpu().numpy())
      all_labels.extend(labels.cpu().numpy())

  print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}")
  print(classification_report(all_labels, all_preds, target_names=le.classes_))

## Print end time
end_time = time.time()
print(f"Training ended at:  {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
print(f"Total training time: {end_time - start_time:.2f}s")
print("\nTraining done! 🎉")