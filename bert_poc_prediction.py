import torch
import pickle

from pathlib import Path
from transformers import BertForSequenceClassification, BertTokenizer

import json
from bert_poc_models import Datasource

# Load fle
GEN_LOG_PATH = Path("./poc-gen-logs")

file_names = [f.name for f in GEN_LOG_PATH.iterdir() if f.is_file() and f.suffix == ".jsonl"]
logs_line = []

for file in file_names:
  file_dir = Path(GEN_LOG_PATH / file)

  with open(file_dir, "r") as f:
    lines = f.readlines()

  logs_line.extend(lines)

print(logs_line)

# Load token and model
SAVE_PATH = Path("./bert-log-classifier")

tokenizer = BertTokenizer.from_pretrained(SAVE_PATH)
model = BertForSequenceClassification.from_pretrained(SAVE_PATH)

with open(SAVE_PATH / "label_encoder.pkl", "rb") as f:
  le = pickle.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# bert_poc_models.py — add these
FRAMEWORK_MAP = {
    "/api-cart":     "pino",
    "/api-order":    "go",
    "/api-product":  "go",
    "/api-customer": "serilog_compact",
    "/api-payment":  "serilog",
}

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

def predict(line: str) -> dict:
    # 1. Parse fluent-bit line
    bracket = line.index("[")
    timestamp, payload = json.loads(line[bracket:])
    inner_log = json.loads(payload["log"])

    # 2. Build Datasource
    data = Datasource(
        container_id=payload["container_id"],
        container_name=payload["container_name"],
        framework=FRAMEWORK_MAP.get(payload["container_name"], "unknown"),
        timestamp=int(timestamp),
        source=payload["source"],
        log=inner_log,
    )

    # 3. Extract text
    text = extract_text(data=data)

    # 4. Tokenize
    tokens = tokenizer(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    # 5. Predict
    with torch.no_grad():
        outputs = model(
            input_ids=tokens["input_ids"].to(device),
            attention_mask=tokens["attention_mask"].to(device),
        )

    predicted_id = torch.argmax(outputs.logits, dim=1).item()
    predicted_label = le.inverse_transform([predicted_id])[0]

    return {
        "text": text,
        "predicted": predicted_label,
    }

# Run on all lines
for line in logs_line:
    result = predict(line.strip())
    print(f"{result['predicted']:<15} | {result['text']}")