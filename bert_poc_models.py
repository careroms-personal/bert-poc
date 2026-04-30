from pydantic import BaseModel
from typing import Dict, Any

class Datasource(BaseModel):
  container_id: str
  container_name: str
  framework: str
  timestamp: int
  source: str
  log: Dict[str, Any]