from pydantic import BaseModel
from typing import Optional

class Observation(BaseModel):
    accuracy: float
    precision: float
    recall: float
    feature_count: int
    scaling: bool
    test_split: float
    model_type: str

class Action(BaseModel):
    type: str

class ResetRequest(BaseModel):
    task_name: Optional[str] = None