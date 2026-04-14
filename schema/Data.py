
from pydantic import BaseModel
from typing import List

class Data(BaseModel):
    ground_truth: List
    