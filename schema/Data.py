from __future__ import annotations

from typing import List

from pydantic import BaseModel


# normalized bounding box coordinates in [0, 1]
class BoundingBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float


# single detected object with its class, confidence, bounding box, and sampled depth
class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox
    median_depth: float


# risk assessment for a single pet–object pair
class PairRisk(BaseModel):
    pet: Detection
    obj: Detection
    proximity_2d: float
    depth_similarity: float
    contact_likelihood: float
    closeness_score: float
    risk_multiplier: float
    risk_score: float


# complete mischief result for one frame including all pair scores and the final verdict
class MischiefResult(BaseModel):
    source: str
    detections: List[Detection]
    pairs: List[PairRisk]
    max_risk_score: float
    risk_level: str
    warning_message: str
