from __future__ import annotations

from typing import List

from pydantic import BaseModel


class BoundingBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float  # all normalized to [0, 1]


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox
    median_depth: float  # sampled from depth map; 0 = far, 1 = near


class PairRisk(BaseModel):
    pet: Detection
    obj: Detection
    proximity_2d: float
    depth_similarity: float
    contact_likelihood: float
    closeness_score: float
    risk_multiplier: float
    risk_score: float


class MischiefResult(BaseModel):
    source: str  # image filename or "video_frame"
    detections: List[Detection]
    pairs: List[PairRisk]
    max_risk_score: float
    risk_level: str  # "HIGH", "MEDIUM", "LOW"
    warning_message: str
