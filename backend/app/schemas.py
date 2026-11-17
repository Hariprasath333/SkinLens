from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str = Field(..., description="Overall API operational status")
    model_ready: bool = Field(..., description="Whether the trained checkpoint is loaded into memory")
    device: str = Field(..., description="Inference compute device (cpu or cuda)")
    model_name: str = Field("SkinLensHybrid", description="Model architecture identifier")
    version: str = Field("1.0.0", description="Application version")
    temperature: float = Field(..., description="Loaded temperature scaling parameter")
    num_classes: int = Field(7, description="Number of target lesion classes")

class ClassItem(BaseModel):
    index: int = Field(..., description="Canonical integer index (0-6)")
    key: str = Field(..., description="Short class abbreviation (e.g. nv)")
    name: str = Field(..., description="Formal medical name of the lesion category")
    description: str = Field(..., description="Clinical description of the category")

class ClassesResponse(BaseModel):
    total: int = Field(..., description="Total number of supported classes")
    classes: List[ClassItem] = Field(..., description="Canonical ordered class list")

class GradCamResponse(BaseModel):
    available: bool = Field(True, description="Whether Grad-CAM was successfully generated")
    target_layer: Optional[str] = Field(None, description="Model layer probed for activations")
    target_class: Optional[str] = Field(None, description="Class category for which the map was generated")
    scope: Optional[str] = Field(None, description="Scope of the explanation")
    original_image: Optional[str] = Field(None, description="Base64 data URL of the resized original image")
    heatmap: Optional[str] = Field(None, description="Base64 data URL of the JET colormap heatmap")
    overlay: Optional[str] = Field(None, description="Base64 data URL of the 0.6*image + 0.4*heatmap overlay")
    disclaimer: Optional[str] = Field(None, description="Medical and scientific disclaimer")
    error: Optional[str] = Field(None, description="Error message if Grad-CAM generation failed")

class PredictResponse(BaseModel):
    item_id: str = Field("case_1", description="Identifier for this prediction case")
    status: str = Field("completed", description="Status ('completed' or 'failed')")
    predicted_class: Optional[str] = Field(None, description="Predicted lesion class key")
    predicted_class_name: Optional[str] = Field(None, description="Full medical name of predicted class")
    raw_confidence: Optional[float] = Field(None, description="Uncalibrated probability of predicted class")
    calibrated_confidence: Optional[float] = Field(None, description="Temperature-calibrated confidence score")
    raw_probabilities: Optional[Dict[str, float]] = Field(None, description="Raw probabilities across all 7 classes")
    calibrated_probabilities: Optional[Dict[str, float]] = Field(None, description="Calibrated probabilities across all 7 classes")
    metadata_used: Dict[str, Any] = Field(default_factory=dict, description="Metadata submitted or defaulted")
    gradcam: Optional[GradCamResponse] = Field(None, description="Grad-CAM visualization if requested")
    error: Optional[str] = Field(None, description="Error details if prediction failed")

class BatchItemResult(BaseModel):
    id: str = Field(..., description="Item identifier corresponding to the uploaded case")
    status: str = Field(..., description="Item status: 'completed' or 'failed'")
    predicted_class: Optional[str] = Field(None, description="Predicted lesion class key")
    predicted_class_name: Optional[str] = Field(None, description="Full medical name of predicted class")
    raw_confidence: Optional[float] = Field(None, description="Raw confidence of predicted class")
    calibrated_confidence: Optional[float] = Field(None, description="Calibrated confidence of predicted class")
    raw_probabilities: Optional[Dict[str, float]] = Field(None, description="Raw 7-class distribution")
    calibrated_probabilities: Optional[Dict[str, float]] = Field(None, description="Calibrated 7-class distribution")
    metadata_used: Dict[str, Any] = Field(default_factory=dict, description="Clinical metadata applied to this item")
    gradcam: Optional[GradCamResponse] = Field(None, description="Per-item Grad-CAM explanation")
    error: Optional[str] = Field(None, description="Failure reason for this item if failed")

class BatchResponse(BaseModel):
    total_items: int = Field(..., description="Total number of images submitted")
    successful_items: int = Field(..., description="Number of successfully analyzed images")
    failed_items: int = Field(..., description="Number of images that encountered errors")
    results: List[BatchItemResult] = Field(..., description="Ordered independent results for each item")

class ExplainResponse(BaseModel):
    predicted_class: str = Field(..., description="Predicted class for the image")
    target_class: str = Field(..., description="Class targeted by the Grad-CAM backward pass")
    target_class_name: str = Field(..., description="Medical name of the targeted class")
    target_class_cal_prob: float = Field(..., description="Calibrated probability of targeted class")
    target_class_raw_prob: float = Field(..., description="Raw probability of targeted class")
    original_image: str = Field(..., description="Base64 data URL of the original input")
    heatmap: str = Field(..., description="Base64 data URL of the activation heatmap")
    overlay: str = Field(..., description="Base64 data URL of the composite overlay")
    target_layer: str = Field(..., description="Probed convolutional head")
    scope: str = Field(..., description="Interpretability scope description")
    disclaimer: str = Field(..., description="Scientific limitation notice")
