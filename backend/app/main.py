import os
import json
import logging
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.inference import InferencePipeline, CLASS_NAMES, CLASS_FULL
from app.gradcam import GradCAMEngine, SCIENTIFIC_LIMITATION_DISCLAIMER
from app.batch import BatchProcessor
from app.schemas import (
    HealthResponse,
    ClassesResponse,
    ClassItem,
    PredictResponse,
    BatchResponse,
    BatchItemResult,
    GradCamResponse,
    ExplainResponse,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("skinlens.api")

# Constants & Limits
MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB
MAX_BATCH_SIZE = 15
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/jpg"}
MAGIC_SIGNATURES = {
    b"\xff\xd8\xff": "image/jpeg",
    b"\x89PNG\r\n\x1a\n": "image/png",
    b"RIFF": "image/webp",
}

CLASS_DESCRIPTIONS = {
    "akiec": "Actinic keratoses and intraepithelial carcinoma (Bowen's disease); precancerous/early non-melanoma skin lesions.",
    "bcc": "Basal cell carcinoma; a common form of non-melanoma skin cancer originating in basal cells.",
    "bkl": "Benign keratosis-like lesions; includes seborrheic keratoses, solar lentigines, and lichen-planus-like keratoses.",
    "df": "Dermatofibroma; benign fibrous cutaneous histiocytoma nodules.",
    "mel": "Melanoma; malignant neoplasm of melanocytic origin with high metastatic potential.",
    "nv": "Melanocytic nevi; common benign moles composed of melanocyte clusters.",
    "vasc": "Vascular lesions; benign angiomas, pyogenic granulomas, and vascular malformations.",
}

# Lifespan initialization to ensure model loads once during startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing SkinLens Inference Engine & Preloading Model...")
    try:
        pipeline = InferencePipeline.get_instance()
        app.state.pipeline = pipeline
        app.state.gradcam_engine = GradCAMEngine(pipeline)
        app.state.batch_processor = BatchProcessor(pipeline)
        logger.info(f"Model successfully loaded on compute device: {pipeline.device}")
    except Exception as e:
        logger.error(f"FATAL: Model initialization failed during startup: {e}")
        app.state.pipeline = None
        app.state.gradcam_engine = None
        app.state.batch_processor = None
    yield
    logger.info("SkinLens Backend shutting down.")

# Instantiate FastAPI application
app = FastAPI(
    title="SkinLens API",
    description=(
        "Production-minded REST API for Multimodal Skin Lesion Classification. "
        "Integrates a hybrid EfficientNetV2-S + Swin Transformer V2 + Metadata Fusion model "
        "trained on the HAM10000 dataset, with temperature scaling and Grad-CAM interpretability."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Configuration
cors_origins_env = os.getenv("CORS_ORIGINS", "")
allowed_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
if cors_origins_env:
    for origin in cors_origins_env.split(","):
        cleaned = origin.strip()
        if cleaned and cleaned not in allowed_origins:
            allowed_origins.append(cleaned)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

def validate_image_bytes(file_bytes: bytes, filename: str = "upload") -> None:
    """Validates file size, non-emptiness, and magic byte signatures."""
    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Uploaded file '{filename}' is empty."
        )
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"File '{filename}' exceeds maximum allowed size of {MAX_FILE_SIZE_BYTES // (1024*1024)}MB."
        )
    # Check header magic signatures
    is_valid_signature = False
    for sig in MAGIC_SIGNATURES:
        if file_bytes.startswith(sig):
            is_valid_signature = True
            break
        # Special check for WebP (RIFF....WEBP)
        if len(file_bytes) >= 12 and file_bytes[:4] == b"RIFF" and file_bytes[8:12] == b"WEBP":
            is_valid_signature = True
            break

    if not is_valid_signature:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File '{filename}' has an unsupported or corrupted image format. Supported formats: JPEG, PNG, WEBP."
        )

# Global Exception Handler to prevent exposing raw stack traces
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred while processing the request."}
    )

# ----------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------

@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def get_health():
    """Health check endpoint reporting API readiness, compute device, and model status."""
    pipeline: Optional[InferencePipeline] = getattr(app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready or failed to load during startup."
        )
    return HealthResponse(
        status="healthy",
        model_ready=True,
        device=str(pipeline.device),
        model_name="SkinLensHybrid (EfficientNetV2-S + Swin Transformer V2)",
        version="1.0.0",
        temperature=pipeline.temperature,
        num_classes=len(CLASS_NAMES)
    )

@app.get("/api/classes", response_model=ClassesResponse, tags=["Model Info"])
async def get_classes():
    """Returns the 7 canonical HAM10000 classification categories with descriptions."""
    class_list = [
        ClassItem(
            index=i,
            key=k,
            name=CLASS_FULL[k],
            description=CLASS_DESCRIPTIONS.get(k, "")
        )
        for i, k in enumerate(CLASS_NAMES)
    ]
    return ClassesResponse(total=len(class_list), classes=class_list)

@app.post("/api/predict", response_model=PredictResponse, tags=["Inference"])
async def predict_single(
    image: UploadFile = File(..., description="Dermoscopic lesion image (JPEG, PNG, WEBP)"),
    age: Optional[float] = Form(None, description="Patient age (0-90)"),
    sex: Optional[str] = Form(None, description="Patient sex ('male', 'female', 'unknown')"),
    localization: Optional[str] = Form(None, description="Anatomical localization"),
    include_gradcam: bool = Form(True, description="Whether to include Grad-CAM explanation")
):
    """
    Analyzes a single dermoscopic image with optional clinical metadata.
    Returns calibrated probabilities across all 7 categories and Grad-CAM explanation.
    """
    pipeline: InferencePipeline = getattr(app.state, "pipeline", None)
    gradcam_engine: GradCAMEngine = getattr(app.state, "gradcam_engine", None)
    if pipeline is None or gradcam_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model engine is unavailable."
        )

    # Validate image bytes
    contents = await image.read()
    validate_image_bytes(contents, filename=image.filename or "upload")

    # Metadata dictionary used
    metadata_used = {"age": age, "sex": sex, "localization": localization}

    # Execute inference
    try:
        inf_result = pipeline.predict(
            image_source=contents,
            age=age,
            sex=sex,
            localization=localization
        )
    except Exception as e:
        logger.error(f"Prediction failed for {image.filename}: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Inference preprocessing failed: {str(e)}"
        )

    # Optional Grad-CAM
    gradcam_data: Optional[GradCamResponse] = None
    if include_gradcam:
        try:
            cam_res = gradcam_engine.explain(
                image_source=contents,
                age=age,
                sex=sex,
                localization=localization,
                target_class=inf_result["predicted_class"]
            )
            gradcam_data = GradCamResponse(
                available=True,
                target_layer=cam_res["target_layer"],
                target_class=cam_res["target_class"],
                scope=cam_res["explanation_scope"],
                original_image=cam_res["original_image_base64"],
                heatmap=cam_res["heatmap_base64"],
                overlay=cam_res["overlay_base64"],
                disclaimer=cam_res["limitation_disclaimer"]
            )
        except Exception as e:
            logger.warning(f"Grad-CAM explanation generation failed: {e}")
            gradcam_data = GradCamResponse(
                available=False,
                disclaimer=SCIENTIFIC_LIMITATION_DISCLAIMER,
                error=f"Grad-CAM generation failed: {str(e)}"
            )

    return PredictResponse(
        item_id=image.filename or "case_1",
        status="completed",
        predicted_class=inf_result["predicted_class"],
        predicted_class_name=inf_result["predicted_class_name"],
        raw_confidence=inf_result["raw_confidence"],
        calibrated_confidence=inf_result["calibrated_confidence"],
        raw_probabilities=inf_result["raw_probabilities"],
        calibrated_probabilities=inf_result["calibrated_probabilities"],
        metadata_used=metadata_used,
        gradcam=gradcam_data
    )

@app.post("/api/predict/batch", response_model=BatchResponse, tags=["Inference"])
async def predict_batch(
    images: List[UploadFile] = File(..., description="List of dermoscopic images"),
    metadata_json: Optional[str] = Form(None, description="Optional JSON array or map containing metadata per image"),
    include_gradcam: bool = Form(True, description="Whether to generate Grad-CAM for each image")
):
    """
    Analyzes multiple dermoscopic images in one batch with independent metadata.
    Provides strict error isolation: failed/corrupted images do not fail the other cases.
    """
    batch_processor: BatchProcessor = getattr(app.state, "batch_processor", None)
    if batch_processor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Batch processor is unavailable."
        )

    if not images:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No image files were supplied in the batch request."
        )

    if len(images) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch size {len(images)} exceeds maximum allowed batch count of {MAX_BATCH_SIZE} images."
        )

    # Parse metadata JSON if provided
    metadata_entries: Dict[str, Any] = {}
    if metadata_json:
        try:
            parsed = json.loads(metadata_json)
            if isinstance(parsed, list):
                # Map by index "0", "1", ...
                metadata_entries = {str(i): m for i, m in enumerate(parsed) if isinstance(m, dict)}
            elif isinstance(parsed, dict):
                metadata_entries = parsed
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid metadata_json parameter format: {str(e)}"
            )

    # Build batch items
    batch_items = []
    for idx, img_file in enumerate(images):
        file_bytes = await img_file.read()
        item_id = img_file.filename or f"image_{idx + 1}"

        # Resolve metadata: look up by filename or by string index
        m_entry = metadata_entries.get(item_id, metadata_entries.get(str(idx), {}))
        age = m_entry.get("age") if isinstance(m_entry, dict) else None
        sex = m_entry.get("sex") if isinstance(m_entry, dict) else None
        loc = m_entry.get("localization") if isinstance(m_entry, dict) else None

        batch_items.append({
            "id": item_id,
            "image": file_bytes,
            "age": age,
            "sex": sex,
            "localization": loc
        })

    # Execute batch processing with error isolation
    batch_output = batch_processor.process_batch(
        items=batch_items,
        generate_explanation=include_gradcam
    )

    # Format into Pydantic schema
    formatted_results: List[BatchItemResult] = []
    for r in batch_output["results"]:
        cam_resp = None
        if r.get("gradcam"):
            g = r["gradcam"]
            if g.get("available") is False:
                cam_resp = GradCamResponse(
                    available=False,
                    disclaimer=g.get("disclaimer"),
                    error=g.get("error")
                )
            else:
                cam_resp = GradCamResponse(
                    available=True,
                    target_layer=g.get("target_layer"),
                    target_class=r.get("predicted_class"),
                    scope=g.get("scope"),
                    original_image=g.get("original_image"),
                    heatmap=g.get("heatmap"),
                    overlay=g.get("overlay"),
                    disclaimer=g.get("disclaimer")
                )

        formatted_results.append(
            BatchItemResult(
                id=r["id"],
                status=r["status"],
                predicted_class=r.get("predicted_class"),
                predicted_class_name=r.get("predicted_class_name"),
                raw_confidence=r.get("raw_confidence"),
                calibrated_confidence=r.get("calibrated_confidence"),
                raw_probabilities=r.get("raw_probabilities"),
                calibrated_probabilities=r.get("calibrated_probabilities"),
                metadata_used=r.get("metadata_used", {}),
                gradcam=cam_resp,
                error=r.get("error")
            )
        )

    return BatchResponse(
        total_items=batch_output["total_items"],
        successful_items=batch_output["successful_items"],
        failed_items=batch_output["failed_items"],
        results=formatted_results
    )

@app.post("/api/explain", response_model=ExplainResponse, tags=["Interpretability"])
async def explain_image(
    image: UploadFile = File(..., description="Dermoscopic lesion image"),
    age: Optional[float] = Form(None, description="Patient age"),
    sex: Optional[str] = Form(None, description="Patient sex"),
    localization: Optional[str] = Form(None, description="Anatomical localization"),
    target_class: Optional[str] = Form(None, description="Optional target class key to explain (default: predicted class)")
):
    """
    Generates a Grad-CAM explanation focusing on the EfficientNetV2-S convolutional head.
    Highlights spatial visual features that influenced the model's prediction.
    """
    gradcam_engine: GradCAMEngine = getattr(app.state, "gradcam_engine", None)
    if gradcam_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Grad-CAM engine is unavailable."
        )

    contents = await image.read()
    validate_image_bytes(contents, filename=image.filename or "upload")

    try:
        cam_res = gradcam_engine.explain(
            image_source=contents,
            age=age,
            sex=sex,
            localization=localization,
            target_class=target_class
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Grad-CAM explanation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Grad-CAM computation failed: {str(e)}"
        )

    return ExplainResponse(
        predicted_class=cam_res["predicted_class"],
        target_class=cam_res["target_class"],
        target_class_name=cam_res["target_class_name"],
        target_class_cal_prob=cam_res["target_class_cal_prob"],
        target_class_raw_prob=cam_res["target_class_raw_prob"],
        original_image=cam_res["original_image_base64"],
        heatmap=cam_res["heatmap_base64"],
        overlay=cam_res["overlay_base64"],
        target_layer=cam_res["target_layer"],
        scope=cam_res["explanation_scope"],
        disclaimer=cam_res["limitation_disclaimer"]
    )
