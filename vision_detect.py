from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import time
from collections import defaultdict

from detector import detect_bgr, decode_image, calculate_audit_id, FaceDetectionError

# -------- Rate limiting (memoria) --------
# IP -> lista de timestamps (segundos)
_request_counts: Dict[str, List[float]] = defaultdict(list)

def rate_limit_check(client_ip: str, limit: int = 10, window: int = 60) -> bool:
    """
    Límite simple: máx 'limit' solicitudes por 'window' segundos por IP.
    Devuelve True si se permite la solicitud, False si está rate limited.
    """
    now = time.time()
    window_start = now - window

    # Limpia timestamps fuera de ventana
    recent = [t for t in _request_counts[client_ip] if t > window_start]
    _request_counts[client_ip] = recent

    if len(recent) < limit:
        _request_counts[client_ip].append(now)
        return True
    return False

# -------- Router --------
router = APIRouter(prefix="/v1/vision", tags=["vision"])

# -------- Modelos Pydantic --------
class Landmark(BaseModel):
    x: float = Field(..., ge=0.0, le=1.0, description="X relativo [0..1]")
    y: float = Field(..., ge=0.0, le=1.0, description="Y relativo [0..1]")

class FaceBox(BaseModel):
    bbox: List[float] = Field(..., min_items=4, max_items=4, description="[x,y,w,h] relativos [0..1]")
    score: float = Field(..., ge=0.0, le=1.0, description="Confianza [0..1]")
    landmarks: Optional[List[Landmark]] = Field(None, description="5 landmarks si están disponibles")

class DetectResponse(BaseModel):
    faces: List[FaceBox] = Field(..., description="Rostros detectados")
    audit_id: str = Field(..., description="Prefijo SHA-256 (32 hex) de la imagen")
    message: str = Field(..., description="Mensaje de resultado")

@router.post("/detect", response_model=DetectResponse)
async def detect_faces(
    request: Request,
    image: UploadFile = File(..., description="Imagen a analizar (JPEG/PNG/WebP)")
):
    """
    Detecta rostros en la imagen subida.
    - Tamaño máx: 10MB
    - Tipos: JPEG, PNG, WebP (se permite 'application/octet-stream' para clientes que no envían MIME correcto)
    - Rate limit: 10 req/min/IP
    """
    # Rate limiting
    client_ip = request.client.host if request.client else "unknown"
    if not rate_limit_check(client_ip, limit=10, window=60):
        raise HTTPException(status_code=429, detail="Rate limit exceeded: 10 requests per minute")

    # Validación de tipo MIME (acepta algunos casos ambiguos)
    allowed_types = {
        "image/jpeg", "image/jpg", "image/png", "image/webp", "application/octet-stream"
    }
    if (image.content_type or "").lower() not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type. Allowed: JPEG, PNG, WebP"
        )

    try:
        # Lee bytes y valida tamaño
        contents = await image.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty image payload")
        if len(contents) > 10 * 1024 * 1024:  # 10 MB
            raise HTTPException(status_code=400, detail="Image too large. Maximum size: 10MB")

        # Audit ID
        audit_id = calculate_audit_id(contents)

        # Decodificación y detección
        img_bgr = decode_image(contents)
        faces_dicts = detect_bgr(img_bgr)

        # Pydantic validará contra FaceBox; convertimos dict->modelo (opcional)
        faces_models = [FaceBox(**f) for f in faces_dicts]

        message = "face detection successful" if faces_models else "no face detected"

        return DetectResponse(
            faces=faces_models,
            audit_id=audit_id,
            message=message
        )

    except FaceDetectionError as e:
        # Errores conocidos del pipeline (decodificación/modelo)
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        # Repropaga HTTPException (400/429/etc.)
        raise
    except Exception as e:
        # Falla inesperada
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

