import os
import cv2
import numpy as np
from typing import List, Dict
import hashlib

class FaceDetectionError(Exception):
    """Custom exception for face detection errors"""
    pass

# Global detector instance
_DETECTOR = None

def get_detector():
    """
    Get or create the global YuNet face detector instance.
    Returns the cv2.FaceDetectorYN object.
    """
    global _DETECTOR
    if _DETECTOR is None:
        model_path = os.getenv("YUNET_PATH", "./models/yunet.onnx")

        if not os.path.exists(model_path):
            raise FaceDetectionError(f"YuNet model not found at {model_path}")

        try:
            # Create detector with environment configurable parameters
            score_thr = float(os.getenv("SCORE_THR", "0.6"))
            nms_thr   = float(os.getenv("NMS_THR", "0.3"))
            top_k     = int(os.getenv("TOP_K",   "5000"))

            # ✅ Correct OpenCV API (Python): FaceDetectorYN_create with positional args
            _DETECTOR = cv2.FaceDetectorYN_create(
                model_path, "", (320, 320), score_thr, nms_thr, top_k
            )
        except Exception as e:
            raise FaceDetectionError(f"Failed to initialize face detector: {str(e)}")

    return _DETECTOR

def decode_image(image_bytes: bytes) -> np.ndarray:
    """
    Decode image bytes to BGR numpy array.

    Raises:
        FaceDetectionError: If image cannot be decoded
    """
    try:
        if not image_bytes:
            raise FaceDetectionError("Empty image payload")
        np_array = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        if img is None:
            raise FaceDetectionError("Failed to decode image - invalid format")
        return img
    except Exception as e:
        raise FaceDetectionError(f"Image decoding failed: {str(e)}")

def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

def detect_bgr(img: np.ndarray) -> List[Dict]:
    """
    Detect faces in BGR image.

    Returns:
        List of detected faces with relative coordinates:
        {
          "bbox": [x_rel, y_rel, w_rel, h_rel],
          "score": float,
          "landmarks": [{"x": float, "y": float}] | None
        }
    """
    if img is None or img.size == 0:
        raise FaceDetectionError("Empty image array")

    detector = get_detector()
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        raise FaceDetectionError("Invalid image dimensions")

    # Set input size for current frame
    detector.setInputSize((w, h))

    # Detect faces
    ok, faces = detector.detect(img)
    if not ok or faces is None:
        return []

    results: List[Dict] = []
    for face in faces:
        # face layout (YuNet): [x, y, w, h, l0x, l0y, ..., l4x, l4y, score]
        f = face.astype(float)

        x, y, bw, bh = f[:4]
        # score index can vary, use fallback
        score = float(f[14]) if len(f) > 14 else (float(f[4]) if len(f) > 4 else 0.0)

        # Convert to relative coordinates [0..1] and clamp
        x_rel = _clamp01(x / w)
        y_rel = _clamp01(y / h)
        w_rel = _clamp01(bw / w)
        h_rel = _clamp01(bh / h)

        # Landmarks (5 points -> 10 values) if present
        landmark_list = None
        if len(f) >= 14:
            lm = f[4:14]  # 10 values
            if lm.size == 10:
                landmark_list = []
                for i in range(0, 10, 2):
                    lx = _clamp01(lm[i]   / w)
                    ly = _clamp01(lm[i+1] / h)
                    landmark_list.append({"x": lx, "y": ly})

        results.append({
            "bbox": [x_rel, y_rel, w_rel, h_rel],
            "score": float(score),
            "landmarks": landmark_list
        })

    return results

def calculate_audit_id(image_bytes: bytes) -> str:
    """
    Calculate SHA-256 audit ID for image (first 32 hex chars).
    """
    sha_hash = hashlib.sha256(image_bytes).hexdigest()
    return sha_hash[:32]


