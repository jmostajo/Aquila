import os
import cv2
import time
from detector import get_detector, detect_bgr

def wait_for_face(
    device: int = 0,
    timeout: float = 10.0,
    consec: int = 3,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    score_thr: float = 0.6
) -> bool:
    """
    Bloquea el arranque hasta detectar rostro(s) en la cámara.

    Params:
        device: índice de cámara (0 por defecto).
        timeout: tiempo máx. de espera en segundos.
        consec: nº de frames consecutivos con detección válida requeridos.
        width/height: resolución solicitada al capturador.
        fps: frames por segundo deseados (solo para pacing del bucle).
        score_thr: umbral mínimo de confianza para considerar una detección válida.

    Return:
        True si se detecta rostro dentro del timeout; False en caso contrario.
    """
    print(f"[face-gate] Start (timeout={timeout}s, consec={consec}, thr={score_thr})")

    # Elegir backend AVFoundation si existe (macOS), si no CAP_ANY.
    avf = getattr(cv2, "CAP_AVFOUNDATION", None)
    backends = [b for b in (avf, cv2.CAP_ANY) if b is not None]

    cap = None
    for backend in backends:
        try:
            cap = cv2.VideoCapture(device, backend)
            if not cap or not cap.isOpened():
                if cap: cap.release()
                cap = None
                continue

            # Intentar fijar propiedades
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS,          fps)

            # Probar lectura
            ok, test_frame = cap.read()
            if ok and test_frame is not None:
                print(f"[face-gate] Camera opened (backend={backend})")
                break

            # Si falla la lectura, probar siguiente backend
            cap.release()
            cap = None
        except Exception as e:
            if cap:
                cap.release()
                cap = None
            print(f"[face-gate] Backend {backend} failed: {e}")

    if not cap or not cap.isOpened():
        print(f"[face-gate] ERROR: Cannot open camera device {device}. "
              f"macOS: revisa Permisos de Cámara para tu terminal.")
        return False

    try:
        # Pre-carga explícita del detector (validará que el modelo exista)
        _ = get_detector()

        start = time.time()
        consecutive = 0
        frame_count = 0
        sleep_dt = 1.0 / max(1, int(fps))  # evita división por cero

        while (time.time() - start) < timeout:
            ok, frame = cap.read()
            if not ok or frame is None:
                # Pequeño backoff si hay frame drop
                time.sleep(0.1)
                continue

            frame_count += 1

            # Detectar rostros en el frame actual
            faces = detect_bgr(frame)

            # ¿Hay al menos un rostro con score suficiente?
            valid = any( (f.get("score", 0.0) >= score_thr) for f in faces )

            if valid:
                consecutive += 1
                print(f"[face-gate] Face detected ({consecutive}/{consec})")
                if consecutive >= consec:
                    print("[face-gate] passed")
                    return True
            else:
                # Reinicia contador si un frame no cumple
                consecutive = 0
                if frame_count % 10 == 0:
                    print(f"[face-gate] Scanning... ({frame_count} frames)")

            # Pacing del bucle (sin bloquear demasiado)
            if sleep_dt > 0:
                time.sleep(sleep_dt)

        print(f"[face-gate] Timeout reached after {timeout}s")
        return False

    except Exception as e:
        print(f"[face-gate] ERROR: {e}")
        return False
    finally:
        try:
            cap.release()
        except Exception:
            pass

