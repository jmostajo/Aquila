# gate_and_run.py — Detecta rostro y, si aprueba, lanza Aquila.py en Streamlit con snapshot
import os
import sys
import time
import webbrowser
from subprocess import Popen
from pathlib import Path
import cv2

from facegate import wait_for_face


def capture_one_frame(device=0, width=640, height=480, fps=30):
    """Toma un frame de la cámara con fallback de backend (macOS AVFoundation -> ANY)."""
    def _open(backend_id):
        cap = cv2.VideoCapture(device, backend_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        return cap

    # Intentar AVFoundation (macOS), si falla usar cualquier backend disponible
    cap = _open(cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        try:
            cap.release()
        except Exception:
            pass
        cap = _open(cv2.CAP_ANY)

    # Pequeño warm-up
    for _ in range(3):
        cap.read()
        time.sleep(0.02)

    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def main():
    # Asegura ruta del modelo YuNet
    os.environ.setdefault("YUNET_PATH", "./models/yunet.onnx")

    # Gate de rostro (usa env vars para parámetros si existen)
    ok = wait_for_face(
        device=int(os.getenv("CAMERA_INDEX", "0")),
        timeout=float(os.getenv("FACE_GATE_TIMEOUT", "10")),
        consec=int(os.getenv("FACE_GATE_CONSEC", "3")),
        width=int(os.getenv("CAM_WIDTH", "640")),
        height=int(os.getenv("CAM_HEIGHT", "480")),
        fps=int(os.getenv("CAM_FPS", "30")),
        score_thr=float(os.getenv("SCORE_THR", "0.6")),
    )
    if not ok:
        print("[launch] Face gate FAILED — aborting.", flush=True)
        sys.exit(1)

    print("[launch] Face gate PASSED — capturing snapshot…", flush=True)
    time.sleep(0.4)  # pequeña pausa para soltar/retomar la cámara

    snapshot_dir = Path("./snapshots")
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    snapshot_path = snapshot_dir / f"facegate_{ts}.jpg"

    frame = capture_one_frame(
        device=int(os.getenv("CAMERA_INDEX", "0")),
        width=int(os.getenv("CAM_WIDTH", "640")),
        height=int(os.getenv("CAM_HEIGHT", "480")),
        fps=int(os.getenv("CAM_FPS", "30")),
    )

    if frame is not None:
        ok_write = cv2.imwrite(str(snapshot_path), frame)
        exists = snapshot_path.exists()
        print(f"[launch] cv2.imwrite -> {ok_write}, exists -> {exists}", flush=True)
        if ok_write and exists:
            abs_path = str(snapshot_path.resolve())
            os.environ["AQUILA_FACE_SNAPSHOT"] = abs_path
            print(f"[launch] Snapshot saved -> {abs_path}", flush=True)
        else:
            print("[launch] ERROR: snapshot not written to disk", flush=True)
    else:
        print("[launch] WARNING: could not capture snapshot after gate.", flush=True)

    # Señales para la UI (Aquila.py)
    os.environ["AQUILA_FACE_OK"] = "1"
    print(f"[launch] AQUILA_FACE_SNAPSHOT={os.environ.get('AQUILA_FACE_SNAPSHOT')}", flush=True)
    print("[launch] Launching Streamlit (Aquila.py)…", flush=True)

    # Lanza tu app en Streamlit
    Popen(
        [sys.executable, "-m", "streamlit", "run", "Aquila.py", "--server.headless=false"],
        env=os.environ
    )

    # Intento de abrir el navegador (Streamlit normalmente ya lo hace)
    time.sleep(1.5)
    try:
        webbrowser.open("http://localhost:8501", new=2)
    except Exception:
        pass


if __name__ == "__main__":
    main()

