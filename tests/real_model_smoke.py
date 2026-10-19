from __future__ import annotations

import io
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PORT = int(os.environ.get("YANZHI_SMOKE_PORT", "8813"))
BASE_URL = f"http://127.0.0.1:{PORT}"
ENV = os.environ.copy()
ENV["MODEL_PATH"] = str(ROOT / "server" / "models" / "face_landmarker.task")
ENV["INDEX_PATH"] = str(ROOT / "index.html")

def main() -> None:
    process = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn", "server.app:app",
            "--host", "127.0.0.1", "--port", str(PORT),
        ],
        cwd=ROOT,
        env=ENV,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.monotonic() + 90
        health = None
        while time.monotonic() < deadline:
            if process.poll() is not None:
                output = process.stdout.read() if process.stdout else ""
                raise RuntimeError(f"Real-model service exited during startup.\n{output}")
            try:
                health = httpx.get(f"{BASE_URL}/health", timeout=2)
                if health.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(1)
        if health is None or health.status_code != 200:
            output = process.stdout.read() if process.stdout else ""
            raise RuntimeError(f"Real-model service did not become healthy within 90 seconds.\n{output}")
        health_data = health.json()
        assert health_data.get("offline") is True, health_data
        assert health_data.get("model") == "face_landmarker.task", health_data
        print(f"Real model loaded: {health_data}")

        image_buffer = io.BytesIO()
        Image.new("RGB", (320, 320), (127, 127, 127)).save(image_buffer, format="PNG")
        response = httpx.post(
            f"{BASE_URL}/api/analyze",
            files={"photo": ("blank.png", image_buffer.getvalue(), "image/png")},
            timeout=60,
        )
        assert response.status_code == 422, (
            f"Expected blank image to be rejected as no-face (422), got "
            f"{response.status_code}: {response.text[:500]}"
        )
        assert "未检测到" in response.json().get("detail", ""), response.text
        print("Real-model inference correctly rejected a no-face image (HTTP 422).")
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        if process.stdout:
            process.stdout.close()

if __name__ == "__main__":
    main()
