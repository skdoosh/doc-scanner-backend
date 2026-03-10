import io
import os

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image

from app.scanner import scan_document

app = FastAPI()


def _read_allowed_origins() -> list[str]:
    # Comma-separated origins, e.g.:
    # "https://skdoosh-blog.netlify.app,http://localhost:1313"
    raw = os.getenv("DOC_SCANNER_ALLOWED_ORIGINS", "*")
    origins = [item.strip() for item in raw.split(",") if item.strip()]
    return origins or ["*"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_read_allowed_origins(),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "service": "doc-scanner-backend"}


@app.post("/scan")
async def scan(
    file: UploadFile = File(...),
    enhance: bool = False,
    max_dimension: int = Query(default=1800, ge=800, le=4000),
    output: str = Query(default="jpeg"),
):
    try:
        contents = await file.read()
        np_img = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(
                status_code=400, detail="Uploaded file could not be decoded as an image"
            )

        scanned = scan_document(
            image,
            enhance=enhance,
            max_output_dimension=max_dimension,
        )

        # OpenCV uses BGR; Pillow expects RGB.
        rgb_scanned = cv2.cvtColor(scanned, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_scanned)
        buf = io.BytesIO()
        fmt = output.lower().strip()
        if fmt not in {"png", "jpeg", "jpg"}:
            raise HTTPException(status_code=400, detail="output must be png or jpeg")

        if fmt == "png":
            pil_img.save(buf, format="PNG", optimize=True)
            media_type = "image/png"
        else:
            pil_img.save(buf, format="JPEG", quality=90, optimize=True)
            media_type = "image/jpeg"
        buf.seek(0)

        return StreamingResponse(buf, media_type=media_type)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
