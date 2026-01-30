import io

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image

from app.scanner import scan_document

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["https://yourblog.com"]
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/scan")
async def scan(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        np_img = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(
                status_code=400, detail="Uploaded file could not be decoded as an image"
            )

        scanned = scan_document(image)

        pil_img = Image.fromarray(scanned)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        buf.seek(0)

        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
