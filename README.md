# doc-scanner-backend

FastAPI backend for document scanning (perspective correction + enhancement).

## Local run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## API

- `GET /healthz`
- `POST /scan` (multipart form with `file`)
  - Optional query params:
    - `enhance=true` to apply contrast enhancement (default: `false`)
    - `max_dimension=1600` to cap output resolution (default: `1800`)
    - `output=jpeg|png` (default: `jpeg`)
  - Default mode preserves original colors.

## Render deploy

This repo includes `render.yaml` for Render Blueprint deployment.

Required env var:
- `DOC_SCANNER_ALLOWED_ORIGINS`
  - Example: `https://skdoosh-blog.netlify.app`
  - Local + prod: `https://skdoosh-blog.netlify.app,http://localhost:1313`
