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

## Render deploy

This repo includes `render.yaml` for Render Blueprint deployment.

Required env var:
- `DOC_SCANNER_ALLOWED_ORIGINS`
  - Example: `https://skdoosh-blog.netlify.app`
  - Local + prod: `https://skdoosh-blog.netlify.app,http://localhost:1313`
