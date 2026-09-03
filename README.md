# LectorBilletes API

API visual orientativa para estimar si un billete chileno podría ser canjeable.
Usa OpenAI Vision; no contiene modelos PyTorch y no autentifica billetes.

## Desarrollo

```bash
python -m venv .venv
pip install -r requirements.txt
uvicorn main:app --reload
```

Copia `.env.example` como `.env` y configura `OPENAI_API_KEY`.

## Railway

Configura `OPENAI_API_KEY`, `OPENAI_MODEL` y `FRONTEND_ORIGINS`. El `Procfile`
inicia Uvicorn usando el puerto entregado por Railway.

## Endpoints

- `GET /health`
- `POST /analyze` con `front_image_base64` y, opcionalmente, `back_image_base64`.
- `POST /predict` se conserva como alias temporal.

La respuesta utiliza `POTENCIALMENTE_CANJEABLE`,
`POTENCIALMENTE_NO_CANJEABLE` o `REQUIERE_REVISION_PRESENCIAL` y siempre
incluye una advertencia de que el resultado no garantiza el canje.
