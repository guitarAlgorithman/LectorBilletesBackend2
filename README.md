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
inicia Uvicorn usando el puerto entregado por Railway. `railway.json` declara el
comando de inicio, el healthcheck `/health` y la política de reinicio.

Variables mínimas:

```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
FRONTEND_ORIGINS=https://tu-frontend.up.railway.app
```

Después del primer despliegue, genera el dominio público del backend y úsalo en
`VITE_API_URL` del servicio frontend. Cuando Railway genere el dominio del
frontend, vuelve aquí y configura ese dominio exacto en `FRONTEND_ORIGINS`.

## Endpoints

- `GET /health`
- `POST /analyze` con `front_image_base64` y, opcionalmente, `back_image_base64`.
- `POST /predict` se conserva como alias temporal.

La respuesta utiliza `POTENCIALMENTE_CANJEABLE`,
`POTENCIALMENTE_NO_CANJEABLE` o `REQUIERE_REVISION_PRESENCIAL` y siempre
incluye una advertencia de que el resultado no garantiza el canje.
