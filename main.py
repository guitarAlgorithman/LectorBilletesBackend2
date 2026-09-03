import base64
import binascii
import json
import os
import time
from collections import defaultdict, deque
from io import BytesIO
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import OpenAI
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field, model_validator

load_dotenv()

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(8 * 1024 * 1024)))
MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "20000000"))
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))
FRONTEND_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_ORIGINS", os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")
    ).split(",")
    if origin.strip()
]

DISCLAIMER = (
    "Estimación automatizada basada únicamente en características visibles. "
    "No autentifica el billete ni garantiza su aceptación o canje. La decisión "
    "definitiva corresponde al Banco Central de Chile o a la entidad receptora."
)

ANALYSIS_INSTRUCTIONS = """
Eres un asistente de evaluación VISUAL y ORIENTATIVA de billetes chilenos. No
autentificas billetes y nunca garantizas el canje. Analiza solo lo observable en
las fotografías y aplica estos criterios del instructivo de clasificación del
Banco Central de Chile:

- POTENCIALMENTE_CANJEABLE: parece un billete chileno identificable; conserva
  aparentemente más del 50 % de su superficie en una sola pieza y el daño
  visible permite su revisión. Puede estar desgastado, rasgado, reparado,
  escrito, manchado o agujereado: esos daños pueden hacerlo no apto para
  circular, pero no necesariamente no canjeable.
- POTENCIALMENTE_NO_CANJEABLE: hay evidencia visual clara de reconstrucción con
  partes de billetes diferentes, entintado de seguridad, tratamiento para
  retirar esa tinta, o no se conserva una porción identificable suficiente.
- REQUIERE_REVISION_PRESENCIAL: fotografía insuficiente, duda relevante,
  quemadura severa, hongos, contaminación, piezas múltiples, superficie cercana
  al 50 %, posible falsificación o cualquier condición imposible de resolver
  visualmente.

No confundas "no apto para circular" con "no canjeable". No evalúes rigidez,
tacto, fluorescencia, contaminación ni autenticidad. Si falta el reverso, puedes
analizar el anverso, pero baja la confianza si la decisión depende de una zona no
visible. Describe evidencia concreta y evita afirmaciones categóricas.
""".strip()

RESULT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "parece_billete_chileno": {"type": "boolean"},
        "denominacion_estimada": {
            "type": "string",
            "enum": ["1000", "2000", "5000", "10000", "20000", "desconocida"],
        },
        "resultado": {
            "type": "string",
            "enum": [
                "POTENCIALMENTE_CANJEABLE",
                "POTENCIALMENTE_NO_CANJEABLE",
                "REQUIERE_REVISION_PRESENCIAL",
            ],
        },
        "confianza_visual": {"type": "string", "enum": ["alta", "media", "baja"]},
        "danos_visibles": {"type": "array", "items": {"type": "string"}},
        "evidencia": {"type": "array", "items": {"type": "string"}},
        "motivo": {"type": "string"},
        "requiere_revision_presencial": {"type": "boolean"},
    },
    "required": [
        "parece_billete_chileno", "denominacion_estimada", "resultado",
        "confianza_visual", "danos_visibles", "evidencia", "motivo",
        "requiere_revision_presencial",
    ],
}


class AnalyzeRequest(BaseModel):
    front_image_base64: str | None = Field(default=None, min_length=32)
    back_image_base64: str | None = Field(default=None, min_length=32)
    image_base64: str | None = Field(default=None, min_length=32)

    @model_validator(mode="after")
    def require_an_image(self):
        if not (self.front_image_base64 or self.image_base64):
            raise ValueError("Debes enviar front_image_base64 o image_base64")
        return self


class AnalysisResult(BaseModel):
    parece_billete_chileno: bool
    denominacion_estimada: Literal["1000", "2000", "5000", "10000", "20000", "desconocida"]
    resultado: Literal[
        "POTENCIALMENTE_CANJEABLE", "POTENCIALMENTE_NO_CANJEABLE",
        "REQUIERE_REVISION_PRESENCIAL",
    ]
    confianza_visual: Literal["alta", "media", "baja"]
    danos_visibles: list[str]
    evidencia: list[str]
    motivo: str
    requiere_revision_presencial: bool
    es_estimacion_visual: bool = True
    autenticidad_evaluada: bool = False
    canje_garantizado: bool = False
    advertencia_legal: str = DISCLAIMER


app = FastAPI(
    title="Orientador visual de canje de billetes chilenos",
    version="2.0.0",
    description="Evaluación visual orientativa mediante OpenAI; no autentifica ni garantiza canje.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

_requests_by_ip: dict[str, deque[float]] = defaultdict(deque)


@app.middleware("http")
async def simple_rate_limit(request: Request, call_next):
    if request.url.path not in {"/analyze", "/predict"}:
        return await call_next(request)
    now = time.monotonic()
    ip = request.client.host if request.client else "unknown"
    bucket = _requests_by_ip[ip]
    while bucket and now - bucket[0] > 60:
        bucket.popleft()
    if len(bucket) >= RATE_LIMIT_PER_MINUTE:
        return JSONResponse(
            status_code=429,
            content={"detail": "Demasiadas consultas. Intenta nuevamente en un minuto."},
        )
    bucket.append(now)
    return await call_next(request)


def _openai_client() -> OpenAI:
    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY no está configurada")
    return OpenAI()


def _normalise_image(value: str, label: str) -> str:
    encoded = value.split(",", 1)[1] if value.startswith("data:") and "," in value else value
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"{label}: base64 inválido") from exc
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"{label}: la imagen excede el máximo de {MAX_IMAGE_BYTES // (1024 * 1024)} MB",
        )
    try:
        with Image.open(BytesIO(raw)) as image:
            image.load()
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise HTTPException(status_code=413, detail=f"{label}: demasiados píxeles")
            image = image.convert("RGB")
            image.thumbnail((2400, 2400))
            output = BytesIO()
            image.save(output, format="JPEG", quality=88, optimize=True)
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=422, detail=f"{label}: archivo de imagen inválido") from exc
    return base64.b64encode(output.getvalue()).decode("ascii")


def _analyse_with_openai(front: str, back: str | None) -> AnalysisResult:
    content = [
        {"type": "input_text", "text": "Analiza el anverso y, si existe, el reverso del mismo billete."},
        {"type": "input_text", "text": "ANVERSO:"},
        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{front}", "detail": "high"},
    ]
    if back:
        content.extend([
            {"type": "input_text", "text": "REVERSO:"},
            {"type": "input_image", "image_url": f"data:image/jpeg;base64,{back}", "detail": "high"},
        ])
    try:
        response = _openai_client().responses.create(
            model=OPENAI_MODEL,
            instructions=ANALYSIS_INSTRUCTIONS,
            input=[{"role": "user", "content": content}],
            text={"format": {
                "type": "json_schema", "name": "evaluacion_canje_billete",
                "strict": True, "schema": RESULT_SCHEMA,
            }},
            max_output_tokens=700,
            store=False,
        )
        return AnalysisResult(**json.loads(response.output_text))
    except HTTPException:
        raise
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=502, detail="OpenAI devolvió un resultado no válido") from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="No fue posible completar el análisis visual") from exc


@app.get("/")
def root():
    return {"service": "Orientador visual de canje de billetes chilenos", "version": "2.0.0", "docs": "/docs"}


@app.get("/health")
def health():
    configured = bool(os.getenv("OPENAI_API_KEY", "").strip())
    return {"ok": configured, "model": OPENAI_MODEL, "openai_key_configured": configured}


@app.post("/analyze", response_model=AnalysisResult)
@app.post("/predict", response_model=AnalysisResult, include_in_schema=False)
def analyse_bill(payload: AnalyzeRequest):
    front = _normalise_image(payload.front_image_base64 or payload.image_base64 or "", "anverso")
    back = _normalise_image(payload.back_image_base64, "reverso") if payload.back_image_base64 else None
    return _analyse_with_openai(front, back)
