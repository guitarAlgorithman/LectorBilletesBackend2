# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
from io import BytesIO
import base64
import json
import os
import time

from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

# ======================
# CONFIG (env-friendly)
# ======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
UMBRAL_NINGUNA = float(os.getenv("UMBRAL_NINGUNA", "0.6"))

MODEL_DENOM_PATH = Path(os.getenv("MODEL_DENOM_PATH", "modelo_denom.pt"))

APTITUD_MODELS = {
    "1000": Path(os.getenv("MODEL_APT_1000", "modelo_aptitud_1000.pt")),
    "2000": Path(os.getenv("MODEL_APT_2000", "modelo_aptitud_2000.pt")),
    "5000": Path(os.getenv("MODEL_APT_5000", "modelo_aptitud_5000.pt")),
    "10000": Path(os.getenv("MODEL_APT_10000", "modelo_aptitud_10000.pt")),
    "20000": Path(os.getenv("MODEL_APT_20000", "modelo_aptitud_20000.pt")),
}
DENOMS_VALIDAS = list(APTITUD_MODELS.keys())

# OpenAI prompt/model desde env
OPENAI_VISION_PROMPT = os.getenv("OPENAI_VISION_PROMPT", "").strip()
OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini").strip()
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "300"))

# Healthcheck OpenAI (opcional)
HEALTHCHECK_OPENAI = os.getenv("HEALTHCHECK_OPENAI", "0").strip() == "1"
OPENAI_HEALTH_MODEL = os.getenv("OPENAI_HEALTH_MODEL", "gpt-4o-mini").strip()
OPENAI_HEALTH_TIMEOUT_S = int(os.getenv("OPENAI_HEALTH_TIMEOUT_S", "8"))

# ======================
# APP + CORS
# ======================
app = FastAPI(title="Detector de Billetes - 2 etapas + OpenAI Vision + Health")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================
# OPENAI CLIENT
# ======================
openai_client = OpenAI()  # requiere OPENAI_API_KEY en env

def openai_vision_bill_json(image_base64: str) -> dict:
    """
    Llama OpenAI Vision y devuelve:
      {"ok": True, "data": <dict>, "raw": <str>, "mode": ...}
    o
      {"ok": False, "error": "..."}
    """
    if not OPENAI_VISION_PROMPT:
        return {"ok": False, "error": "OPENAI_VISION_PROMPT no está configurado en el entorno"}

    data_url = f"data:image/jpeg;base64,{image_base64}"
    prompt = OPENAI_VISION_PROMPT

    schema = {
        "name": "billete_chile",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "descripcion": {"type": "string"},
                "es_billete": {"type": "boolean"},
                "denominacion_estimada": {
                    "type": "string",
                    "enum": ["1000", "2000", "5000", "10000", "20000", "ninguna"],
                },
                "motivos": {"type": "string"},
            },
            "required": ["descripcion", "es_billete", "denominacion_estimada", "motivos"],
        },
    }

    err_responses = None

    # ---------- Intento 1: Responses API + json_schema ----------
    try:
        if hasattr(openai_client, "responses"):
            r = openai_client.responses.create(
                model=OPENAI_VISION_MODEL,
                response_format={"type": "json_schema", "json_schema": schema},
                input=[{
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": data_url},
                    ],
                }],
                max_output_tokens=OPENAI_MAX_TOKENS,
            )
            raw = (getattr(r, "output_text", "") or "").strip()
            if not raw:
                return {"ok": False, "error": "OpenAI devolvió vacío (responses)", "raw": raw, "mode": "responses_json_schema"}

            # Con json_schema debería ser parseable
            parsed = json.loads(raw)
            return {"ok": True, "data": parsed, "raw": raw, "mode": "responses_json_schema"}

    except Exception as e:
        err_responses = f"{type(e).__name__}: {e}"

    # ---------- Intento 2: Chat Completions + json_object (fallback) ----------
    try:
        r = openai_client.chat.completions.create(
            model=OPENAI_VISION_MODEL,
            response_format={"type": "json_object"},
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }],
            max_tokens=OPENAI_MAX_TOKENS,
        )

        raw = (r.choices[0].message.content or "").strip()
        if not raw:
            return {
                "ok": False,
                "error": f"OpenAI devolvió vacío (chat). responses_error={err_responses}",
                "raw": raw,
                "mode": "chat_json_object",
            }

        parsed = json.loads(raw)

        # filtro defensivo: solo 4 llaves
        filtered = {
            "descripcion": str(parsed.get("descripcion", "")),
            "es_billete": bool(parsed.get("es_billete", False)),
            "denominacion_estimada": parsed.get("denominacion_estimada", "ninguna"),
            "motivos": str(parsed.get("motivos", "")),
        }
        if filtered["denominacion_estimada"] not in ["1000", "2000", "5000", "10000", "20000", "ninguna"]:
            filtered["denominacion_estimada"] = "ninguna"

        return {"ok": True, "data": filtered, "raw": raw, "mode": f"chat_json_object(responses_error={err_responses})"}

    except Exception as e:
        err_chat = f"{type(e).__name__}: {e}"
        return {"ok": False, "error": f"OpenAI failed. responses_error={err_responses} | chat_error={err_chat}"}

# ======================
# TRANSFORMS
# ======================
infer_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# ======================
# LOAD MODELS (startup)
# ======================
denom_model = None
class_names_denom = []

print(f"📌 Cargando modelo de denominación en {DEVICE}...")
try:
    if not MODEL_DENOM_PATH.exists():
        raise FileNotFoundError(f"No existe {MODEL_DENOM_PATH}")

    ckpt_denom = torch.load(MODEL_DENOM_PATH, map_location=DEVICE)
    class_names_denom = ckpt_denom["classes"]

    denom_model = models.resnet18(weights=None)
    denom_model.fc = nn.Linear(denom_model.fc.in_features, len(class_names_denom))
    denom_model.load_state_dict(ckpt_denom["state_dict"])
    denom_model.to(DEVICE)
    denom_model.eval()

    print("✔️ Modelo de denominación cargado.")
    print("   Clases:", class_names_denom)

except Exception as e:
    print(f"❌ Error cargando modelo de denominación: {type(e).__name__}: {e}")

apto_models = {}

def load_apto_model(weights_path: Path):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    state = torch.load(weights_path, map_location=DEVICE)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model

print("📌 Cargando modelos de aptitud...")
for denom, path in APTITUD_MODELS.items():
    try:
        if path.exists():
            apto_models[denom] = load_apto_model(path)
            print(f"   ✔ {denom} -> {path}")
        else:
            print(f"   ⚠ NO se encontró modelo de aptitud para {denom}: {path}")
    except Exception as e:
        print(f"   ❌ Error cargando aptitud {denom} ({path}): {type(e).__name__}: {e}")
print("✔️ Modelos de aptitud listos (si existen).")

# ======================
# SCHEMAS
# ======================
class PredictRequest(BaseModel):
    image_base64: str

# ======================
# HELPERS
# ======================
def decode_image_from_base64(b64_str: str) -> Image.Image:
    img_bytes = base64.b64decode(b64_str)
    return Image.open(BytesIO(img_bytes)).convert("RGB")

def infer_denominacion(img: Image.Image):
    """
    Devuelve:
      denom_final: '1000'...'20000' o 'ninguna'
      denom_pred: salida cruda
      prob_max: confianza de la clase predicha
      probs: lista probs
    """
    if denom_model is None or not class_names_denom:
        raise RuntimeError("Modelo de denominación no está cargado")

    x = infer_tf(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = denom_model(x)
        probs = torch.softmax(logits, dim=1)[0]

    prob_max, idx = torch.max(probs, dim=0)
    prob_max = prob_max.item()
    denom_pred = class_names_denom[idx.item()]

    denom_final = "ninguna" if (denom_pred == "desconocido" or prob_max < UMBRAL_NINGUNA) else denom_pred
    return denom_final, denom_pred, prob_max, probs.cpu().numpy().tolist()

def infer_aptitud(img: Image.Image, denom: str):
    if denom not in apto_models:
        return None, None, None

    model = apto_models[denom]
    x = infer_tf(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    prob_max, idx = torch.max(probs, dim=0)
    prob_max = prob_max.item()

    # IMPORTANTE: tu convención fue idx 0 = apto, idx 1 = no_apto
    clase = "apto" if idx.item() == 0 else "no_apto"
    return clase, prob_max, probs.cpu().numpy().tolist()

# ======================
# HEALTH HELPERS
# ======================
def _check_models_loaded() -> dict:
    out = {
        "denom_model": {
            "loaded": denom_model is not None,
            "path": str(MODEL_DENOM_PATH),
            "exists": MODEL_DENOM_PATH.exists(),
            "classes": class_names_denom,
        },
        "aptitud_models": {},
        "all_required_apt_loaded": True,
        "any_apt_loaded": False,
    }

    for denom, path in APTITUD_MODELS.items():
        loaded = denom in apto_models
        exists = path.exists()
        out["aptitud_models"][denom] = {
            "loaded": loaded,
            "path": str(path),
            "exists": exists,
        }
        if loaded:
            out["any_apt_loaded"] = True
        if not exists or (exists and not loaded):
            out["all_required_apt_loaded"] = False

    return out

def _check_openai() -> dict:
    key_present = bool(os.getenv("OPENAI_API_KEY", "").strip())
    out = {
        "api_key_present": key_present,
        "prompt_configured": bool(OPENAI_VISION_PROMPT),
        "model": OPENAI_VISION_MODEL,
        "has_responses_api": hasattr(openai_client, "responses"),
        "deep_check_enabled": HEALTHCHECK_OPENAI,
        "ok": False,
    }

    if not key_present:
        out["error"] = "OPENAI_API_KEY no está configurada"
        return out

    # superficial OK si hay key
    out["ok"] = True

    if not HEALTHCHECK_OPENAI:
        return out

    # deep check: llamada mínima (barata)
    t0 = time.time()
    try:
        if hasattr(openai_client, "responses"):
            r = openai_client.responses.create(
                model=OPENAI_HEALTH_MODEL,
                input="Responde SOLO 'ok'.",
                max_output_tokens=5,
            )
            txt = (getattr(r, "output_text", "") or "").strip().lower()
            out["latency_ms"] = int((time.time() - t0) * 1000)
            out["ok"] = ("ok" in txt)
            out["mode"] = "responses"
            out["raw"] = txt[:50]
            if not out["ok"]:
                out["error"] = "Respuesta inesperada (deep check responses)"
            return out

        r = openai_client.chat.completions.create(
            model=OPENAI_HEALTH_MODEL,
            messages=[{"role": "user", "content": "Responde SOLO 'ok'."}],
            max_tokens=5,
        )
        txt = (r.choices[0].message.content or "").strip().lower()
        out["latency_ms"] = int((time.time() - t0) * 1000)
        out["ok"] = ("ok" in txt)
        out["mode"] = "chat.completions"
        out["raw"] = txt[:50]
        if not out["ok"]:
            out["error"] = "Respuesta inesperada (deep check chat)"
        return out

    except Exception as e:
        out["latency_ms"] = int((time.time() - t0) * 1000)
        out["ok"] = False
        out["error"] = f"{type(e).__name__}: {e}"
        return out

# ======================
# ENDPOINTS
# ======================
@app.get("/")
def root():
    return {"info": "API billetes 2 etapas OK + OpenAI + Health"}

@app.get("/health")
def health():
    models_status = _check_models_loaded()
    openai_status = _check_openai()

    # criterio: denom cargado + existe + al menos 1 aptitud cargado
    ok_models = (
        models_status["denom_model"]["loaded"]
        and models_status["denom_model"]["exists"]
        and models_status["any_apt_loaded"]
    )
    ok = ok_models and openai_status["ok"]

    return {
        "ok": ok,
        "device": DEVICE,
        "umbral_ninguna": UMBRAL_NINGUNA,
        "models": models_status,
        "openai": openai_status,
    }

@app.get("/debug-models")
def debug_models():
    return {
        "device": DEVICE,
        "umbral_ninguna": UMBRAL_NINGUNA,
        "openai": {
            "model": OPENAI_VISION_MODEL,
            "prompt_configurado": bool(OPENAI_VISION_PROMPT),
            "prompt_len": len(OPENAI_VISION_PROMPT),
            "prompt_head": OPENAI_VISION_PROMPT[:80],
            "has_responses_api": hasattr(openai_client, "responses"),
        },
        "modelo_denominacion": {
            "ruta": str(MODEL_DENOM_PATH),
            "exists": MODEL_DENOM_PATH.exists(),
            "clases": class_names_denom,
            "estado": "cargado" if denom_model is not None else "no cargado",
        },
        "modelos_aptitud": {
            denom: {"cargado": (denom in apto_models), "ruta": str(path), "exists": path.exists()}
            for denom, path in APTITUD_MODELS.items()
        },
    }

@app.post("/predict")
def predict_final(req: PredictRequest):
    # 1) decode
    try:
        img = decode_image_from_base64(req.image_base64)
    except Exception:
        oa = openai_vision_bill_json(req.image_base64)
        return {
            "denominacion": "ninguna",
            "apto": None,
            "confianza": 0.0,
            "detalle": "Base64 inválido o imagen corrupta",
            "openai": oa,
        }

    # 2) denom
    try:
        denom_final, denom_model_out, prob_denom, probs_denom = infer_denominacion(img)
    except Exception as e:
        oa = openai_vision_bill_json(req.image_base64)
        return {
            "denominacion": "ninguna",
            "apto": None,
            "confianza": 0.0,
            "detalle": f"Error en modelo de denominación: {type(e).__name__}: {e}",
            "openai": oa,
        }

    # 3) aptitud
    apto_bool = None
    confianza = float(prob_denom)
    detalle = "OK"

    if denom_final == "ninguna" or denom_final not in DENOMS_VALIDAS:
        detalle = "No se detectó una denominación válida"
    else:
        clase_apto, prob_apto, _ = infer_aptitud(img, denom_final)
        if clase_apto is None:
            detalle = "Modelo de aptitud no disponible para esta denominación"
        else:
            apto_bool = (clase_apto == "apto")
            confianza = float(prob_apto)

    # 4) openai always
    oa = openai_vision_bill_json(req.image_base64)

    return {
        "denominacion": denom_final,
        "apto": apto_bool,
        "confianza": confianza,
        "detalle": detalle,

        # debug (déjalo mientras pruebas; bórralo si quieres)
        "denominacion_modelo": denom_model_out,
        "prob_denom": float(prob_denom),
        "probs_denom": probs_denom,

        "openai": oa,
    }
