# src/api/handler.py
import json
from pathlib import Path
from joblib import load
import pandas as pd

MODEL_CANDIDATES = [Path("/opt/models/xgb.joblib"), Path("models/xgb.joblib")]
_model = None

def _load_model():
    """Carrega o modelo treinado de caminhos conhecidos e faz cache em memória.

    Returns:
      Any: Objeto de modelo carregado via joblib.
    """
    global _model
    if _model is None:
        for p in MODEL_CANDIDATES:
            if p.exists():
                _model = load(p)
                break
        if _model is None:
            raise RuntimeError("Modelo não encontrado em /opt/models ou ./models")
    return _model

def handler(event, context):
    """
    Espera JSON:
      {"features": { "margem_suprimento_w": 1.23, "ghi_mean_w": 5.4, ... }}
    Retorna:
      {"classe_risco": "baixo|medio|alto"}
    """
    model = _load_model()
    body = event.get("body")
    if isinstance(body, str):
        body = json.loads(body)
    feats = body.get("features", {})
    import pandas as pd
    X = pd.DataFrame([feats])
    pred = model.predict(X)[0]
    return {"statusCode": 200, "body": json.dumps({"classe_risco": pred})}
