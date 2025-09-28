# src/api/handler.py
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from joblib import load

MODEL_CANDIDATES = [
    Path("/opt/models/xgb.joblib"),
    Path("/opt/models/logreg.joblib"),
    Path("models/xgb.joblib"),
    Path("models/logreg.joblib"),
]
_model = None


def _load_model() -> Any:
    """Carrega o modelo treinado de caminhos conhecidos e faz cache em memoria."""
    global _model
    if _model is None:
        for candidate in MODEL_CANDIDATES:
            if candidate.exists():
                _model = load(candidate)
                break
        if _model is None:
            raise RuntimeError(
                "Modelo nao encontrado; configure /opt/models ou ./models com o joblib."
            )
    return _model


def _prepare_features(model: Any, feats: Dict[str, Any]) -> pd.DataFrame:
    """Monta DataFrame alinhado com as colunas usadas no treinamento."""
    X = pd.DataFrame([feats])
    expected = getattr(model, "selected_features_", None)
    if expected:
        X = X.reindex(columns=list(expected), fill_value=pd.NA)
    return X


def _decode_prediction(model: Any, pred: Any) -> Any:
    """Converte codigos inteiros de volta para o rotulo textual quando possivel."""
    inv_map = getattr(model, "inv_label_mapping_", None)
    if isinstance(inv_map, Dict):
        return inv_map.get(pred, pred)
    return pred


def handler(event, context):
    """Handler AWS Lambda para inferencia do risco semanal de deficit de energia."""
    model = _load_model()
    body = event.get("body") if isinstance(event, dict) else None
    if isinstance(body, str):
        body = json.loads(body or "{}")
    elif body is None:
        body = event if isinstance(event, dict) else {}

    if not isinstance(body, dict):
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "Payload invalido; envie JSON no campo 'body'."}),
        }

    feats = body.get("features", {})
    if not isinstance(feats, dict):
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "Campo 'features' deve ser um objeto JSON."}),
        }

    X = _prepare_features(model, feats)
    pred = model.predict(X)[0]
    decoded = _decode_prediction(model, pred)

    return {
        "statusCode": 200,
        "body": json.dumps({"classe_risco": decoded}),
    }
