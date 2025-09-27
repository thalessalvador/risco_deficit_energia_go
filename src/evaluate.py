# src/evaluate.py
from __future__ import annotations
from pathlib import Path
import yaml
import pandas as pd
from joblib import load
from sklearn.metrics import classification_report, confusion_matrix

from src.data_loader import load_all_sources
from src.feature_engineer import build_features_weekly
from src.train import (
    rotular_semana,
    encode_labels,
    rotular_semana_com_thresholds,
    _normalize_pred_to_domain,
    _postprocess_with_hard_rules,
)


def main(config_path="configs/config.yaml", model_name="xgb"):
    """Carrega modelo, gera rótulos alinhados e produz relatório de avaliação.

    Args:
      config_path (str): Caminho para `configs/config.yaml`.
      model_name (str): Nome do modelo salvo em `models/` (ex.: "xgb").

    Returns:
      None
    """
    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    feat_dir = Path(cfg["paths"]["features_dir"])
    test_path = feat_dir / "features_test_holdout.parquet"
    train_path = feat_dir / "features_trainval.parquet"
    if test_path.exists() and train_path.exists():
        Xw = pd.read_parquet(test_path)
        train_ref = pd.read_parquet(train_path)
        print(f"[eval] Avaliando no conjunto hold-out: {len(Xw)} semanas.")
    else:
        # Fallback para avaliação in-sample caso o hold-out não exista
        data = load_all_sources(cfg)
        Xw = build_features_weekly(data, cfg)
        train_ref = Xw
        print(
            "[eval] Aviso: hold-out não encontrado; avaliando no dataset completo (in-sample)."
        )
    Xw = Xw.dropna(axis=1, how="all")
    H = int(cfg.get("problem", {}).get("forecast_horizon_weeks", 1))

    model = load(Path(cfg["paths"]["models_dir"]) / f"{model_name}.joblib")
    thr = getattr(model, "label_thresholds_", None)
    if thr is not None:
        # Usa thresholds persistidos do treino
        y = rotular_semana_com_thresholds(Xw, cfg, thr)
    else:
        # Fallback: usa o conjunto de treino como referência para quantis
        y = rotular_semana(Xw, cfg, ref_df=train_ref)

    if H > 0:
        y = y.shift(-H)
    idx_ok = y.dropna().index
    X = Xw.loc[idx_ok]
    y = y.loc[idx_ok]

    pred_raw = model.predict(X)
    pred = _normalize_pred_to_domain(pred_raw, model)
    pred = _postprocess_with_hard_rules(pred, X, cfg)

    lbl_map = getattr(model, "label_mapping_", {"baixo": 0, "medio": 1, "alto": 2})
    inv_map = getattr(model, "inv_label_mapping_", {0: "baixo", 1: "medio", 2: "alto"})

    try:
        y_enc = encode_labels(y)
    except Exception:
        y_enc = y.map(lbl_map)

    try:
        pred_enc = encode_labels(pred)
    except Exception:
        pred_enc = pred.map(lbl_map)

    label_order = [0, 1, 2]
    target_names = [inv_map[i] for i in label_order]

    rep = classification_report(
        y_enc, pred_enc, labels=label_order, target_names=target_names, digits=3
    )
    cm = confusion_matrix(y_enc, pred_enc, labels=label_order)
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)

    rep_dir = Path(cfg["paths"]["reports_dir"])
    rep_dir.mkdir(parents=True, exist_ok=True)
    with open(rep_dir / f"report_{model_name}.txt", "w", encoding="utf-8") as f:
        f.write(rep + "\n")
        f.write("Matriz de confusão (linhas=verdadeiro, colunas=previsto):\n")
        f.write(pd.DataFrame(cm, index=target_names, columns=target_names).to_string())

    print(rep)
    print("\nMatriz de confusão (linhas=verdadeiro, colunas=previsto):")
    print(cm_df.to_string())


if __name__ == "__main__":
    main()
