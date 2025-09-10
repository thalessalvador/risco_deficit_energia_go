# src/evaluate.py
from __future__ import annotations
from pathlib import Path
import yaml
import pandas as pd
from joblib import load
from sklearn.metrics import classification_report, confusion_matrix

from src.data_loader import load_all_sources
from src.feature_engineer import build_features_weekly
from src.train import rotular_semana, encode_labels

def main(config_path="configs/config.yaml", model_name="xgb"):
    """Carrega modelo, gera rótulos alinhados e produz relatório de avaliação.

    Args:
      config_path (str): Caminho para `configs/config.yaml`.
      model_name (str): Nome do modelo salvo em `models/` (ex.: "xgb").

    Returns:
      None
    """
    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    data = load_all_sources(cfg)
    Xw = build_features_weekly(data, cfg)
    Xw = Xw.dropna(axis=1, how="all")
    H = int(cfg.get("problem", {}).get("forecast_horizon_weeks", 1))
    y = rotular_semana(Xw, cfg, ref_df=Xw)
    if H > 0:
        y = y.shift(-H)
    idx_ok = y.dropna().index
    X = Xw.loc[idx_ok]
    y = y.loc[idx_ok]

    model = load(Path(cfg["paths"]["models_dir"]) / f"{model_name}.joblib")
    pred = model.predict(X)

    # Tenta usar mapeamento persistido no modelo; fallback para padrão
    lbl_map = getattr(model, "label_mapping_", {"baixo": 0, "medio": 1, "alto": 2})
    inv_map = getattr(model, "inv_label_mapping_", {0: "baixo", 1: "medio", 2: "alto"})

    # Encoda y verdadeiro para inteiros, garantindo consistência com o modelo
    try:
        y_enc = encode_labels(y)
    except Exception:
        # Fallback para .map caso o encode rígido falhe por algum motivo
        y_enc = y.map(lbl_map)
    label_order = [0, 1, 2]
    target_names = [inv_map[i] for i in label_order]

    rep = classification_report(y_enc, pred, labels=label_order, target_names=target_names, digits=3)
    cm  = confusion_matrix(y_enc, pred, labels=label_order)

    rep_dir = Path(cfg["paths"]["reports_dir"]); rep_dir.mkdir(parents=True, exist_ok=True)
    with open(rep_dir / f"report_{model_name}.txt","w",encoding="utf-8") as f:
        f.write(rep + "\n")
        f.write("Matriz de confusão (linhas=verdadeiro, colunas=previsto):\n")
        f.write(pd.DataFrame(cm, index=target_names, columns=target_names).to_string())

    print(rep)
    print("\nMatriz de confusão:\n", cm)

if __name__ == "__main__":
    main()
