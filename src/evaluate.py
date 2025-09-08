# src/evaluate.py
from __future__ import annotations
from pathlib import Path
import yaml
import pandas as pd
from joblib import load
from sklearn.metrics import classification_report, confusion_matrix

from data_loader import load_all_sources
from feature_engineer import build_features_weekly
from train import rotular_semana

def main(config_path="configs/config.yaml", model_name="xgb"):
    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    data = load_all_sources(cfg)
    Xw = build_features_weekly(data, cfg).ffill().bfill()
    y = rotular_semana(Xw, cfg)

    model = load(Path(cfg["paths"]["models_dir"]) / f"{model_name}.joblib")
    pred = model.predict(Xw)

    rep = classification_report(y, pred, target_names=["baixo","medio","alto"], labels=["baixo","medio","alto"], digits=3)
    cm  = confusion_matrix(y, pred, labels=["baixo","medio","alto"])

    rep_dir = Path(cfg["paths"]["reports_dir"]); rep_dir.mkdir(parents=True, exist_ok=True)
    with open(rep_dir / f"report_{model_name}.txt","w",encoding="utf-8") as f:
        f.write(rep + "\n")
        f.write("Matriz de confusão (linhas=verdadeiro, colunas=previsto):\n")
        f.write(pd.DataFrame(cm, index=["baixo","medio","alto"], columns=["baixo","medio","alto"]).to_string())

    print(rep)
    print("\nMatriz de confusão:\n", cm)

if __name__ == "__main__":
    main()
