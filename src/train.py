# src/train.py
from __future__ import annotations
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from joblib import dump

from data_loader import load_all_sources
from feature_engineer import build_features_weekly

def rotular_semana(df: pd.DataFrame, cfg) -> pd.Series:
    r = cfg["problem"]["label_rules"]
    margem_col = r["coluna_margem"]
    q_baixo, q_med = r["q_baixo"], r["q_medio"]

    if margem_col not in df.columns:
        raise ValueError(f"Coluna '{margem_col}' não encontrada para rotulagem.")

    margem = df[margem_col].copy()
    thr_baixo = margem.quantile(q_baixo)
    thr_med   = margem.quantile(q_med)

    y = pd.Series(index=df.index, dtype=object)
    y[margem <= thr_baixo] = "alto"
    y[(margem > thr_baixo) & (margem <= thr_med)] = "medio"
    y[margem > thr_med] = "baixo"

    # Override por cortes
    if r.get("usar_override_cortes", True):
        cortes_cols = [c for c in r["colunas_corte"] if c in df.columns]
        if cortes_cols:
            corte_total = df[cortes_cols].sum(axis=1)
            y[corte_total > r.get("corte_thr_mwh", 0)] = "alto"

    # Override hidrológico (estiagem)
    if r.get("usar_override_hidro", True):
        ear_col = r.get("coluna_ear")
        ena_col = r.get("coluna_ena")
        if ear_col in df.columns:
            ear_thr = df[ear_col].quantile(r.get("ear_q_baixo", 0.2))
            y[df[ear_col] <= ear_thr] = "alto"
        if ena_col in df.columns:
            ena_thr = df[ena_col].quantile(r.get("ena_q_baixo", 0.2))
            k = int(r.get("janelas_consecutivas_ena", 2))
            ena_low = (df[ena_col] <= ena_thr).rolling(k).sum() >= k
            y[ena_low.fillna(False)] = "alto"

    return y

def make_model(model_cfg):
    if model_cfg["type"] == "logistic_regression":
        return Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(**model_cfg["params"]))
        ])
    elif model_cfg["type"] == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(**model_cfg["params"])
    else:
        raise ValueError(f"Modelo não suportado: {model_cfg['type']}")

def main(config_path="configs/config.yaml"):
    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    out_dir = Path(cfg["paths"]["models_dir"]); out_dir.mkdir(parents=True, exist_ok=True)

    data = load_all_sources(cfg)
    Xw = build_features_weekly(data, cfg)
    Xw = Xw.dropna(axis=1, how="all").ffill().bfill()

    y = rotular_semana(Xw, cfg)
    X = Xw.loc[y.index]

    resultados = []
    tss = TimeSeriesSplit(n_splits=cfg["modeling"]["cv"]["n_splits"])
    for mcfg in cfg["modeling"]["models"]:
        model = make_model(mcfg)
        f1s, bals = [], []
        for tr_idx, te_idx in tss.split(X):
            Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
            ytr, yte = y.iloc[tr_idx], y.iloc[te_idx]
            model.fit(Xtr, ytr)
            pred = model.predict(Xte)
            f1s.append(f1_score(yte, pred, average="macro"))
            bals.append(balanced_accuracy_score(yte, pred))
        resultados.append({"modelo": mcfg["name"], "f1_macro": float(np.mean(f1s)), "balanced_acc": float(np.mean(bals))})

        model.fit(X, y)
        dump(model, out_dir / f"{mcfg['name']}.joblib")

    rep_dir = Path(cfg["paths"]["reports_dir"]); rep_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(resultados).to_csv(rep_dir / "cv_scores.csv", index=False)
    print(pd.DataFrame(resultados))

if __name__ == "__main__":
    main()
