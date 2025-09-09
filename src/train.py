# src/train.py
from __future__ import annotations
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from joblib import dump

from data_loader import load_all_sources
from feature_engineer import build_features_weekly


def rotular_semana(df: pd.DataFrame, cfg, ref_df: pd.DataFrame | None = None) -> pd.Series:
    r = cfg["problem"]["label_rules"]
    margem_col = r["coluna_margem"]
    q_baixo, q_med = r["q_baixo"], r["q_medio"]

    if margem_col not in df.columns:
        raise ValueError(f"Coluna '{margem_col}' não encontrada para rotulagem.")

    base = ref_df if ref_df is not None else df
    margem = df[margem_col].copy()
    thr_baixo = base[margem_col].quantile(q_baixo)
    thr_med = base[margem_col].quantile(q_med)

    y = pd.Series(index=df.index, dtype=object)
    y[margem <= thr_baixo] = "alto"
    y[(margem > thr_baixo) & (margem <= thr_med)] = "medio"
    y[margem > thr_med] = "baixo"

    # Ajuste por cortes (curtailment) — interpretação: superávit local/operacional
    cd = r.get("curtail_downgrade", {})
    if cd.get("usar_downgrade_cortes", True):
        ratio_thr = float(cd.get("corte_ratio_thr", 0.05))
        req_no_import = bool(cd.get("requer_saldo_importador_nao_positivo", True))
        ratio = df.get("ratio_corte_renovavel_w")
        saldo_import = df.get("saldo_importador_mwh_sum_w")
        if ratio is not None:
            cond_ratio = ratio.fillna(0) > ratio_thr
            cond_import = True
            if req_no_import and (saldo_import is not None):
                cond_import = saldo_import.fillna(0) <= 0  # não está importando líquido
            cond = cond_ratio & cond_import
            # "descer" um nível de risco onde as condições forem verdadeiras
            y.loc[cond & (y == "alto")] = "medio"
            y.loc[cond & (y == "medio")] = "baixo"

    # Ajuste hidrológico (estiagem)
    if r.get("usar_override_hidro", True):
        ear_col = r.get("coluna_ear")
        ena_col = r.get("coluna_ena")
        if ear_col in df.columns:
            ear_thr = base[ear_col].quantile(r.get("ear_q_baixo", 0.2))
            y[df[ear_col] <= ear_thr] = "alto"
        if ena_col in df.columns:
            ena_thr = base[ena_col].quantile(r.get("ena_q_baixo", 0.2))
            k = int(r.get("janelas_consecutivas_ena", 2))
            ena_low = (df[ena_col] <= ena_thr).rolling(k).sum() >= k
            y[ena_low.fillna(False)] = "alto"

    return y


def make_model(model_cfg):
    if model_cfg["type"] == "logistic_regression":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=False)),
                ("clf", LogisticRegression(**model_cfg["params"])),
            ]
        )
    elif model_cfg["type"] == "xgboost":
        from xgboost import XGBClassifier

        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", XGBClassifier(**model_cfg["params"]))
        ])
    else:
        raise ValueError(f"Modelo não suportado: {model_cfg['type']}")


def main(config_path="configs/config.yaml"):
    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    out_dir = Path(cfg["paths"]["models_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_all_sources(cfg)
    Xw = build_features_weekly(data, cfg)
    # limpa colunas completamente vazias; imputação fica no pipeline (evita vazamento)
    Xw = Xw.dropna(axis=1, how="all")

    H = int(cfg.get("problem", {}).get("forecast_horizon_weeks", 1))
    X = Xw

    resultados = []
    tss = TimeSeriesSplit(n_splits=cfg["modeling"]["cv"]["n_splits"])
    for mcfg in cfg["modeling"]["models"]:
        model = make_model(mcfg)
        f1s, bals = [], []
        for tr_idx, te_idx in tss.split(X):
            Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
            # Rotulagem sem vazamento: thresholds calculados no treino, aplicados em treino e teste
            ytr = rotular_semana(Xtr, cfg, ref_df=Xtr)
            yte = rotular_semana(Xte, cfg, ref_df=Xtr)
            # Horizonte T+H: alinhamento para prever t+H
            if H > 0:
                ytr = ytr.shift(-H)
                yte = yte.shift(-H)
            # Alinha para índices com rótulo disponível
            tr_idx_ok = ytr.dropna().index
            te_idx_ok = yte.dropna().index
            Xtr_ok, ytr_ok = Xtr.loc[tr_idx_ok], ytr.loc[tr_idx_ok]
            Xte_ok, yte_ok = Xte.loc[te_idx_ok], yte.loc[te_idx_ok]

            if len(Xtr_ok) == 0 or len(Xte_ok) == 0:
                continue

            model.fit(Xtr_ok, ytr_ok)
            pred = model.predict(Xte_ok)
            f1s.append(f1_score(yte_ok, pred, average="macro"))
            bals.append(balanced_accuracy_score(yte_ok, pred))
        resultados.append(
            {
                "modelo": mcfg["name"],
                "f1_macro": float(np.mean(f1s)),
                "balanced_acc": float(np.mean(bals)),
            }
        )

        # re-ajusta no conjunto completo e salva artefato
        y_full = rotular_semana(X, cfg, ref_df=X)
        if H > 0:
            y_full = y_full.shift(-H)
        idx_ok = y_full.dropna().index
        X_full_ok, y_full_ok = X.loc[idx_ok], y_full.loc[idx_ok]
        model.fit(X_full_ok, y_full_ok)
        dump(model, out_dir / f"{mcfg['name']}.joblib")

    rep_dir = Path(cfg["paths"]["reports_dir"])
    rep_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(resultados).to_csv(rep_dir / "cv_scores.csv", index=False)
    print(pd.DataFrame(resultados))


if __name__ == "__main__":
    main()
