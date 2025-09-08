# src/feature_engineer.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List

def _p05(x): return np.nanpercentile(x, 5)
def _p95(x): return np.nanpercentile(x, 95)

AGG_FUNCS = {
    "mean": "mean", "sum": "sum", "max": "max", "min": "min", "std": "std",
    "p95": _p95, "p05": _p05
}
AGG_NAME = {v: k for k, v in AGG_FUNCS.items()}
AGG_NAME[_p95] = "p95"
AGG_NAME[_p05] = "p05"

def weekly_aggregate(df: pd.DataFrame, hows: List[str]) -> pd.DataFrame:
    funcs = [AGG_FUNCS[h] for h in hows]
    w = df.resample("W").agg(funcs)
    cols = []
    for (col, func) in w.columns.to_flat_index():
        agg = func if isinstance(func, str) else AGG_NAME.get(func, "agg")
        cols.append(f"{col}_{agg}_w")
    w.columns = cols
    return w

def add_lags_rolls(dfw: pd.DataFrame, lags: List[int], rolls: List[int]) -> pd.DataFrame:
    out = dfw.copy()
    for col in dfw.columns:
        for L in lags:
            out[f"{col}_lag{L}w"] = dfw[col].shift(L)
        for R in rolls:
            out[f"{col}_r{R}w_mean"] = dfw[col].rolling(R, min_periods=1).mean()
            out[f"{col}_r{R}w_std"]  = dfw[col].rolling(R, min_periods=1).std()
    return out

def build_features_weekly(data: Dict[str, pd.DataFrame], cfg: Dict) -> pd.DataFrame:
    hows = cfg["aggregation"]["features"]["daily_aggs"]
    lags = cfg["aggregation"]["features"]["lags_weeks"]
    rolls = cfg["aggregation"]["features"]["rolling_weeks"]

    bases = []

    if "carga" in data:
        bases.append(weekly_aggregate(data["carga"][["carga_mwh"]], hows))

    if "ger_fontes" in data:
        bases.append(weekly_aggregate(
            data["ger_fontes"][["ger_hidreletrica_mwh","ger_eolica_mwh","ger_fv_mwh","ger_termica_mwh"]],
            hows
        ))

    if "intercambio" in data:
        bases.append(weekly_aggregate(
            data["intercambio"][["import_mwh","export_mwh"]],
            hows
        ))

    if "ena" in data:
        bases.append(weekly_aggregate(data["ena"][["ena_mwmed"]], hows))
    if "ear" in data:
        bases.append(weekly_aggregate(data["ear"][["ear_pct"]], hows))

    if "cortes_eolica" in data:
        bases.append(weekly_aggregate(data["cortes_eolica"][["corte_eolica_mwh"]], hows))
    if "cortes_fv" in data:
        bases.append(weekly_aggregate(data["cortes_fv"][["corte_fv_mwh"]], hows))

    if "clima" in data:
        clima = data["clima"].copy()
        if "precipitacao_mm" in clima.columns:
            clima["precip_14d_mm"] = clima["precipitacao_mm"].rolling(14, min_periods=1).sum()
            clima["precip_30d_mm"] = clima["precipitacao_mm"].rolling(30, min_periods=1).sum()
        sel = [c for c in ["ghi","temp2m_c","precipitacao_mm","precip_14d_mm","precip_30d_mm"] if c in clima.columns]
        bases.append(weekly_aggregate(clima[sel], hows))

    Xw = pd.concat(bases, axis=1).sort_index()
    Xw = Xw.loc[~Xw.index.duplicated(keep="last")]

    def pick(prefix: str):
        return [c for c in Xw.columns if c.startswith(prefix)]
    ger_cols = pick("ger_hidreletrica_mwh_") + pick("ger_eolica_mwh_") + pick("ger_fv_mwh_") + pick("ger_termica_mwh_")
    imp_cols = pick("import_mwh_")
    exp_cols = pick("export_mwh_")
    if ger_cols:
        Xw["geracao_total_mwh_sum_w"] = Xw[[c for c in ger_cols if c.endswith("_sum_w")]].sum(axis=1)
    if imp_cols:
        Xw["import_total_mwh_sum_w"] = Xw[[c for c in imp_cols if c.endswith("_sum_w")]].sum(axis=1)
    if exp_cols:
        Xw["export_total_mwh_sum_w"] = Xw[[c for c in exp_cols if c.endswith("_sum_w")]].sum(axis=1)
    if {"geracao_total_mwh_sum_w","import_total_mwh_sum_w","export_total_mwh_sum_w"} <= set(Xw.columns):
        Xw["margem_suprimento_w"] = Xw["geracao_total_mwh_sum_w"] + Xw["import_total_mwh_sum_w"] - Xw["export_total_mwh_sum_w"]
        Xw["margem_suprimento_min_w"] = Xw["margem_suprimento_w"]

    Xw = add_lags_rolls(Xw, lags, rolls)

    Xw = Xw.sort_index().ffill().bfill()
    return Xw
