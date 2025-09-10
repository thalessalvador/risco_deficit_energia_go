# src/feature_engineer.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List


def _p05(x):
    """Percentil 5 ignorando NaNs.

    Args:
      x (array-like): Série numérica.

    Returns:
      float: Valor do percentil 5.
    """
    return np.nanpercentile(x, 5)


def _p95(x):
    """Percentil 95 ignorando NaNs.

    Args:
      x (array-like): Série numérica.

    Returns:
      float: Valor do percentil 95.
    """
    return np.nanpercentile(x, 95)


AGG_FUNCS = {
    "mean": "mean",
    "sum": "sum",
    "max": "max",
    "min": "min",
    "std": "std",
    "p95": _p95,
    "p05": _p05,
}
AGG_NAME = {v: k for k, v in AGG_FUNCS.items()}
AGG_NAME[_p95] = "p95"
AGG_NAME[_p05] = "p05"


def weekly_aggregate(df: pd.DataFrame, hows: List[str]) -> pd.DataFrame:
    """Agrega um DataFrame diário para semanal aplicando funções especificadas.

    Args:
      df (pandas.DataFrame): Dados diários com índice datetime.
      hows (list[str]): Lista de agregações (ex.: "mean", "sum", "p95").

    Returns:
      pandas.DataFrame: Colunas agregadas com sufixo `_w` e nome da função.
    """
    funcs = [AGG_FUNCS[h] for h in hows]
    w = df.resample("W").agg(funcs)
    cols = []
    for col, func in w.columns.to_flat_index():
        agg = func if isinstance(func, str) else AGG_NAME.get(func, "agg")
        cols.append(f"{col}_{agg}_w")
    w.columns = cols
    return w


def add_lags_rolls(
    dfw: pd.DataFrame, lags: List[int], rolls: List[int]
) -> pd.DataFrame:
    """Cria lags (em semanas) e janelas móveis para cada coluna semanal.

    Args:
      dfw (pandas.DataFrame): Features semanais agregadas.
      lags (list[int]): Lags em semanas a criar (ex.: [1,2,4]).
      rolls (list[int]): Tamanhos de janelas para médias/desvios móveis.

    Returns:
      pandas.DataFrame: DataFrame com colunas adicionais de lags e janelas móveis.
    """
    # Evita fragmentação do DataFrame: acumula novas colunas e concatena ao final
    new_cols = {}
    for col in dfw.columns:
        s = dfw[col]
        for L in lags:
            new_cols[f"{col}_lag{L}w"] = s.shift(L)
        for R in rolls:
            roll = s.rolling(R, min_periods=1)
            new_cols[f"{col}_r{R}w_mean"] = roll.mean()
            new_cols[f"{col}_r{R}w_std"] = roll.std()
    if new_cols:
        return pd.concat([dfw, pd.DataFrame(new_cols, index=dfw.index)], axis=1, copy=False)
    return dfw.copy()


def build_features_weekly(data: Dict[str, pd.DataFrame], cfg: Dict) -> pd.DataFrame:
    """Constrói a feature store semanal a partir dos insumos diários.

    - Agrega D→W (múltiplas funções),
    - Deriva métricas (margem, saldo, razão de corte),
    - Cria lags e janelas.

    Args:
      data (dict[str, pandas.DataFrame]): Dicionário com DataFrames diários.
      cfg (dict): Configurações (agregações, lags, janelas).

    Returns:
      pandas.DataFrame: Features semanais indexadas por semana.
    """
    hows = cfg["aggregation"]["features"]["daily_aggs"]
    lags = cfg["aggregation"]["features"]["lags_weeks"]
    rolls = cfg["aggregation"]["features"]["rolling_weeks"]

    bases = []

    if "carga" in data:
        bases.append(weekly_aggregate(data["carga"][["carga_mwh"]], hows))

    if "ger_fontes" in data:
        bases.append(
            weekly_aggregate(
                data["ger_fontes"][
                    [
                        "ger_hidreletrica_mwh",
                        "ger_eolica_mwh",
                        "ger_fv_mwh",
                        "ger_termica_mwh",
                    ]
                ],
                hows,
            )
        )

    if "intercambio" in data:
        bases.append(
            weekly_aggregate(data["intercambio"][["import_mwh", "export_mwh"]], hows)
        )

    if "ena" in data:
        bases.append(weekly_aggregate(data["ena"][["ena_mwmed"]], hows))
    if "ear" in data:
        bases.append(weekly_aggregate(data["ear"][["ear_pct"]], hows))

    if "cortes_eolica" in data:
        bases.append(
            weekly_aggregate(data["cortes_eolica"][["corte_eolica_mwh"]], hows)
        )
    if "cortes_fv" in data:
        bases.append(weekly_aggregate(data["cortes_fv"][["corte_fv_mwh"]], hows))

    if "clima" in data:
        clima = data["clima"].copy()
        if "precipitacao_mm" in clima.columns:
            clima["precip_14d_mm"] = (
                clima["precipitacao_mm"].rolling(14, min_periods=1).sum()
            )
            clima["precip_30d_mm"] = (
                clima["precipitacao_mm"].rolling(30, min_periods=1).sum()
            )
        sel = [
            c
            for c in [
                "ghi",
                "temp2m_c",
                "precipitacao_mm",
                "precip_14d_mm",
                "precip_30d_mm",
            ]
            if c in clima.columns
        ]
        bases.append(weekly_aggregate(clima[sel], hows))

    Xw = pd.concat(bases, axis=1).sort_index()
    Xw = Xw.loc[~Xw.index.duplicated(keep="last")]

    def pick(prefix: str):
        return [c for c in Xw.columns if c.startswith(prefix)]

    ger_cols = (
        pick("ger_hidreletrica_mwh_")
        + pick("ger_eolica_mwh_")
        + pick("ger_fv_mwh_")
        + pick("ger_termica_mwh_")
    )
    imp_cols = pick("import_mwh_")
    exp_cols = pick("export_mwh_")
    if ger_cols:
        Xw["geracao_total_mwh_sum_w"] = Xw[
            [c for c in ger_cols if c.endswith("_sum_w")]
        ].sum(axis=1)
    if imp_cols:
        Xw["import_total_mwh_sum_w"] = Xw[
            [c for c in imp_cols if c.endswith("_sum_w")]
        ].sum(axis=1)
    if exp_cols:
        Xw["export_total_mwh_sum_w"] = Xw[
            [c for c in exp_cols if c.endswith("_sum_w")]
        ].sum(axis=1)
    if {
        "geracao_total_mwh_sum_w",
        "import_total_mwh_sum_w",
        "export_total_mwh_sum_w",
    } <= set(Xw.columns):
        Xw["margem_suprimento_w"] = (
            Xw["geracao_total_mwh_sum_w"]
            + Xw["import_total_mwh_sum_w"]
            - Xw["export_total_mwh_sum_w"]
        )
        Xw["margem_suprimento_min_w"] = Xw["margem_suprimento_w"]
        Xw["saldo_importador_mwh_sum_w"] = (
            Xw["import_total_mwh_sum_w"] - Xw["export_total_mwh_sum_w"]
        )

    # total de cortes renováveis e razão vs potencial renovável
    corte_cols = [
        c
        for c in Xw.columns
        if c.startswith("corte_eolica_mwh_") and c.endswith("_sum_w")
    ] + [
        c for c in Xw.columns if c.startswith("corte_fv_mwh_") and c.endswith("_sum_w")
    ]
    if corte_cols:
        Xw["corte_renovavel_mwh_sum_w"] = Xw[corte_cols].sum(axis=1)
    ren_cols = [
        c
        for c in Xw.columns
        if c.startswith("ger_eolica_mwh_") and c.endswith("_sum_w")
    ] + [c for c in Xw.columns if c.startswith("ger_fv_mwh_") and c.endswith("_sum_w")]
    if ren_cols:
        Xw["ger_renovavel_mwh_sum_w"] = Xw[ren_cols].sum(axis=1)
    if {"corte_renovavel_mwh_sum_w", "ger_renovavel_mwh_sum_w"} <= set(Xw.columns):
        Xw["ratio_corte_renovavel_w"] = Xw["corte_renovavel_mwh_sum_w"] / (
            Xw["corte_renovavel_mwh_sum_w"] + Xw["ger_renovavel_mwh_sum_w"]
        ).replace({0: np.nan})

    Xw = add_lags_rolls(Xw, lags, rolls)

    # Preenchimento somente para trás (sem olhar o futuro)
    Xw = Xw.sort_index().ffill()
    return Xw
