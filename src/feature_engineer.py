# src/feature_engineer.py
from __future__ import annotations
import warnings
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
    # Evita erros de horário de verão (DST) ao resamplear: usa UTC se índice for tz‑aware
    dfr = df
    try:
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            dfr = df.tz_convert("UTC")
    except Exception:
        # fallback conservador: remove tz
        try:
            dfr = df.tz_convert("UTC").tz_localize(None)
        except Exception:
            dfr = df
    w = dfr.resample("W").agg(funcs)
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
        return pd.concat(
            [dfw, pd.DataFrame(new_cols, index=dfw.index)], axis=1, copy=False
        )
    return dfw.copy()


def build_features_weekly(data: Dict[str, pd.DataFrame], cfg: Dict) -> pd.DataFrame:
    """Constrói a feature store semanal a partir dos insumos diários.

    - Agrega D->W (múltiplas funções),
    - Deriva métricas (margem, saldo, razão de corte),
    - Cria lags e janelas.

    Args:
      data (dict[str, pandas.DataFrame]): Dicionário com DataFrames diários.
      cfg (dict): Configurações (agregações, lags, janelas).

    Returns:
      pandas.DataFrame: Features semanais indexadas por semana.
    """
    hows = cfg["aggregation"]["features"]["daily_aggs"]
    min_ratio = float(cfg["aggregation"]["features"].get("min_nonnull_ratio", 0.5))
    weekly_min_ratio = cfg["aggregation"]["features"].get(
        "min_nonnull_ratio_weekly", None
    )
    try:
        weekly_min_ratio = (
            float(weekly_min_ratio) if weekly_min_ratio is not None else None
        )
    except Exception:
        weekly_min_ratio = None
    lags = cfg["aggregation"]["features"]["lags_weeks"]
    rolls = cfg["aggregation"]["features"]["rolling_weeks"]

    bases = []
    daily_margin_parts: Dict[str, pd.Series] = {}

    def _filter_present_and_dense(df: pd.DataFrame, cols: List[str]) -> List[str]:
        cols = [c for c in cols if c in df.columns]
        if not cols:
            return []
        ratios = df[cols].notna().mean()
        keep = [c for c in cols if float(ratios.get(c, 0.0)) >= min_ratio]
        dropped = sorted(set(cols) - set(keep))
        if dropped:
            warnings.warn(
                f"Removendo colunas com baixa cobertura (<{min_ratio:.0%}) em feature engineering: {dropped}"
            )
        return keep

    if "carga" in data:
        sel = _filter_present_and_dense(data["carga"], ["carga_mwh"])
        if sel:
            bases.append(weekly_aggregate(data["carga"][sel], hows))

    if "ger_fontes" in data:
        sel = _filter_present_and_dense(
            data["ger_fontes"],
            [
                "ger_hidreletrica_mwh",
                "ger_eolica_mwh",
                "ger_fv_mwh",
                "ger_termica_mwh",
            ],
        )
        if sel:
            gf = data["ger_fontes"][sel]
            bases.append(weekly_aggregate(gf, hows))
            daily_margin_parts["geracao_total"] = gf.sum(axis=1, min_count=1)

    if "intercambio" in data:
        sel = _filter_present_and_dense(
            data["intercambio"], ["import_mwh", "export_mwh"]
        )
        if sel:
            interc = data["intercambio"][sel]
            bases.append(weekly_aggregate(interc, hows))
            if "import_mwh" in interc.columns:
                daily_margin_parts["importacao"] = interc["import_mwh"]
            if "export_mwh" in interc.columns:
                daily_margin_parts["exportacao"] = interc["export_mwh"]

    if "ena" in data:
        sel = _filter_present_and_dense(data["ena"], ["ena_mwmed"])
        if sel:
            bases.append(weekly_aggregate(data["ena"][sel], hows))
    if "ear" in data:
        sel = _filter_present_and_dense(data["ear"], ["ear_pct"])
        if sel:
            bases.append(weekly_aggregate(data["ear"][sel], hows))

    if "cortes_eolica" in data:
        sel = _filter_present_and_dense(data["cortes_eolica"], ["corte_eolica_mwh"])
        if sel:
            bases.append(weekly_aggregate(data["cortes_eolica"][sel], hows))
    if "cortes_fv" in data:
        sel = _filter_present_and_dense(data["cortes_fv"], ["corte_fv_mwh"])
        if sel:
            bases.append(weekly_aggregate(data["cortes_fv"][sel], hows))

    if "clima" in data:
        clima = data["clima"].copy()
        if "precipitacao_mm" in clima.columns:
            clima["precip_14d_mm"] = (
                clima["precipitacao_mm"].rolling(14, min_periods=1).sum()
            )
            clima["precip_30d_mm"] = (
                clima["precipitacao_mm"].rolling(30, min_periods=1).sum()
            )
            clima["precip_90d_mm"] = (
                clima["precipitacao_mm"].rolling(90, min_periods=1).sum()
            )
            clima["precip_180d_mm"] = (
                clima["precipitacao_mm"].rolling(180, min_periods=1).sum()
            )
        cand = [
            "ghi",
            "temp2m_c",
            "precipitacao_mm",
            "precip_14d_mm",
            "precip_30d_mm",
            "precip_90d_mm",
            "precip_180d_mm",
        ]
        sel = _filter_present_and_dense(clima, cand)
        if sel:
            bases.append(weekly_aggregate(clima[sel], hows))

    Xw = pd.concat(bases, axis=1).sort_index()
    Xw = Xw.loc[~Xw.index.duplicated(keep="last")]


    margin_sum_weekly = None
    margin_min_weekly = None
    if daily_margin_parts:
        daily_margin_df = pd.DataFrame(daily_margin_parts).sort_index()
        if not daily_margin_df.empty:
            base = daily_margin_df.get("geracao_total")
            if base is None:
                base = pd.Series(dtype=float)
            imports = daily_margin_df.get("importacao")
            if imports is None:
                imports = pd.Series(dtype=float)
            exports = daily_margin_df.get("exportacao")
            if exports is None:
                exports = pd.Series(dtype=float)

            idx = daily_margin_df.index
            margin_daily = pd.Series(0.0, index=idx, dtype=float)
            if not base.empty:
                margin_daily = margin_daily.add(base.astype(float), fill_value=0.0)
            if not imports.empty:
                margin_daily = margin_daily.add(imports.astype(float), fill_value=0.0)
            if not exports.empty:
                margin_daily = margin_daily.sub(exports.astype(float), fill_value=0.0)

            coverage = pd.DataFrame(
                {
                    "base": base.reindex(idx),
                    "imports": imports.reindex(idx),
                    "exports": exports.reindex(idx),
                }
            )
            valid_mask = coverage.notna().any(axis=1)
            margin_daily = margin_daily.where(valid_mask)
            margin_daily = margin_daily.dropna()
            if not margin_daily.empty:
                margin_daily = margin_daily.sort_index()
                margin_sum_weekly = margin_daily.resample("W").sum()
                margin_min_weekly = margin_daily.resample("W").min()

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
        fallback_margin = (
            Xw["geracao_total_mwh_sum_w"]
            + Xw["import_total_mwh_sum_w"]
            - Xw["export_total_mwh_sum_w"]
        )
        if margin_sum_weekly is not None:
            aligned_sum = margin_sum_weekly.reindex(Xw.index)
            Xw["margem_suprimento_w"] = aligned_sum.fillna(fallback_margin)
        else:
            Xw["margem_suprimento_w"] = fallback_margin

        if margin_min_weekly is not None:
            aligned_min = margin_min_weekly.reindex(Xw.index)
            Xw["margem_suprimento_min_w"] = aligned_min.fillna(
                Xw["margem_suprimento_w"]
            )
        elif "margem_suprimento_min_w" in Xw.columns:
            Xw["margem_suprimento_min_w"] = Xw["margem_suprimento_min_w"].fillna(
                Xw["margem_suprimento_w"]
            )
        else:
            Xw["margem_suprimento_min_w"] = Xw["margem_suprimento_w"]

        Xw["saldo_importador_mwh_sum_w"] = (
            Xw["import_total_mwh_sum_w"] - Xw["export_total_mwh_sum_w"]
        )

    # Métricas de adequação simplificadas (energia):
    # - margem vs. carga (MWh)
    # - razão de margem de reserva (≈ proxy para PNS/Reserva Operativa)
    # - ENS semanal aproximada (energia não suprida) e sua razão vs. demanda semanal
    # - LOLP empírico (52 semanas) como frequência de semanas com margem negativa
    carga_sum_cols = [
        c for c in Xw.columns if c.startswith("carga_mwh_") and c.endswith("_sum_w")
    ]
    if "margem_suprimento_w" in Xw.columns and carga_sum_cols:
        carga_sum_col = carga_sum_cols[0]
        # margem vs. carga (energia): suprimento - demanda
        Xw["margem_vs_carga_w"] = Xw["margem_suprimento_w"] - Xw[carga_sum_col]
        # razão de margem de reserva (% da demanda semanal)
        denom = Xw[carga_sum_col].replace({0: np.nan})
        Xw["reserve_margin_ratio_w"] = Xw["margem_vs_carga_w"] / denom
        # ENS semanal aproximada (energia não suprida)
        Xw["ens_week_mwh"] = (-Xw["margem_vs_carga_w"]).clip(lower=0)
        Xw["ens_week_ratio"] = Xw["ens_week_mwh"] / denom
        # LOLP empírico (freq. de semanas com déficit em janela móvel de 52 semanas)
        Xw["lolp_52w"] = (
            (Xw["margem_vs_carga_w"] < 0).rolling(52, min_periods=12).mean()
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

    # Filtro de cobertura semanal: descarta colunas com baixa proporção de não nulos
    if weekly_min_ratio is not None:
        ratios_w = Xw.notna().mean()
        keep_cols = ratios_w[ratios_w >= weekly_min_ratio].index.tolist()
        dropped_cols = sorted(set(Xw.columns) - set(keep_cols))
        if dropped_cols:
            warnings.warn(
                f"Removendo features semanais com baixa cobertura (<{weekly_min_ratio:.0%}): {dropped_cols[:20]}"
                + (" …" if len(dropped_cols) > 20 else "")
            )
        Xw = Xw[keep_cols]

    Xw = add_lags_rolls(Xw, lags, rolls)

    # Preenchimento somente para trás (sem olhar o futuro)
    Xw = Xw.sort_index().ffill()
    return Xw
