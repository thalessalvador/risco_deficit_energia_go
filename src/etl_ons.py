from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import unicodedata
import warnings

import numpy as np
import pandas as pd


def _norm_text(s: str) -> str:
    """Normaliza texto: minúsculas, sem acentos e espaçamentos compactados.

    Args:
      s (str): Texto de entrada.

    Returns:
      str: Texto normalizado para comparações/mapeamentos.
    """
    s = s.strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("/", "").replace("-", "_")
    s = s.replace(" ", "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nomes de colunas para facilitar detecção (minúsculos/sem acentos).

    Args:
      df (pandas.DataFrame): DataFrame original.

    Returns:
      pandas.DataFrame: Cópia com colunas normalizadas.
    """
    df = df.copy()
    df.columns = [_norm_text(c) for c in df.columns]
    return df


SUBMERCADO_ALIASES = {
    "seco",
    "se_co",
    "se",
    "sudeste_centrooeste",
    "sudeste_centro_oeste",
    "sudeste_centro_oeste_seco",
    "sudeste_centrooeste_seco",
    "sudeste",
    "se/co",
}


def _match_submercado(value: str, alvo: str = "seco") -> bool:
    """Verifica se um valor textual corresponde ao submercado alvo (com aliases).

    Args:
      value (str): Texto a testar (ex.: "SE", "SUDESTE/CENTRO-OESTE").
      alvo (str): Alvo normalizado (ex.: "SE/CO").

    Returns:
      bool: True se houver correspondência.
    """
    v = _norm_text(str(value))
    a = _norm_text(alvo)
    if v == a:
        return True
    if v in SUBMERCADO_ALIASES and a in SUBMERCADO_ALIASES:
        return True
    return False


def _infer_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Infere um índice datetime a partir de colunas comuns de data/hora.

    Tenta `din_instante`, `data_hora`, `datahora`, `datetime`, `data` ou a
    combinação `data`+`hora`.

    Args:
      df (pandas.DataFrame): DataFrame com colunas de data/hora.

    Returns:
      pandas.DataFrame: Mesmo DataFrame, reindexado por datetime.
    """
    df = df.copy()
    cols = set(df.columns)
    # candidatos comuns no ONS
    candidates = [
        "din_instante",
        "datahora",
        "data_hora",
        "data_e_hora",
        "datetime",
        "data",
    ]
    dt = None
    for c in candidates:
        if c in cols:
            dt = pd.to_datetime(df[c], errors="coerce")
            break
    if dt is None and {"data", "hora"} <= cols:
        dt = pd.to_datetime(
            df["data"].astype(str) + " " + df["hora"].astype(str), errors="coerce"
        )
    if dt is None:
        # try any column that looks like date
        cand = [c for c in df.columns if "data" in c or "date" in c or "dia" in c]
        if cand:
            dt = pd.to_datetime(df[cand[0]], errors="coerce")
        else:
            raise ValueError(
                "Não foi possível inferir coluna de data/hora do arquivo bruto."
            )
    df = df.set_index(dt).dropna(axis=0, subset=[dt.name])
    df.index.name = "data"
    return df


def _read_csv_auto(path: Path) -> pd.DataFrame:
    """Lê CSV detectando separador e com fallback de encoding/engine.

    Estratégia:
    - Encodings tentados: utf-8-sig, utf-8, latin1, cp1252
    - Separadores tentados: sniff, ';', ',', '\t', '|'
    - Usa engine 'python' (on_bad_lines='skip') e, por fim, engine 'c' para ';' e ','
    """
    encodings = ["utf-8-sig", "utf-8", "latin1", "cp1252"]
    seps = [None, ";", ",", "\t", "|"]
    last_err = None
    for enc in encodings:
        for sep in seps:
            try:
                return pd.read_csv(
                    path, sep=sep, engine="python", encoding=enc, on_bad_lines="skip"
                )
            except Exception as e:
                last_err = e
                continue
    for enc in encodings:
        for sep in [";", ","]:
            try:
                return pd.read_csv(path, sep=sep, engine="c", encoding=enc)
            except Exception as e:
                last_err = e
                continue
    raise last_err if last_err else RuntimeError(f"Falha ao ler CSV: {path}")


def _ensure_daily_sum(df: pd.DataFrame) -> pd.DataFrame:
    """Reamostra para diário somando os valores (ex.: MWmed/h -> MWh/dia).

    Args:
      df (pandas.DataFrame): Série horária.

    Returns:
      pandas.DataFrame: Série diária pela soma.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Índice temporal não é DatetimeIndex")
    # Para séries de energia horárias em MWmed, somar no dia ≈ MWh/dia
    return df.resample("D").sum(min_count=1)


def _ensure_daily_mean(df: pd.DataFrame) -> pd.DataFrame:
    """Reamostra para diário usando média dos valores.

    Args:
      df (pandas.DataFrame): Série temporal.

    Returns:
      pandas.DataFrame: Série diária pela média.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Índice temporal não é DatetimeIndex")
    return df.resample("D").mean()


def etl_balanco_subsistema_horario(
    path: Path, out_dir: Path, submercado: str
) -> Optional[Path]:
    """Transforma Balanço por Subsistema (horário) em geração diária por fonte.

    Espera colunas de geração (hidráulica/térmica/eólica/solar) e subsistema
    (id ou nome). Gera `ons_geracao_fontes_diaria.csv` em MWh/dia.

    Args:
      path (Path): Arquivo bruto do ONS (horário).
      out_dir (Path): Diretório de saída.
      submercado (str): Submercado alvo (ex.: "SE/CO").

    Returns:
      Path|None: Caminho do CSV gerado ou None se não aplicável.
    """
    if not path.exists():
        return None
    df0 = _read_csv_auto(path)
    df0 = _normalize_columns(df0)

    # identificar coluna de subsistema
    sub_cols = [
        c
        for c in df0.columns
        if c in {"submercado", "subsistema", "id_subsistema", "nom_subsistema"}
    ]
    if not sub_cols:
        warnings.warn(
            "Coluna de subsistema não encontrada; assumindo arquivo já filtrado para SE/CO."
        )
        df = df0
    else:
        sub_col = sub_cols[0]
        mask = df0[sub_col].apply(lambda x: _match_submercado(str(x), submercado))
        df = df0.loc[mask].copy()
    if df.empty:
        warnings.warn("Arquivo de balanço sem linhas para o submercado informado.")
        return None

    # inferir datetime
    df = _infer_datetime_index(df)

    # heurística para colunas de geração por fonte
    col_map = {}
    for c in df.columns:
        nc = _norm_text(c)
        if any(k in nc for k in ["hidro", "hidraul"]):
            col_map.setdefault("ger_hidreletrica_mwh", c)
        elif "term" in nc:
            col_map.setdefault("ger_termica_mwh", c)
        elif "eolic" in nc:
            col_map.setdefault("ger_eolica_mwh", c)
        elif any(k in nc for k in ["solar", "fotov", "fv"]):
            col_map.setdefault("ger_fv_mwh", c)

    want = ["ger_hidreletrica_mwh", "ger_termica_mwh", "ger_eolica_mwh", "ger_fv_mwh"]
    missing = [k for k in want if k not in col_map]
    if missing:
        warnings.warn(
            f"Colunas de geração ausentes no balanço (faltando {missing}); arquivo pode ter layout diferente."
        )

    sel = {k: v for k, v in col_map.items()}
    if not sel:
        warnings.warn("Nenhuma coluna de geração reconhecida; pulando balanço.")
        return None

    dfd = df[list(sel.values())].rename(columns={v: k for k, v in sel.items()})
    dfd = _ensure_daily_sum(dfd)
    dfd = dfd.reset_index()
    out = out_dir / "ons_geracao_fontes_diaria.csv"
    dfd.to_csv(out, index=False)
    return out


def etl_intercambio_horario(
    path: Path, out_dir: Path, submercado: str
) -> Optional[Path]:
    """Transforma Intercâmbios (horário) em import/export diários do submercado.

    Args:
      path (Path): Arquivo bruto do ONS (horário).
      out_dir (Path): Diretório de saída.
      submercado (str): Submercado alvo.

    Returns:
      Path|None: CSV `ons_intercambio_diario.csv` gerado ou None.
    """
    if not path.exists():
        return None
    df = _read_csv_auto(path)
    df = _normalize_columns(df)
    # identificar colunas from/to e valor
    from_cols = [
        c
        for c in df.columns
        if c
        in {
            "de",
            "from",
            "origem",
            "subsistema_de",
            "id_subsistema_origem",
            "nom_subsistema_origem",
        }
        or ("origem" in c)
    ]
    to_cols = [
        c
        for c in df.columns
        if c
        in {
            "para",
            "to",
            "destino",
            "subsistema_para",
            "id_subsistema_destino",
            "nom_subsistema_destino",
        }
        or ("destino" in c)
    ]
    # detectar coluna de valor: contém 'mwmed' ou 'valor' ou 'intercambio'
    val_cols = [
        c
        for c in df.columns
        if c in {"valor", "mwmed", "mw", "potencia"}
        or ("mwmed" in c)
        or ("intercambio" in c)
    ]
    if not from_cols or not to_cols or not val_cols:
        warnings.warn(
            "Não foi possível identificar colunas de 'de/para/valor' em Intercâmbio; pulando."
        )
        return None
    c_from, c_to, c_val = from_cols[0], to_cols[0], val_cols[0]

    # inferir datetime
    df = _infer_datetime_index(df)

    is_imp = df[c_to].apply(lambda x: _match_submercado(str(x), submercado))
    is_exp = df[c_from].apply(lambda x: _match_submercado(str(x), submercado))

    imp = df.loc[is_imp, [c_val]].rename(columns={c_val: "import_mwh"})
    exp = df.loc[is_exp, [c_val]].rename(columns={c_val: "export_mwh"})

    impd = _ensure_daily_sum(imp)
    expd = _ensure_daily_sum(exp)
    outd = impd.join(expd, how="outer").fillna(0.0).reset_index()

    out = out_dir / "ons_intercambio_diario.csv"
    outd.to_csv(out, index=False)
    return out


def etl_ena_diaria(path: Path, out_dir: Path, submercado: str) -> Optional[Path]:
    """Padroniza ENA Diário por Subsistema em `ons_ena_diaria.csv` (ena_mwmed).

    Args:
      path (Path): Arquivo diário do ONS.
      out_dir (Path): Diretório de saída.
      submercado (str): Submercado alvo.

    Returns:
      Path|None: Caminho do CSV gerado ou None.
    """
    if not path.exists():
        return None
    df = _read_csv_auto(path)
    df = _normalize_columns(df)
    # filtra submercado
    sub_cols = [c for c in df.columns if c in {"submercado", "subsistema"}]
    if sub_cols:
        sub_col = sub_cols[0]
        df = df.loc[df[sub_col].apply(lambda x: _match_submercado(str(x), submercado))]
    if df.empty:
        warnings.warn("ENA diário sem linhas para o submercado.")
        return None

    # data e valor
    df = _normalize_columns(df)
    # detectar coluna de valor
    val_col = None
    for c in df.columns:
        nc = _norm_text(c)
        if "ena" in nc and (
            "mw" in nc or "mwmed" in nc or "energia" in nc or "valor" in nc
        ):
            val_col = c
            break
    if val_col is None:
        # fallback a 'valor' puro
        if "valor" in df.columns:
            val_col = "valor"
        else:
            # assume já com nome desejado
            val_col = "ena_mwmed" if "ena_mwmed" in df.columns else None
    if val_col is None:
        warnings.warn("Coluna de ENA não reconhecida; pulando.")
        return None

    # inferir data
    if "data" not in df.columns:
        if "din_instante" in df.columns:
            df = df.rename(columns={"din_instante": "data"})
        else:
            cand = [c for c in df.columns if "data" in c or "date" in c]
            if cand:
                df = df.rename(columns={cand[0]: "data"})
            else:
                warnings.warn("Coluna de data não encontrada para ENA; pulando.")
                return None
    dfd = df[["data", val_col]].rename(columns={val_col: "ena_mwmed"}).copy()
    dfd["data"] = pd.to_datetime(dfd["data"], errors="coerce")
    dfd = dfd.dropna(subset=["data"]).sort_values("data")
    out = out_dir / "ons_ena_diaria.csv"
    dfd.to_csv(out, index=False)
    return out


def etl_ear_diaria(path: Path, out_dir: Path, submercado: str) -> Optional[Path]:
    """Padroniza EAR Diário por Subsistema em `ons_ear_diaria.csv` (ear_pct).

    Args:
      path (Path): Arquivo diário do ONS.
      out_dir (Path): Diretório de saída.
      submercado (str): Submercado alvo.

    Returns:
      Path|None: Caminho do CSV gerado ou None.
    """
    if not path.exists():
        return None
    df = _read_csv_auto(path)
    df = _normalize_columns(df)
    # filtra submercado
    sub_cols = [c for c in df.columns if c in {"submercado", "subsistema"}]
    if sub_cols:
        sub_col = sub_cols[0]
        df = df.loc[df[sub_col].apply(lambda x: _match_submercado(str(x), submercado))]
    if df.empty:
        warnings.warn("EAR diário sem linhas para o submercado.")
        return None

    # detectar coluna EAR (%)
    val_col = None
    for c in df.columns:
        nc = _norm_text(c)
        if "ear" in nc and (
            "pct" in nc or "%" in c or "percent" in nc or "_" not in nc
        ):
            val_col = c
            break
    if val_col is None:
        if "valor" in df.columns:
            val_col = "valor"
        elif "ear_pct" in df.columns:
            val_col = "ear_pct"
        else:
            warnings.warn("Coluna EAR não reconhecida; pulando.")
            return None

    # inferir data
    if "data" not in df.columns:
        if "din_instante" in df.columns:
            df = df.rename(columns={"din_instante": "data"})
        else:
            cand = [c for c in df.columns if "data" in c or "date" in c]
            if cand:
                df = df.rename(columns={cand[0]: "data"})
            else:
                warnings.warn("Coluna de data não encontrada para EAR; pulando.")
                return None
    dfd = df[["data", val_col]].rename(columns={val_col: "ear_pct"}).copy()
    dfd["data"] = pd.to_datetime(dfd["data"], errors="coerce")
    dfd = dfd.dropna(subset=["data"]).sort_values("data")
    out = out_dir / "ons_ear_diaria.csv"
    dfd.to_csv(out, index=False)
    return out


def _explode_month_to_daily(
    df: pd.DataFrame, value_col: str, date_col: str = "data"
) -> pd.DataFrame:
    """Distribui valor mensal uniformemente pelos dias do mês (série diária).

    Args:
      df (pandas.DataFrame): Tabela mensal com colunas de data e valor.
      value_col (str): Nome da coluna de valor mensal (MWh).
      date_col (str): Nome da coluna de data mensal.

    Returns:
      pandas.DataFrame: Série diária com soma dos valores por dia do mês.
    """
    dfe = []
    for _, row in df.iterrows():
        d = pd.to_datetime(row[date_col])
        start = pd.Timestamp(year=d.year, month=d.month, day=1)
        end = start + pd.offsets.MonthEnd(0)
        days = pd.date_range(start, end, freq="D")
        val = float(row[value_col]) if pd.notnull(row[value_col]) else 0.0
        per_day = val / len(days) if len(days) else 0.0
        dfe.append(pd.DataFrame({"data": days, value_col: per_day}))
    if not dfe:
        return pd.DataFrame(columns=["data", value_col])
    out = pd.concat(dfe, ignore_index=True)
    out = out.groupby("data", as_index=False)[value_col].sum()
    return out


def etl_constrained_off_mensal(
    path: Path, out_dir: Path, fonte: str, submercado: str
) -> Optional[Path]:
    """Padroniza Constrained-off mensal (eólica/FV) em série diária (MWh).

    Args:
      path (Path): Arquivo mensal do ONS.
      out_dir (Path): Diretório de saída.
      fonte (str): "eolica" ou "fv".
      submercado (str): Submercado alvo.

    Returns:
      Path|None: Caminho do CSV diário gerado ou None.
    """
    if not path.exists():
        return None
    df = _read_csv_auto(path)
    df = _normalize_columns(df)

    # filtra submercado se disponível
    sub_cols = [
        c
        for c in df.columns
        if c in {"submercado", "subsistema", "id_subsistema", "nom_subsistema"}
    ]
    if sub_cols:
        sub_col = sub_cols[0]
        df = df.loc[df[sub_col].apply(lambda x: _match_submercado(str(x), submercado))]
    if df.empty:
        warnings.warn(f"Constrained-off {fonte} sem linhas para o submercado.")
        return None

    # detectar data mensal e valor (MWh)
    if "data" not in df.columns:
        if "din_instante" in df.columns:
            df = df.rename(columns={"din_instante": "data"})
        elif "competencia" in df.columns:
            df = df.rename(columns={"competencia": "data"})
        elif {"ano", "mes"} <= set(df.columns):
            df["data"] = pd.to_datetime(
                dict(year=df["ano"].astype(int), month=df["mes"].astype(int), day=1)
            )
        else:
            warnings.warn("Coluna mensal de data não encontrada nos cortes; pulando.")
            return None

    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    df = df.dropna(subset=["data"]).sort_values("data")

    # 1) Tenta identificar coluna de MWh diretamente
    vcol = None
    for c in df.columns:
        nc = _norm_text(c)
        if any(k in nc for k in ["mwh", "energia_nao_gerada", "corte", "restricao"]):
            vcol = c
            break

    if vcol is not None:
        dfd = df[["data", vcol]].copy()
        # garante numérico; valores não numéricos viram NaN -> tratados como 0 na explosão
        dfd[vcol] = pd.to_numeric(dfd[vcol], errors="coerce")
        daily = _explode_month_to_daily(dfd, vcol, date_col="data")
        daily = daily.rename(columns={vcol: f"corte_{fonte}_mwh"})
    else:
        # 2) Caso não exista MWh explícito, calcula a partir de estimada - verificada (MWmed)
        if {"val_geracaoestimada", "val_geracaoverificada"} <= set(df.columns):
            tmp = df[["data", "val_geracaoestimada", "val_geracaoverificada"]].copy()
            tmp["diff_mwmed"] = (
                tmp["val_geracaoestimada"] - tmp["val_geracaoverificada"]
            ).clip(lower=0)
            # agrega por mês e converte para MWh (24 * dias_do_mes)
            g = tmp.groupby(tmp["data"].dt.to_period("M"))[["diff_mwmed"]].sum()
            g.index = g.index.to_timestamp(how="start")
            days = g.index.to_series().apply(lambda d: (d + pd.offsets.MonthEnd(0)).day)
            mwh = g["diff_mwmed"] * 24 * days
            dfd = pd.DataFrame({"data": mwh.index, f"corte_{fonte}_mwh": mwh.values})
            daily = _explode_month_to_daily(dfd, f"corte_{fonte}_mwh", date_col="data")
        else:
            warnings.warn(
                "Coluna de MWh inexistente e não foi possível calcular a partir de estimada/verificada; pulando."
            )
            return None

    out = out_dir / f"ons_cortes_{fonte}_diario.csv"
    daily.to_csv(out, index=False)
    return out


def etl_constrained_off_mensal(
    path: Path, out_dir: Path, fonte: str, submercado: str
) -> Optional[Path]:
    """(Override) Constrained-off (eólica/FV) -> série diária (MWh) a partir de intradiário ou mensal.

    Suporta:
      - Arquivos mensais agregados (MWh) -> explode uniformemente por dia do mês.
      - Séries intra-diárias (ex.: 30 min) com referência/geração -> integra energia não gerada.
    """
    if not path.exists():
        return None
    df = _read_csv_auto(path)
    df = _normalize_columns(df)

    # filtra submercado se disponível
    sub_cols = [
        c
        for c in df.columns
        if c in {"submercado", "subsistema", "id_subsistema", "nom_subsistema"}
    ]
    if sub_cols:
        sub_col = sub_cols[0]
        df = df.loc[df[sub_col].apply(lambda x: _match_submercado(str(x), submercado))]
    if df.empty:
        warnings.warn(f"Constrained-off {fonte} sem linhas para o submercado.")
        return None

    # coluna de data
    original_cols = set(df.columns)
    if "data" not in df.columns:
        if "din_instante" in df.columns:
            df = df.rename(columns={"din_instante": "data"})
        elif "competencia" in df.columns:
            df = df.rename(columns={"competencia": "data"})
        elif {"ano", "mes"} <= original_cols:
            df["data"] = pd.to_datetime(
                dict(year=df["ano"].astype(int), month=df["mes"].astype(int), day=1)
            )
        else:
            warnings.warn("Coluna de data não encontrada nos cortes; pulando.")
            return None

    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    df = df.dropna(subset=["data"]).sort_values("data")

    # Heurística de intra-diário: presença de colunas de geração e passo médio < 6h
    has_gen = any(
        c in df.columns
        for c in [
            "val_geracao",
            "val_geracaolimitada",
            "val_geracaoreferencia",
            "val_geracaoreferenciafinal",
        ]
    )
    if len(df) >= 10:
        dt = df["data"].sort_values().diff().dt.total_seconds().dropna()
        med_hours = float(np.median(dt) / 3600.0) if len(dt) else None
    else:
        med_hours = None
    is_intraday = bool(has_gen and med_hours is not None and med_hours < 6)

    if is_intraday:
        df2 = df.copy()
        for c in [
            "val_geracao",
            "val_geracaolimitada",
            "val_disponibilidade",
            "val_geracaoreferencia",
            "val_geracaoreferenciafinal",
        ]:
            if c in df2.columns:
                df2[c] = pd.to_numeric(df2[c], errors="coerce")

        # referência prioritária
        ref_col = None
        for cand in [
            "val_geracaoreferenciafinal",
            "val_geracaoreferencia",
            "val_disponibilidade",
        ]:
            if cand in df2.columns and pd.api.types.is_numeric_dtype(df2[cand]):
                ref_col = cand
                break
        gen_col = "val_geracao" if "val_geracao" in df2.columns else None
        lim_col = (
            "val_geracaolimitada" if "val_geracaolimitada" in df2.columns else None
        )

        if ref_col is None or (gen_col is None and lim_col is None):
            warnings.warn(
                "Cortes (intradiário): sem referência/geração suficientes; pulando."
            )
            return None

        if lim_col is not None and pd.api.types.is_numeric_dtype(df2[lim_col]):
            base_diff = (df2[ref_col] - df2[lim_col]).clip(lower=0)
        else:
            base_diff = (df2[ref_col] - df2[gen_col]).clip(lower=0)

        # integra energia (MWh) por dia
        df2 = df2.set_index(pd.to_datetime(df2["data"]))
        dt_next = df2.index.to_series().shift(-1) - df2.index.to_series()
        dt_h = dt_next.dt.total_seconds() / 3600.0
        med_dt = float(dt_h.dropna().median()) if dt_h.notna().any() else 0.5
        if not np.isfinite(med_dt) or med_dt <= 0:
            med_dt = 0.5
        dt_h = dt_h.fillna(med_dt)
        energy_mwh = base_diff.fillna(0.0) * dt_h.values
        daily = energy_mwh.groupby(df2.index.floor("D")).sum().reset_index()
        daily.columns = ["data", f"corte_{fonte}_mwh"]
    else:
        # mensal agregado
        vcol = None
        for c in df.columns:
            nc = _norm_text(c)
            if any(
                k in nc for k in ["mwh", "energia_nao_gerada", "corte", "restricao"]
            ):
                if pd.api.types.is_numeric_dtype(df[c]):
                    vcol = c
                    break

        if vcol is not None:
            dfd = df[["data", vcol]].copy()
            dfd[vcol] = pd.to_numeric(dfd[vcol], errors="coerce").fillna(0.0)
            daily = _explode_month_to_daily(dfd, vcol, date_col="data")
            daily = daily.rename(columns={vcol: f"corte_{fonte}_mwh"})
        elif {"val_geracaoestimada", "val_geracaoverificada"} <= set(df.columns):
            tmp = df[["data", "val_geracaoestimada", "val_geracaoverificada"]].copy()
            for c in ["val_geracaoestimada", "val_geracaoverificada"]:
                tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
            tmp["diff_mwmed"] = (
                tmp["val_geracaoestimada"] - tmp["val_geracaoverificada"]
            ).clip(lower=0)
            g = tmp.groupby(tmp["data"].dt.to_period("M"))[["diff_mwmed"]].sum()
            g.index = g.index.to_timestamp(how="start")
            days = g.index.to_series().apply(lambda d: (d + pd.offsets.MonthEnd(0)).day)
            mwh = g["diff_mwmed"] * 24 * days
            dfd = pd.DataFrame({"data": mwh.index, f"corte_{fonte}_mwh": mwh.values})
            daily = _explode_month_to_daily(dfd, f"corte_{fonte}_mwh", date_col="data")
        else:
            warnings.warn(
                "Cortes (mensal): sem coluna de MWh e sem pares estimada/verificada; pulando."
            )
            return None

    out = out_dir / f"ons_cortes_{fonte}_diario.csv"
    daily.to_csv(out, index=False)
    return out


def etl_carga(input_path: Path, out_dir: Path, submercado: str) -> Optional[Path]:
    """Padroniza carga para diário com colunas `data` e `carga_mwh`.

    - Se vier horária (MWmed/h): soma por dia (≈ MWh/dia).
    - Se vier diária: apenas normaliza nomes e ordena.

    Args:
      input_path (Path): Arquivo de carga (diário ou horário).
      out_dir (Path): Diretório de saída.
      submercado (str): Submercado alvo.

    Returns:
      Path|None: Caminho do CSV diário gerado ou None.
    """
    if not input_path.exists():
        return None
    df = _read_csv_auto(input_path)
    df = _normalize_columns(df)

    # filtra submercado se existir
    sub_cols = [
        c
        for c in df.columns
        if c in {"submercado", "subsistema", "id_subsistema", "nom_subsistema"}
    ]
    if sub_cols:
        sub_col = sub_cols[0]
        df = df.loc[df[sub_col].apply(lambda x: _match_submercado(str(x), submercado))]
    if df.empty:
        warnings.warn("Carga sem linhas para o submercado.")
        return None

    # detectar valor e data/hora
    preferred_val_cols = ["val_cargaglobalsmmg", "val_cargaglobal"]
    val_col = next((c for c in preferred_val_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])), None)
    if val_col is None:
        for c in df.columns:
            nc = _norm_text(c)
            if ("cargaverificada" in nc or "carga" in nc or "demanda" in nc) and ("mwmed" in nc or "mw" in nc or "valor" in nc):
                if pd.api.types.is_numeric_dtype(df[c]):
                    val_col = c
                    break
    if val_col is None and "carga_mwh" in df.columns:
        val_col = "carga_mwh"
    if val_col is None and "valor" in df.columns and pd.api.types.is_numeric_dtype(df["valor"]):
        val_col = "valor"
    if val_col is None:
        warnings.warn("Coluna de valor de carga não detectada; pulando carga.")
        return None

    # tenta inferir se é horário ou semi-horário
    is_hourly = False
    if "din_instante" in df.columns:
        df = df.rename(columns={"din_instante": "data"})
        is_hourly = True
    elif "din_referenciautc" in df.columns:
        df = df.rename(columns={"din_referenciautc": "data"})
        is_hourly = True
    elif "dat_referencia" in df.columns:
        df = df.rename(columns={"dat_referencia": "data"})
    else:
        has_hora = any(c in df.columns for c in ["hora", "hr", "h"]) or any("hora" in c for c in df.columns)
        if has_hora:
            is_hourly = True
    if is_hourly:
        df = _infer_datetime_index(df)
        outd = _ensure_daily_sum(df[[val_col]].rename(columns={val_col: "carga_mwh"}))
        outd = outd.reset_index()
    else:
        if "data" not in df.columns:
            cand = [c for c in df.columns if "data" in c or "date" in c]
            if cand:
                df = df.rename(columns={cand[0]: "data"})
        dfd = df[["data", val_col]].rename(columns={val_col: "carga_mwh"}).copy()
        dfd["data"] = pd.to_datetime(dfd["data"], errors="coerce")
        outd = dfd.dropna(subset=["data"]).sort_values("data")

    out = out_dir / "ons_carga_diaria.csv"
    outd.to_csv(out, index=False)
    return out


"""
ETL para dados do ONS: converte arquivos brutos (horários/mensais) em CSVs diários padronizados
esperados pelo pipeline. Meteorologia (NASA ou outros provedores) vive em `src/meteo.py`.
"""


def main():
    """CLI do ETL ONS: converte brutos em diários padronizados (CSV)."""
    ap = argparse.ArgumentParser(
        description="ETL ONS -> CSVs diários esperados pelo pipeline."
    )
    ap.add_argument(
        "--raw-dir", default="data/raw", help="Diretório com arquivos brutos baixados."
    )
    ap.add_argument(
        "--out-dir",
        default="data/raw",
        help="Diretório de saída para CSVs diários padronizados.",
    )
    ap.add_argument(
        "--submercado", default="SE/CO", help="Submercado alvo (ex.: 'SE/CO')."
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Sobrescreve arquivos de saída se existirem.",
    )
    args = ap.parse_args()

    raw = Path(args.raw_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Entradas brutas esperadas (nomes sugeridos)
    paths = {
        "balanco": raw / "ons_balanco_subsistema_horario.csv",
        "intercambio": raw / "ons_intercambios_entre_subsistemas_horario.csv",
        "carga": raw / "ons_carga.csv",  # aceita diário ou horário
        "ena": raw / "ons_ena_diario_subsistema.csv",
        "ear": raw / "ons_ear_diario_subsistema.csv",
        "corte_eolica": raw / "ons_constrained_off_eolica_mensal.csv",
        "corte_fv": raw / "ons_constrained_off_fv_mensal.csv",
    }

    # Executa ETLs conforme disponíveis
    produced: Dict[str, Optional[Path]] = {}
    produced["ger_fontes"] = etl_balanco_subsistema_horario(
        paths["balanco"], out, args.submercado
    )
    produced["intercambio"] = etl_intercambio_horario(
        paths["intercambio"], out, args.submercado
    )
    produced["carga"] = etl_carga(paths["carga"], out, args.submercado)
    produced["ena"] = etl_ena_diaria(paths["ena"], out, args.submercado)
    produced["ear"] = etl_ear_diaria(paths["ear"], out, args.submercado)
    produced["cortes_eolica"] = etl_constrained_off_mensal(
        paths["corte_eolica"], out, "eolica", args.submercado
    )
    produced["cortes_fv"] = etl_constrained_off_mensal(
        paths["corte_fv"], out, "fv", args.submercado
    )

    # feedback
    for k, v in produced.items():
        if v is not None:
            print(f"[OK] {k}: {v}")
        else:
            print(f"[--] {k}: arquivo de entrada ausente ou não reconhecido; ignorado.")


if __name__ == "__main__":
    main()
