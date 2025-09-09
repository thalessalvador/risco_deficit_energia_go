from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import unicodedata
import warnings

import numpy as np
import pandas as pd


def _norm_text(s: str) -> str:
    """Lower, strip, remove accents and collapse spaces/underscores."""
    s = s.strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("/", "").replace("-", "_")
    s = s.replace(" ", "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_norm_text(c) for c in df.columns]
    return df


SUBMERCADO_ALIASES = {
    "seco",
    "se_co",
    "sudeste_centrooeste",
    "sudeste_centro_oeste",
    "sudeste_centro_oeste_seco",
    "sudeste_centrooeste_seco",
    "se/co",
}


def _match_submercado(value: str, alvo: str = "seco") -> bool:
    v = _norm_text(str(value))
    a = _norm_text(alvo)
    if v == a:
        return True
    if v in SUBMERCADO_ALIASES and a in SUBMERCADO_ALIASES:
        return True
    return False


def _infer_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Try to create a DatetimeIndex from common date/hour columns.

    - If 'data' exists and is datetime-like, use it.
    - If 'data' and 'hora' exist, combine.
    - If 'datetime' exists, use it.
    - Otherwise, raise.
    """
    df = df.copy()
    cols = set(df.columns)
    if "data" in cols and np.issubdtype(df["data"].dtype, np.datetime64):
        dt = pd.to_datetime(df["data"], errors="coerce")
    elif {"data", "hora"} <= cols:
        dt = pd.to_datetime(
            df["data"].astype(str) + " " + df["hora"].astype(str), errors="coerce"
        )
    elif "datetime" in cols:
        dt = pd.to_datetime(df["datetime"], errors="coerce")
    else:
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


def _ensure_daily_sum(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Índice temporal não é DatetimeIndex")
    # Para séries de energia horárias em MWmed, somar no dia ≈ MWh/dia
    return df.resample("D").sum(min_count=1)


def _ensure_daily_mean(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Índice temporal não é DatetimeIndex")
    return df.resample("D").mean()


def etl_balanco_subsistema_horario(
    path: Path, out_dir: Path, submercado: str
) -> Optional[Path]:
    """Transforma Balanço de Energia por Subsistema (horário) em geração diária por fonte.

    Espera colunas com geração por fonte (hidráulica/térmica/eólica/solar) e uma coluna de subsistema.
    Saída: ons_geracao_fontes_diaria.csv com colunas em MWh/dia.
    """
    if not path.exists():
        return None
    df0 = pd.read_csv(path)
    df0 = _normalize_columns(df0)

    # identificar coluna de subsistema
    sub_cols = [c for c in df0.columns if c in {"submercado", "subsistema"}]
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
    """Transforma Intercâmbios Entre Subsistemas (horário) em import/export diário do submercado.

    Espera colunas que indiquem 'de' e 'para' subsistema e uma coluna de valor (MWmed).
    Saída: ons_intercambio_diario.csv com import_mwh/export_mwh por dia.
    """
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = _normalize_columns(df)
    # identificar colunas from/to e valor
    from_cols = [
        c for c in df.columns if c in {"de", "from", "origem", "subsistema_de"}
    ]
    to_cols = [
        c for c in df.columns if c in {"para", "to", "destino", "subsistema_para"}
    ]
    val_cols = [c for c in df.columns if c in {"valor", "mwmed", "mw", "potencia"}]
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
    """Padroniza ENA Diário por Subsistema → ons_ena_diaria.csv com coluna ena_mwmed."""
    if not path.exists():
        return None
    df = pd.read_csv(path)
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
    """Padroniza EAR Diário por Subsistema → ons_ear_diaria.csv com coluna ear_pct."""
    if not path.exists():
        return None
    df = pd.read_csv(path)
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
    """Distribui um valor mensal uniformemente pelos dias do mês, retornando série diária."""
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
    """Padroniza Constrained-off mensal (eólica/FV) → série diária em MWh.

    fonte: 'eolica' ou 'fv'. Saída: ons_cortes_<fonte>_diario.csv com coluna corte_<fonte>_mwh
    """
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = _normalize_columns(df)

    # filtra submercado se disponível
    sub_cols = [c for c in df.columns if c in {"submercado", "subsistema"}]
    if sub_cols:
        sub_col = sub_cols[0]
        df = df.loc[df[sub_col].apply(lambda x: _match_submercado(str(x), submercado))]
    if df.empty:
        warnings.warn(f"Constrained-off {fonte} sem linhas para o submercado.")
        return None

    # detectar data mensal e valor (MWh)
    if "data" not in df.columns:
        # tenta 'competencia', 'mes'
        if "competencia" in df.columns:
            df = df.rename(columns={"competencia": "data"})
        elif {"ano", "mes"} <= set(df.columns):
            df["data"] = pd.to_datetime(
                dict(year=df["ano"].astype(int), month=df["mes"].astype(int), day=1)
            )
        else:
            warnings.warn("Coluna mensal de data não encontrada nos cortes; pulando.")
            return None
    # valor
    val_candidates = [
        c
        for c in df.columns
        if any(k in c for k in ["mwh", "energia_nao_gerada", "corte", "restricao"])
    ]
    if not val_candidates:
        warnings.warn("Coluna de valor (MWh) não encontrada nos cortes; pulando.")
        return None
    vcol = val_candidates[0]

    dfd = df[["data", vcol]].copy()
    dfd["data"] = pd.to_datetime(dfd["data"], errors="coerce")
    dfd = dfd.dropna(subset=["data"]).sort_values("data")
    daily = _explode_month_to_daily(dfd, vcol, date_col="data")
    daily = daily.rename(columns={vcol: f"corte_{fonte}_mwh"})
    out = out_dir / f"ons_cortes_{fonte}_diario.csv"
    daily.to_csv(out, index=False)
    return out


def etl_carga(input_path: Path, out_dir: Path, submercado: str) -> Optional[Path]:
    """Padroniza carga diária para ter coluna 'carga_mwh' e 'data'.

    Se vier horária (carga MWmed/h), soma para diário. Se já vier diário, apenas renomeia/ordena.
    """
    if not input_path.exists():
        return None
    df = pd.read_csv(input_path)
    df = _normalize_columns(df)

    # filtra submercado se existir
    sub_cols = [c for c in df.columns if c in {"submercado", "subsistema"}]
    if sub_cols:
        sub_col = sub_cols[0]
        df = df.loc[df[sub_col].apply(lambda x: _match_submercado(str(x), submercado))]
    if df.empty:
        warnings.warn("Carga sem linhas para o submercado.")
        return None

    # detectar valor e data/hora
    val_col = None
    for c in df.columns:
        nc = _norm_text(c)
        if any(k in nc for k in ["carga", "demanda"]) and any(
            k in nc for k in ["mw", "mwh", "mwmed", "valor"]
        ):
            val_col = c
            break
    if val_col is None:
        # fallback comum: 'carga_mwh' ou 'valor'
        val_col = (
            "carga_mwh"
            if "carga_mwh" in df.columns
            else ("valor" if "valor" in df.columns else None)
        )
    if val_col is None:
        warnings.warn("Coluna de valor de carga não detectada; pulando carga.")
        return None

    # tenta inferir se é horário ou diário
    is_hourly = False
    has_hora = any(c in df.columns for c in ["hora", "hr", "h"]) or any(
        "hora" in c for c in df.columns
    )
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


def maybe_fetch_nasa_power(out_dir: Path, overwrite: bool = False) -> Optional[Path]:
    """Opcional: baixa e agrega NASA POWER diário para Goiás e salva clima_go_diario.csv.

    Requer 'requests'. Se não houver rede ou a lib não estiver instalada, apenas retorna None.
    """
    out_path = out_dir / "clima_go_diario.csv"
    if out_path.exists() and not overwrite:
        return out_path
    try:
        import requests  # type: ignore
    except Exception:
        warnings.warn(
            "Biblioteca 'requests' não instalada; pulando fetch da NASA POWER."
        )
        return None

    # pontos dentro dos limites de goiás (aprox)
    # pontos = [(-19,-51), (-18,-53), (-18,-52), (-18,-51), (-18,-50), (-18.00,-49.70), (-18,-49), (-18,-48), (-17,-53), (-17,-52), (-17,-51), (-17,-50), (-17,-49), (-17,-48), (-16.70,-49.30), (-16,-52), (-16,-51), (-16,-50), (-16,-49), (-15.90,-48.20), (-15,-51), (-15,-50), (-15,-49), (-15,-48), (-15,-47), (-14,-50), (-14,-49), (-14,-48), (-14,-47), (-13.60,-46.90), (-13,-50), (-13,-49)]
    ini, fim = "2018-01-01", pd.Timestamp.today().strftime("%Y-%m-%d")

    def baixa(lat: float, lon: float) -> pd.DataFrame:
        url = (
            "https://power.larc.nasa.gov/api/temporal/daily/point"
            f"?parameters=ALLSKY_SFC_SW_DWN,T2M,PRECTOTCORR&community=RE&longitude={lon}&latitude={lat}"
            f"&start={ini.replace('-', '')}&end={fim.replace('-', '')}&format=JSON"
        )
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        j = r.json()["properties"]["parameter"]
        df = pd.DataFrame(
            {
                "data": pd.to_datetime(pd.Series(j["ALLSKY_SFC_SW_DWN"]).index),
                "ghi": list(j["ALLSKY_SFC_SW_DWN"].values()),
                "temp2m_c": list(j["T2M"].values()),
                "precipitacao_mm": list(j["PRECTOTCORR"].values()),
            }
        )
        return df

    dfs = [baixa(lat, lon) for lat, lon in pontos]
    df = dfs[0][["data", "ghi", "temp2m_c", "precipitacao_mm"]].copy()
    for d in dfs[1:]:
        df[["ghi", "temp2m_c", "precipitacao_mm"]] += d[
            ["ghi", "temp2m_c", "precipitacao_mm"]
        ]
    df[["ghi", "temp2m_c", "precipitacao_mm"]] /= len(dfs)
    df.to_csv(out_path, index=False)
    return out_path


def main():
    ap = argparse.ArgumentParser(
        description="ETL ONS/NASA → CSVs diários esperados pelo pipeline."
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
        "--fetch-nasa",
        action="store_true",
        help="Baixa NASA POWER e salva clima_go_diario.csv.",
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

    if args.fetch_nasa:
        try:
            produced["clima"] = maybe_fetch_nasa_power(out, overwrite=args.overwrite)
        except Exception as e:
            warnings.warn(f"Falha ao baixar NASA POWER: {e}")

    # feedback
    for k, v in produced.items():
        if v is not None:
            print(f"[OK] {k}: {v}")
        else:
            print(f"[--] {k}: arquivo de entrada ausente ou não reconhecido; ignorado.")


if __name__ == "__main__":
    main()
