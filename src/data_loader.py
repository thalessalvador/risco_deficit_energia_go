# src/data_loader.py
from __future__ import annotations
import pandas as pd
from pathlib import Path
from typing import Dict

TZ = "America/Sao_Paulo"

def read_csv_timeseries(path: Path, dt_col: str = "data", to_tz: str = TZ) -> pd.DataFrame:
    """Lê um CSV temporal e retorna um DataFrame com índice datetime na timezone alvo.

    Args:
      path (Path): Caminho do arquivo CSV a ser lido.
      dt_col (str): Nome da coluna de data/hora no CSV (padrão: "data").
      to_tz (str): Timezone de destino para conversão (padrão: America/Sao_Paulo).

    Returns:
      pandas.DataFrame: Série temporal com índice datetime (tz-aware) e colunas numéricas.
    """
    df = pd.read_csv(path)
    if dt_col not in df.columns:
        raise ValueError(f"Coluna de data '{dt_col}' não encontrada em {path.name}")
    df[dt_col] = pd.to_datetime(df[dt_col], utc=True, errors="coerce")
    df = df.set_index(dt_col).sort_index()
    try:
        df = df.tz_convert(to_tz)
    except Exception:
        df = df.tz_localize("UTC").tz_convert(to_tz)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def ensure_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Garante frequência diária contínua com média diária e preenchimento curto.

    Args:
      df (pandas.DataFrame): DataFrame com índice datetime.

    Returns:
      pandas.DataFrame: Série diária contínua (média por dia) com ffill limitado a 7 dias.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Índice temporal inválido")
    # Converter para UTC se índice tiver timezone para evitar problemas de DST
    try:
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df = df.tz_convert("UTC")
    except Exception:
        pass
    if (df.index.freq is None) or (df.index.freq != "D"):
        df = df.resample("D").mean()
    return df.ffill(limit=7)

def load_all_sources(cfg: Dict) -> Dict[str, pd.DataFrame]:
    """Carrega todos os dados padronizados a partir de `data/raw` conforme `configs`.

    Espera arquivos diários já padronizados pelo ETL e retorna um dicionário de
    DataFrames diários alinhados pelo índice de data.

    Args:
      cfg (dict): Configurações do projeto (paths etc.).

    Returns:
      dict[str, pandas.DataFrame]: Map com chaves como "carga", "ger_fontes", etc.
    """
    raw = Path(cfg["paths"]["raw_dir"])

    files = {
        "carga": raw / "ons_carga_diaria.csv",
        "ger_fontes": raw / "ons_geracao_fontes_diaria.csv",
        "intercambio": raw / "ons_intercambio_diario.csv",
        "ena": raw / "ons_ena_diaria.csv",
        "ear": raw / "ons_ear_diaria.csv",
        "cortes_eolica": raw / "ons_cortes_eolica_diario.csv",
        "cortes_fv": raw / "ons_cortes_fv_diario.csv",
        "clima": raw / "clima_go_diario.csv",
    }

    data = {}
    for name, path in files.items():
        if path.exists():
            df = read_csv_timeseries(path, dt_col="data")
            df = ensure_daily(df)
            data[name] = df
    if not data:
        raise FileNotFoundError(f"Nenhum CSV encontrado em {raw}.")
    return data
