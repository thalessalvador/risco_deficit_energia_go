from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple
import warnings
import pandas as pd


def fetch_power_nasa(
    out_dir: Path | str,
    overwrite: bool = False,
    pontos: Optional[List[Tuple[float, float]]] = None,
    inicio: str = "2018-01-01",
    fim: Optional[str] = None,
) -> Optional[Path]:
    """Baixa meteorologia diária via NASA POWER e salva `clima_go_diario.csv`.

    Args:
      out_dir (Path|str): Diretório de saída.
      overwrite (bool): Se True, sobrescreve arquivo existente.
      pontos (list[tuple[float,float]]|None): Lista (lat,lon) para média espacial.
      inicio (str): Data inicial (YYYY-MM-DD).
      fim (str|None): Data final (YYYY-MM-DD); padrão = hoje.

    Returns:
      Path|None: Caminho do CSV gerado ou None se falhar/sem rede.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "clima_go_diario.csv"
    if out_path.exists() and not overwrite:
        return out_path

    try:
        import requests  # type: ignore
    except Exception:
        warnings.warn("Biblioteca 'requests' não instalada; pulando meteorologia NASA POWER.")
        return None

    if pontos is None:
        # grade simples sobre GO
        pontos = [(-16.7, -49.3), (-15.9, -48.2), (-18.0, -49.7), (-13.6, -46.9)]

    if fim is None:
        fim = pd.Timestamp.today().strftime("%Y-%m-%d")

    def baixa(lat: float, lon: float) -> pd.DataFrame:
        url = (
            "https://power.larc.nasa.gov/api/temporal/daily/point"
            f"?parameters=ALLSKY_SFC_SW_DWN,T2M,PRECTOTCORR&community=RE&longitude={lon}&latitude={lat}"
            f"&start={inicio.replace('-', '')}&end={fim.replace('-', '')}&format=JSON"
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
    if not dfs:
        return None
    df = dfs[0][["data", "ghi", "temp2m_c", "precipitacao_mm"]].copy()
    for d in dfs[1:]:
        df[["ghi", "temp2m_c", "precipitacao_mm"]] += d[["ghi", "temp2m_c", "precipitacao_mm"]]
    df[["ghi", "temp2m_c", "precipitacao_mm"]] /= len(dfs)
    df.to_csv(out_path, index=False)
    return out_path


def fetch_meteorologia(out_dir: Path | str, provider: str = "nasa_power", **kwargs) -> Optional[Path]:
    """Ponto de entrada genérico para obter meteorologia de diferentes provedores.

    Args:
      out_dir (Path|str): Diretório de saída.
      provider (str): Identificador do provedor (ex.: "nasa_power").
      **kwargs: Parâmetros específicos do provedor.

    Returns:
      Path|None: Caminho do CSV gerado ou None se provedor não suportado.
    """
    provider = (provider or "").lower()
    if provider in {"nasa_power", "nasapower", "nasa"}:
        return fetch_power_nasa(out_dir, **kwargs)
    warnings.warn(f"Provedor de meteorologia '{provider}' não suportado.")
    return None
