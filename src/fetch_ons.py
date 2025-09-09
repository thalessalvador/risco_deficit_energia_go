from __future__ import annotations
import argparse
import io
import os
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


CKAN_BASE = "https://dados.ons.org.br/api/3/action"


def _norm(s: str) -> str:
    import unicodedata

    s = s.strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _pick_resource(resources: List[dict], prefer_formats: Tuple[str, ...] = ("CSV", "XLSX", "ZIP")) -> Optional[dict]:
    if not resources:
        return None
    # sort by preferred format and latest (if 'last_modified' present)
    def key(r):
        fmt = (r.get("format") or r.get("mimetype") or "").upper()
        try:
            fmt_idx = prefer_formats.index(fmt)
        except ValueError:
            fmt_idx = len(prefer_formats)
        last = r.get("last_modified") or r.get("created") or ""
        return (fmt_idx, last)

    return sorted(resources, key=key)[0]


class CkanClient:
    def __init__(self, base: str = CKAN_BASE, timeout: int = 60):
        self.base = base.rstrip("/")
        self.timeout = timeout

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        import requests

        url = f"{self.base}/{path.lstrip('/')}"
        r = requests.get(url, params=params, timeout=self.timeout)
        r.raise_for_status()
        j = r.json()
        if not j.get("success"):
            raise RuntimeError(f"CKAN call failed: {url}")
        return j["result"]

    def search_package(self, query: str, rows: int = 10) -> List[dict]:
        res = self._get("package_search", {"q": query, "rows": rows})
        return res.get("results", [])

    def show_package(self, package_id: str) -> dict:
        return self._get("package_show", {"id": package_id})


@dataclass
class DatasetSpec:
    query: str
    out_name: str
    note: str = ""
    resource_filter: Optional[str] = None  # regex on resource name or description


DEFAULT_SPECS: Dict[str, DatasetSpec] = {
    # Titles on ONS portal may vary slightly; queries aim to be robust in PT/EN
    "balanco": DatasetSpec(
        query="Balanço de Energia nos Subsistemas",
        out_name="ons_balanco_subsistema_horario.csv",
        note="Usado para derivar geração por fonte (hidro/term/eolica/fv).",
        resource_filter=r"(?i)hor(a|á)ria|hour|hora",
    ),
    "intercambio": DatasetSpec(
        query="Intercâmbios Entre Subsistemas",
        out_name="ons_intercambios_entre_subsistemas_horario.csv",
        note="Fluxos entre subsistemas para calcular import/export do SE/CO.",
        resource_filter=r"(?i)hor(a|á)ria|hour|hora",
    ),
    "ena": DatasetSpec(
        query="ENA Diário por Subsistema",
        out_name="ons_ena_diario_subsistema.csv",
        note="ENA diária por subsistema (mwmed).",
        resource_filter=r"(?i)di(á|a)rio|daily",
    ),
    "ear": DatasetSpec(
        query="EAR Diário por Subsistema",
        out_name="ons_ear_diario_subsistema.csv",
        note="EAR diária por subsistema (%).",
        resource_filter=r"(?i)di(á|a)rio|daily",
    ),
    "corte_eolica": DatasetSpec(
        query="Restrição de Operação por Constrained-off de Usinas Eólicas",
        out_name="ons_constrained_off_eolica_mensal.csv",
        note="Cortes eólicos (mensal).",
        resource_filter=r"(?i)mensal|monthly",
    ),
    "corte_fv": DatasetSpec(
        query="Restrição de Operação por Constrained-off de Usinas Fotovoltaicas",
        out_name="ons_constrained_off_fv_mensal.csv",
        note="Cortes FV (mensal).",
        resource_filter=r"(?i)mensal|monthly",
    ),
    "carga": DatasetSpec(
        query="Carga Verificada",
        out_name="ons_carga.csv",
        note="Carga verificada (horária ou diária).",
        resource_filter=None,
    ),
}


def _download_resource(url: str, out_path: Path) -> Path:
    import requests

    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    # try to infer filename from headers
    cd = r.headers.get("content-disposition", "")
    fname = None
    m = re.search(r"filename=([^;]+)", cd)
    if m:
        fname = m.group(1).strip('"')
    # stream into memory or file depending on size
    content = r.content

    # detect type
    url_lower = url.lower()
    if (fname and fname.lower().endswith(".zip")) or url_lower.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            # pick the largest CSV/XLSX inside
            infos = z.infolist()
            infos = sorted(infos, key=lambda i: i.file_size, reverse=True)
            choice = None
            for info in infos:
                name = info.filename.lower()
                if name.endswith(".csv") or name.endswith(".xlsx"):
                    choice = info
                    break
            if choice is None:
                # fallback: first file
                choice = infos[0]
            data = z.read(choice)
            tmp = out_path.with_suffix(Path(choice.filename).suffix)
            tmp.write_bytes(data)
            return tmp
    else:
        out_path.write_bytes(content)
        return out_path


def _maybe_convert_to_csv(in_path: Path, out_csv: Path) -> Path:
    ext = in_path.suffix.lower()
    if ext == ".csv":
        if in_path != out_csv:
            out_csv.write_bytes(in_path.read_bytes())
        return out_csv
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(in_path)
        df.to_csv(out_csv, index=False)
        return out_csv
    else:
        # try to read as CSV regardless
        try:
            df = pd.read_csv(in_path)
            df.to_csv(out_csv, index=False)
            return out_csv
        except Exception:
            # leave as is
            return in_path


def fetch_one(client: CkanClient, spec: DatasetSpec, out_dir: Path, verbose: bool = True) -> Optional[Path]:
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)
    pkgs = client.search_package(spec.query, rows=20)
    if not pkgs:
        if verbose:
            print(f"[CKAN] Nenhum pacote encontrado para query: {spec.query}")
        return None

    # pick best package by normalized title similarity
    qn = _norm(spec.query)
    def score(pkg):
        title = _norm(pkg.get("title", ""))
        s = 0
        if qn in title or title in qn:
            s += 10
        # prefer most recent metadata_modified
        mm = pkg.get("metadata_modified") or ""
        return (s, mm)

    pkg = sorted(pkgs, key=score, reverse=True)[0]
    pkg_full = client.show_package(pkg["id"]) if "id" in pkg else pkg
    resources = pkg_full.get("resources", [])

    # optional filter by resource name/description
    if spec.resource_filter:
        rx = re.compile(spec.resource_filter)
        resources = [r for r in resources if rx.search((r.get("name") or "") + " " + (r.get("description") or ""))]
        if not resources:
            resources = pkg_full.get("resources", [])

    res = _pick_resource(resources)
    if not res:
        if verbose:
            print(f"[CKAN] Nenhum recurso adequado para: {spec.query}")
        return None

    url = res.get("url") or res.get("download_url")
    if not url:
        if verbose:
            print(f"[CKAN] Recurso sem URL para: {spec.query}")
        return None

    tmp_path = out_dir / (spec.out_name + ".tmp")
    if verbose:
        print(f"[CKAN] Baixando {spec.query} → {spec.out_name}")
    downloaded = _download_resource(url, tmp_path)
    final_target = out_dir / spec.out_name
    out = _maybe_convert_to_csv(downloaded, final_target)
    # cleanup tmp
    try:
        if tmp_path.exists():
            tmp_path.unlink()
        if downloaded != tmp_path and downloaded.exists() and downloaded.suffix != ".csv":
            # keep downloaded non-csv as reference with .orig extension
            downloaded.rename(final_target.with_suffix(final_target.suffix + ".orig"))
    except Exception:
        pass
    return out


def fetch_all(out_dir: Path, datasets: Optional[List[str]] = None, verbose: bool = True) -> Dict[str, Optional[Path]]:
    if datasets is None:
        datasets = list(DEFAULT_SPECS.keys())
    client = CkanClient()
    results: Dict[str, Optional[Path]] = {}
    for key in datasets:
        spec = DEFAULT_SPECS[key]
        try:
            p = fetch_one(client, spec, out_dir, verbose=verbose)
        except Exception as e:
            print(f"[ERRO] {key}: {e}")
            p = None
        results[key] = p
    return results


def main():
    ap = argparse.ArgumentParser(description="Baixa dados do ONS (CKAN) e salva em data/raw.")
    ap.add_argument("--out-dir", default="data/raw", help="Diretório onde salvar arquivos brutos.")
    ap.add_argument("--datasets", nargs="*", default=[], help=f"Quais datasets baixar: {list(DEFAULT_SPECS.keys())}")
    ap.add_argument("--all", action="store_true", help="Baixar todos os datasets suportados.")
    args = ap.parse_args()

    out = Path(args.out_dir)
    _ensure_dir(out)
    if args.all or not args.datasets:
        keys = list(DEFAULT_SPECS.keys())
    else:
        keys = args.datasets

    res = fetch_all(out, keys)
    for k, p in res.items():
        status = "OK" if p is not None else "--"
        print(f"[{status}] {k}: {p if p else 'não baixado'}")


if __name__ == "__main__":
    main()

