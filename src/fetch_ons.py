from __future__ import annotations
import argparse
import io
import os
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import pandas as pd


CKAN_BASE = "https://dados.ons.org.br/api/3/action"
SITE_BASE = CKAN_BASE.split("/api/3", 1)[0]


def _norm(s: str) -> str:
    """Normaliza texto (minúsculas, sem acentos, sem pontuação extra).

    Args:
      s (str): Texto de entrada.

    Returns:
      str: Texto normalizado.
    """
    import unicodedata

    s = s.strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _ensure_dir(p: Path) -> None:
    """Garante que o diretório exista (cria recursivamente)."""
    p.mkdir(parents=True, exist_ok=True)


def _pick_resource(resources: List[dict], prefer_formats: Tuple[str, ...] = ("CSV", "XLSX", "ZIP")) -> Optional[dict]:
    """Escolhe o melhor recurso (CSV/XLSX/ZIP mais recente) entre os disponíveis.

    Args:
      resources (list[dict]): Recursos do pacote CKAN.
      prefer_formats (tuple[str,...]): Ordem de preferência por formato.

    Returns:
      dict|None: Recurso escolhido ou None se vazio.
    """
    if not resources:
        return None
    # ordena por formato preferido e data mais recente
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
        """Executa GET no endpoint CKAN e retorna o campo `result`."""
        import requests

        url = f"{self.base}/{path.lstrip('/')}"
        r = requests.get(url, params=params, timeout=self.timeout)
        r.raise_for_status()
        j = r.json()
        if not j.get("success"):
            raise RuntimeError(f"CKAN call failed: {url}")
        return j["result"]

    def search_package(self, query: str, rows: int = 10) -> List[dict]:
        """Busca pacotes por texto livre."""
        res = self._get("package_search", {"q": query, "rows": rows})
        return res.get("results", [])

    def show_package(self, package_id: str) -> dict:
        """Obtém metadados completos de um pacote por id."""
        return self._get("package_show", {"id": package_id})


@dataclass
class DatasetSpec:
    """Especificação de dataset a baixar do CKAN.

    Atributos:
      query (str): Texto de busca do pacote.
      out_name (str): Nome do arquivo de saída (CSV) em `data/raw`.
      note (str): Observação/uso interno.
      resource_filter (str|None): Regex para filtrar recursos por nome/descrição.
    """
    query: str
    out_name: str
    note: str = ""
    resource_filter: Optional[str] = None


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
    """Baixa um recurso (CSV/XLSX/ZIP) de forma robusta e retorna o caminho local.

    - Faz download em streaming para não estourar memória em arquivos grandes.
    - Se for ZIP, extrai o maior CSV/XLSX contido e retorna o caminho extraído.
    - Caso contrário, retorna o caminho do arquivo baixado (pode ser .csv ou .xlsx).
    """
    import requests

    # Baixa para arquivo temporário no disco
    tmp_file = out_path  # já recebemos um caminho temporário do chamador
    with requests.get(url, stream=True, timeout=(15, 300)) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        read = 0
        last_mb_print = 0
        with open(tmp_file, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):  # 1MB
                if not chunk:
                    continue
                f.write(chunk)
                read += len(chunk)
                # imprime progresso a cada 50MB
                if total and (read - last_mb_print) >= 50 * 1024 * 1024:
                    mb = read / (1024 * 1024)
                    tot_mb = total / (1024 * 1024)
                    print(f"  [download] {mb:.0f}/{tot_mb:.0f} MB")
                    last_mb_print = read

    # ZIP? Usa header para decidir
    try:
        is_zip = zipfile.is_zipfile(tmp_file)
    except Exception:
        is_zip = False

    if is_zip:
        with zipfile.ZipFile(tmp_file, "r") as z:
            infos = sorted(z.infolist(), key=lambda i: i.file_size, reverse=True)
            choice = None
            for info in infos:
                name = info.filename.lower()
                if name.endswith(".csv") or name.endswith(".xlsx"):
                    choice = info
                    break
            if choice is None:
                choice = infos[0]
            data = z.read(choice)
            extracted = out_path.with_suffix(Path(choice.filename).suffix)
            with open(extracted, "wb") as f:
                f.write(data)
        return extracted
    else:
        return tmp_file


def _maybe_convert_to_csv(in_path: Path, out_csv: Path) -> Path:
    """Converte XLS/XLSX para CSV; para ZIPs já extraídos, garante saída .csv."""
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
        # try to read as CSV regardless (detect separator)
        try:
            df = pd.read_csv(in_path, sep=None, engine="python")
            df.to_csv(out_csv, index=False)
            return out_csv
        except Exception:
            # tenta como Excel mesmo com extensão desconhecida
            try:
                df = pd.read_excel(in_path)
                df.to_csv(out_csv, index=False)
                return out_csv
            except Exception:
                # deixa como está; o chamador pode lidar
                return in_path


def _append_csv_files(sources: List[Path], target: Path) -> Path:
    """Concatena múltiplos CSVs (mesmo schema) em um único arquivo `target`.

    Faz append em disco para evitar alto uso de memória.
    """
    if not sources:
        raise ValueError("Nenhuma fonte para concatenar")
    with open(target, "wb") as out:
        for i, src in enumerate(sources):
            with open(src, "rb") as f:
                if i == 0:
                    out.write(f.read())
                else:
                    # pula a primeira linha (cabeçalho)
                    first = f.readline()
                    out.write(f.read())
    return target


def _parse_iso_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s.split("Z")[0], fmt)
        except Exception:
            continue
    return None


def _extract_period_ym_from_resource(r: dict) -> Optional[Tuple[int, Optional[int]]]:
    """Extrai (ano, mes?) do recurso a partir do nome/descrição/URL ou metadados.

    Retorna (YYYY, MM|None) ou None se não conseguir inferir.
    """
    key = _period_key_from_resource(r)
    m = re.match(r"^(\d{4})(?:-(\d{2}))?$", key)
    if m:
        y = int(m.group(1))
        mth = int(m.group(2)) if m.group(2) else None
        return y, mth
    return None


def _parse_since_until(since: Optional[str], until: Optional[str]) -> Tuple[Optional[Tuple[int, Optional[int]]], Optional[Tuple[int, Optional[int]]]]:
    def _p(val: Optional[str]) -> Optional[Tuple[int, Optional[int]]]:
        if not val:
            return None
        m = re.match(r"^(\d{4})(?:-(\d{2}))?$", val)
        if not m:
            return None
        y = int(m.group(1))
        mth = int(m.group(2)) if m.group(2) else None
        return y, mth

    return _p(since), _p(until)


def _resource_matches_since_until(r: dict, since: Optional[str], until: Optional[str]) -> bool:
    """Decide se um recurso entra no intervalo.

    Regra:
    - Primeiro tenta inferir período do recurso (YYYY[-MM]) pelo nome/URL.
    - Se conseguir, compara com since/until parseados como YYYY[-MM].
    - Se não conseguir, cai no filtro simplificado por substring no nome/descrição.
    """
    if not since and not until:
        return True

    # tenta por período explícito no nome/URL
    r_ym = _extract_period_ym_from_resource(r)
    s_ym, u_ym = _parse_since_until(since, until)
    if r_ym:
        ry, rm = r_ym
        ok_since = True
        ok_until = True
        if s_ym:
            sy, sm = s_ym
            ok_since = (ry > sy) or (ry == sy and (sm is None or (rm is not None and rm >= sm)))
        if u_ym:
            uy, um = u_ym
            ok_until = (ry < uy) or (ry == uy and (um is None or (rm is not None and rm <= um)))
        return ok_since and ok_until

    # fallback por substring
    name = (r.get("name") or "") + " " + (r.get("description") or "")
    ok_since = True if not since else (since in name)
    ok_until = True if not until else (until in name)
    return ok_since and ok_until


def _list_resources(client: CkanClient, query: str, resource_filter: Optional[str], verbose: bool) -> Tuple[dict, List[dict]]:
    pkgs = client.search_package(query, rows=50)
    if not pkgs:
        if verbose:
            print(f"[CKAN] Nenhum pacote encontrado para query: {query}")
        return {}, []

    # escolhe melhor pacote por similaridade com o título normalizado
    qn = _norm(query)
    def score(pkg):
        title = _norm(pkg.get("title", ""))
        s = 0
        if qn in title or title in qn:
            s += 10
        mm = pkg.get("metadata_modified") or ""
        return (s, mm)

    pkg = sorted(pkgs, key=score, reverse=True)[0]
    pkg_full = client.show_package(pkg["id"]) if "id" in pkg else pkg
    resources = pkg_full.get("resources", [])
    # filtro opcional por regex em nome/descrição do recurso
    if resource_filter:
        rx = re.compile(resource_filter)
        filtered = [r for r in resources if rx.search((r.get("name") or "") + " " + (r.get("description") or ""))]
        # se o filtro ficou muito restritivo, volte à lista completa
        resources = filtered or resources
    return pkg_full, resources


def _pref_index(fmt: Optional[str], prefer: Tuple[str, ...] = ("CSV", "XLSX", "ZIP")) -> int:
    f = (fmt or "").upper()
    try:
        return prefer.index(f)
    except ValueError:
        return len(prefer)


_RE_YM = re.compile(r"(20\d{2})[-_/.]?(0[1-9]|1[0-2])")
_RE_Y = re.compile(r"(19\d{2}|20\d{2})")


def _period_key_from_resource(r: dict) -> str:
    """Extrai uma chave de período (YYYY-MM ou YYYY) do nome/descrição/URL do recurso.

    Fallback para ano-mês de last_modified/created quando possível.
    """
    name = (r.get("name") or "") + " " + (r.get("description") or "") + " " + (r.get("url") or "")
    m = _RE_YM.search(name)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    m = _RE_Y.search(name)
    if m:
        return m.group(1)
    lm = _parse_iso_dt(r.get("last_modified") or r.get("created"))
    if lm:
        return f"{lm.year:04d}-{lm.month:02d}"
    # fallback: usa o próprio nome
    return (r.get("name") or "").strip() or (r.get("id") or "unknown")


def _group_prefer_by_period(resources: List[dict], prefer: Tuple[str, ...] = ("CSV", "XLSX", "ZIP")) -> List[dict]:
    """Agrupa recursos por período e escolhe o formato preferido em cada grupo."""
    groups: Dict[str, List[dict]] = {}
    for r in resources:
        k = _period_key_from_resource(r)
        groups.setdefault(k, []).append(r)
    chosen: List[dict] = []
    for k, items in groups.items():
        items_sorted = sorted(items, key=lambda r: (_pref_index(r.get("format"), prefer), r.get("last_modified") or r.get("created") or ""))
        chosen.append(items_sorted[0])
    # ordena por período (string YYYY[-MM] ordena lexicograficamente)
    chosen = sorted(chosen, key=lambda r: _period_key_from_resource(r))
    return chosen


def _is_data_resource(r: dict) -> bool:
    """Retorna True se o recurso parece ser dado (CSV/XLSX/ZIP) e não documentação.

    Exclui PDFs/JSON e nomes que contenham 'dicionario', 'dictionary', 'glossario',
    'metadados', etc.
    """
    fmt = (r.get("format") or r.get("mimetype") or "").upper()
    url = (r.get("url") or r.get("download_url") or "").lower()
    name = ((r.get("name") or "") + " " + (r.get("description") or "")).lower()
    if any(bad in name for bad in ["dicionario", "dictionary", "glossario", "metadado", "metadata"]):
        return False
    if fmt in {"CSV", "XLSX", "XLS", "ZIP"}:
        return True
    if url.endswith(".csv") or url.endswith(".xlsx") or url.endswith(".xls") or url.endswith(".zip"):
        return True
    return False


def fetch_one(client: CkanClient, spec: DatasetSpec, out_dir: Path, verbose: bool = True, since: Optional[str] = None, until: Optional[str] = None, overwrite: bool = False) -> Optional[Path]:
    """Baixa um dataset do ONS e salva com nome padronizado em `out_dir`.

    Args:
      client (CkanClient): Cliente CKAN.
      spec (DatasetSpec): Especificação do dataset.
      out_dir (Path): Diretório de saída.
      verbose (bool): Se True, imprime progresso.

    Returns:
      Path|None: Caminho para o CSV gerado ou None se não encontrado.
    """
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)
    pkg_full, resources = _list_resources(client, spec.query, spec.resource_filter, verbose)
    if not resources:
        return None

    # mantém apenas recursos de dados (descarta PDFs/dicionários)
    resources = [r for r in resources if _is_data_resource(r)]
    # tenta escolher o mais recente dentro do intervalo (se houver since/until)
    if since or until:
        candidates = [r for r in resources if _resource_matches_since_until(r, since, until)] or resources
        res = _pick_resource(candidates)
    else:
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

    if verbose:
        page_slug = pkg_full.get("name") or pkg_full.get("id") or ""
        page_url = f"{SITE_BASE}/dataset/{page_slug}" if page_slug else SITE_BASE
        res_name = (res.get("name") or "").strip()
        res_fmt = (res.get("format") or res.get("mimetype") or "").upper()
        res_last = res.get("last_modified") or res.get("created") or ""
        print(f"[CKAN] Pacote: {pkg_full.get('title','(sem título)')} → {page_url}")
        print(f"[CKAN] Recurso: {res_name} [{res_fmt}] {res_last}")
        print(f"[CKAN] URL: {url}")

    tmp_path = out_dir / (spec.out_name + ".tmp")
    if verbose:
        print(f"[CKAN] Baixando {spec.query} → {spec.out_name}")
    final_target = out_dir / spec.out_name
    if final_target.exists() and not overwrite:
        if verbose:
            print(f"[CKAN] Já existe {final_target.name}; pulando (use --overwrite para baixar novamente)")
        return final_target

    downloaded = _download_resource(url, tmp_path)
    out = _maybe_convert_to_csv(downloaded, final_target)
    # limpeza de temporários
    try:
        if tmp_path.exists():
            tmp_path.unlink()
        if downloaded != tmp_path and downloaded.exists() and downloaded.suffix != ".csv":
            # mantém original não-CSV como referência (.orig)
            downloaded.rename(final_target.with_suffix(final_target.suffix + ".orig"))
    except Exception:
        pass
    return out


def fetch_many_and_concat(
    client: CkanClient,
    spec: DatasetSpec,
    out_dir: Path,
    since: Optional[str] = None,
    until: Optional[str] = None,
    verbose: bool = True,
    overwrite: bool = False,
) -> Optional[Path]:
    """Baixa todos os recursos do pacote (filtrados) e concatena em um único CSV.

    Útil para datasets mensais (um recurso por mês). `since`/`until` aceitam
    prefixos no nome do recurso (ex.: "2022-" ou "2022-04").
    """
    pkg_full, resources = _list_resources(client, spec.query, spec.resource_filter, verbose)
    if not resources:
        return None
    # ordena recursos por data de criação/atualização ou por nome
    def rkey(r):
        return (r.get("last_modified") or r.get("created") or r.get("name") or "")
    resources = sorted(resources, key=rkey)
    # mantém apenas recursos de dados (descarta PDFs/dicionários)
    resources = [r for r in resources if _is_data_resource(r)]
    # aplica filtros since/until
    resources = [r for r in resources if _resource_matches_since_until(r, since, until)]
    # agrupa por período e prefere CSV > XLSX > ZIP
    resources = _group_prefer_by_period(resources)

    tmp_csvs: List[Path] = []
    for r in resources:
        url = r.get("url") or r.get("download_url")
        if not url:
            continue
        if verbose:
            print(f"[CKAN] Recurso mensal: {r.get('name','(sem nome)')} → {url}")
        tmp_path = out_dir / f"{spec.out_name}.{abs(hash(url))}.tmp"
        downloaded = _download_resource(url, tmp_path)
        # converte para CSV se necessário
        csv_path = downloaded if downloaded.suffix.lower() == ".csv" else downloaded.with_suffix(".csv")
        csv_path = _maybe_convert_to_csv(downloaded, csv_path)
        tmp_csvs.append(csv_path)
        # limpeza de temporários (arquivo baixado e .tmp)
        try:
            if downloaded.exists() and downloaded != csv_path:
                downloaded.unlink()
        except Exception:
            pass
        try:
            if tmp_path.exists() and tmp_path != downloaded:
                tmp_path.unlink()
        except Exception:
            pass

    if not tmp_csvs:
        return None
    final = out_dir / spec.out_name
    if final.exists() and not overwrite:
        if verbose:
            print(f"[CKAN] Já existe {final.name}; pulando concatenação (use --overwrite)")
        return final
    _append_csv_files(tmp_csvs, final)
    # limpeza de temporários
    for p in tmp_csvs:
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass
    return final


def fetch_all(out_dir: Path, datasets: Optional[List[str]] = None, verbose: bool = True, since: Optional[str] = None, until: Optional[str] = None, overwrite: bool = False) -> Dict[str, Optional[Path]]:
    """Baixa todos os datasets especificados (ou os padrão) para `out_dir`.

    Args:
      out_dir (Path): Diretório de saída.
      datasets (list[str]|None): Subconjunto de chaves de `DEFAULT_SPECS`.
      verbose (bool): Se True, imprime progresso.

    Returns:
      dict[str, Path|None]: Mapa de dataset → arquivo baixado (ou None).
    """
    if datasets is None:
        datasets = list(DEFAULT_SPECS.keys())
    client = CkanClient()
    results: Dict[str, Optional[Path]] = {}
    for key in datasets:
        spec = DEFAULT_SPECS[key]
        try:
            # Para conjuntos mensais (constrained-off), baixamos e concatenamos todos recursos
            # Para a maioria dos datasets, há um recurso por mês/ano; concatena todos dentro do período
            if key in {"corte_eolica", "corte_fv", "balanco", "intercambio", "ena", "ear", "carga"}:
                p = fetch_many_and_concat(client, spec, out_dir, since=since, until=until, verbose=verbose, overwrite=overwrite)
            else:
                p = fetch_one(client, spec, out_dir, verbose=verbose, since=since, until=until, overwrite=overwrite)
        except Exception as e:
            print(f"[ERRO] {key}: {e}")
            p = None
        results[key] = p
    return results


def main():
    """CLI para baixar dados do ONS via CKAN em `data/raw`."""
    ap = argparse.ArgumentParser(description="Baixa dados do ONS (CKAN) e salva em data/raw.")
    ap.add_argument("--out-dir", default="data/raw", help="Diretório onde salvar arquivos brutos.")
    ap.add_argument("--datasets", nargs="*", default=[], help=f"Quais datasets baixar: {list(DEFAULT_SPECS.keys())}")
    ap.add_argument("--since", default=None, help="Filtro (prefixo) no nome do recurso, ex.: 2022- ou 2022-04")
    ap.add_argument("--overwrite", action="store_true", help="Força re-download mesmo se arquivo final já existir.")
    ap.add_argument("--all", action="store_true", help="Baixar todos os datasets suportados.")
    args = ap.parse_args()

    out = Path(args.out_dir)
    _ensure_dir(out)
    if args.all or not args.datasets:
        keys = list(DEFAULT_SPECS.keys())
    else:
        keys = args.datasets

    res = fetch_all(out, keys, since=args.since, overwrite=args.overwrite)
    for k, p in res.items():
        status = "OK" if p is not None else "--"
        print(f"[{status}] {k}: {p if p else 'não baixado'}")


if __name__ == "__main__":
    main()
