from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional
import sys
import yaml

# Imports do projeto
from src.fetch_ons import fetch_all
from src import etl_ons
from src.meteo import fetch_meteorologia
from src.data_loader import load_all_sources
from src.feature_engineer import build_features_weekly
from src import train as train_mod
from src import evaluate as eval_mod


def _since_to_date(since: Optional[str]) -> Optional[str]:
    """Converte since (YYYY ou YYYY-MM) para data inicial YYYY-MM-01."""
    if not since:
        return None
    try:
        parts = since.split("-")
        if len(parts) == 1 and len(parts[0]) == 4:
            y = int(parts[0])
            return f"{y:04d}-01-01"
        elif len(parts) >= 2:
            y = int(parts[0])
            m = int(parts[1])
            return f"{y:04d}-{m:02d}-01"
    except Exception:
        return None
    return None


def run_data(
    raw_dir: str,
    submercado: str,
    fetch_nasa: bool = False,
    overwrite: bool = False,
    since: Optional[str] = None,
    config_path: Optional[str] = None,
    use_s3: bool = False,
) -> None:
    """Executa download (CKAN), ETL do ONS e, opcionalmente, meteorologia.

    Args:
      raw_dir (str): Diretório de dados brutos (`data/raw`).
      submercado (str): Submercado alvo (ex.: "SE/CO").
      fetch_nasa (bool): Se True, inclui meteorologia (NASA POWER).
      overwrite (bool): Se True, permite sobrescrever saídas.
    """

    cfg = None
    if config_path:
        try:
            cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
        except Exception:
            cfg = None

    if use_s3:
        print("[data] Buscando dados do S3...")
        s3_cfg = (cfg.get("s3") if cfg else {}) or {}
        if not s3_cfg.get("enabled", False):
            print("[data] S3 não está habilitado no config.yaml. Usando modo local/API.")
            use_s3 = False

    if not use_s3:
        raw = Path(raw_dir)
        raw.mkdir(parents=True, exist_ok=True)
    else:
        import os
        from src.s3_utils import download_file_from_s3
        bucket = s3_cfg.get("bucket")
        # Sempre usar prefixo 'raw/' para leitura dos dados brutos
        prefix = "raw/"
        aws_access_key_id = s3_cfg.get("aws_access_key_id")
        aws_secret_access_key = s3_cfg.get("aws_secret_access_key")
        region = s3_cfg.get("region")
        raw = Path(raw_dir)
        raw.mkdir(parents=True, exist_ok=True)
        arquivos = [
            "ons_balanco_subsistema_horario.csv",
            "ons_intercambios_entre_subsistemas_horario.csv",
            "ons_carga.csv",
            "ons_ena_diario_subsistema.csv",
            "ons_ear_diario_subsistema.csv",
            "ons_constrained_off_eolica_mensal.csv",
            "ons_constrained_off_fv_mensal.csv",
        ]
        for arq in arquivos:
            print(f"Baixando arquivo {arq} do S3 (raw)...")
            s3_key = arq
            local_path = str(raw / arq)
            ok = download_file_from_s3(
                bucket, s3_key, local_path,
                prefix=prefix,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=region,
            )
            if ok:
                print(f"[S3] Baixado: {arq} -> {local_path}")
            else:
                print(f"[S3] ERRO ao baixar: {arq} -> {local_path}")


    # parâmetros de região (YAML opcional)
    regions = (cfg.get("regions") if cfg else {}) or {}
    subm_eff = submercado or regions.get("submercado") or "SE/CO"
    carga_area = regions.get("carga_area")

    if not use_s3:
        res = fetch_all(raw, since=since, overwrite=overwrite)
        for k, p in res.items():
            print(f"   - {k}: {p if p else 'não baixado'}")

    # (Re)baixa carga com a área configurada, se fornecida
    if carga_area:
        try:
            from src.fetch_ons import fetch_carga_api
            print(f"[data] Baixando Carga Verificada para área: {carga_area}")
            fetch_carga_api(raw, since=since, area=str(carga_area), overwrite=overwrite, verbose=True)
        except Exception as e:
            print(f"[data] Aviso: fetch de carga com area={carga_area} falhou: {e}")

    # 2) Rodar ETL -> diários padronizados
    print(f"[data] Rodando ETL ONS -> diários (submercado={subm_eff})")
    paths = {
        "balanco": raw / "ons_balanco_subsistema_horario.csv",
        "intercambio": raw / "ons_intercambios_entre_subsistemas_horario.csv",
        "carga": raw / "ons_carga.csv",
        "ena": raw / "ons_ena_diario_subsistema.csv",
        "ear": raw / "ons_ear_diario_subsistema.csv",
        "corte_eolica": raw / "ons_constrained_off_eolica_mensal.csv",
        "corte_fv": raw / "ons_constrained_off_fv_mensal.csv",
    }

    etl_ons.etl_balanco_subsistema_horario(paths["balanco"], raw, subm_eff)
    etl_ons.etl_intercambio_horario(paths["intercambio"], raw, subm_eff)
    etl_ons.etl_carga(paths["carga"], raw, subm_eff)
    etl_ons.etl_ena_diaria(paths["ena"], raw, subm_eff)
    etl_ons.etl_ear_diaria(paths["ear"], raw, subm_eff)
    etl_ons.etl_constrained_off_mensal(paths["corte_eolica"], raw, "eolica", subm_eff)
    etl_ons.etl_constrained_off_mensal(paths["corte_fv"], raw, "fv", subm_eff)

    if fetch_nasa:
        try:
            inicio = _since_to_date(since)
            pontos = regions.get("meteo_points") if regions else None
            out = fetch_meteorologia(
                raw, provider="nasa_power", overwrite=overwrite, inicio=inicio, pontos=pontos
            )
            print(f"[data] Meteorologia: {out if out else 'não baixado'}")
        except Exception as e:
            print(f"[data] Meteorologia falhou: {e}")


def run_features(config_path: str) -> None:
    """Gera a feature store semanal e salva em parquet.

    Args:
      config_path (str): Caminho para o YAML de configuração.
    """
    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    data = load_all_sources(cfg)
    Xw = build_features_weekly(data, cfg)
    out_dir = Path(cfg["paths"]["features_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "features_weekly.parquet"
    Xw.to_parquet(out)
    print("[features] salvo:", out, Xw.shape)

    # Salva também no S3 (camada bronze), se configurado
    s3_cfg = cfg.get("s3", {})
    if s3_cfg.get("enabled", False):
        from src.s3_utils import upload_file_to_s3
        upload_file_to_s3(
            str(out),
            s3_cfg.get("bucket"),
            out.name,
            prefix="bronze/",
            aws_access_key_id=s3_cfg.get("aws_access_key_id"),
            aws_secret_access_key=s3_cfg.get("aws_secret_access_key"),
            region_name=s3_cfg.get("region"),
        )
        print(f"[S3] Features salvas em bronze: {out.name}")
    else:
        print("[S3] Upload para bronze não realizado: S3 desabilitado no config.yaml.")


def run_train(config_path: str) -> None:
    """Treina modelos definidos no YAML e salva artefatos."""
    train_mod.main(config_path=config_path)


def run_eval(config_path: str, model_name: str) -> None:
    """Avalia um modelo salvo e gera relatório.

    Args:
      config_path (str): Caminho para o YAML de configuração.
      model_name (str): Nome do modelo salvo (ex.: "xgb").
    """
    eval_mod.main(config_path=config_path, model_name=model_name)


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parseia argumentos da CLI unificada do projeto."""
    p = argparse.ArgumentParser(
        description="CLI unificada do projeto (dados -> features -> treino -> avaliação)"
    )
    p.add_argument(
        "--use-s3",
        action="store_true",
        help="Se definido, busca os dados do S3 conforme configuração do YAML. Por padrão, usa API/local."
    )
    # Suporta tanto positional action quanto --action
    p.add_argument(
        "action",
        nargs="?",
        choices=["data", "features", "train", "eval", "all"],
        help="Ação a executar",
    )
    p.add_argument(
        "--action",
        dest="action_opt",
        choices=["data", "features", "train", "eval", "all"],
        help="Ação a executar",
    )

    p.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Caminho para o YAML de configuração",
    )

    # Opções de dados/ETL
    p.add_argument("--raw-dir", default="data/raw", help="Diretório de dados brutos")
    p.add_argument(
        "--submercado", default="SE/CO", help="Submercado alvo (ex.: 'SE/CO')"
    )
    p.add_argument(
        "--since",
        default=None,
        help="Baixar dados a partir deste ano/mês (ex.: 2022 ou 2022-01)",
    )
    # Clima
    p.add_argument(
        "--incluir-meteorologia",
        action="store_true",
        help="Inclui meteorologia (NASA POWER) no passo de dados",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Sobrescreve saídas do ETL (inclui NASA)",
    )

    # Opções de avaliação
    p.add_argument(
        "--model", default="xgb", help="Nome do modelo para avaliação (ex.: xgb)"
    )

    args = p.parse_args(argv)
    # Consolida action
    args.action = args.action_opt or args.action
    if not args.action:
        p.error("Informe a ação: data | features | train | eval | all")
    return args


def main(argv: list[str] | None = None) -> None:
    """Função principal da CLI: direciona para as ações selecionadas."""
    args = parse_args(sys.argv[1:] if argv is None else argv)

    include_met = bool(getattr(args, "incluir_meteorologia", False))

    if args.action == "data":
        run_data(
            args.raw_dir,
            args.submercado,
            fetch_nasa=include_met,
            overwrite=args.overwrite,
            since=args.since,
            config_path=args.config,
            use_s3=args.use_s3,
        )
    elif args.action == "features":
        run_features(args.config)
    elif args.action == "train":
        run_train(args.config)
    elif args.action == "eval":
        run_eval(args.config, args.model)
    elif args.action == "all":
        run_data(
            args.raw_dir,
            args.submercado,
            fetch_nasa=include_met,
            overwrite=args.overwrite,
            since=args.since,
            config_path=args.config,
            use_s3=args.use_s3,
        )
        run_features(args.config)
        run_train(args.config)
        run_eval(args.config, args.model)
    else:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
