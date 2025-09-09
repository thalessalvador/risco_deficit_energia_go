from __future__ import annotations
import argparse
from pathlib import Path
import sys
import yaml

# Imports do projeto
from src.fetch_ons import fetch_all
from src import etl_ons
from src.data_loader import load_all_sources
from src.feature_engineer import build_features_weekly
from src import train as train_mod
from src import evaluate as eval_mod


def run_data(raw_dir: str, submercado: str, fetch_nasa: bool = False, overwrite: bool = False) -> None:
    raw = Path(raw_dir)
    raw.mkdir(parents=True, exist_ok=True)

    # 1) Baixar brutos do ONS
    print("[data] Baixando dados do ONS (CKAN) para:", raw.resolve())
    res = fetch_all(raw)
    for k, p in res.items():
        print(f"   - {k}: {p if p else 'não baixado'}")

    # 2) Rodar ETL → diários padronizados
    print(f"[data] Rodando ETL ONS → diários (submercado={submercado})")
    paths = {
        "balanco": raw / "ons_balanco_subsistema_horario.csv",
        "intercambio": raw / "ons_intercambios_entre_subsistemas_horario.csv",
        "carga": raw / "ons_carga.csv",
        "ena": raw / "ons_ena_diario_subsistema.csv",
        "ear": raw / "ons_ear_diario_subsistema.csv",
        "corte_eolica": raw / "ons_constrained_off_eolica_mensal.csv",
        "corte_fv": raw / "ons_constrained_off_fv_mensal.csv",
    }

    etl_ons.etl_balanco_subsistema_horario(paths["balanco"], raw, submercado)
    etl_ons.etl_intercambio_horario(paths["intercambio"], raw, submercado)
    etl_ons.etl_carga(paths["carga"], raw, submercado)
    etl_ons.etl_ena_diaria(paths["ena"], raw, submercado)
    etl_ons.etl_ear_diaria(paths["ear"], raw, submercado)
    etl_ons.etl_constrained_off_mensal(paths["corte_eolica"], raw, "eolica", submercado)
    etl_ons.etl_constrained_off_mensal(paths["corte_fv"], raw, "fv", submercado)

    if fetch_nasa:
        try:
            out = etl_ons.maybe_fetch_nasa_power(raw, overwrite=overwrite)
            print(f"[data] NASA POWER: {out if out else 'não baixado'}")
        except Exception as e:
            print(f"[data] NASA POWER falhou: {e}")


def run_features(config_path: str) -> None:
    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    data = load_all_sources(cfg)
    Xw = build_features_weekly(data, cfg)
    out_dir = Path(cfg["paths"]["features_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "features_weekly.parquet"
    Xw.to_parquet(out)
    print("[features] salvo:", out, Xw.shape)


def run_train(config_path: str) -> None:
    train_mod.main(config_path=config_path)


def run_eval(config_path: str, model_name: str) -> None:
    eval_mod.main(config_path=config_path, model_name=model_name)


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CLI unificada do projeto (dados → features → treino → avaliação)")
    # Suporta tanto positional action quanto --action
    p.add_argument("action", nargs="?", choices=["data", "features", "train", "eval", "all"], help="Ação a executar")
    p.add_argument("--action", dest="action_opt", choices=["data", "features", "train", "eval", "all"], help="Ação a executar")

    p.add_argument("--config", default="configs/config.yaml", help="Caminho para o YAML de configuração")

    # Opções de dados/ETL
    p.add_argument("--raw-dir", default="data/raw", help="Diretório de dados brutos")
    p.add_argument("--submercado", default="SE/CO", help="Submercado alvo (ex.: 'SE/CO')")
    # Clima
    p.add_argument("--incluir-meteorologia", action="store_true", help="Inclui meteorologia (NASA POWER) no passo de dados")
    p.add_argument("--overwrite", action="store_true", help="Sobrescreve saídas do ETL (inclui NASA)")

    # Opções de avaliação
    p.add_argument("--model", default="xgb", help="Nome do modelo para avaliação (ex.: xgb)")

    args = p.parse_args(argv)
    # Consolida action
    args.action = args.action_opt or args.action
    if not args.action:
        p.error("Informe a ação: data | features | train | eval | all")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    include_met = bool(getattr(args, "incluir_meteorologia", False))

    if args.action == "data":
        run_data(args.raw_dir, args.submercado, fetch_nasa=include_met, overwrite=args.overwrite)
    elif args.action == "features":
        run_features(args.config)
    elif args.action == "train":
        run_train(args.config)
    elif args.action == "eval":
        run_eval(args.config, args.model)
    elif args.action == "all":
        run_data(args.raw_dir, args.submercado, fetch_nasa=include_met, overwrite=args.overwrite)
        run_features(args.config)
        run_train(args.config)
        run_eval(args.config, args.model)
    else:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
