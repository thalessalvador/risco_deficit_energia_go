PY=python

# Parâmetros configuráveis
SUBMERCADO ?= SE/CO
ETL_FLAGS ?=
ifdef OVERWRITE
ETL_FLAGS += --overwrite
endif

download:
	$(PY) -m src.fetch_ons --all || $(PY) src/fetch_ons.py --all

ingest:
	$(PY) -m src.etl_ons --raw-dir data/raw --out-dir data/raw --submercado "$(SUBMERCADO)" $(ETL_FLAGS) || $(PY) src/etl_ons.py --raw-dir data/raw --out-dir data/raw --submercado "$(SUBMERCADO)" $(ETL_FLAGS)

data:
	$(MAKE) download
	$(PY) -m src.etl_ons --raw-dir data/raw --out-dir data/raw --submercado "$(SUBMERCADO)" --fetch-nasa $(ETL_FLAGS) || $(PY) src/etl_ons.py --raw-dir data/raw --out-dir data/raw --submercado "$(SUBMERCADO)" --fetch-nasa $(ETL_FLAGS)

features:
	$(PY) - <<'PY'
from yaml import safe_load
from src.data_loader import load_all_sources
from src.feature_engineer import build_features_weekly
from pathlib import Path
cfg = safe_load(open("configs/config.yaml"))
data = load_all_sources(cfg)
Xw = build_features_weekly(data, cfg)
Path(cfg["paths"]["features_dir"]).mkdir(parents=True, exist_ok=True)
Xw.to_parquet(Path(cfg["paths"]["features_dir"])/"features_weekly.parquet")
print("Saved features_weekly.parquet", Xw.shape)
PY

train:
	$(PY) -m src.train --config configs/config.yaml || $(PY) src/train.py

eval:
	$(PY) -m src.evaluate --config configs/config.yaml --model_name xgb || $(PY) src/evaluate.py
