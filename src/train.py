# src/train.py
from __future__ import annotations
import yaml
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from joblib import dump

from src.data_loader import load_all_sources
from src.feature_engineer import build_features_weekly

# Mapeamento estável de rótulos (string → inteiro)
LABEL_MAP = {"baixo": 0, "medio": 1, "alto": 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def encode_labels(y: pd.Series) -> pd.Series:
    """Encoda rótulos de string para inteiros estáveis 0/1/2.

    Lança erro se encontrar valores fora de {'baixo','medio','alto'}.
    """
    cats = list(LABEL_MAP.keys())
    c = pd.Categorical(y, categories=cats, ordered=True)
    if (c.codes < 0).any():
        unknown = set(pd.Series(y).iloc[np.where(c.codes < 0)[0]].unique())
        raise ValueError(f"Rótulos desconhecidos encontrados: {unknown}")
    return pd.Series(c.codes, index=y.index, dtype=int)


def _prefix_param_grid(grid: dict, prefix: str = "clf__") -> dict:
    """Adiciona prefixo de etapa do Pipeline (ex.: 'clf__') nas chaves do grid.

    Aceita chaves já prefixadas e retorna um novo dicionário.
    """
    if not grid:
        return {}
    out = {}
    for k, v in grid.items():
        out[k if k.startswith(prefix) else f"{prefix}{k}"] = v
    return out


"""
Funções auxiliares para tuning simples via GridSearchCV.
"""


def rotular_semana(df: pd.DataFrame, cfg, ref_df: pd.DataFrame | None = None) -> pd.Series:
    """Gera rótulos semanais (baixo|medio|alto) com base na margem e ajustes.

    - Usa quantis de `coluna_margem` definidos em `cfg`.
    - Pode calcular thresholds a partir de `ref_df` (ex.: conjunto de treino) para evitar vazamento.
    - Aplica ajustes por cortes (downgrade) e hidrologia (override por EAR/ENA).

    Args:
      df (pandas.DataFrame): Features semanais da janela alvo.
      cfg (dict): Configurações (label_rules etc.).
      ref_df (pandas.DataFrame|None): DataFrame de referência para cálculos de quantis.

    Returns:
      pandas.Series: Série categórica com rótulos por semana.
    """
    r = cfg["problem"]["label_rules"]
    margem_col = r["coluna_margem"]
    q_baixo, q_med = r["q_baixo"], r["q_medio"]

    # Fallback robusto de coluna de margem: tenta alternativas se a coluna pedida não existir
    if margem_col not in df.columns:
        candidates = [
            margem_col,
            "margem_vs_carga_w",
            "margem_suprimento_w",
            "margem_suprimento_min_w",
        ]
        margem_col = next((c for c in candidates if c in df.columns), None)
        if margem_col is None:
            raise ValueError(
                f"Coluna de margem não encontrada para rotulagem. Tentadas: {candidates}"
            )

    base = ref_df if ref_df is not None else df
    margem = df[margem_col].copy()
    thr_baixo = base[margem_col].quantile(q_baixo)
    thr_med = base[margem_col].quantile(q_med)

    y = pd.Series(index=df.index, dtype=object)
    y[margem <= thr_baixo] = "alto"
    y[(margem > thr_baixo) & (margem <= thr_med)] = "medio"
    y[margem > thr_med] = "baixo"

    # Ajuste por cortes (curtailment) — interpretação: superávit local/operacional
    cd = r.get("curtail_downgrade", {})
    if cd.get("usar_downgrade_cortes", True):
        ratio_thr = float(cd.get("corte_ratio_thr", 0.05))
        req_no_import = bool(cd.get("requer_saldo_importador_nao_positivo", True))
        ratio = df.get("ratio_corte_renovavel_w")
        saldo_import = df.get("saldo_importador_mwh_sum_w")
        if ratio is not None:
            cond_ratio = ratio.fillna(0) > ratio_thr
            cond_import = True
            if req_no_import and (saldo_import is not None):
                cond_import = saldo_import.fillna(0) <= 0  # não está importando líquido
            cond = cond_ratio & cond_import
            # "descer" um nível de risco onde as condições forem verdadeiras
            y.loc[cond & (y == "alto")] = "medio"
            y.loc[cond & (y == "medio")] = "baixo"

    # Ajuste hidrológico (estiagem)
    if r.get("usar_override_hidro", True):
        ear_col = r.get("coluna_ear")
        ena_col = r.get("coluna_ena")
        if ear_col in df.columns:
            ear_thr = base[ear_col].quantile(r.get("ear_q_baixo", 0.2))
            y[df[ear_col] <= ear_thr] = "alto"
        if ena_col in df.columns:
            ena_thr = base[ena_col].quantile(r.get("ena_q_baixo", 0.2))
            k = int(r.get("janelas_consecutivas_ena", 2))
            ena_low = (df[ena_col] <= ena_thr).rolling(k).sum() >= k
            y[ena_low.fillna(False)] = "alto"

    # Regras duras (proxies dos critérios EPE/MME/ONS) — Comentário
    #
    # Mapeamento pragmático semanal (determinístico) para critérios anuais:
    # - Adequação de potência/Reserva Operativa (PNS):
    #     reserve_margin_ratio_w = (suprimento − demanda) / demanda.
    #     Se < 5% (reserva abaixo de 5% da demanda semanal), marcar "alto".
    # - ENS (energia não suprida) vs. demanda:
    #     ens_week_ratio = ENS_semana / demanda_semana.
    #     Se ≥ 5%, marcar "alto" (proxy do CVaR_1% no limite regulatório).
    # - LOLP (probabilidade de perda de carga):
    #     lolp_52w = frequência móvel (52s) de semanas com déficit.
    #     Se > 5%, marcar "alto" (proxy anual).
    #
    # Observação: estas regras duras atuam como overrides conservadores, após
    # quantis e ajustes, aproximando os critérios oficiais sem simulação de cenários.
    if bool(r.get("usar_regras_duras", True)):
        reserva_frac = float(r.get("reserva_operativa_frac", 0.05))
        ens_thr = float(r.get("ens_ratio_thr", 0.05))
        lolp_thr = float(r.get("lolp_thr", 0.05))

        rm = df.get("reserve_margin_ratio_w")
        if rm is not None:
            y[rm < reserva_frac] = "alto"

        ens_ratio = df.get("ens_week_ratio")
        if ens_ratio is not None:
            y[ens_ratio >= ens_thr] = "alto"

        lolp = df.get("lolp_52w")
        if lolp is not None:
            y[lolp > lolp_thr] = "alto"

    return y

def compute_label_thresholds(cfg, ref_df: pd.DataFrame) -> dict:
    """Calcula thresholds fixos de rotulagem a partir do conjunto de treino/validação.

    Retorna um dicionário JSON-serializável contendo valores de quantis para a margem,
    thresholds para EAR/ENA (se existirem), parâmetros de curtailment e metadados.
    """
    r = cfg["problem"]["label_rules"]
    margem_col = r["coluna_margem"]
    q_baixo = float(r["q_baixo"]) if "q_baixo" in r else 0.1
    q_medio = float(r["q_medio"]) if "q_medio" in r else 0.4

    # Fallback robusto de coluna de margem na base de referência
    if margem_col not in ref_df.columns:
        candidates = [
            margem_col,
            "margem_vs_carga_w",
            "margem_suprimento_w",
            "margem_suprimento_min_w",
        ]
        margem_col = next((c for c in candidates if c in ref_df.columns), None)
    has_margem = bool(margem_col and margem_col in ref_df.columns)
    thr_baixo = float(ref_df[margem_col].quantile(q_baixo)) if has_margem else None
    thr_medio = float(ref_df[margem_col].quantile(q_medio)) if has_margem else None

    ear_col = r.get("coluna_ear")
    ear_q = float(r.get("ear_q_baixo", 0.2)) if ear_col else None
    ear_thr = float(ref_df[ear_col].quantile(ear_q)) if ear_col and ear_col in ref_df.columns else None

    ena_col = r.get("coluna_ena")
    ena_q = float(r.get("ena_q_baixo", 0.2)) if ena_col else None
    ena_thr = float(ref_df[ena_col].quantile(ena_q)) if ena_col and ena_col in ref_df.columns else None
    k = int(r.get("janelas_consecutivas_ena", 2))

    cd = r.get("curtail_downgrade", {}) or {}
    ratio_thr = float(cd.get("corte_ratio_thr", 0.05))
    req_no_import = bool(cd.get("requer_saldo_importador_nao_positivo", True))

    thresholds = {
        "margem": {
            "col": margem_col,
            "q_baixo": q_baixo,
            "q_baixo_value": thr_baixo,
            "q_medio": q_medio,
            "q_medio_value": thr_medio,
        },
        "ear": {
            "col": ear_col,
            "q_baixo": ear_q,
            "q_baixo_value": ear_thr,
        },
        "ena": {
            "col": ena_col,
            "q_baixo": ena_q,
            "q_baixo_value": ena_thr,
            "janelas_consecutivas": k,
        },
        "curtail": {
            "ratio_thr": ratio_thr,
            "requer_saldo_importador_nao_positivo": req_no_import,
        },
        "meta": {
            "train_start": str(ref_df.index.min()) if len(ref_df.index) else None,
            "train_end": str(ref_df.index.max()) if len(ref_df.index) else None,
        },
    }
    # Critérios/proxies regulatórios persistidos para uso consistente em avaliação/inferência
    thresholds["regulatory_proxies"] = {
        "usar_regras_duras": bool(r.get("usar_regras_duras", True)),
        "reserva_operativa_frac": float(r.get("reserva_operativa_frac", 0.05)),
        "ens_ratio_thr": float(r.get("ens_ratio_thr", 0.05)),
        "lolp_thr": float(r.get("lolp_thr", 0.05)),
    }
    return thresholds

def rotular_semana_com_thresholds(df: pd.DataFrame, cfg, thresholds: dict) -> pd.Series:
    """Rotula com thresholds fixos (sem recalcular quantis).

    Aplica a mesma lógica de rotular_semana usando valores numéricos persistidos.
    """
    r = cfg["problem"]["label_rules"]
    margem_info = thresholds.get("margem", {})
    margem_col = margem_info.get("col", r.get("coluna_margem"))
    thr_baixo = margem_info.get("q_baixo_value")
    thr_medio = margem_info.get("q_medio_value")

    if margem_col not in df.columns:
        candidates = [
            margem_col,
            "margem_vs_carga_w",
            "margem_suprimento_w",
            "margem_suprimento_min_w",
        ]
        margem_col = next((c for c in candidates if c in df.columns), None)
    if (margem_col is None) or (thr_baixo is None) or (thr_medio is None):
        raise ValueError("Thresholds/coluna de margem ausentes para rotulagem.")

    margem = df[margem_col].copy()
    y = pd.Series(index=df.index, dtype=object)
    y[margem <= thr_baixo] = "alto"
    y[(margem > thr_baixo) & (margem <= thr_medio)] = "medio"
    y[margem > thr_medio] = "baixo"

    # ajuste por curtailment
    cd = thresholds.get("curtail", {})
    ratio_thr = float(cd.get("ratio_thr", 0.05))
    req_no_import = bool(cd.get("requer_saldo_importador_nao_positivo", True))
    ratio = df.get("ratio_corte_renovavel_w")
    saldo_import = df.get("saldo_importador_mwh_sum_w")
    if ratio is not None:
        cond_ratio = ratio.fillna(0) > ratio_thr
        cond_import = True
        if req_no_import and (saldo_import is not None):
            cond_import = saldo_import.fillna(0) <= 0
        cond = cond_ratio & cond_import
        y.loc[cond & (y == "alto")] = "medio"
        y.loc[cond & (y == "medio")] = "baixo"

    # overrides hidrológicos
    ear_info = thresholds.get("ear", {})
    ear_col = ear_info.get("col")
    ear_thr = ear_info.get("q_baixo_value")
    if ear_col and (ear_col in df.columns) and (ear_thr is not None):
        y[df[ear_col] <= ear_thr] = "alto"

    ena_info = thresholds.get("ena", {})
    ena_col = ena_info.get("col")
    ena_thr = ena_info.get("q_baixo_value")
    k = int(ena_info.get("janelas_consecutivas", r.get("janelas_consecutivas_ena", 2)))
    if ena_col and (ena_col in df.columns) and (ena_thr is not None):
        ena_low = (df[ena_col] <= ena_thr).rolling(k).sum() >= k
        y[ena_low.fillna(False)] = "alto"

    # Regras duras com thresholds persistidos (evita drift entre treino e avaliação)
    reg = thresholds.get("regulatory_proxies", {})
    if bool(reg.get("usar_regras_duras", True)):
        reserva_frac = float(reg.get("reserva_operativa_frac", 0.05))
        ens_thr = float(reg.get("ens_ratio_thr", 0.05))
        lolp_thr = float(reg.get("lolp_thr", 0.05))

        rm = df.get("reserve_margin_ratio_w")
        if rm is not None:
            y[rm < reserva_frac] = "alto"

        ens_ratio = df.get("ens_week_ratio")
        if ens_ratio is not None:
            y[ens_ratio >= ens_thr] = "alto"

        lolp = df.get("lolp_52w")
        if lolp is not None:
            y[lolp > lolp_thr] = "alto"

    return y

def make_model(model_cfg):
    """Cria o estimador conforme o bloco `model_cfg` (logreg ou xgboost).

    Inclui imputação no pipeline para evitar vazamento na fase de treino.

    Args:
      model_cfg (dict): Especificação com chaves `type` e `params`.

    Returns:
      sklearn.pipeline.Pipeline | xgboost.XGBClassifier: Estimador configurado.
    """
    if model_cfg["type"] == "logistic_regression":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=False)),
                ("clf", LogisticRegression(**model_cfg["params"])),
            ]
        )
    elif model_cfg["type"] == "xgboost":
        from xgboost import XGBClassifier

        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", XGBClassifier(**model_cfg["params"]))
        ])
    else:
        raise ValueError(f"Modelo não suportado: {model_cfg['type']}")


def main(config_path="configs/config.yaml"):
    """Treina os modelos definidos em `configs/config.yaml` com CV temporal.

    - Constrói features semanais, gera rótulos por fold (sem vazamento) e avalia com TSS.
    - Reajusta em todo o conjunto e salva artefatos `.joblib` e métricas de CV.

    Args:
      config_path (str): Caminho para o arquivo de configuração YAML.
    """
    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    out_dir = Path(cfg["paths"]["models_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_all_sources(cfg)
    Xw = build_features_weekly(data, cfg)
    # limpa colunas completamente vazias; imputação fica no pipeline (evita vazamento)
    Xw = Xw.dropna(axis=1, how="all")

    # --- Hold-out split for final evaluation ---
    holdout_frac = float(cfg.get("modeling", {}).get("holdout_fraction", 0.2))
    holdout_frac = 0.0 if holdout_frac < 0 else holdout_frac
    test_size = int(np.ceil(len(Xw) * holdout_frac)) if holdout_frac > 0 else 0
    if test_size >= len(Xw):
        test_size = max(1, len(Xw) // 5)
    if test_size > 0:
        train_val_df = Xw.iloc[:-test_size]
        test_df = Xw.iloc[-test_size:]
    else:
        train_val_df = Xw.copy()
        test_df = Xw.iloc[0:0]
    feats_dir = Path(cfg["paths"]["features_dir"])
    feats_dir.mkdir(parents=True, exist_ok=True)
    train_val_df.to_parquet(feats_dir / "features_trainval.parquet")
    if len(test_df) > 0:
        test_df.to_parquet(feats_dir / "features_test_holdout.parquet")
    print(f"[train] Hold-out saved: train_val={len(train_val_df)} weeks, test={len(test_df)} weeks in {feats_dir}.")

    # thresholds fixos a partir do conjunto de treino/validação
    thresholds = compute_label_thresholds(cfg, train_val_df)

    H = int(cfg.get("problem", {}).get("forecast_horizon_weeks", 1))
    X = train_val_df

    resultados = []
    tss = TimeSeriesSplit(n_splits=cfg["modeling"]["cv"]["n_splits"])
    for mcfg in cfg["modeling"]["models"]:
        model = make_model(mcfg)
        f1s, bals = [], []
        # tuning (opcional) por modelo
        tune_cfg = mcfg.get("tuning", {}) or {}
        use_tuning = bool(tune_cfg.get("use", False))
        inner_splits = int(tune_cfg.get("cv_splits", 3))
        scoring = tune_cfg.get("scoring", ["f1_macro", "balanced_accuracy"]) or "f1_macro"
        refit_metric = tune_cfg.get("refit", "f1_macro")
        # grid único (sem duas fases)
        param_grid = _prefix_param_grid(tune_cfg.get("param_grid", {}))

        for tr_idx, te_idx in tss.split(X):
            Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
            # Rotulagem sem vazamento: thresholds calculados no treino, aplicados em treino e teste
            ytr = rotular_semana(Xtr, cfg, ref_df=Xtr)
            yte = rotular_semana(Xte, cfg, ref_df=Xtr)
            # Horizonte T+H: alinhamento para prever t+H
            if H > 0:
                ytr = ytr.shift(-H)
                yte = yte.shift(-H)
            # Alinha para índices com rótulo disponível
            tr_idx_ok = ytr.dropna().index
            te_idx_ok = yte.dropna().index
            Xtr_ok, ytr_ok = Xtr.loc[tr_idx_ok], ytr.loc[tr_idx_ok]
            Xte_ok, yte_ok = Xte.loc[te_idx_ok], yte.loc[te_idx_ok]

            if len(Xtr_ok) == 0 or len(Xte_ok) == 0:
                continue

            # Encoda rótulos para inteiros (0=baixo,1=medio,2=alto)
            ytr_enc = encode_labels(ytr_ok)
            yte_enc = encode_labels(yte_ok)

            if use_tuning and param_grid:
                inner_cv = TimeSeriesSplit(n_splits=inner_splits)
                gs = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    scoring=scoring,
                    refit=refit_metric,
                    cv=inner_cv,
                    n_jobs=-1,
                    error_score=np.nan,
                )
                gs.fit(Xtr_ok, ytr_enc)
                best_est = gs.best_estimator_
                pred = best_est.predict(Xte_ok)
            else:
                model.fit(Xtr_ok, ytr_enc)
                pred = model.predict(Xte_ok)
            f1s.append(f1_score(yte_enc, pred, average="macro"))
            bals.append(balanced_accuracy_score(yte_enc, pred))
        resultados.append(
            {
                "modelo": mcfg["name"],
                "f1_macro": float(np.mean(f1s)),
                "balanced_acc": float(np.mean(bals)),
            }
        )

        # re-ajusta no conjunto de treino/validacao e salva artefato
        y_full = rotular_semana(X, cfg, ref_df=X)
        if H > 0:
            y_full = y_full.shift(-H)
        idx_ok = y_full.dropna().index
        X_full_ok, y_full_ok = X.loc[idx_ok], y_full.loc[idx_ok]
        # Encoda rótulos no ajuste final
        y_full_enc = encode_labels(y_full_ok)
        if use_tuning and param_grid:
            inner_cv = TimeSeriesSplit(n_splits=inner_splits)
            gs = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                scoring=scoring,
                refit=refit_metric,
                cv=inner_cv,
                n_jobs=-1,
                error_score=np.nan,
            )
            gs.fit(X_full_ok, y_full_enc)
            final_model = gs.best_estimator_
            # opcional: salvar melhores params
            try:
                best_params_path = Path(cfg["paths"]["reports_dir"]) / f"tuning_{mcfg['name']}_best_params.json"
                best_params_path.parent.mkdir(parents=True, exist_ok=True)
                with open(best_params_path, "w", encoding="utf-8") as jf:
                    json.dump(gs.best_params_, jf, ensure_ascii=False, indent=2)
            except Exception:
                pass
        else:
            model.fit(X_full_ok, y_full_enc)
            final_model = model

        # Anexa mapeamentos ao artefato para uso na avaliação/inferência
        try:
            setattr(final_model, "label_mapping_", LABEL_MAP)
            setattr(final_model, "inv_label_mapping_", INV_LABEL_MAP)
            setattr(final_model, "label_thresholds_", thresholds)
        except Exception:
            pass

        # grava thresholds também em JSON (um por modelo)
        try:
            with open(out_dir / f"label_thresholds_{mcfg['name']}.json", "w", encoding="utf-8") as jf:
                json.dump(thresholds, jf, ensure_ascii=False, indent=2)
        except Exception:
            pass

        dump(final_model, out_dir / f"{mcfg['name']}.joblib")

    rep_dir = Path(cfg["paths"]["reports_dir"])
    rep_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(resultados).to_csv(rep_dir / "cv_scores.csv", index=False)
    print(pd.DataFrame(resultados))


if __name__ == "__main__":
    main()
