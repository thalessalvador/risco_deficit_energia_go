from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, List

import numpy as np
import pandas as pd
import yaml
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Dependência opcional (xgboost); mantém o código funcional sem ela.
try:
    from xgboost import XGBClassifier  # type: ignore

    _HAS_XGB = True
except Exception:  # pragma: no cover
    _HAS_XGB = False

# Componentes do projeto
from src.data_loader import load_all_sources
from src.feature_engineer import build_features_weekly
from src.models.contiguous import ContiguousLabelClassifier


# ============================================================
# ===================== CONFIG STRUCT ========================
# ============================================================


@dataclass
class Paths:
    raw_dir: str
    features_dir: str
    models_dir: str
    reports_dir: str


# ============================================================
# ================== RÓTULOS E THRESHOLDS ====================
# ============================================================

LABELS = ("baixo", "medio", "alto")
LABEL_TO_INT = {"baixo": 0, "medio": 1, "alto": 2}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}


def encode_labels(y: Iterable[str]) -> np.ndarray:
    """Mapeia 'baixo'|'medio'|'alto' → 0|1|2 (compatível com evaluate.py)."""
    y = pd.Series(y).astype("string")
    return y.map(LABEL_TO_INT).to_numpy()


def _pick_margin_column(cfg: Dict) -> str:
    rules = cfg.get("problem", {}).get("label_rules", {})
    col = rules.get("coluna_margem") or rules.get("coluna_margem_semana")
    return col or "margem_suprimento_min_w"  # fallback


def compute_label_thresholds(cfg: Dict, ref_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calcula os thresholds (quantis) a partir de um DataFrame de referência (sem oversample).
    Assumimos: valores MENORES de margem => MAIOR risco.
    """
    rules = cfg.get("problem", {}).get("label_rules", {})
    q_baixo = float(rules.get("q_baixo", 0.10))  # quantil para 'alto' risco
    q_medio = float(rules.get("q_medio", 0.40))  # quantil para 'medio' risco

    col_margem = _pick_margin_column(cfg)
    if col_margem not in ref_df.columns:
        raise KeyError(
            f"[rotulagem] Coluna de margem '{col_margem}' não encontrada nas features."
        )

    s = ref_df[col_margem].dropna().astype(float)
    if s.empty:
        raise ValueError(
            "[rotulagem] Série de margem vazia; não é possível calcular quantis."
        )

    t_low = float(np.nanquantile(s, q_baixo))
    t_med = float(np.nanquantile(s, q_medio))
    if not (t_low <= t_med):  # segurança contra distribuições patológicas
        t_low, t_med = sorted([t_low, t_med])

    thresholds = {
        "t_low": t_low,
        "t_med": t_med,
        "col": col_margem,
        "q_baixo": q_baixo,
        "q_medio": q_medio,
    }

    if bool(rules.get("usar_override_hidro", False)):
        hydro_cfg: Dict[str, Any] = {
            "usar_override_hidro": True,
            "janelas_consecutivas_ena": int(rules.get("janelas_consecutivas_ena", 1)),
        }
        ear_col = rules.get("coluna_ear")
        if ear_col:
            hydro_cfg["ear_col"] = ear_col
            hydro_cfg["ear_q_baixo"] = float(rules.get("ear_q_baixo", 0.2))
            if ear_col in ref_df.columns:
                ear_series = ref_df[ear_col].dropna().astype(float)
                if not ear_series.empty:
                    hydro_cfg["ear_threshold"] = float(
                        np.nanquantile(ear_series, hydro_cfg["ear_q_baixo"])
                    )
                else:
                    hydro_cfg["ear_threshold"] = None
                    warnings.warn(
                        f"[rotulagem] Serie EAR '{ear_col}' vazia; override hidrologico por EAR ignorado."
                    )
            else:
                hydro_cfg["ear_threshold"] = None
                warnings.warn(
                    f"[rotulagem] Coluna EAR '{ear_col}' nao encontrada nas features; override hidrologico por EAR ignorado."
                )
        ena_col = rules.get("coluna_ena")
        if ena_col:
            hydro_cfg["ena_col"] = ena_col
            hydro_cfg["ena_q_baixo"] = float(rules.get("ena_q_baixo", 0.2))
            if ena_col in ref_df.columns:
                ena_series = ref_df[ena_col].dropna().astype(float)
                if not ena_series.empty:
                    hydro_cfg["ena_threshold"] = float(
                        np.nanquantile(ena_series, hydro_cfg["ena_q_baixo"])
                    )
                else:
                    hydro_cfg["ena_threshold"] = None
                    warnings.warn(
                        f"[rotulagem] Serie ENA '{ena_col}' vazia; override hidrologico por ENA ignorado."
                    )
            else:
                hydro_cfg["ena_threshold"] = None
                warnings.warn(
                    f"[rotulagem] Coluna ENA '{ena_col}' nao encontrada nas features; override hidrologico por ENA ignorado."
                )
        thresholds["hydro"] = hydro_cfg

    return thresholds


def _apply_label_adjustments(
    base_labels: pd.Series,
    Xw: pd.DataFrame,
    cfg: Dict,
    thresholds: Dict[str, Any],
) -> pd.Series:
    """Aplica ajustes pos-quantis (cortes e hidrologia) nos rotulos base."""
    if base_labels.empty:
        return base_labels
    y_idx = base_labels.map(LABEL_TO_INT).astype(float)
    y_idx = _apply_curtailment_downgrade(y_idx, Xw, cfg)
    y_idx = _apply_hydrology_overrides(y_idx, Xw, cfg, thresholds)
    adjusted = y_idx.astype("Int64").map(INT_TO_LABEL)
    return adjusted.astype("string")


def _apply_curtailment_downgrade(
    y_idx: pd.Series, Xw: pd.DataFrame, cfg: Dict
) -> pd.Series:
    rules = cfg.get("problem", {}).get("label_rules", {}) or {}
    curtail_cfg = rules.get("curtail_downgrade", {}) or {}
    if not curtail_cfg.get("usar_downgrade_cortes", False):
        return y_idx

    ratio_col = "ratio_corte_renovavel_w"
    if ratio_col not in Xw.columns:
        return y_idx

    ratio_series = Xw[ratio_col].astype(float)
    thr = float(curtail_cfg.get("corte_ratio_thr", 0.05))
    mask = (ratio_series >= thr).fillna(False)

    if curtail_cfg.get("requer_saldo_importador_nao_positivo", False):
        saldo_col = "saldo_importador_mwh_sum_w"
        if saldo_col in Xw.columns:
            saldo_series = Xw[saldo_col].astype(float)
            mask &= (saldo_series <= 0).fillna(False)
        else:
            warnings.warn(
                "[rotulagem] Flag 'requer_saldo_importador_nao_positivo' ativa, mas coluna 'saldo_importador_mwh_sum_w' ausente; downgrade por cortes ignorado."
            )
            mask &= False

    # Filtro opcional para rebaixar apenas de 'medio' para 'baixo'
    if curtail_cfg.get("rebaixar_apenas_de_medio_para_baixo", False):
        mask &= y_idx == LABEL_TO_INT["medio"]

    mask &= y_idx.notna()

    if not mask.any():
        return y_idx

    out = y_idx.copy()
    out.loc[mask] = (out.loc[mask] - 1).clip(lower=LABEL_TO_INT["baixo"])
    return out


def _apply_hydrology_overrides(
    y_idx: pd.Series, Xw: pd.DataFrame, cfg: Dict, thresholds: Dict[str, Any]
) -> pd.Series:
    rules = cfg.get("problem", {}).get("label_rules", {}) or {}
    if not rules.get("usar_override_hidro", False):
        return y_idx

    hydro_cfg = thresholds.get("hydro", {}) or {}
    mask = pd.Series(False, index=Xw.index, dtype=bool)

    ear_col = hydro_cfg.get("ear_col") or rules.get("coluna_ear")
    ear_thr = hydro_cfg.get("ear_threshold")
    if ear_col and ear_thr is not None:
        if ear_col in Xw.columns:
            ear_series = Xw[ear_col].astype(float)
            ear_low = (ear_series <= ear_thr).fillna(False)
            mask |= ear_low
        else:
            warnings.warn(
                f"[rotulagem] Coluna de EAR '{ear_col}' ausente nas features; override hidrologico ignorado."
            )

    ena_col = hydro_cfg.get("ena_col") or rules.get("coluna_ena")
    ena_thr = hydro_cfg.get("ena_threshold")
    consec = int(
        hydro_cfg.get("janelas_consecutivas_ena")
        if hydro_cfg.get("janelas_consecutivas_ena") is not None
        else rules.get("janelas_consecutivas_ena", 1)
    )
    if consec < 1:
        consec = 1

    if ena_col and ena_thr is not None:
        if ena_col in Xw.columns:
            ena_series = Xw[ena_col].astype(float)
            ena_low = (ena_series <= ena_thr).fillna(False)
            if consec <= 1:
                ena_bad = ena_low
            else:
                ena_bad = (
                    ena_low.astype(int).rolling(window=consec, min_periods=consec).sum()
                    == consec
                )
            mask |= ena_bad.fillna(False)
        else:
            warnings.warn(
                f"[rotulagem] Coluna de ENA '{ena_col}' ausente nas features; override hidrologico ignorado."
            )

    mask &= y_idx.notna()

    if not mask.any():
        return y_idx

    out = y_idx.copy()
    out.loc[mask] = LABEL_TO_INT["alto"]
    return out




def _apply_hard_rules(
    y_idx: pd.Series, Xw: pd.DataFrame, cfg: Dict
) -> pd.Series:
    rules = cfg.get("problem", {}).get("label_rules", {}) or {}
    if not rules.get("usar_regras_duras", False):
        return y_idx

    mask = pd.Series(False, index=y_idx.index, dtype=bool)

    rm_frac = rules.get("reserva_operativa_frac")
    if rm_frac is not None:
        try:
            rm_frac = float(rm_frac)
        except Exception:
            rm_frac = None
    ens_thr = rules.get("ens_ratio_thr")
    if ens_thr is not None:
        try:
            ens_thr = float(ens_thr)
        except Exception:
            ens_thr = None
    lolp_thr = rules.get("lolp_thr")
    if lolp_thr is not None:
        try:
            lolp_thr = float(lolp_thr)
        except Exception:
            lolp_thr = None

    if rm_frac is not None:
        col = "reserve_margin_ratio_w"
        if col in Xw.columns:
            bad = (Xw[col].astype(float) < rm_frac).fillna(False)
            mask |= bad
        else:
            warnings.warn(
                f"[rotulagem] Coluna '{col}' ausente; regra dura de reserva operativa ignorada."
            )

    if ens_thr is not None:
        col = "ens_week_ratio"
        if col in Xw.columns:
            bad = (Xw[col].astype(float) >= ens_thr).fillna(False)
            mask |= bad
        else:
            warnings.warn(
                f"[rotulagem] Coluna '{col}' ausente; regra dura de ENS ignorada."
            )

    if lolp_thr is not None:
        col = "lolp_52w"
        if col in Xw.columns:
            bad = (Xw[col].astype(float) >= lolp_thr).fillna(False)
            mask |= bad
        else:
            warnings.warn(
                f"[rotulagem] Coluna '{col}' ausente; regra dura de LOLP ignorada."
            )

    mask &= y_idx.notna()
    if not mask.any():
        return y_idx

    out = y_idx.copy()
    out.loc[mask] = LABEL_TO_INT["alto"]
    return out


def _postprocess_with_hard_rules(
    y_labels: pd.Series, Xw: pd.DataFrame, cfg: Dict
) -> pd.Series:
    rules = cfg.get("problem", {}).get("label_rules", {}) or {}
    if not rules.get("usar_regras_duras", False):
        return y_labels

    y_idx = y_labels.map(LABEL_TO_INT).astype(float)
    y_idx = _apply_hard_rules(y_idx, Xw, cfg)
    return y_idx.astype("Int64").map(INT_TO_LABEL).astype("string")


def _maybe_tune_estimator(
    base_estimator,
    model_cfg: Dict,
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
) -> Tuple[Pipeline, Optional[Dict[str, Any]]]:
    tuning_cfg = (model_cfg.get("tuning") or {})
    if not tuning_cfg.get("use"):
        return base_estimator, None

    param_grid = tuning_cfg.get("param_grid") or {}
    if not param_grid:
        return base_estimator, None

    mask = y.notna()
    if not mask.any():
        warnings.warn(
            f"[tuning:{model_name}] Nenhum rotulo disponivel; pulando GridSearchCV."
        )
        return base_estimator, None

    X_fit = X.loc[mask]
    y_fit = encode_labels(y.loc[mask])

    step_name = None
    if isinstance(base_estimator, Pipeline):
        step_name = base_estimator.steps[-1][0]

    def _map_param(param_name: str) -> str:
        if step_name:
            return f"base_estimator__{step_name}__{param_name}"
        return f"base_estimator__{param_name}"

    mapped_grid = {_map_param(k): v for k, v in param_grid.items()}

    scoring_cfg = tuning_cfg.get("scoring") or ["f1_macro"]
    if isinstance(scoring_cfg, (str, bytes)):
        scoring_list = [scoring_cfg]
    else:
        scoring_list = list(scoring_cfg)

    if len(scoring_list) == 1:
        scoring_arg = scoring_list[0]
        refit = tuning_cfg.get("refit", True)
    else:
        scoring_arg = scoring_list
        refit = tuning_cfg.get("refit") or scoring_list[0]

    cv_splits = int(tuning_cfg.get("cv_splits", 3))
    inner_cv = TimeSeriesSplit(n_splits=cv_splits)

    grid_estimator = ContiguousLabelClassifier(
        base_estimator=clone(base_estimator),
        fallback_strategy="most_frequent",
    )

    grid = GridSearchCV(
        estimator=grid_estimator,
        param_grid=mapped_grid,
        scoring=scoring_arg,
        refit=refit,
        cv=inner_cv,
        n_jobs=tuning_cfg.get("n_jobs", None),
        verbose=int(tuning_cfg.get("verbose", 0)),
    )

    try:
        grid.fit(X_fit, y_fit)
    except Exception as exc:
        warnings.warn(f"[tuning:{model_name}] Falha no GridSearchCV: {exc}.")
        return base_estimator, None

    best_params = grid.best_params_
    prefix_parts = ["base_estimator__"]
    if step_name:
        prefix_parts.append(f"{step_name}__")
    prefix = "".join(prefix_parts)
    pretty_params = {
        key[len(prefix):] if key.startswith(prefix) else key: value
        for key, value in best_params.items()
    }
    print(f"[tuning:{model_name}] melhores parametros: {pretty_params}")

    best_base = clone(grid.best_estimator_.base_estimator)
    return best_base, pretty_params


def rotular_semana_com_thresholds(
    Xw: pd.DataFrame, cfg: Dict, thresholds: Dict[str, Any]
) -> pd.Series:
    """
    Aplica rótulos fixos ('baixo'|'medio'|'alto') com base em thresholds pré-calculados.
    Regra: margem <= t_low → 'alto'; <= t_med → 'medio'; acima → 'baixo'.
    """
    col = thresholds["col"]
    if col not in Xw.columns:
        raise KeyError(f"[rotular_semana_com_thresholds] Coluna '{col}' ausente.")
    x = Xw[col].astype(float)

    t_low = thresholds["t_low"]
    t_med = thresholds["t_med"]

    y = pd.Series(index=Xw.index, dtype="string")
    y.loc[x <= t_low] = "alto"
    y.loc[(x > t_low) & (x <= t_med)] = "medio"
    y.loc[x > t_med] = "baixo"

    y = _apply_label_adjustments(y, Xw, cfg, thresholds)
    return y


def rotular_semana(
    Xw: pd.DataFrame, cfg: Dict, ref_df: Optional[pd.DataFrame] = None
) -> pd.Series:
    """Rotula usando quantis do ref_df (ou do próprio Xw quando ref_df=None)."""
    if ref_df is None:
        ref_df = Xw
    thr = compute_label_thresholds(cfg, ref_df)
    return rotular_semana_com_thresholds(Xw, cfg, thr)


# ============================================================
# ================== OVERSAMPLING CONTROLADO =================
# ============================================================


def _reindex_unique_if_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Garante índice único caso seja DateTimeIndex e duplicatas surjam após concat."""
    if isinstance(df.index, pd.DatetimeIndex):
        dups = df.index.duplicated(keep=False)
        if dups.any():
            new_index = df.index.to_series()
            counts = {}
            for i, ts in enumerate(new_index):
                c = counts.get(ts, 0)
                counts[ts] = c + 1
                if c > 0:
                    new_index.iat[i] = ts + pd.Timedelta(microseconds=c)
            df = df.copy()
            df.index = pd.DatetimeIndex(new_index.values)
    return df


def oversample_minority(
    X: pd.DataFrame,
    y: pd.Series,
    min_fraction: float = 0.25,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Duplica linhas das classes com participação < min_fraction até atingir o piso.
    Usar APENAS em treino/validação (nunca no hold-out).
    """
    if min_fraction <= 0 or min_fraction >= 1:
        return X, y

    rng = np.random.default_rng(random_state)
    X_aug = X
    y_aug = y.astype("string")

    n_total = len(y_aug)
    if n_total == 0:
        return X_aug, y_aug

    counts = y_aug.value_counts()
    for cls in LABELS:
        cnt = int(counts.get(cls, 0))
        frac = cnt / n_total if n_total else 0.0
        if frac >= min_fraction or cnt == 0:
            continue

        # k mínimo para atingir o piso no novo total:
        need = (min_fraction * n_total - cnt) / (1.0 - min_fraction)
        k = int(np.ceil(max(0.0, need)))

        idx_cls = y_aug[y_aug == cls].index
        if len(idx_cls) == 0:
            continue
        picks = rng.choice(idx_cls, size=k, replace=True)

        X_aug = pd.concat([X_aug, X.loc[picks]], axis=0)
        y_aug = pd.concat([y_aug, y_aug.loc[picks]], axis=0)
        n_total = len(y_aug)

    X_aug, y_aug = _reindex_unique_if_datetime(X_aug), y_aug
    return X_aug, y_aug


# ============================================================
# ===================== MODELOS SUPORTADOS ===================
# ============================================================


def _make_estimator(model_cfg: Dict):
    """Cria um Pipeline apropriado para cada modelo, com imputer e scaler quando necessário."""
    mtype = (model_cfg.get("type") or "").lower()
    params = model_cfg.get("params", {}) or {}

    if mtype in ("logistic_regression", "logreg", "lr"):
        base = LogisticRegression(
            C=params.get("C", 1.0),
            max_iter=params.get("max_iter", 500),
            solver=params.get("solver", "saga"),
            class_weight=params.get("class_weight", "balanced"),
            n_jobs=params.get("n_jobs", None),
        )
        pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", base),
            ]
        )
        return pipe

    if mtype in ("xgboost", "xgb"):
        if not _HAS_XGB:
            warnings.warn(
                "[train] xgboost nao instalado; caindo para LogisticRegression."
            )
            # reaproveita o caminho da logreg (com scaler, que não atrapalha)
            return _make_estimator({"type": "logistic_regression", "params": params})

        base = XGBClassifier(
            n_estimators=params.get("n_estimators", 300),
            max_depth=params.get("max_depth", 4),
            learning_rate=params.get("learning_rate", 0.05),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            reg_lambda=params.get("reg_lambda", 1.0),
            reg_alpha=params.get("reg_alpha", 0.1),
            scale_pos_weight=params.get("scale_pos_weight", 1),
            random_state=params.get("random_state", 42),
            n_jobs=params.get("n_jobs", -1),
            tree_method=params.get("tree_method", "hist"),
        )
        pipe = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),  # XGB não precisa de scaler
                ("model", base),
            ]
        )
        return pipe

    raise ValueError(f"[train] Tipo de modelo não suportado: {mtype}")


# ============================================================
# ===================== UTIL/MÉTRICAS ========================
# ============================================================


def _normalize_pred_to_domain(y_pred_raw, est) -> pd.Series:
    """Normaliza predições para rótulos canônicos ('baixo'|'medio'|'alto')."""
    ypred = pd.Series(y_pred_raw)
    if np.issubdtype(ypred.dtype, np.number):
        ypred = ypred.astype(int).map(INT_TO_LABEL)

    if not ypred.isin(LABELS).all() and hasattr(est, "inv_label_mapping_"):
        invmap = getattr(est, "inv_label_mapping_", {})
        try:
            ypred2 = pd.Series(y_pred_raw).map(invmap)
            ypred = ypred.where(ypred.isin(LABELS), ypred2)
        except Exception:
            pass

    return ypred.astype("string")


def _append_metrics_row(
    paths: Paths, model_name: str, f1: float, ba: float
) -> pd.DataFrame:
    """Acrescenta/gera reports/metrics_cv.csv com linha {modelo, f1_macro, balanced_acc} e retorna o DF."""
    reports_dir = Path(paths.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / "metrics_cv.csv"

    row = pd.DataFrame([{"modelo": model_name, "f1_macro": f1, "balanced_acc": ba}])
    if csv_path.exists():
        old = pd.read_csv(csv_path)
        out = pd.concat([old, row], ignore_index=True)
    else:
        out = row
    out.to_csv(csv_path, index=False)
    return out


# ============================================================
# ============ SPLIT HOLDOUT (cauda temporal) ================
# ============================================================


def _split_holdout(
    Xw: pd.DataFrame, holdout_fraction: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Separa hold-out pela cauda temporal (último bloco)."""
    if holdout_fraction <= 0 or holdout_fraction >= 0.9:
        return Xw, pd.DataFrame(index=pd.DatetimeIndex([], name=Xw.index.name))
    n = len(Xw)
    n_hold = max(1, int(np.floor(n * holdout_fraction)))
    Xtr = Xw.iloc[:-n_hold]
    Xte = Xw.iloc[-n_hold:]
    return Xtr, Xte


# ============================================================
# ======================= TREINO DE 1 MODELO =================
# ============================================================


def _train_one_model(
    model_cfg: Dict,
    model_name: str,
    cfg: Dict,
    paths: Paths,
    X_trainval_base: pd.DataFrame,
    X_holdout: pd.DataFrame,
    thresholds: Dict[str, Any],
    y_base: pd.Series,
    y_hold: pd.Series,
) -> Tuple[str, float, float]:
    """Treina 1 modelo (CV + final), salva o .joblib e devolve (model_name, f1_macro_médio, bal_acc_médio)."""

    # Oversample config
    os_cfg = cfg.get("modeling", {}).get("oversample", {}) or {}
    os_use = bool(os_cfg.get("use", False))
    os_min_fraction = float(os_cfg.get("min_fraction", 0.25))
    os_rs = int(os_cfg.get("random_state", 42))
    os_in_cv = bool(os_cfg.get("in_cv", True))  # conservador por padrão

    # Para compat: sempre gravar o parquet base como features_trainval.parquet
    feat_dir = Path(paths.features_dir)
    X_trainval_base.to_parquet(feat_dir / "features_trainval.parquet", index=True)

    # Estimador (pipeline com imputer e, se logreg, scaler)
    base_estimator = _make_estimator(model_cfg)
    base_estimator, tuned_params = _maybe_tune_estimator(
        base_estimator=base_estimator,
        model_cfg=model_cfg,
        model_name=model_name,
        X=X_trainval_base,
        y=y_base,
    )
    params_path = Path(paths.reports_dir) / f"tuning_{model_name}.json"
    params_path.parent.mkdir(parents=True, exist_ok=True)
    if tuned_params:
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(tuned_params, f, ensure_ascii=False, indent=2)
    else:
        if params_path.exists():
            params_path.unlink()

    est = ContiguousLabelClassifier(
        base_estimator=base_estimator, fallback_strategy="most_frequent"
    )

    # CV
    n_splits = int(cfg.get("modeling", {}).get("cv", {}).get("n_splits", 5))
    tss = TimeSeriesSplit(n_splits=n_splits)

    cv_metrics = []
    for fold, (itr, ite) in enumerate(tss.split(X_trainval_base), start=1):
        Xtr, Xte = X_trainval_base.iloc[itr], X_trainval_base.iloc[ite]
        ytr = rotular_semana_com_thresholds(Xtr, cfg, thresholds)
        yte = rotular_semana_com_thresholds(Xte, cfg, thresholds)

        # Filtro anti-NaN de labels
        mask_tr = ytr.notna()
        mask_te = yte.notna()
        dropped_tr = int((~mask_tr).sum())
        dropped_te = int((~mask_te).sum())
        if dropped_tr or dropped_te:
            print(
                f"[cv:{model_name}] Fold {fold}: removendo NaN em labels -> treino:{dropped_tr} | teste:{dropped_te}"
            )
        Xtr, ytr = Xtr.loc[mask_tr], ytr.loc[mask_tr]
        Xte, yte = Xte.loc[mask_te], yte.loc[mask_te]
        if len(yte) == 0 or len(ytr) == 0:
            print(f"[cv:{model_name}] Fold {fold}: sem amostras após filtro; pulando.")
            continue

        if os_use and os_in_cv:
            Xtr, ytr = oversample_minority(
                Xtr, ytr, min_fraction=os_min_fraction, random_state=os_rs
            )

        # Ajuste do modelo (pipeline fará imputação/escala usando APENAS Xtr)
        est.fit(Xtr, encode_labels(ytr))

        # Predição e normalização
        y_pred_raw = est.predict(Xte)
        y_pred = _normalize_pred_to_domain(y_pred_raw, est)
        y_pred = _postprocess_with_hard_rules(y_pred, Xte, cfg)

        # Remover quaisquer predições fora do domínio/NaN
        mask_ok = y_pred.isin(LABELS)
        if not mask_ok.all():
            n_bad = int((~mask_ok).sum())
            print(
                f"[cv:{model_name}] Fold {fold}: removendo {n_bad} predições fora do domínio/NaN."
            )
            y_pred = y_pred.loc[mask_ok]
            yte = yte.loc[mask_ok]
            Xte = Xte.loc[mask_ok]
        if len(yte) == 0:
            print(
                f"[cv:{model_name}] Fold {fold}: teste vazio após normalização; pulando."
            )
            continue

        y_true = encode_labels(yte)
        y_pred_enc = encode_labels(y_pred)
        fold_f1 = f1_score(y_true, y_pred_enc, average="macro", zero_division=0)
        fold_ba = balanced_accuracy_score(y_true, y_pred_enc)
        cv_metrics.append((fold_f1, fold_ba))
        print(
            f"[cv:{model_name}] Fold {fold}/{n_splits}: f1_macro={fold_f1:.3f} | bal_acc={fold_ba:.3f}"
        )

    if cv_metrics:
        mf1 = float(np.mean([m[0] for m in cv_metrics]))
        mba = float(np.mean([m[1] for m in cv_metrics]))
        print(f"[cv:{model_name}] MÉDIAS: f1_macro={mf1:.3f} | bal_acc={mba:.3f}")
    else:
        mf1 = float("nan")
        mba = float("nan")
        print(f"[cv:{model_name}] Nenhum fold válido para métricas.")

    # Treino final (em todo train_val_base)
    X_final = X_trainval_base.copy()
    y_final = y_base.copy()
    mask_fin = y_final.notna()
    n_drop_fin = int((~mask_fin).sum())
    if n_drop_fin > 0:
        print(
            f"[train:{model_name}] Removendo {n_drop_fin} amostras com label NaN no ajuste final."
        )
    X_final, y_final = X_final.loc[mask_fin], y_final.loc[mask_fin]

    if os_use and not os_in_cv:
        X_final, y_final = oversample_minority(
            X_final, y_final, min_fraction=os_min_fraction, random_state=os_rs
        )

    est.fit(X_final, encode_labels(y_final))

    # metadados no próprio objeto
    setattr(est, "label_thresholds_", thresholds)
    setattr(est, "label_mapping_", LABEL_TO_INT)
    setattr(est, "inv_label_mapping_", INT_TO_LABEL)

    # Persistência
    model_path = Path(paths.models_dir) / f"{model_name}.joblib"
    dump(est, model_path)
    print(f"[train:{model_name}] Modelo salvo em: {model_path}")

    # Persistir thresholds por modelo (e o genérico para compatibilidade)
    with open(
        Path(paths.models_dir) / f"label_thresholds_{model_name}.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(thresholds, f, ensure_ascii=False, indent=2)
    with open(
        Path(paths.models_dir) / "label_thresholds.json", "w", encoding="utf-8"
    ) as f:
        json.dump(thresholds, f, ensure_ascii=False, indent=2)

    return model_name, mf1, mba


# ============================================================
# ============================ MAIN ==========================
# ============================================================


def main(
    config_path: str = "configs/config.yaml", model_name: Optional[str] = None
) -> None:
    # -------------------- Carregar config --------------------
    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))

    paths = Paths(
        raw_dir=cfg["paths"]["raw_dir"],
        features_dir=cfg["paths"]["features_dir"],
        models_dir=cfg["paths"]["models_dir"],
        reports_dir=cfg["paths"]["reports_dir"],
    )
    feat_dir = Path(paths.features_dir)
    feat_dir.mkdir(parents=True, exist_ok=True)
    Path(paths.models_dir).mkdir(parents=True, exist_ok=True)
    Path(paths.reports_dir).mkdir(parents=True, exist_ok=True)

    # -------------------- Preparar features (UMA vez) --------
    data = load_all_sources(cfg)
    Xw = build_features_weekly(data, cfg)
    Xw = Xw.dropna(axis=1, how="all").sort_index()
    if Xw.empty:
        raise RuntimeError("[train] Nenhuma feature semanal disponível.")

    # -------------------- Split hold-out (UMA vez) -----------
    hold_frac = float(cfg.get("modeling", {}).get("holdout_fraction", 0.2))
    train_val_base, holdout = _split_holdout(Xw, hold_frac)

    # Persistir parquets de referência (pré-oversample) para auditoria
    (feat_dir / "features_trainval_base.parquet").parent.mkdir(
        parents=True, exist_ok=True
    )
    train_val_base.to_parquet(feat_dir / "features_trainval_base.parquet", index=True)
    holdout.to_parquet(feat_dir / "features_test_holdout.parquet", index=True)

    print(
        f"[train] Tamanhos: train_val_base={len(train_val_base)} | holdout={len(holdout)}"
    )

    # -------------------- Thresholds fixos (UMA vez) ---------
    thresholds = compute_label_thresholds(cfg, train_val_base)
    with open(
        Path(paths.models_dir) / "label_thresholds.json", "w", encoding="utf-8"
    ) as f:
        json.dump(thresholds, f, ensure_ascii=False, indent=2)

    # -------------------- Rótulos base (UMA vez) -------------
    y_base = rotular_semana_com_thresholds(train_val_base, cfg, thresholds)
    y_hold = rotular_semana_com_thresholds(holdout, cfg, thresholds)

    # Log das distribuicoes base/holdout
    def _dist(s: pd.Series) -> Dict[str, str]:
        s = s.dropna()
        cnt = s.value_counts().reindex(LABELS).fillna(0).astype(int)
        total = int(cnt.sum())
        pct = (cnt / max(total, 1) * 100.0).round(2)
        return {k: f"{int(cnt[k])} ({pct[k]}%)" for k in LABELS}

    print("[train] Distribuicao (train_val_base):", _dist(y_base))
    print("[train] Distribuicao (holdout):      ", _dist(y_hold))

    # -------------------- Quais modelos treinar? -------------
    models_cfg: List[Dict] = cfg.get("modeling", {}).get("models", [])
    selected: List[Dict] = []

    if model_name:
        m = None
        for mc in models_cfg:
            if (mc.get("name") or "").lower() == model_name.lower():
                m = mc
                break
        if m is None:
            m = {"name": model_name, "type": model_name, "params": {}}
        selected = [m]
    else:
        if not models_cfg:
            models_cfg = [
                {"name": "logreg", "type": "logistic_regression", "params": {}},
                {"name": "xgb", "type": "xgboost", "params": {}},
            ]
        selected = models_cfg

    # -------------------- Loop de modelos --------------------
    all_rows = []
    for mc in selected:
        name = (mc.get("name") or mc.get("type") or "model").lower()
        print(f"\n=========== Treinando modelo: {name} ===========")
        mdl_name, mf1, mba = _train_one_model(
            model_cfg=mc,
            model_name=name,
            cfg=cfg,
            paths=paths,
            X_trainval_base=train_val_base,
            X_holdout=holdout,
            thresholds=thresholds,
            y_base=y_base,
            y_hold=y_hold,
        )
        _append_metrics_row(paths, mdl_name, mf1, mba)
        all_rows.append({"modelo": mdl_name, "f1_macro": mf1, "balanced_acc": mba})

    # -------------------- Tabela consolidada -----------------
    metrics_df = pd.DataFrame(all_rows)
    print("\n=== Metricas CV consolidadas ===")
    with pd.option_context("display.max_columns", None, "display.width", 120):
        print(
            metrics_df.sort_values(by="f1_macro", ascending=False).to_string(
                index=False
            )
        )


if __name__ == "__main__":
    # Ex.: python src/train.py
    #     python src/train.py configs/config.yaml xgb
    import sys

    cfg_path = "configs/config.yaml"
    mname: Optional[str] = None
    if len(sys.argv) >= 2:
        cfg_path = sys.argv[1]
    if len(sys.argv) >= 3:
        mname = sys.argv[2]
    main(cfg_path, mname)
