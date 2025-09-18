# src/models/contiguous.py
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.dummy import DummyClassifier


class ContiguousLabelClassifier(BaseEstimator, ClassifierMixin):
    """Normaliza rótulos antes do treinamento e restaura os códigos originais.

    Args:
      base_estimator: Estimador compatível com a API do scikit-learn encapsulado pelo wrapper.
      fallback_strategy (str): Estratégia usada pelo DummyClassifier quando apenas uma classe permanece.
    """

    def __init__(self, base_estimator, fallback_strategy: str = "most_frequent"):
        self.base_estimator = base_estimator
        self.fallback_strategy = fallback_strategy

    def get_params(self, deep: bool = True) -> dict:
        params = {
            "base_estimator": self.base_estimator,
            "fallback_strategy": self.fallback_strategy,
        }
        if deep and hasattr(self.base_estimator, "get_params"):
            base_params = self.base_estimator.get_params(deep=True)
            params.update(base_params)
            params.update({f"base_estimator__{k}": v for k, v in base_params.items()})
        return params

    def set_params(self, **params):
        base_updates = {}
        own_updates = {}
        base_param_names = set()
        if hasattr(self.base_estimator, "get_params"):
            base_param_names = set(self.base_estimator.get_params(deep=True).keys())
        for key, value in params.items():
            if key.startswith("base_estimator__"):
                base_updates[key.split("__", 1)[1]] = value
            elif key in {"base_estimator", "fallback_strategy"}:
                own_updates[key] = value
            elif key in base_param_names:
                base_updates[key] = value
            else:
                own_updates[key] = value
        if base_updates:
            self.base_estimator.set_params(**base_updates)
        return super().set_params(**own_updates)

    def fit(self, X, y, **fit_params):
        """Ajusta o estimador encapsulado com rótulos densificados.

        Args:
          X (array-like): Conjunto de observações usado no ajuste.
          y (array-like): Rótulos originais (0, 1, 2) associados às observações.
          **fit_params: Argumentos adicionais repassados ao estimador base.

        Returns:
          ContiguousLabelClassifier: Instância ajustada.
        """
        y_arr = np.asarray(y)
        if y_arr.size == 0:
            raise ValueError("Nenhum rótulo disponível para treino.")

        uniques = np.unique(y_arr)
        uniques_sorted = np.sort(uniques)
        self._forward_map_ = {orig: idx for idx, orig in enumerate(uniques_sorted)}
        self._inverse_map_ = {idx: orig for orig, idx in self._forward_map_.items()}
        self.classes_ = uniques_sorted
        self._is_dummy_ = len(uniques_sorted) == 1

        if self._is_dummy_:
            self._estimator_ = DummyClassifier(strategy=self.fallback_strategy)
            y_dense = np.zeros_like(y_arr, dtype=int)
        else:
            self._estimator_ = clone(self.base_estimator)
            y_dense = np.vectorize(self._forward_map_.get)(y_arr).astype(int)
            self._configure_estimator(len(uniques_sorted))

        if "eval_set" in fit_params and fit_params["eval_set"] is not None:
            fit_params = fit_params.copy()
            fit_params["eval_set"] = [
                (X_eval, self._encode(eval_y)) for X_eval, eval_y in fit_params["eval_set"]
            ]

        self._estimator_.fit(X, y_dense, **fit_params)
        return self

    def _configure_estimator(self, n_classes: int) -> None:
        """Configura parâmetros específicos para estimadores suportados.

        Args:
          n_classes (int): Número de classes distintas após a densificação.
        """
        if hasattr(self._estimator_, "get_xgb_params"):
            if n_classes <= 2:
                self._estimator_.set_params(objective="binary:logistic")
                try:
                    self._estimator_.set_params(num_class=None)
                except ValueError:
                    pass
            else:
                self._estimator_.set_params(objective="multi:softprob", num_class=n_classes)

    def _encode(self, y):
        arr = np.asarray(y)
        if self._is_dummy_:
            return np.zeros_like(arr, dtype=int)
        return np.vectorize(self._forward_map_.get)(arr).astype(int)

    def _decode(self, y_dense):
        arr = np.asarray(y_dense)
        return np.vectorize(self._inverse_map_.get)(arr)

    def predict(self, X):
        dense = self._estimator_.predict(X)
        return self._decode(dense)

    def predict_proba(self, X):
        if not hasattr(self._estimator_, "predict_proba"):
            raise AttributeError("Estimador base não possui predict_proba.")
        return self._estimator_.predict_proba(X)

    def decision_function(self, X):
        if not hasattr(self._estimator_, "decision_function"):
            raise AttributeError("Estimador base não possui decision_function.")
        return self._estimator_.decision_function(X)

    def score(self, X, y):
        y_dense = self._encode(y)
        return self._estimator_.score(X, y_dense)

    def __getattr__(self, item):
        if item.endswith("_") and hasattr(self.__dict__.get("_estimator_", None), item):
            return getattr(self._estimator_, item)
        raise AttributeError(item)
