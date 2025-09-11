from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd


def main():
    base = Path('data/features')
    test_p = base / 'features_test_holdout.parquet'
    train_p = base / 'features_trainval.parquet'
    full_p = base / 'features_weekly.parquet'

    paths = [p for p in [test_p, train_p, full_p] if p.exists()]
    if not paths:
        print('No feature parquet files found in data/features')
        sys.exit(1)

    for p in paths:
        df = pd.read_parquet(p)
        print(f"\n== {p.name} ==")
        print('shape:', df.shape)
        na_all = [c for c in df.columns if df[c].isna().all()]
        print('NA-all cols:', len(na_all))
        if na_all:
            print('first 40 NA-all columns:')
            for c in na_all[:40]:
                print('  -', c)
        # summarize corte features
        corte_cols = [c for c in df.columns if c.startswith('corte_')]
        if corte_cols:
            print('\nCorte columns summary (non-null counts, min/max sample):')
            sub = df[corte_cols]
            nn = sub.notna().sum().sort_values().head(10)
            print('lowest non-null counts:')
            for k,v in nn.items():
                print(f'  {k}: {v}')
            # show a few head rows for key metrics
            for key in ['corte_eolica_mwh_sum_w', 'corte_fv_mwh_sum_w', 'ratio_corte_renovavel_w']:
                if key in sub.columns:
                    s = sub[key]
                    print(f"\n{key}: non-null={s.notna().sum()}, unique={s.nunique()}")
                    print(' head:', s.head(5).to_list())
                    print(' tail:', s.tail(5).to_list())


if __name__ == '__main__':
    main()

