# Risco Semanal de Déficit de Energia em Goiás (SE/CO)

**Objetivo:** classificar, em **horizonte semanal**, o **risco de déficit de energia** em Goiás (baixo | médio | alto) usando dados **operacionais do ONS** e **clima** (leve).  
**Disciplinas atendidas:** Machine Learning, Cloud (AWS) e Modelagem de Dados.
**Alunos:** Thales Salvador, Miguel Toledo, Carlos Henrique.

## Estrutura do repositório

```
project/
├─ src/
│  ├─ data_loader.py
│  ├─ feature_engineer.py
│  ├─ train.py
│  ├─ evaluate.py
│  ├─ etl_ons.py                # ETL: brutos do ONS/NASA → diários padronizados
│  ├─ fetch_ons.py              # Downloader via CKAN (dados.ons.org.br)
│  └─ api/handler.py              # stub p/ AWS Lambda
├─ configs/config.yaml
├─ requirements.txt
├─ Makefile
├─ README.md
└─ data/
   └─ raw/                        # coloque aqui os CSVs brutos (ver tabela abaixo)
```

> Timezone padrão: **America/Sao_Paulo**.

## Início Rápido

1) Ambiente e dependências
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

2) Baixar dados e gerar insumos diários (ONS + NASA)
```bash
make data                        # download (CKAN) + ETL (+ clima NASA)
```

3) Gerar features semanais e treinar
```bash
make features                    # => data/features/features_weekly.parquet
make train                       # => models/*.joblib e reports/cv_scores.csv
```

4) (Opcional) Avaliar
```bash
make eval
```

Detalhes sobre fontes, nomes de arquivos e parametrização em `README_data.md`.

## Dados utilizados

Coloque estes **CSVs diários** em `data/raw/`:

| Arquivo CSV | Colunas mínimas | Fonte |
|---|---|---|
| `ons_carga_diaria.csv` | `data`, `carga_mwh` | ONS – carga verificada (SE/CO) |
| `ons_geracao_fontes_diaria.csv` | `data`, `ger_hidreletrica_mwh`, `ger_eolica_mwh`, `ger_fv_mwh`, `ger_termica_mwh` | ONS – geração por **fonte** (SE/CO) |
| `ons_intercambio_diario.csv` | `data`, `import_mwh`, `export_mwh` | ONS – intercâmbios (SE/CO) |
| `ons_ena_diaria.csv` | `data`, `ena_mwmed` | ONS – ENA (SE/CO) |
| `ons_ear_diaria.csv` | `data`, `ear_pct` | ONS – EAR (SE/CO) |
| `ons_cortes_eolica_diario.csv` | `data`, `corte_eolica_mwh` | ONS – cortes eólicos (constrained-off) |
| `ons_cortes_fv_diario.csv` | `data`, `corte_fv_mwh` | ONS – cortes FV (constrained-off) |
| `clima_go_diario.csv` | `data`, `ghi`, `temp2m_c`, `precipitacao_mm` | NASA POWER (GO agregado) |

> Se algum dataset vier horário/semi-horário, **agregue para diário** (média ou soma, conforme a métrica) antes de salvar, ou deixe que os scripts reamostrem.

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Como rodar

1) **Gerar features semanais**
```bash
make features
# => salva data/features/features_weekly.parquet
```

2) **Treinar modelos (LogReg e XGBoost)**
```bash
make train
# => salva models/logreg.joblib e models/xgb.joblib
# => gera reports/cv_scores.csv com F1-macro e Balanced Accuracy (TimeSeriesSplit)
```

3) **Avaliar e gerar relatório**
```bash
make eval
# => reports/report_xgb.txt (classification_report + matriz de confusão)
```

## Configuração (configs/config.yaml)

- `problem.horizon`: **weekly** (semanal).  
- `problem.label_rules`:
  - `coluna_margem`: **margem_suprimento_min_w** (proxy da margem semanal).
  - `q_baixo`/`q_medio`: definem os **quantis** para rotular **alto/médio/baixo**.
  - **Ajustes por cortes**: cortes eólico/FV **diminuem** o risco quando indicam **superávit renovável** (razão de corte semanal acima do limiar **e** sem importação líquida).
  - **Ajustes por hidrologia**: **EAR** muito baixa **ou** **ENA** muito baixa por `k` semanas → **alto**.
- `aggregation.features`: agregações de **diário→semanal**, **lags** e **janelas móveis**.
- `modeling.models`: especifica **Logistic Regression** e **XGBoost**.

> Você pode ajustar quantis/limiares no YAML sem alterar código.

## Como rotulamos (resumo)

1. Calculamos a **margem de suprimento semanal**:  
   `geração_total (hidro+eólica+FV+term) + importações − exportações`.  
2. Aplicamos **quantis** sobre o **mínimo**/proxy semanal para definir:  
   - **alto** (≤ Q10), **médio** (Q10–Q40], **baixo** (> Q40).  
3. **Ajustes**:  
   - **Cortes** (constrained-off) eólico/FV: **reduzem** o risco quando representam superávit renovável (razão de corte > limiar e sem importação líquida).  
   - **Hidrologia**: **EAR** muito baixa **ou** **ENA** muito baixa por `k` semanas → **alto**.

## Saídas

- `data/features/features_weekly.parquet` — feature store semanal.  
- `models/*.joblib` — artefatos dos modelos.  
- `reports/cv_scores.csv` — métricas de CV.  
- `reports/report_<modelo>.txt` — relatório final + matriz de confusão.

## API (stub para AWS Lambda)

Arquivo: `src/api/handler.py`.  
Empacote o modelo em `./models` (ou em `/opt/models/` na imagem Lambda).  
Exemplo de payload:

```json
{
  "features": {
    "ghi_mean_w": 5.1,
    "temp2m_c_max_w": 33.4,
    "precip_14d_mm_sum_w": 12.0,
    "margem_suprimento_w": 1.23
  }
}
```

Resposta:
```json
{ "classe_risco": "baixo" }
```

> Para produção, proteja o endpoint (API Gateway) e **versione** o modelo.

## Validação & Boas práticas

- **TimeSeriesSplit** (sem vazamento).  
- Padronização de timezone e reamostragem **D→W**.  
- Seeds fixas (`seed: 42`).  
- Colunas **em português** (consistentes com os CSVs e o código).

## Problemas comuns

- **Coluna ausente**: verifique o nome exato no CSV (tabela acima).  
- **Datas inválidas**: formato deve ser `YYYY-MM-DD`.  
- **Faltas longas**: o pipeline preenche curtas (até 7 dias). Falhas maiores → revisar a origem.

## Créditos dos dados

- **ONS** – Portal de Dados Abertos (carga, geração por fonte, intercâmbios, ENA/EAR, cortes).  
- **NASA POWER** – API diária (GHI, temperatura, precipitação).
