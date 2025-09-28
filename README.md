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
│  ├─ etl_ons.py                # ETL: brutos do ONS -> diários padronizados
│  ├─ meteo.py                  # Meteorologia (NASA POWER) – fetch/process
│  ├─ fetch_ons.py              # Downloader via CKAN (dados.ons.org.br)
│  └─ api/handler.py              # stub p/ AWS Lambda
├─ configs/config.yaml
├─ requirements.txt
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
python main.py data --incluir-meteorologia  # download (CKAN) + ETL (+ clima NASA)
# Limitar período: usa default em configs/config.yaml (download.since), ou:
# python main.py data --since 2022
```

3) Gerar features semanais e treinar
```bash
python main.py features          # => data/features/features_weekly.parquet
python main.py train             # => models/*.joblib e reports/cv_scores.csv
```

4) (Opcional) Avaliar
```bash
python main.py eval --model xgb
```

Comando único:
```bash
python main.py all --incluir-meteorologia
```

Detalhes sobre fontes, nomes de arquivos e parametrização em `README_data.md`.

## Dados utilizados

Coloque estes **CSVs diários** em `data/raw/` (colunas efetivamente usadas no pipeline):

| Arquivo CSV | Colunas usadas | Fonte |
|---|---|---|
| `ons_carga_diaria.csv` | `data`, `carga_mwh` | ONS – carga verificada (SE/CO) |
| `ons_geracao_fontes_diaria.csv` | `data`, `ger_hidreletrica_mwh`, `ger_eolica_mwh`, `ger_fv_mwh`, `ger_termica_mwh` | ONS – geração por **fonte** (SE/CO) |
| `ons_intercambio_diario.csv` | `data`, `import_mwh`, `export_mwh` | ONS – intercâmbios (SE/CO) |
| `ons_ena_diaria.csv` | `data`, `ena_mwmed` | ONS – ENA (SE/CO) |
| `ons_ear_diaria.csv` | `data`, `ear_pct` | ONS – EAR (SE/CO) |
| `ons_cortes_eolica_diario.csv` | `data`, `corte_eolica_mwh` | ONS – cortes eólicos (constrained-off) |
| `ons_cortes_fv_diario.csv` | `data`, `corte_fv_mwh` | ONS – cortes FV (constrained-off) |
| `clima_go_diario.csv` | `data`, `ghi`, `temp2m_c`, `precipitacao_mm` | NASA POWER (GO agregado) |

Observação: para clima criamos derivadas diárias antes do D->W: `precip_14d_mm` e `precip_30d_mm`.

> Se algum dataset vier horário/semi-horário, **agregue para diário** (média ou soma, conforme a métrica) antes de salvar, ou deixe que os scripts reamostrem.

## Features geradas

As features semanais seguem um padrão configurável em `configs/config.yaml` (agregações D->W: `mean`, `sum`, `max`, `min`, `std`, `p95`, `p05`; lags: `[1,2,4]`; janelas móveis: `[2,4]`). Abaixo, o mapeamento das colunas de entrada para as famílias de features geradas.

| Origem | Colunas (D) | Features semanais (padrão) | Observações/derivadas |
|---|---|---|---|
| Carga | `carga_mwh` | `carga_mwh_<agg>_w` para cada agregação | Usada para `margem_vs_carga_w`, `reserve_margin_ratio_w`, `ens_week_*` |
| Geração por fonte | `ger_hidreletrica_mwh`, `ger_eolica_mwh`, `ger_fv_mwh`, `ger_termica_mwh` | `ger_<fonte>_mwh_<agg>_w` | Soma semanal: `geracao_total_mwh_sum_w` |
| Intercâmbio | `import_mwh`, `export_mwh` | `import_mwh_<agg>_w`, `export_mwh_<agg>_w` | Somas: `import_total_mwh_sum_w`, `export_total_mwh_sum_w`; saldo: `saldo_importador_mwh_sum_w` |
| ENA | `ena_mwmed` | `ena_mwmed_<agg>_w` | — |
| EAR | `ear_pct` | `ear_pct_<agg>_w` | — |
| Cortes eólicos/FV | `corte_eolica_mwh`, `corte_fv_mwh` | `corte_eolica_mwh_<agg>_w`, `corte_fv_mwh_<agg>_w` | Derivadas: `corte_renovavel_mwh_sum_w`, `ratio_corte_renovavel_w` |
| Clima (NASA) | `ghi`, `temp2m_c`, `precipitacao_mm`, `precip_14d_mm`, `precip_30d_mm` | `<col>_<agg>_w` para cada coluna disponível | `precip_14d_mm`/`precip_30d_mm` são construídas no diário e então agregadas |

Derivadas operacionais (semanais):
- **`geracao_total_mwh_sum_w`**: Soma da geração de todas as fontes (hidro, eólica, FV, térmica) na semana.
- **`import_total_mwh_sum_w`**: Total de energia importada de outros subsistemas na semana.
- **`export_total_mwh_sum_w`**: Total de energia exportada para outros subsistemas na semana.
- **`saldo_importador_mwh_sum_w`**: Saldo líquido de energia (`importações - exportações`). Um valor positivo indica que o subsistema foi um importador líquido na semana.
- **`margem_suprimento_w`**: Representa a oferta total de energia disponível na semana. É calculada como `geração_total + importações - exportações`.
- **`margem_suprimento_min_w`**: Proxy da margem de suprimento, atualmente uma cópia da `margem_suprimento_w`.
- **`margem_vs_carga_w`**: **Feature central para a rotulagem.** É a diferença entre a oferta e a demanda (`margem_suprimento_w - carga_mwh_sum_w`). Um valor negativo indica um déficit de energia na semana.
- **`reserve_margin_ratio_w`**: Razão da margem de reserva, calculada como `margem_vs_carga_w / carga_mwh_sum_w`. Indica a folga (ou déficit) como uma porcentagem da demanda.
- **`ens_week_mwh`**: "Energy Not Supplied" (Energia Não Suprida). É uma estimativa do déficit, calculada como o valor negativo da `margem_vs_carga_w` (zerado se a margem for positiva).
- **`ens_week_ratio`**: Razão do ENS em relação à carga total da semana (`ens_week_mwh / carga_mwh_sum_w`).
- **`lolp_52w`**: "Loss of Load Probability". É uma probabilidade empírica de perda de carga, calculada como a frequência de semanas com `margem_vs_carga_w` negativa em uma janela móvel de 52 semanas.
 
Expansões temporais (para cada feature semanal base):
- Lags: sufixos `_<feature>_lag{L}w` para `L` em `[1,2,4]`.
- Janelas móveis: `_<feature>_r{R}w_mean` e `_<feature>_r{R}w_std` para `R` em `[2,4]`.

## Coleta de dados e ETL (detalhes)

### Breve descrição das fontes (visão operacional)
- ONS – Carga verificada (SE/CO): sinal da demanda real (agregamos para semanal; criamos lags e janelas).
- ONS – Geração por fonte (SE/CO): oferta efetiva por hidro/eólica/FV/térmica (evita baixar tudo por usina do país; se necessário, somar por fonte).
- ONS – Intercâmbios do SE/CO: importações/exportações entram na margem de suprimento.
- ONS – ENA e EAR (diário, SE/CO): hidrologia (entrada d’água e estoque), fortes condutores do risco.
- ONS – Constrained-off (cortes) eólica/FV: indicador de superávit/limitação de escoamento (não é déficit). Usamos a razão semanal `cortes/(cortes+geração)` como sinal para reduzir risco quando não há importação líquida.
- NASA POWER (diário agregado p/ GO): GHI, temperatura e precipitação (impactos em FV, carga e hidrologia). Coleta leve por API.

NOTA: DEFAULT_SPECS em fech_ons.py define todas as fontes a serem baixadas no ONS. para incluir ou remover alguma, edite esta parte do arquivo.

### Onde baixar (referência)
- ONS (manual): baixe CSV/XLSX das páginas de Carga Verificada, Balanço de Energia nos Subsistemas (serve como “Geração por Fonte”), Intercâmbios Entre Subsistemas, ENA Diário por Subsistema, EAR Diário por Subsistema e Constrained‑off (eólica/FV) para o submercado SE/CO.
  - Salve os brutos em `data/raw/` com nomes sugeridos:
    - `ons_carga.csv` (diário ou horário)
    - `ons_balanco_subsistema_horario.csv` (horário)
    - `ons_intercambios_entre_subsistemas_horario.csv` (horário)
    - `ons_ena_diario_subsistema.csv` (diário)
    - `ons_ear_diario_subsistema.csv` (diário)
    - `ons_constrained_off_eolica_mensal.csv` (mensal)
    - `ons_constrained_off_fv_mensal.csv` (mensal)
  - Depois rode o ETL para gerar os diários padronizados usados pelo pipeline (`python main.py data`).

  NOTA: O software já baixa automaticamente. Veja a próxima seção.

### Automatizando o download e o ETL
- Tudo de uma vez (ONS + ETL + NASA):
  - `python main.py all --incluir-meteorologia`
- Apenas preparar dados (ONS + ETL, sem NASA):
  - `python main.py data`

Parâmetros úteis:
- Submercado (padrão `SE/CO`):
  - `python main.py data --submercado "SE/CO"`
- Incluir meteorologia (NASA):
  - `python main.py data --incluir-meteorologia`
- Limitar período baixado (ou usar `download.since` no YAML):
  - `python main.py data --since 2022`
- Não sobrescrever arquivos já baixados (padrão):
  - omita `--overwrite`
- Forçar re-download e reprocessamento:
  - `python main.py data --overwrite`

Notas:
- O downloader (`src/fetch_ons.py`) baixa e concatena os recursos mensais/anuais dentro do período, gerando CSVs consolidados em `data/raw/` com os nomes esperados pelo ETL.
- O ETL (`src/etl_ons.py`) é tolerante a variações comuns de nomes de colunas; para layouts muito diferentes, ajuste os mapeamentos no código ou normalize os CSVs conforme as tabelas de “Dados utilizados”.

### Exemplo rápido (Python) – NASA POWER
```python
import pandas as pd, requests
from datetime import date

# grade simples de pontos sobre GO (lat,lon) – ajuste se quiser
pontos = [(-16.7,-49.3), (-15.9,-48.2), (-18.0,-49.7), (-13.6,-46.9)]
ini, fim = "2018-01-01", str(date.today())

def baixa_ponto(lat, lon):
    url = ("https://power.larc.nasa.gov/api/temporal/daily/point"
           f"?parameters=ALLSKY_SFC_SW_DWN,T2M,PRECTOTCORR"
           f"&community=RE&longitude={lon}&latitude={lat}"
           f"&start={ini.replace('-','')}&end={fim.replace('-','')}&format=JSON")
    r = requests.get(url, timeout=60)
    j = r.json()["properties"]["parameter"]
    return pd.DataFrame({
        "data": pd.to_datetime(pd.Series(j["ALLSKY_SFC_SW_DWN"]).index),
        "ghi": list(j["ALLSKY_SFC_SW_DWN"].values()),
        "temp2m_c": list(j["T2M"].values()),
        "precipitacao_mm": list(j["PRECTOTCORR"].values()),
    })

dfs = [baixa_ponto(*p) for p in pontos]
df = dfs[0][["data","ghi","temp2m_c","precipitacao_mm"]].copy()
for d in dfs[1:]:
    df[["ghi","temp2m_c","precipitacao_mm"]] += d[["ghi","temp2m_c","precipitacao_mm"]]
df[["ghi","temp2m_c","precipitacao_mm"]] /= len(dfs)
df.to_csv("data/raw/clima_go_diario.csv", index=False)
```

### Observações de consistência
- Manter nomes de colunas exatamente como definidos nas tabelas acima.
- Se vier horário/semi‑horário, agregue para diário (soma para energia; média para índices) ou deixe os scripts reamostrarem.
- O ETL é tolerante a variações comuns de nomes; para layouts muito diferentes, ajustar `src/etl_ons.py`.

## Parâmetros de região (config.yaml)
- `regions.submercado`: submercado alvo para o ETL (ex.: "SE/CO", "NE", "N", "S"). A CLI `--submercado` tem precedência.
- `regions.carga_area`: código da área para a API de Carga Verificada (ex.: "SECO", "NE", ou códigos específicos como "GO"/"PESE").
- `regions.meteo_points`: lista de pontos (lat,lon) para agregação da meteorologia (NASA POWER). Ajuste para outra UF/região conforme necessário.

Nota: se `regions.carga_area` for definido, o passo de dados rebaixa `ons_carga.csv` com a área informada antes de rodar o ETL.

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Como rodar

1) **Gerar features semanais**
```bash
python main.py features
# => salva data/features/features_weekly.parquet
```

2) **Treinar modelos (LogReg e XGBoost)**
```bash
python main.py train
# => salva models/logreg.joblib e models/xgb.joblib
# => gera reports/cv_scores.csv com F1-macro e Balanced Accuracy (TimeSeriesSplit)
```

3) **Avaliar e gerar relatório**
```bash
python main.py eval --model xgb
# => reports/report_xgb.txt (classification_report + matriz de confusão)
```

## Configuração (configs/config.yaml)

- `problem.horizon`: **weekly** (semanal).  
- `problem.label_rules`:
  - `coluna_margem`: **margem_vs_carga_w** (saldo suprimento - carga da semana).
  - `q_baixo`/`q_medio`: definem os **quantis** para rotular **alto/médio/baixo**.
  - **Ajustes por cortes**: cortes eólico/FV **reduzem** o risco quando indicam **superávit renovável** (razão de corte semanal acima do limiar **e** sem importação líquida).
  - **Ajustes por hidrologia**: **EAR** muito baixa **ou** **ENA** muito baixa por `k` semanas -> **alto**.
- `aggregation.features`: agregações de **diário->semanal**, **lags** e **janelas móveis**.
  - `features.min_nonnull_ratio`: descarta colunas diárias com menos de X% de preenchimento (padrão 50%) antes de gerar features.
- `modeling.models`: especifica **Logistic Regression** e **XGBoost**.

> Você pode ajustar quantis/limiares no YAML sem alterar código.

## Como rotulamos (resumo)

1. Calculamos a **margem de suprimento semanal**:  
   `geração_total (hidro+eólica+FV+term) + importações − exportações`.  
2. Aplicamos **quantis** sobre o **mínimo**/proxy semanal para definir:  
   - **alto** (≤ Q10), **médio** (Q10–Q40], **baixo** (> Q40).  
3. **Ajustes**:  
   - **Cortes** (constrained-off) eólico/FV: **reduzem** o risco quando representam superávit renovável (razão de corte > limiar e sem importação líquida).  
   - **Hidrologia**: **EAR** muito baixa **ou** **ENA** muito baixa por `k` semanas -> **alto**.

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
- Padronização de timezone e reamostragem **D->W**.  
- Seeds fixas (`seed: 42`).  
- Colunas **em português** (consistentes com os CSVs e o código).

## Problemas comuns

- **Coluna ausente**: verifique o nome exato no CSV (tabela acima).  
- **Datas inválidas**: formato deve ser `YYYY-MM-DD`.  
- **Faltas longas**: o pipeline preenche curtas (até 7 dias). Falhas maiores -> revisar a origem.

## Créditos dos dados

- **ONS** – Portal de Dados Abertos (carga, geração por fonte, intercâmbios, ENA/EAR, cortes).  
- **NASA POWER** – API diária (GHI, temperatura, precipitação).
