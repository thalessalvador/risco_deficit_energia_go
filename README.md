# Risco Semanal de Deficit de Energia em Goias (SE/CO)

## Modelagem e Transformações com dbt

O projeto utiliza dbt para modelagem e transformação dos dados no Snowflake.

### Como rodar os modelos dbt

1. Instale as dependências (já incluso em requirements.txt):
  ```bash
  pip install -r requirements.txt
  ```

2. Configure o arquivo `profiles.yml` em `C:/Users/<seu_usuario>/.dbt/profiles.yml` com as credenciais do Snowflake (veja exemplo em `dbt_energia/` ou peça ao time).

3. Navegue até a pasta do projeto dbt:
  ```bash
  cd dbt_energia
  ```

4. Execute os modelos:
  ```bash
  dbt run
  ```
  Isso irá criar/atualizar as tabelas no Snowflake, incluindo `dim_calendario` e `fato_metricas_energia`.

5. Para rodar apenas um modelo específico:
  ```bash
  dbt run --select fato_metricas_energia
  ```

6. Para validar e documentar:
  ```bash
  dbt test
  dbt docs generate
  dbt docs serve
  ```
  O comando `dbt docs serve` abrirá uma interface web com a documentação e lineage dos modelos.

### Observações
- Os modelos dbt estão em `dbt_energia/models/`.
- Ajuste os aliases ou inclua mais colunas conforme necessário nos arquivos `.sql`.
- O filtro `where DATA is not null` nos modelos garante que apenas registros válidos sejam considerados.
- Para agendar execuções automáticas, utilize dbt Cloud, Airflow ou outro orquestrador.


**Objetivo:** classificar, em **horizonte semanal**, o **risco de deficit de energia** em Goias (baixo | medio | alto) usando dados **operacionais do ONS** e **clima** (leve).  
**Disciplinas atendidas:** Machine Learning, Cloud (AWS) e Modelagem de Dados.
**Alunos:** Thales Salvador, Miguel Toledo, Carlos Henrique.

## Estrutura do repositorio

```
project/
 src/
   data_loader.py
   feature_engineer.py
   train.py
   evaluate.py
   etl_ons.py                # ETL: brutos do ONS -> diarios padronizados
   meteo.py                  # Meteorologia (NASA POWER)  fetch/process
  fetch_ons.py              # Downloader via CKAN (dados.ons.org.br)
  s3_utils.py               # Utilitários para leitura/escrita no S3 (raw/bronze)
  api/handler.py              # stub p/ AWS Lambda
 configs/config.yaml
 requirements.txt
 README.md
 data/
    raw/                        # coloque aqui os CSVs brutos (ver tabela abaixo)
```

> Timezone padrao: **America/Sao_Paulo**.


## Uso com AWS S3
> Os arquivos brutos foram baixados e enviados para o S3 utilizando um Jupyter Notebook rodando no Amazon SageMaker.

Para referência e reuso, recomenda-se salvar o notebook de ingestão em:

- Diretório: `data/`
- Arquivo sugerido: `ingestao_s3_sagemaker.ipynb`

Assim, basta copiar o notebook para `data/ingestao_s3_sagemaker.ipynb` para documentar e reproduzir o processo de ingestão no futuro.
> **Nota:** Os arquivos presentes em `raw/` no S3 passam pelo mesmo fluxo de ingestão, ETL e processamento do pipeline que os dados baixados diretamente das fontes externas (ONS/NASA). Ou seja, ao optar por buscar do S3, o restante do pipeline permanece idêntico, apenas mudando a origem dos arquivos brutos.

O projeto permite baixar os dados diretamente de um bucket S3, caso você já possua os arquivos padronizados no formato esperado. Para isso:

1. Configure a seção `s3` no arquivo `configs/config.yaml` com as credenciais, bucket, região e prefixo corretos. Exemplo:

```yaml
s3:
  enabled: true -- altere aqui se usará ou não o s3
  bucket: "nome-do-seu-bucket"
  prefix: "caminho/opcional/"
  aws_access_key_id: "SUA_AWS_ACCESS_KEY_ID"
  aws_secret_access_key: "SUA_AWS_SECRET_ACCESS_KEY"
  region: "us-east-1"
```

2. Execute o pipeline de dados com a flag `--use-s3`:

```bash
python main.py data --use-s3
```


**Importante sobre camadas S3:**

- Para leitura de dados brutos, o pipeline sempre busca arquivos do S3 na pasta/prefixo `raw/`.
- Para gravação de arquivos processados (features, outputs), utilize sempre o prefixo `bronze/` ao salvar no S3.
- O upload automático das features para o S3 só ocorre se o S3 estiver habilitado (`enabled: true` no bloco `s3` do `config.yaml`). Caso contrário, o arquivo será salvo apenas localmente.

Exemplo de leitura (download) de dados brutos:
```python
from src.s3_utils import download_file_from_s3
ok = download_file_from_s3(
  bucket, "ons_carga.csv", "data/raw/ons_carga.csv", prefix="raw/", ...)
```

Exemplo de gravação (upload) de features processadas:
```python
from src.s3_utils import upload_file_to_s3
upload_file_to_s3(
  "data/features/features_weekly.parquet", bucket, "features_weekly.parquet", prefix="bronze/", ...)
```

Os seguintes arquivos serão buscados do S3 (em formato CSV):

- ons_balanco_subsistema_horario.csv
- ons_intercambios_entre_subsistemas_horario.csv
- ons_carga.csv
- ons_ena_diario_subsistema.csv
- ons_ear_diario_subsistema.csv
- ons_constrained_off_eolica_mensal.csv
- ons_constrained_off_fv_mensal.csv

Eles serão baixados para o diretório local especificado em `--raw-dir` (padrão: `data/raw`).

Se a flag não for usada, o comportamento padrão permanece: os dados são baixados automaticamente das fontes do ONS via API/CKAN.

---

## Inicio Rapido

Antes de iniciar:

1. Faça upload do notebook `data/ingestao_s3_sagemaker.ipynb` para o ambiente do SageMaker e execute-o para realizar a ingestão dos dados no S3.
2. No bucket S3, crie as pastas (prefixos) `raw/` (para dados brutos) e `bronze/` (para arquivos processados), se ainda não existirem.

Depois:

1) Ambiente e dependencias
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

2) Baixar dados e gerar insumos diarios (ONS + NASA)
```bash
python main.py data --incluir-meteorologia  # download (CKAN) + ETL (+ clima NASA)
# Limitar periodo: usa default em configs/config.yaml (download.since), ou:
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

Comando unico:
```bash
python main.py all --incluir-meteorologia
```

Detalhes sobre fontes, nomes de arquivos e parametrizacao em `README_data.md`.

## Dados utilizados

O Software coloca estes **CSVs diarios** em `data/raw/` ou você mesmo pode colocar (colunas efetivamente usadas no pipeline):

| Arquivo CSV | Colunas usadas | Fonte |
|---|---|---|
| `ons_carga_diaria.csv` | `data`, `carga_mwh` | ONS  carga verificada (SE/CO) |
| `ons_geracao_fontes_diaria.csv` | `data`, `ger_hidreletrica_mwh`, `ger_eolica_mwh`, `ger_fv_mwh`, `ger_termica_mwh` | ONS  geracao por **fonte** (SE/CO) |
| `ons_intercambio_diario.csv` | `data`, `import_mwh`, `export_mwh` | ONS  intercambios (SE/CO) |
| `ons_ena_diaria.csv` | `data`, `ena_mwmed` | ONS  ENA (SE/CO) |
| `ons_ear_diaria.csv` | `data`, `ear_pct` | ONS  EAR (SE/CO) |
| `ons_cortes_eolica_diario.csv` | `data`, `corte_eolica_mwh` | ONS  cortes eolicos (constrained-off) |
| `ons_cortes_fv_diario.csv` | `data`, `corte_fv_mwh` | ONS  cortes FV (constrained-off) |
| `clima_go_diario.csv` | `data`, `ghi`, `temp2m_c`, `precipitacao_mm` | NASA POWER (GO agregado) |

Observacao: para clima criamos derivadas diarias antes do D->W: `precip_14d_mm`, `precip_30d_mm`, `precip_90d_mm` e `precip_180d_mm`.

> Se algum dataset vier horario/semi-horario, **agregue para diario** (media ou soma, conforme a metrica) antes de salvar, ou deixe que os scripts reamostrem.

## Features geradas

As features semanais seguem um padrao configuravel em `configs/config.yaml` (agregacoes D->W: `mean`, `sum`, `max`, `min`, `std`, `p95`, `p05`; lags: `[1,2,4]`; janelas moveis: `[2,4]`). Abaixo, o mapeamento das colunas de entrada para as familias de features geradas.

| Origem | Colunas (D) | Features semanais (padrao) | Observacoes/derivadas |
|---|---|---|---|
| Carga | `carga_mwh` | `carga_mwh_<agg>_w` para cada agregacao | Usada para `margem_vs_carga_w`, `reserve_margin_ratio_w`, `ens_week_*` |
| Geracao por fonte | `ger_hidreletrica_mwh`, `ger_eolica_mwh`, `ger_fv_mwh`, `ger_termica_mwh` | `ger_<fonte>_mwh_<agg>_w` | Soma semanal: `geracao_total_mwh_sum_w` |
| Intercambio | `import_mwh`, `export_mwh` | `import_mwh_<agg>_w`, `export_mwh_<agg>_w` | Somas: `import_total_mwh_sum_w`, `export_total_mwh_sum_w`; saldo: `saldo_importador_mwh_sum_w` |
| ENA | `ena_mwmed` | `ena_mwmed_<agg>_w` |  |
| EAR | `ear_pct` | `ear_pct_<agg>_w` |  |
| Cortes eolicos/FV | `corte_eolica_mwh`, `corte_fv_mwh` | `corte_eolica_mwh_<agg>_w`, `corte_fv_mwh_<agg>_w` | Derivadas: `corte_renovavel_mwh_sum_w`, `ratio_corte_renovavel_w` |
| Clima (NASA) | `ghi`, `temp2m_c`, `precipitacao_mm`, `precip_14d_mm`, `precip_30d_mm` | `<col>_<agg>_w` para cada coluna disponivel | `precip_14d_mm`/`precip_30d_mm` sao construidas no diario e entao agregadas |

Derivadas operacionais (semanais):
- **`geracao_total_mwh_sum_w`**: Soma da geracao de todas as fontes (hidro, eolica, FV, termica) na semana.
- **`import_total_mwh_sum_w`**: Total de energia importada de outros subsistemas na semana.
- **`export_total_mwh_sum_w`**: Total de energia exportada para outros subsistemas na semana.
- **`saldo_importador_mwh_sum_w`**: Saldo liquido de energia (`importacoes - exportacoes`). Um valor positivo indica que o subsistema foi um importador liquido na semana.
- **`margem_suprimento_w`**: Representa a oferta total de energia disponivel na semana. E calculada como `geracao_total + importacoes - exportacoes`.
- **`margem_suprimento_min_w`**: Proxy da margem de suprimento, atualmente uma copia da `margem_suprimento_w`.
- **`margem_vs_carga_w`**: **Feature central para a rotulagem.** E a diferenca entre a oferta e a demanda (`margem_suprimento_w - carga_mwh_sum_w`). Um valor negativo indica um deficit de energia na semana.
- **`reserve_margin_ratio_w`**: Razao da margem de reserva, calculada como `margem_vs_carga_w / carga_mwh_sum_w`. Indica a folga (ou deficit) como uma porcentagem da demanda.
- **`ens_week_mwh`**: "Energy Not Supplied" (Energia Nao Suprida). E uma estimativa do deficit, calculada como o valor negativo da `margem_vs_carga_w` (zerado se a margem for positiva).
- **`ens_week_ratio`**: Razao do ENS em relacao a carga total da semana (`ens_week_mwh / carga_mwh_sum_w`).
- **`lolp_52w`**: "Loss of Load Probability". E uma probabilidade empirica de perda de carga, calculada como a frequencia de semanas com `margem_vs_carga_w` negativa em uma janela movel de 52 semanas.
 
Expansoes temporais (para cada feature semanal base):
- Lags: sufixos `_<feature>_lag{L}w` para `L` em `[1,2,4]`.
- Janelas moveis: `_<feature>_r{R}w_mean` e `_<feature>_r{R}w_std` para `R` em `[2,4]`.

## Coleta de dados e ETL (detalhes)

### Breve descricao das fontes (visao operacional)
- ONS  Carga verificada (SE/CO): sinal da demanda real (agregamos para semanal; criamos lags e janelas).
- ONS  Geracao por fonte (SE/CO): oferta efetiva por hidro/eolica/FV/termica (evita baixar tudo por usina do pais; se necessario, somar por fonte).
- ONS  Intercambios do SE/CO: importacoes/exportacoes entram na margem de suprimento.
- ONS  ENA e EAR (diario, SE/CO): hidrologia (entrada dagua e estoque), fortes condutores do risco.
- ONS  Constrained-off (cortes) eolica/FV: indicador de superavit/limitacao de escoamento (nao e deficit). Usamos a razao semanal `cortes/(cortes+geracao)` como sinal para reduzir risco quando nao ha importacao liquida.
- NASA POWER (diario agregado p/ GO): GHI, temperatura e precipitacao (impactos em FV, carga e hidrologia). Coleta leve por API.

NOTA: DEFAULT_SPECS em fech_ons.py define todas as fontes a serem baixadas no ONS. para incluir ou remover alguma, edite esta parte do arquivo.

### Onde baixar (referencia)
- ONS (manual): baixe CSV/XLSX das paginas de Carga Verificada, Balanco de Energia nos Subsistemas (serve como Geracao por Fonte), Intercambios Entre Subsistemas, ENA Diario por Subsistema, EAR Diario por Subsistema e Constrainedoff (eolica/FV) para o submercado SE/CO.
  - Salve os brutos em `data/raw/` com nomes sugeridos:
    - `ons_carga.csv` (diario ou horario)
    - `ons_balanco_subsistema_horario.csv` (horario)
    - `ons_intercambios_entre_subsistemas_horario.csv` (horario)
    - `ons_ena_diario_subsistema.csv` (diario)
    - `ons_ear_diario_subsistema.csv` (diario)
    - `ons_constrained_off_eolica_mensal.csv` (mensal)
    - `ons_constrained_off_fv_mensal.csv` (mensal)
  - Depois rode o ETL para gerar os diarios padronizados usados pelo pipeline (`python main.py data`).

  NOTA: O software ja baixa automaticamente. Veja a proxima secao.

### Automatizando o download e o ETL
- Tudo de uma vez (ONS + ETL + NASA):
  - `python main.py all --incluir-meteorologia`
- Apenas preparar dados (ONS + ETL, sem NASA):
  - `python main.py data`

Parametros uteis:
- Submercado (padrao `SE/CO`):
  - `python main.py data --submercado "SE/CO"`
- Incluir meteorologia (NASA):
  - `python main.py data --incluir-meteorologia`
- Limitar periodo baixado (ou usar `download.since` no YAML):
  - `python main.py data --since 2022`
- Nao sobrescrever arquivos ja baixados (padrao):
  - omita `--overwrite`
- Forcar re-download e reprocessamento:
  - `python main.py data --overwrite`

Notas:
- O downloader (`src/fetch_ons.py`) baixa e concatena os recursos mensais/anuais dentro do periodo, gerando CSVs consolidados em `data/raw/` com os nomes esperados pelo ETL.
- O ETL (`src/etl_ons.py`) e tolerante a variacoes comuns de nomes de colunas; para layouts muito diferentes, ajuste os mapeamentos no codigo ou normalize os CSVs conforme as tabelas de Dados utilizados.

### Exemplo rapido (Python)  NASA POWER
```python
import pandas as pd, requests
from datetime import date

# grade simples de pontos sobre GO (lat,lon)  ajuste se quiser
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

### Observacoes de consistencia
- Manter nomes de colunas exatamente como definidos nas tabelas acima.
- Se vier horario/semihorario, agregue para diario (soma para energia; media para indices) ou deixe os scripts reamostrarem.
- O ETL e tolerante a variacoes comuns de nomes; para layouts muito diferentes, ajustar `src/etl_ons.py`.

## Parametros de regiao (config.yaml)
- `regions.submercado`: submercado alvo para o ETL (ex.: "SE/CO", "NE", "N", "S"). A CLI `--submercado` tem precedencia.
- `regions.carga_area`: codigo da area para a API de Carga Verificada (ex.: "SECO", "NE", ou codigos especificos como "GO"/"PESE").
- `regions.meteo_points`: lista de pontos (lat,lon) para agregacao da meteorologia (NASA POWER). Ajuste para outra UF/regiao conforme necessario.

Nota: se `regions.carga_area` for definido, o passo de dados rebaixa `ons_carga.csv` com a area informada antes de rodar o ETL.

## Instalacao

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Como rodar (após concluída a etapa "data")

1) **Gerar features semanais**
```bash
python main.py features
# => salva data/features/features_weekly.parquet
```

2) **Treinar modelos (LogReg, Random Forest, XGBoost)**
```bash
python main.py train
# => salva models/logreg.joblib, models/rf.joblib e models/xgb.joblib
# => gera reports/cv_scores.csv com F1-macro e Balanced Accuracy (TimeSeriesSplit)
```

3) **Avaliar e gerar relatorio**
```bash
python main.py eval --model xgb   # ou --model logreg / --model rf
# => reports/report_<model>.txt (classification_report + matriz de confusao)
```

## Configuracao (configs/config.yaml)

- `problem.horizon`: **weekly** (semanal).  
- `problem.label_rules`:
  - `coluna_margem`: **margem_vs_carga_w** (saldo suprimento - carga da semana).
  - `q_baixo`/`q_medio`: definem os **quantis** para rotular **alto/medio/baixo**.
  - **Ajustes por cortes**: cortes eolico/FV **reduzem** o risco quando indicam **superavit renovavel** (razao de corte semanal acima do limiar **e** sem importacao liquida).
  - **Ajustes por hidrologia**: **EAR** muito baixa **ou** **ENA** muito baixa por `k` semanas -> **alto**.
- `aggregation.features`: agregacoes de **diario->semanal**, **lags** e **janelas moveis**.
  - `features.min_nonnull_ratio`: descarta colunas diarias com menos de X% de preenchimento (padrao 50%) antes de gerar features.
- `modeling.models`: especifica **Logistic Regression**, **Random Forest** e **XGBoost**.
- `modeling.feature_selection`: permite aplicar filtro nas features usando o JSON de importancias do XGB final (`use`, `source`, `keep_top_k`, `keep_list`, `min_importance`).

> Voce pode ajustar quantis/limiares no YAML sem alterar codigo.

## Como rotulamos (resumo)

1. Usamos a coluna `margem_vs_carga_w` (diferença entre oferta e carga semanal) como proxy principal de risco; valores negativos indicam déficit.  
2. Aplicamos os quantis configurados (`q_baixo: 0.05`, `q_medio: 0.20`):  
   - **alto** (<= P5), **medio** (P5-P20], **baixo** (> P20).  
3. **Ajustes** complementares garantem aderência normativa e sinais hidrológicos:  
   - **Cortes** (constrained-off) eólico/FV: podem **rebaixar** o risco quando há superávit renovável (`ratio_corte_renovavel_w >= 0.05` e saldo importador <= 0).  
   - **Hidrologia crítica**: **EAR** <= P20 ou **ENA** <= P20 por `k` semanas consecutivas endurecem para **alto**.  
   - **Regras duras (EPE/MME/ONS)**: violações de `lolp_52w >= 0.05`, `ens_week_ratio >= 0.05` ou `reserve_margin_ratio_w < 0.05` forçam o rótulo **alto**, independentemente dos quantis.

## Auditoria de Rotulagem

- `main.py train` exporta `reports/label_audit_train.csv` apos calcular os rotulos.
- O CSV traz rotulos base (quantis), pos-cortes, pos-regras duras e o rotulo final usado no treino.
- Registra os thresholds adotados (margem, EAR, ENA, razoes de corte) e flags indicando qual regra alterou cada semana.
- Serve para auditoria/QA dos rotulos sem reprocessar os Parquets.

## Saidas

- `data/features/features_weekly.parquet`  feature store semanal.  
- `models/*.joblib`  artefatos dos modelos.  
- `reports/cv_scores.csv`  metricas de CV.  
- `reports/report_<modelo>.txt`  relatorio final + matriz de confusao.  
- `reports/label_audit_train.csv`  auditoria dos rotulos gerados (quantis, regras, thresholds).  
- `reports/feature_importances_<modelo>.json`  ranking de importancia de features do modelo final.

## Selecao de features com XGB

1. Execute `main.py train` com `modeling.feature_selection.use: false` para gerar o ranking completo em `reports/feature_importances_xgb.json`.
2. Ajuste o bloco `modeling.feature_selection` no YAML apontando para esse JSON (`source`) e defina `keep_top_k`, `keep_list` ou `min_importance` conforme o experimento.
3. Reative o treino (`use: true`) para treinar apenas com as colunas selecionadas; novas execucoes sobrescrevem o JSON com as importancias do modelo filtrado.

Use `keep_list` para colunas obrigatorias, `keep_top_k` para limitar pelo ranking e `min_importance` como piso numerico. Quando o bloco estiver desativado o pipeline volta a usar todas as features.

> Observacao: sempre que quiser aumentar `keep_top_k`, gere um ranking completo novamente (rodando `main.py train` com `feature_selection.use: false` ou apontando `source` para um JSON cheio). Reduzir o top_k nao exige esse passo.

## API (stub para AWS Lambda)

Arquivo: `src/api/handler.py`.  
Utilize `xgb.joblib`, `rf.joblib`  ou `logreg.joblib` (treinados via `main.py train`) para gerar uma API Lambda.

### Passo a passo sugerido
Um passo a passo completo em formato de README.md foi criado na pasta src/api deste projeto.

>Importante:
> Para producao, proteja o endpoint (API Gateway) e versione o modelo.
> Apenas as features usadas no treino precisam estar presentes.


## Validacao & Boas praticas

- **TimeSeriesSplit** (sem vazamento).  
- Padronizacao de timezone e reamostragem **D->W**.  
- Seeds fixas (`seed: 42`).  
- Colunas **em portugues** (consistentes com os CSVs e o codigo).

## Problemas comuns

- **Coluna ausente**: verifique o nome exato no CSV (tabela acima).  
- **Datas invalidas**: formato deve ser `YYYY-MM-DD`.  
- **Faltas longas**: o pipeline preenche curtas (ate 7 dias). Falhas maiores -> revisar a origem.

## Creditos dos dados

- **ONS**  Portal de Dados Abertos (carga, geracao por fonte, intercambios, ENA/EAR, cortes).  
- **NASA POWER**  API diaria (GHI, temperatura, precipitacao).
