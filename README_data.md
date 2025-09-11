# README_data - Coleta e padronização 

## Breve descrição das fontes
ONS – Carga verificada (SE/CO)
Sinal da demanda real. (Agregamos para semanal; criamos lags e médias móveis.)

ONS – Geração por fonte no SE/CO (se existir; senão, usar por usina e somar por fonte)
Sinal da oferta efetiva (hidro/eólica/FV/térmica). Evita baixar tudo por usina do país.

ONS – Intercâmbios do SE/CO
Importações/Exportações → entram na margem de suprimento.

ONS – ENA & EAR (diário, SE/CO)
Hidrologia (entrada d’água e estoque). Excelentes “condutores” do risco.

ONS – Constrained-off (cortes) eólica/FV
Indicador de **superávit/limitação de escoamento** (não é déficit). Calculamos a **razão de corte renovável** semanal:  
`ratio_corte_renovavel_w = corte_renovavel / (corte_renovavel + geracao_renovavel)`.  
Se a razão > limiar (padrão **5%**) e **não houver importação líquida** na semana (saldo_importador ≤ 0), usamos como **sinal de redução de risco** (alto→médio; médio→baixo).

Clima (1 fonte leve) — NASA POWER (diário agregado p/ GO)
GHI (radiação), temperatura e **precipitação** (impacto em FV, carga e hidrologia). Coleta simples por API, baixo volume.


## Onde baixar (resumo)

- **ONS** (manual): baixe CSV/XLSX das páginas de **Carga Verificada**, **Balanço de Energia nos Subsistemas** (serve como "Geração por Fonte"), **Intercâmbios Entre Subsistemas**, **ENA Diário por Subsistema**, **EAR Diário por Subsistema** e **Constrained-off** (eólica/FV) do **Submercado Sudeste/Centro-Oeste**. 

  - Salve os arquivos brutos (como vêm do portal) em `data/raw/` com estes nomes sugeridos:  
    - `ons_carga.csv` (pode ser diário ou horário)  
    - `ons_balanco_subsistema_horario.csv` (horário)  
    - `ons_intercambios_entre_subsistemas_horario.csv` (horário)  
    - `ons_ena_diario_subsistema.csv` (diário)  
    - `ons_ear_diario_subsistema.csv` (diário)  
    - `ons_constrained_off_eolica_mensal.csv` (mensal)  
    - `ons_constrained_off_fv_mensal.csv` (mensal)

  - Em seguida, rode o ETL para gerar os arquivos diários padronizados que o pipeline usa:  
    - `python main.py data`  
    Isso produzirá em `data/raw/`:  
    - `ons_carga_diaria.csv`  
    - `ons_geracao_fontes_diaria.csv`  
    - `ons_intercambio_diario.csv`  
    - `ons_ena_diaria.csv`  
    - `ons_ear_diaria.csv`  
    - `ons_cortes_eolica_diario.csv`  
    - `ons_cortes_fv_diario.csv`

- **NASA POWER** (API): pode ser baixado via CLI principal (`python main.py data --incluir-meteorologia`) ou manualmente. Resultado: `clima_go_diario.csv` com colunas `data, ghi, temp2m_c, precipitacao_mm`.

### Exemplo rápido (Python) para NASA POWER

```python
import pandas as pd
import requests
from datetime import date

# grade simples de pontos sobre GO (lat,lon) - ajuste se quiser
pontos = [(-16.7,-49.3), (-15.9,-48.2), (-18.0,-49.7), (-13.6,-46.9)]
ini, fim = "2018-01-01", str(date.today())

def baixa_ponto(lat, lon):
    url = ("https://power.larc.nasa.gov/api/temporal/daily/point"
           f"?parameters=ALLSKY_SFC_SW_DWN,T2M,PRECTOTCORR"
           f"&community=RE&longitude={lon}&latitude={lat}"
           f"&start={ini.replace('-','')}&end={fim.replace('-','')}&format=JSON")
    r = requests.get(url, timeout=60)
    j = r.json()["properties"]["parameter"]
    df = pd.DataFrame({
        "data": pd.to_datetime(pd.Series(j["ALLSKY_SFC_SW_DWN"]).index),
        "ghi": list(j["ALLSKY_SFC_SW_DWN"].values()),
        "temp2m_c": list(j["T2M"].values()),
        "precipitacao_mm": list(j["PRECTOTCORR"].values())
    })
    return df

dfs = [baixa_ponto(*p) for p in pontos]
# média espacial sobre os pontos
df = dfs[0][["data","ghi","temp2m_c","precipitacao_mm"]].copy()
for d in dfs[1:]:
    df[["ghi","temp2m_c","precipitacao_mm"]] += d[["ghi","temp2m_c","precipitacao_mm"]]
df[["ghi","temp2m_c","precipitacao_mm"]] /= len(dfs)

df.to_csv("data/raw/clima_go_diario.csv", index=False)
print("Salvo em data/raw/clima_go_diario.csv", df.shape)
```

### Automatizando o download e o ETL
- Tudo de uma vez (ONS + ETL + NASA):  
  - `python main.py all --incluir-meteorologia`

- Apenas preparar dados (ONS + ETL, sem NASA):  
  - `python main.py data`

- Parâmetros úteis:
  - Submercado (padrão `SE/CO`): `python main.py data --submercado "SE/CO"`
  - Incluir meteorologia (NASA): `python main.py data --incluir-meteorologia`
  - Limitar período baixado (default em `configs/config.yaml: download.since`):
    - `python main.py data --since 2022` 
  - Não sobrescrever arquivos já baixados (padrão): omitindo `--overwrite` o downloader pula conjuntos cujo CSV final já exista.
  - Forçar re-download: `python main.py data --overwrite`

Notas:
- O ETL é tolerante a variações comuns de nomes de colunas. Se algum arquivo do ONS vier com layout muito diferente, ajuste os nomes conforme os sugeridos acima ou me avise que eu amplio os mapeamentos no `src/etl_ons.py`.
- O downloader (`src/fetch_ons.py`) baixa e concatena todos os recursos mensais/anuais dentro do período (ex.: `since=2022`), gerando CSVs consolidados em `data/raw/` com os nomes esperados pelo ETL.

### Observações de consistência
- Manter **nomes de colunas exatamente** como definidos (o código depende deles).  
- Se algum dataset do ONS vier horário, **agregue para diário** (soma p/ energia; média p/ índices).  
- O `feature_engineer.py` agrega **D→W**, calcula **margem de suprimento**, o **saldo importador** e a **razão de corte renovável**, e cria **lags/janelas**.
