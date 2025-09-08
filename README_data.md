# README_data - Coleta e padronização 

## Onde baixar (resumo)

- **ONS** (manual): baixe CSV/XLSX das páginas de **Carga Verificada**, **Geração por Fonte**, **Intercâmbios**, **ENA (diária)**, **EAR (diária)** e **Constrained-off** (eólica/FV) do **Submercado Sudeste/Centro-Oeste**.  
  - Salve com estes nomes em `data/raw/`:  
    - `ons_carga_diaria.csv`  
    - `ons_geracao_fontes_diaria.csv`  
    - `ons_intercambio_diario.csv`  
    - `ons_ena_diaria.csv`  
    - `ons_ear_diaria.csv`  
    - `ons_cortes_eolica_diario.csv`  
    - `ons_cortes_fv_diario.csv`

- **NASA POWER** (API): gere **diário** para Goiás e salve como `clima_go_diario.csv` com as colunas `data, ghi, temp2m_c, precipitacao_mm`.

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

### Observações de consistência
- Manter **nomes de colunas exatamente** como definidos (o código depende deles).  
- Se algum dataset do ONS vier horário, **agregue para diário** (soma p/ energia; média p/ índices).  
- O `feature_engineer.py` agrega **D→W**, calcula **margem de suprimento** e cria **lags/janelas**.
