# Datasets utilizados e campos relevantes

## Definição de alto/médio/baixo segundo literatura
A literatura oficial (EPE/MME, em alinhamento com o ONS/CNPE) usa como critérios de segurança/adequação:

Risco anual de perda de carga (LOLP) ≤ 5%
Esse é o critério clássico do CNPE (Res. 1/2004), reafirmado no relatório do MME/EPE: “a LOLP deve ser menor ou igual a 5% para o SIN em base anual” (horizonte do PDE). 
Serviços e Informações do Brasil

Profundidade de energia não suprida medida por CVaR (ENS)
Propõe-se CVaR₁% (ENS) ≤ 5% da demanda anual (para o SIN; e checar que nenhum subsistema exceda 5% se o corte não for “gerenciável”). A Tabela 4 do relatório ilustra os valores nas simulações e formaliza a diretriz.

Adequação de potência (PNS) via CVaR e Reserva Operativa
Para potência não suprida (PNS), usa-se CVaR₅% (PNS) ≤ Reserva Operativa associada à demanda, que atualmente é 5% da demanda máxima. A leitura prática é: pelo menos 95% do tempo a reserva operativa deve cobrir a demanda; em até 5% dos cenários pode haver redução dessa reserva e, em menos de 5%, cortes de curta duração.

## Balanco_energia_subsistema

### id_subsistema
O que é: Código identificador do subsistema do SIN (Sistema Interligado Nacional).
Contexto: O Brasil é dividido em 4 subsistemas (Norte, Nordeste, Sul e Sudeste/Centro-Oeste). Goiás está no SE/CO.
Uso no projeto: filtrar apenas dados do SE/CO para refletir a realidade goiana.

### nom_subsistema
Tipo: Texto (até 20 posições)
O que é: Nome por extenso do subsistema (ex: "Sudeste/Centro-Oeste").
Uso: Identificação mais legível, mas em análises usa-se mais o id_subsistema.

### din_instante
Tipo: Data/hora (YYYY-MM-DD HH:MM:SS)
O que é: Momento exato da medição.
Obs: A granularidade original é horária.
Uso no projeto: pode ser agregado para diário ou semanal.

### val_gerhidraulica
Tipo: Float
O que é: Geração hidrelétrica medida (MW médios).
Permite nulo/zero, não permite negativo.
Uso: Representa a participação das usinas hidrelétricas no balanço do subsistema. Essencial porque Goiás depende fortemente de hidros.

### val_gertermica
Tipo: Float
O que é: Geração termelétrica (carvão, gás, óleo, biomassa).
Uso: Complementa o sistema em momentos de seca ou pico de demanda.
Relevância: Baixa no estado de Goiás, mas relevante para entender segurança do sistema.

### val_gereolica
Tipo: Float
O que é: Geração eólica verificada (MW médios).
Uso: Importante para estudar complementaridade com solar. Goiás ainda tem pouca eólica, mas Nordeste pode ajudar via intercâmbio.

### val_gersolar
Tipo: Float
O que é: Geração solar fotovoltaica (MW médios).
Uso: É o principal interesse do governo de Goiás (forte potencial solar). Fundamental para os modelos preditivos.

### val_carga
Tipo: Float
O que é: Carga total do subsistema, ou seja, a demanda elétrica (MW médios).
Obs: Não permite valores nulos.
Uso no projeto: comparar oferta vs demanda -> indicador direto de risco de déficit.

### val_intercambio
Tipo: Float
O que é: Intercâmbio líquido de energia do subsistema (MW médios).
Positivo -> importação de energia.
Negativo -> exportação de energia.
Uso: Goiás pode depender de importações (do Nordeste, por exemplo).


## Carga verificada (periodicidade semi-horária)

### cod_areacarga (TEXTO ≤4):
 código da área de carga (ex.: GO, SECO, PESE). Use GO para focar Goiás; SECO quando quiser consolidar no submercado.

### dat_referencia (DATETIME YYYY-MM-DD): 
data “do dia” associado ao registro. Útil para agregações diárias/semanais.

### din_referenciautc (DATETIME YYYY-MM-DDTHH:MM:SSZ): 
timestamp UTC+0 do final do intervalo de 30 min. Converta para America/Sao_Paulo (UTC-3) antes de juntar com outras fontes.

### val_cargaglobal: 
carga global (total). Pode aparecer zerada e, excepcionalmente, negativa (sinal registra ajustes/consistências em alguns cenários).

### val_cargaglobalsmmg: 
carga global líquida de MMGD (o que a rede precisa atender). Para análises operacionais/risco de déficit, esta é a melhor proxy da demanda do sistema. Também aceita zero e negativo.

### val_cargammgd: 
parcela atendida por micro e mini geração distribuída (MMGD). Pode ser zero; não pode ser negativa. Útil para estimar “carga oculta” e efeito fotovoltaico distribuído.

### val_cargaglobalcons: 
carga global consistida usada nos modelos de previsão (após ajustes). Zero permitido; negativo não. Serve para estudos em que você queira replicar a visão “consistida” do ONS.

### val_consistencia: 
valor da consistência (ajuste aplicado). Zero permitido; pode ser negativo (ajuste para baixo). Útil para auditoria de qualidade e análise de viés entre medido vs. consistido.

### val_cargasupervisionada: 
parcela supervisionada pelo ONS (gerações tipo I, IIA, IIB, IIC + intercâmbios). Zero permitido; pode ser negativa (efeito líquido com intercâmbio/ajustes). Ajuda a separar o que o ONS vê diretamente.

### val_carganaosupervisionada: 
parcela não supervisionada (medição CCEE, geração tipo III). Zero permitido; negativo não. Diferencia a origem dos dados de carga.

Observações de integridade (do dicionário): 
todos os campos acima não permitem nulos; vários aceitam zero; negativos podem ocorrer em val_cargaglobal, val_cargaglobalsmmg, val_consistencia, val_cargasupervisionada (trate como ajustes/sinais contábeis no pré-processamento).

## EAR por Subsistema
EAR (Energia Armazenada) é a energia associada ao volume de água nos reservatórios, convertível em geração tanto na própria usina quanto nas usinas a jusante (cascata). O dataset é diário e traz também a EAR máxima (capacidade plena) e a EAR verificada em % — um indicador direto do “nível dos reservatórios”

### id_subsistema (TEXTO, 2 posições)
Código do subsistema: tipicamente N, NE, S, SE (SE inclui Centro-Oeste). Para Goiás, use o subsistema SE como proxy hidrológica regional.

### nom_subsistema (TEXTO, 20)
Nome por extenso (ex.: “Sudeste/Centro-Oeste”). Útil para leitura humana.

### ear_data (DATETIME YYYY-MM-DD)
Data observada (periodicidade diária). É a chave temporal para juntar com suas agregações diárias de carga/oferta.

### ear_max_subsistema (FLOAT, unidade = MWmês)
Capacidade máxima de armazenamento energético do subsistema (reservatórios cheios). Não permite nulo/zero/negativo. Use como denominador para normalizações.

### ear_verif_subsistema_mwmes (FLOAT, MWmês)
EAR verificada no dia (absoluta). Não permite nulo/zero/negativo. É sua medida bruta de estoque hídrico convertível em geração.

### ear_verif_subsistema_percentual (FLOAT, %)
EAR verificada em percentagem da capacidade (equivale a verif_mwmes / max * 100). Não permite nulo/zero/negativo. Indicador-régua para sazonalidade/estresse hídrico.

Observação de integridade: para os três campos numéricos acima, o dicionário registra não permitir valores nulos, zerados ou negativos. MWmês é energia (potência × tempo). Para comparar com grandezas em MWh, converta: EAR_MWh = ear_verif_subsistema_mwmes × horas_no_mês_da_data.

Insight útil: “dias de atendimento” aproximados em base diária:
dias = EAR_MWh / (Demanda_diária_MWh), onde Demanda_diária_MWh = val_carga_media_diaria (MW) × 24.


## ENA Por Subsistema
Peça-chave pra enxergar a “força da água” chegando aos reservatórios e antecipar estresse hídrico.
ENA (Energia Natural Afluente) é a energia “produzível” a partir das vazões naturais que chegam aos reservatórios.
ENA bruta: vazões naturais × produtividades (assumindo 65% do volume útil).
ENA armazenável: ENA bruta descontadas as vertidas (o que de fato pode virar estoque/geração depois). 

### id_subsistema
Tipo/unidade: Texto (2 posições).
O que é: código do subsistema (p.ex., N, NE, S, SE – Sudeste inclui Centro-Oeste).
Integridade: não permite nulo, zero ou negativo.

### nom_subsistema
Tipo/unidade: Texto (20 posições).
O que é: nome do subsistema por extenso.
Integridade: não permite nulo, zero ou negativo.

### ena_data
Tipo/unidade: Data (YYYY-MM-DD).
O que é: dia observado (periodicidade diária).
Integridade: não permite nulo, zero ou negativo.

### ena_bruta_regiao_mwmed
Tipo/unidade: Float em MWmês (energia).
O que é: ENA bruta absoluta do subsistema no dia.
Integridade: não permite nulo, zero ou negativo.
Obs: converta MWmês->MWh (multiplique pelas horas do mês da data) quando comparar com cargas/gerações em MWh; 

### ena_bruta_regiao_percentualmlt
Tipo/unidade: Float, % MLT (média de longo termo).
O que é: ENA bruta normalizada pela MLT (100% ~ regime hidrológico médio).
Integridade: não permite nulo/zero; negativo pode ocorrer (casos de ajuste).
Obs: feature forte para anomalia hidrológica (seca > risco), use lags (D-1/D-7) e Δ% semanal/mensal.

### ena_armazenavel_regiao_mwmed
Tipo/unidade: Float em MWmês.
O que é: ENA armazenável absoluta (desconta vertimentos).
Integridade: não permite nulo/zero; pode ser negativa (ajustes).
Obs: reflete melhor o que pode virar EAR/geração futura; crie tendência e taxa de variação.

### ena_armazenavel_regiao_percentualmlt
Tipo/unidade: Float, % MLT.
O que é: ENA armazenável em % da MLT.
Integridade: não permite nulo/zero; pode ser negativa (ajustes).
Obs: indicador de estresse hídrico; combinado com EAR% ajuda a capturar queda/recuperação de reservatórios.

## Intercâmbio Nacional
Mede, hora a hora, os fluxos de potência ativa nas linhas de fronteira entre subsistemas do SIN; o valor reportado é a soma desses fluxos em MWmed. Útil para saber quem importa/exporta energia a cada hora. Observações do dicionário: (i) a relação de linhas de fronteira está no “Relatório Quadrimestral de Limites de Intercâmbio (Newave)” no SINtegre; (ii) não inclui o intercâmbio do Subsistema Sul com países vizinhos (isso aparece no produto “intercâmbio do SIN”).

### din_instante
Tipo/unidade: DATETIME (YYYY-MM-DD HH:MM:SS), início do período de agregação horária.
O que é: timestamp da medição (janela de 1 hora).
Integridade: não permite nulo; zero/negativo não se aplicam (é data).
obs: chave temporal para agregar por dia/semana e casar com Carga (semi-horária -> diária) e Balanço (horária -> diária). Atenção à coerência de fuso quando juntar com “Carga Verificada” (que vinha em UTC).

### id_subsistema_origem
Tipo/unidade: TEXTO (3).
O que é: código do subsistema de origem do fluxo (p.ex., SE, NE, S, N).
Integridade: não permite nulo.
Obs: identificador para montar matriz hora-a-hora de intercâmbio entre pares (origem -> destino). .

### nom_subsistema_origem
Tipo/unidade: TEXTO (20).
O que é: nome por extenso do subsistema de origem.
Integridade: não permite nulo.


### id_subsistema_destino
Tipo/unidade: TEXTO (3).
O que é: código do subsistema de destino do fluxo.
Integridade: não permite nulo.
Obs: junto com a origem, define o par (aresta) da rede de intercâmbio naquela hora.

### nom_subsistema_destino
Tipo/unidade: TEXTO (20).
O que é: nome por extenso do subsistema de destino.
Integridade: não permite nulo.

### val_intercambiomwmed
Tipo/unidade: FLOAT (MWmed).
O que é: intercâmbio verificado (soma dos fluxos nas linhas de fronteira) entre origem -> destino, na base horária.
Integridade: não permite nulo; permite zero; permite negativo.
Obs:Operacionalmente, o sinal acompanha a direção líquida: valores positivos representam fluxo no sentido origem -> destino naquela hora; negativos indicam reversão (fluxo efetivo no sentido oposto).
Para obter a importação líquida de um subsistema S na hora t, use a regra prática:
net_import_S(t) = Σ[val (destino = S)] − Σ[val (origem = S)].
Essa fórmula é robusta mesmo quando alguns pares vierem negativos (o sinal do próprio valor já ajusta o saldo). Em seguida, some as horas do dia para ter o saldo diário e use na classificação.

## Restrição de Constrained off em usinas eólicas
O dataset traz, em base horária, as restrições de operação (constrained-off) em usinas eólicas Tipo I, II-B e II-C. É onde você enxerga curtailment por motivos elétricos/rede, requisitos de confiabilidade, razões energéticas ou condicionantes de acesso.

### id_subsistema
Tipo: Texto (2).
O que é: código do subsistema (N, NE, S, SE).
Integridade: não permite nulo/zerado/negativo.
Obs: para Goiás, concentre no SE (Sudeste/CO) para visão sistêmica; combine com id_estado="GO" quando quiser granularidade estadual.

### nom_subsistema
Tipo: Texto (até 60).
O que é: nome por extenso do subsistema.
Integridade: não permite nulo/zerado/negativo.


### id_estado
Tipo: Texto (2).
O que é: UF da usina/conjunto (ex.: GO).
Integridade: não permite nulo/zerado/negativo.
Obs: filtrar GO para recorte estadual

### nom_estado
Tipo: Texto (30).
O que é: nome do estado por extenso.
Integridade: não permite nulo/zerado/negativo.

### nom_usina
Tipo: Texto (60).
O que é: nome da usina ou do conjunto de usinas (quando agrupadas).
Integridade: permite nulo.

### id_ons
Tipo: Texto (6).
O que é: identificador da usina/conjunto no ONS.
Integridade: não permite nulo/zerado/negativo.
Obs: chave técnica estável para joins e deduplicações. (Campo adicionado na v1.1.)

### ceg
Tipo: Texto (30).
O que é: Código do Empreendimento de Geração (ANEEL). Em conjuntos de usinas, vem “-” (sem CEG).
Integridade: não permite nulo/zerado/negativo.
Obs: chave de referência regulatória (ANEEL); trate “-” como missing em agrupamentos por empreendimento. (Campo adicionado na v1.1.)

### din_instante
Tipo: DATETIME (YYYY-MM-DD HH:MM:SS).
O que é: data/hora de referência (janela horária).
Integridade: não permite nulo.
Obs: chave temporal para agregar dia/semana e casar com Balanço/Carga/ENA/EAR.

### val_geracao
Tipo: Float (MWmed).
O que é: geração verificada da usina/conjunto naquela hora.
Integridade: não permite nulo; permite zero; não permite negativo.
Obs: base para calcular curtailment (diferença vs. referência/disponibilidade).

### val_geracaolimitada
Tipo: Float (MWmed).
O que é: geração limitada por restrição. É a potência média horária (MWmed) que deixou de ser gerada por causa de uma restrição naquele intervalo. É o “curtailment” registrado: quanto foi cortado/limitado, não a geração efetiva. O dicionário define: “Valor da Geração Limitada por alguma Restrição, em MWmed” (pode ser nulo, pode ser zero, não pode ser negativo).
Integridade: permite nulo; permite zero; não permite negativo.
Se val_geracaolimitada > 0, houve corte por restrição (elétrica/rede, confiabilidade, energética ou acesso).
Se val_geracaolimitada = 0, não houve corte naquele horário (mesmo que o vento fosse baixo).
Nulo ≠ zero: nulo significa “sem valor informado”; não assuma ausência de corte — trate como missing.

### val_disponibilidade
Tipo: Float (MWmed).
O que é: disponibilidade verificada em tempo real (capacidade disponível).
Integridade: permite nulo; permite zero; não permite negativo.
Obs: ajuda a diferenciar “vento fraco” (baixa disponibilidade) de restrição sistêmica (alta disponibilidade, mas baixa geração).

### val_geracaoreferencia
Tipo: Float (MWmed).
O que é: é uma estimativa ex-ante da geração eólica esperada naquela hora, em MW médios. Ele serve como baseline para comparar com a geração efetiva (val_geracao) e identificar perdas por limitação (curtailment). O dicionário define literalmente como “Valor da Geração de referência (ou estimada), em MWmed”. Quando houver, prefira a versão final (val_geracaoreferenciafinal), que é a mesma referência após consistências/ajustes
Integridade: permite nulo; permite zero; não permite negativo.

### val_geracaoreferenciafinal
Tipo: Float (MWmed).
O que é: geração de referência final (após consistências).
Integridade: permite nulo; permite zero; não permite negativo.
Obs: melhor base para calcular energia/potência curtailed: curtailed ≈ max(0, ref_final − geração).

### cod_razaorestricao
Tipo: Texto (3).
O que é: razão da restrição:
REL — indisponibilidade externa (elétrica)
O que significa na operação: limitação por condições elétricas da rede (sobrecarga térmica, limites de tensão/estabilidade, manutenções, falhas, proteção, reconfigurações). A usina poderia gerar (vento presente), mas a rede não comporta.
Sinais no dado: val_disponibilidade alto, val_geracao baixo e val_geracaolimitada > 0; frequentemente com cod_origemrestricao = LOC (problema local) ou SIS (restrição da malha em nível de subsistema).
Implicações/mitigação: reforço de rede, manobras operativas, controle de tensão, remedial actions; no curto prazo, o curtailment reduz oferta efetiva renovável.


CNF — atendimento a requisitos de confiabilidade
O que significa na operação: cortes para cumprir critérios de segurança N-1, níveis de inércia/reserva girante, controle de frequência/voltagem, ou outras margens de confiabilidade do ONS. Não é “pane” na rede, é segurança sistêmica.
Sinais no dado: curtailment mesmo sem evidências de sobrecarga local; ocorre em janelas de maior risco sistêmico; costuma vir com cod_origemrestricao = SIS.
Implicações/mitigação: despacho de fontes síncronas, ajustes de reserva, coordenação de controle; impacta a penetração renovável instantânea.

ENE — razão energética
O que significa na operação (interpretação prática): cortes motivados por condição energética agregada (ex.: excedente momentâneo sem escoamento/armazenamento; coordenação hidrotérmica; evitar vertimento; balanço energético do subsistema). Não é limitação de equipamento específico, mas gestão de energia.
Sinais no dado: val_geracaolimitada > 0 em períodos de baixa carga/alto aporte (vento/hídrica) e pouca folga de transmissão/armazenamento; origem pode ser SIS.
Implicações/mitigação: coordenação com intercâmbio, despacho térmico/hídrico e limites operativos; em geral, aparece com sazonalidade/pluviometria.

PAR — restrição indicada em parecer de acesso
O que significa na operação: limitação formal prevista no parecer de acesso (antes da conclusão de obras/adequações). É comum em comissionamento ou enquanto reforços de rede não ficam prontos.
Sinais no dado: curtailment recorrente até um teto de injeção definido; geralmente cod_origemrestricao = LOC.
Implicações/mitigação: depende da execução das obras de acesso/rede; quando concluídas, a restrição tende a cessar.
No modelo: use PAR para capturar restrições contratuais/temporárias; explica cortes persistentes sem mudança climática/operativa.
Integridade: permite nulo.


### cod_origemrestricao
Tipo: Texto (3).
O que é: origem da restrição:
LOC = local;
SIS = sistêmica.
Integridade: permite nulo.
Obs: separa problemas locais (e.g., proteção, seccionamento) de limitações sistêmicas (e.g., rede saturada). Excelente feature para explicar risco.


## Restrição de Constrained off em usinas fotovoltaicas

O dataset traz, em base horária, as restrições de operação (constrained-off) em usinas fotovoltaicas Tipo I, II-B e II-C — onde você enxerga curtailment por motivos elétricos/rede, requisitos de confiabilidade, razões energéticas ou condicionantes de acesso.

### id_subsistema
Tipo: Texto (2).
O que é: código do subsistema (N, NE, S, SE).
Integridade: não permite nulo/zerado/negativo.
Obs: para Goiás, concentre no SE (Sudeste/CO) para visão sistêmica.

### nom_subsistema
Tipo: Texto (até 60).
O que é: nome por extenso do subsistema.
Integridade: não permite nulo/zerado/negativo.

### id_estado
Tipo: Texto (2).
O que é: UF da usina/conjunto (ex.: GO).
Integridade: não permite nulo/zerado/negativo.
Obs: filtre GO para recorte estadual.

### nom_estado
Tipo: Texto (30).
O que é: nome do estado por extenso.
Integridade: não permite nulo/zerado/negativo.

### nom_usina
Tipo: Texto (60).
O que é: nome da usina ou do conjunto de usinas (quando agrupadas).
Integridade: permite nulo.

### id_ons
Tipo: Texto (6).
O que é: identificador da usina/conjunto no ONS.
Integridade: não permite nulo/zerado/negativo.
Obs: chave técnica estável para joins e deduplicações.

### ceg
Tipo: Texto (30).
O que é: Código do Empreendimento de Geração (ANEEL).
Integridade: não permite nulo/zerado/negativo.
Obs: em conjuntos de usinas pode haver representação especial; trate ausências como missing em agrupamentos por empreendimento.

### din_instante
Tipo: DATETIME (YYYY-MM-DD HH:MM:SS).
O que é: data/hora de referência (janela horária).
Integridade: não permite nulo.
Obs: chave temporal para agregar dia/semana e casar com Balanço/Carga/ENA/EAR.

### val_geracao
Tipo: Float (MWmed).
O que é: geração verificada da usina/conjunto naquela hora.
Integridade: não permite nulo; permite zero; não permite negativo.
Obs: base para calcular curtailment (diferença vs. referência/disponibilidade).

### val_geracaolimitada
Tipo: Float (MWmed).
O que é: geração limitada por restrição (potência média horária que deixou de ser gerada por uma restrição).
Integridade: permite nulo; permite zero; não permite negativo.
Obs: se > 0, houve corte por restrição (elétrica/rede, confiabilidade, energética ou acesso); se = 0, não houve corte naquele horário; nulo ≠ zero (trate como missing).

### val_disponibilidade
Tipo: Float (MWmed).
O que é: disponibilidade verificada em tempo real (capacidade disponível).
Integridade: permite nulo; permite zero; não permite negativo.
Obs: ajuda a diferenciar “recurso solar baixo” de restrição sistêmica (disponibilidade alta, mas geração baixa).

### val_geracaoreferencia
Tipo: Float (MWmed).
O que é: estimativa ex-ante da geração esperada na hora (baseline).
Integridade: permite nulo; permite zero; não permite negativo.
Obs: compare com val_geracao para inferir curtailed quando val_geracaolimitada vier nulo; prefira a versão final abaixo quando disponível.

### val_geracaoreferenciafinal
Tipo: Float (MWmed).
O que é: geração de referência final (após consistências).
Integridade: permite nulo; permite zero; não permite negativo.
Obs: melhor base para estimar curtailed por gap: ≈ max(0, ref_final − geração).

### cod_razaorestricao
Tipo: Texto (3).
O que é: razão da restrição — domínios:
REL — indisponibilidade externa (elétrica)
CNF — atendimento a requisitos de confiabilidade
ENE — razão energética
PAR — restrição indicada em parecer de acesso
Integridade: permite nulo.
Obs (leitura operacional):
• REL: limitação por condições elétricas da rede (sobrecarga, tensão/estabilidade, manutenção/contingência).
• CNF: cortes para cumprir margens de segurança (N-1, inércia/reserva, frequência/voltagem).
• ENE: motivação energética agregada (ex.: excedente sem escoamento/armazenamento).
• PAR: limitação formal prevista no parecer de acesso (comissionamento/obras pendentes).

### cod_origemrestricao
Tipo: Texto (3).
O que é: origem da restrição — domínios: LOC (local) e SIS (sistêmica).
Integridade: permite nulo.
Obs: separa problemas locais (ex.: acesso/seccionamento) de limitações sistêmicas (rede/subsistema).


## Meteorologia (NASA POWER)
Série diária agregada sobre pontos em Goiás (API NASA POWER), usada como covariáveis de clima.

Arquivo produzido: `data/raw/clima_go_diario.csv` (ver coleta em `src/meteo.py`).

Origem oficial: POWER Data Access, endpoint temporal-daily point.
- Base URL: `https://power.larc.nasa.gov/api/temporal/daily/point`
- Parâmetros utilizados: `ALLSKY_SFC_SW_DWN` (GHI), `T2M` (temperatura a 2 m), `PRECTOTCORR` (precipitação corrigida)
- Comunidade: `RE` (Renewables)
- Agregação espacial: média de 4 pontos representativos sobre GO
  - Pontos padrão no código: (-16.7, -49.3), (-15.9, -48.2), (-18.0, -49.7), (-13.6, -46.9)

Observações gerais e integridade
- Periodicidade: diária; série contínua por `data` (YYYY-MM-DD).
- Timezone: timestamps tratados como datas (sem hora); agregação e joins no fuso America/Sao_Paulo via pipeline.
- Unidades conforme documentação NASA POWER (comunidade RE, diário):
  - `ALLSKY_SFC_SW_DWN` -> kWh/m²/dia (GHI diário)
  - `T2M` -> °C (temperatura média diária a 2 m)
  - `PRECTOTCORR` -> mm/dia (precipitação total corrigida)
- Valores faltantes: podem ocorrer em dias isolados; o pipeline permite preenchimento curto (até 7 dias) para joins.

Campos no CSV do projeto (após a média espacial)

### data
Tipo/unidade: Data (YYYY-MM-DD).
O que é: dia de referência da observação meteorológica.
Integridade: não permite nulo.

### ghi
Tipo/unidade: Float (kWh/m²/dia).
O que é: Irradiação Global Horizontal (ALLSKY_SFC_SW_DWN) diária média sobre os pontos em GO.
Uso no projeto: proxy direta para potencial fotovoltaico; entra nas agregações semanais (mean/sum/max, etc.).
Notas: valores típicos 3–7 kWh/m²/dia; sazonalidade marcada (seca/chuvosa).

### temp2m_c
Tipo/unidade: Float (°C).
O que é: Temperatura média diária do ar a 2 metros (T2M).
Uso no projeto: influencia carga (resfriamento) e desempenho FV; agregada semanalmente.
Notas: já vem em graus Celsius via API (comunidade RE).

### precipitacao_mm
Tipo/unidade: Float (mm/dia).
O que é: Precipitação diária total corrigida (PRECTOTCORR).
Uso no projeto: usada diretamente e como base para acumulados móveis (14 e 30 dias) antes da agregação semanal.
Notas: acumulados móveis criados no pipeline diário -> semanal em `src/feature_engineer.py`.

Derivados usados nas features (criados no pipeline)
- `precip_14d_mm`: soma móvel de 14 dias da precipitação (mm)
- `precip_30d_mm`: soma móvel de 30 dias (mm)

Boas práticas/atenção
- Coerência temporal: cruzar com séries do ONS já em frequência diária antes do D->W.
- Variação espacial: pontos padrão podem ser ajustados se desejar focos regionais; manter número de pontos para comparabilidade.
- Consistência de unidades: manter nomes/colunas exatamente como no CSV (`ghi`, `temp2m_c`, `precipitacao_mm`).
