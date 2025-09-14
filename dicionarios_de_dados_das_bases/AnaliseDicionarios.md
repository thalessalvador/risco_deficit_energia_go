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
Uso no projeto: comparar oferta vs demanda → indicador direto de risco de déficit.

### val_intercambio
Tipo: Float
O que é: Intercâmbio líquido de energia do subsistema (MW médios).
Positivo → importação de energia.
Negativo → exportação de energia.
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

