-- models/dim_calendario.sql


with limites as (
    select 
        min(DATA) as data_min,
        max(DATA) as data_max
    from ENERGIA.STAGING.BRONZE
),
datas as (
    select 
        dateadd(day, seq4(), data_min) as data
    from limites,
         table(generator(rowcount => datediff(day, data_min, data_max) + 1))
)

select
    data,
    day(data) as dia_do_mes,
    case month(data)
        when 1 then 'Janeiro'
        when 2 then 'Fevereiro'
        when 3 then 'Março'
        when 4 then 'Abril'
        when 5 then 'Maio'
        when 6 then 'Junho'
        when 7 then 'Julho'
        when 8 then 'Agosto'
        when 9 then 'Setembro'
        when 10 then 'Outubro'
        when 11 then 'Novembro'
        when 12 then 'Dezembro'
    end as nome_mes_pt,
    month(data) as numero_mes,
    year(data) as ano,
    quarter(data) as trimestre,
    case 
        when month(data) between 1 and 6 then 1
        else 2
    end as semestre,
    case 
        when month(data) between 1 and 4 then 1
        when month(data) between 5 and 8 then 2
        else 3
    end as quadrimestre,
    date_trunc('month', data) as primeiro_dia_mes
from datas
