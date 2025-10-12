# 🚀 API de Predição de Risco de Déficit de Energia — AWS Lambda + Docker

Este projeto demonstra o deploy do modelo no **AWS Lambda** utilizando uma **imagem Docker** hospedada no **Amazon ECR (Elastic Container Registry) - Serverless**.  
A aplicação expõe uma **API HTTP** via **Lambda Function URL**, que recebe um JSON com as features e retorna a classe de risco (`alto`, `médio`, `baixo`).

---

## Estrutura do Projeto

```
risco-deficit-energia-api/
│
├── Dockerfile
├── requirements.txt
├── models/
│   └── xgb.joblib
└── src/
    ├── api/
    │   └── handler.py
    └── models/
        └── contiguous.py
```

---

## 1. Configuração Inicial

Antes de tudo certifique-se de estar usando um AWS Linux (EC2 ou Cloudshell). Instale e configure as ferramentas abaixo (caso não tenha instalado):

```bash
aws configure
docker --version
```

---

## 2. Construindo o Container Docker

### 2.1 Dockerfile

```dockerfile
FROM public.ecr.aws/lambda/python:3.12

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ${LAMBDA_TASK_ROOT}/src/
COPY models/ ${LAMBDA_TASK_ROOT}/models/

CMD ["src.api.handler.handler"]
```

### 2.2 requirements.txt

```
pandas
numpy
joblib
scikit-learn==1.7.2
xgboost
```

---

## 3. Build da Imagem e Upload no ECR

```bash
aws ecr create-repository --repository-name risco-energia-api || true

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=us-east-1
REPO="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/risco-energia-api"

aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin "$REPO"

docker build -t risco-energia-api .
docker tag risco-energia-api:latest "$REPO:latest"
docker push "$REPO:latest"
```

---

## 4. Criar a Função Lambda

```bash
ROLE_ARN=$(aws iam get-role --role-name LabRole --query 'Role.Arn' --output text)

aws lambda create-function   --function-name RISCO-ENERGIA-API-IMG   --package-type Image   --code ImageUri="$REPO:latest"   --role "$ROLE_ARN"   --architectures x86_64   --memory-size 1024   --timeout 15

aws lambda update-function-configuration   --function-name RISCO-ENERGIA-API-IMG   --image-config 'Command=["src.api.handler.handler"]'
```

---

## 5. Publicar Endpoint HTTP (Function URL)

```bash
aws lambda create-function-url-config   --function-name RISCO-ENERGIA-API-IMG   --auth-type NONE

aws lambda get-function-url-config   --function-name RISCO-ENERGIA-API-IMG
```

Exemplo:
```
https://conlogt7ujcn435nfbipyqoa5y0mooxq.lambda-url.us-east-1.on.aws/
```

---

## 6. Testando a API

```bash
curl -X POST "https://conlogt7ujcn435nfbipyqoa5y0mooxq.lambda-url.us-east-1.on.aws/"   -H "Content-Type: application/json"   -d '{
    "features": {
      "ena_mwmed__p95_w_r4w_mean": 6050.0,
      "ena_mwmed_max_w_r4w_mean": 6185.0,
      "ear_pct_mean_w": 67.3,
      "ear_pct_min_w": 52.4,
      "ear_pct__p05_w": 49.1,
      "carga_mwh__p05_w_r2w_mean": 4420.0,
      "ena_mwmed__p95_w_r2w_mean": 5925.0,
      "ens_week_mwh": 85.0,
      "ear_pct_sum_w": 463.2,
      "margem_vs_carga_w": 1.18,
      "ena_mwmed_sum_w_r2w_mean": 41230.0,
      "ena_mwmed_mean_w_r2w_mean": 5975.0,
      "ear_pct__p95_w": 81.4,
      "ear_pct_max_w": 88.6,
      "carga_mwh_sum_w": 30970.0,
      "ear_pct_mean_w_r2w_mean": 69.2,
      "carga_mwh_min_w_r2w_mean": 4230.0,
      "carga_mwh_mean_w": 4425.0,
      "ena_mwmed_mean_w_r4w_mean": 5880.0,
      "carga_mwh_sum_w_r2w_mean": 29840.0,
      "ena_mwmed_max_w_lag1w": 6320.0,
      "carga_mwh_mean_w_r2w_mean": 4480.0,
      "ena_mwmed__p95_w_lag1w": 6180.0,
      "carga_mwh_max_w_r2w_mean": 4785.0,
      "ear_pct_min_w_r2w_mean": 55.8
    }
  }'
```

Saída esperada:

```json
{"classe_risco": "alto"}
```

---

## 7. Logs e Depuração

```bash
aws logs tail /aws/lambda/RISCO-ENERGIA-API-IMG --follow
```

---

## 8. Atualizar a Imagem

```bash
docker build -t risco-energia-api .
docker tag risco-energia-api:latest "$REPO:latest"
docker push "$REPO:latest"

aws lambda update-function-code   --function-name RISCO-ENERGIA-API-IMG   --image-uri "$REPO:latest"
```



