import boto3
import os
from botocore.exceptions import NoCredentialsError


def get_s3_client(aws_access_key_id=None, aws_secret_access_key=None, region_name=None):
    return boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id or os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=aws_secret_access_key or os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=region_name or os.getenv('AWS_REGION')
    )


def download_file_from_s3(bucket, s3_key, local_path, prefix="", aws_access_key_id=None, aws_secret_access_key=None, region_name=None):
    s3 = get_s3_client(aws_access_key_id, aws_secret_access_key, region_name)
    s3_key = f"{prefix}{s3_key}" if prefix else s3_key
    try:
        s3.download_file(bucket, s3_key, local_path)
        return True
    except NoCredentialsError:
        print('Credenciais AWS não encontradas.')
        return False
    except Exception as e:
        print(f'Erro ao baixar arquivo do S3: {e}')
        return False


def upload_file_to_s3(local_path, bucket, s3_key, prefix="", aws_access_key_id=None, aws_secret_access_key=None, region_name=None):
    s3 = get_s3_client(aws_access_key_id, aws_secret_access_key, region_name)
    s3_key = f"{prefix}{s3_key}" if prefix else s3_key
    try:
        s3.upload_file(local_path, bucket, s3_key)
        return True
    except NoCredentialsError:
        print('Credenciais AWS não encontradas.')
        return False
    except Exception as e:
        print(f'Erro ao enviar arquivo para o S3: {e}')
        return False


def read_parquet_from_s3(bucket, s3_key, aws_access_key_id=None, aws_secret_access_key=None, region_name=None):
    import pandas as pd
    import io
    s3 = get_s3_client(aws_access_key_id, aws_secret_access_key, region_name)
    try:
        obj = s3.get_object(Bucket=bucket, Key=s3_key)
        return pd.read_parquet(io.BytesIO(obj['Body'].read()))
    except Exception as e:
        print(f'Erro ao ler parquet do S3: {e}')
        return None


def read_csv_from_s3(bucket, s3_key, prefix="", aws_access_key_id=None, aws_secret_access_key=None, region_name=None, **kwargs):
    import pandas as pd
    import io
    s3 = get_s3_client(aws_access_key_id, aws_secret_access_key, region_name)
    s3_key = f"{prefix}{s3_key}" if prefix else s3_key
    try:
        obj = s3.get_object(Bucket=bucket, Key=s3_key)
        return pd.read_csv(io.BytesIO(obj['Body'].read()), **kwargs)
    except Exception as e:
        print(f'Erro ao ler csv do S3: {e}')
        return None
