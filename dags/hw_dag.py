import datetime as dt
import os
import sys

from modules import predict
from modules.pipeline import pipeline
from airflow.models import DAG
from airflow.operators.python import PythonOperator

path = os.path.expanduser('~/airflow_hw')
# Добавим путь к коду проекта в переменную окружения, чтобы он был доступен python-процессу
os.environ['PROJECT_PATH'] = path
# Добавим путь к коду проекта в $PATH, чтобы импортировать функции
sys.path.insert(0, path)

args = {
    'owner': 'airflow',
    'start_date': dt.datetime(2022, 6, 10),
    'retries': 1,
    'retry_delay': dt.timedelta(minutes=1),
    'depends_on_past': False,
}


with DAG(
        dag_id='car_price_prediction',
        schedule="00 15 * * *",
        default_args=args
) as dag:
    pipeline_process = PythonOperator(
        task_id='pipeline',
        python_callable=pipeline,
    )
    predict_process = PythonOperator(
        task_id='predict',
        python_callable=predict,
    )

    pipeline_process >> predict_process