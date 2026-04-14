import os
import sys
from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy import create_engine
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.models import Variable

# Ensure Airflow can discover our custom modules mounted at /opt/airflow/data
sys.path.append('/opt/airflow')

from data.ingestion.yfinance_fetcher import YFinanceFetcher
from data.ingestion.s3_uploader import S3Uploader
from data.transforms.cleaner import DataCleaner
from data.transforms.feature_engineer import FeatureEngineer

# Tickers to track
TARGET_TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "XLF", "XLK", "XLE"]

# Setup Prometheus metric
registry = CollectorRegistry()
pipeline_success_metric = Gauge(
    'pipeline_last_success_timestamp',
    'Last time the ingest_ohlcv DAG succeeded',
    registry=registry
)

default_args = {
    'owner': 'quant',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1), # Change to desired backtest start
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'ingest_ohlcv_daily',
    default_args=default_args,
    description='Fetch OHLCV, transform, and load to PostgreSQL',
    schedule_interval=timedelta(days=1),
    catchup=False,
)

def extract_and_upload(**kwargs):
    """
    Fetches raw data using yfinance and uploads to LocalStack S3.
    """
    execution_date = kwargs['ds']
    
    # We fetch a slightly longer window to allow calculation of rolling features (e.g. 21d vol)
    # So if running daily, we still want to grab enough context.
    # For a daily incremental load, one might just pull last 30 days and upsert.
    # To keep this simple, we'll pull the last 60 days.
    start = (pd.to_datetime(execution_date) - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    end = (pd.to_datetime(execution_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    
    fetcher = YFinanceFetcher(TARGET_TICKERS)
    raw_df = fetcher.fetch_historical_data(start, end)
    
    if raw_df is None or raw_df.empty:
        raise ValueError("No data returned from YFinance")
    
    # Save temporarily to local airflow volume
    raw_path = f"/opt/airflow/data/raw_ohlcv_{execution_date}.csv"
    raw_df.to_csv(raw_path, index=False)
    
    # Upload to S3 (LocalStack)
    s3 = S3Uploader(bucket_name="quantfolio-raw-data")
    s3.upload_csv(raw_path, object_name=f"raw/ohlcv_{execution_date}.csv")
    
    return raw_path

def transform_data(**kwargs):
    ti = kwargs['ti']
    raw_path = ti.xcom_pull(task_ids='extract_raw_data')
    
    raw_df = pd.read_csv(raw_path)
    
    cleaner = DataCleaner()
    clean_df = cleaner.clean_ohlcv(raw_df)
    
    engineer = FeatureEngineer()
    feat_df = engineer.engineer_features(clean_df)
    
    feat_path = raw_path.replace("raw_", "feat_")
    feat_df.to_csv(feat_path, index=False)
    
    return feat_path

def load_to_postgres(**kwargs):
    ti = kwargs['ti']
    feat_path = ti.xcom_pull(task_ids='transform_data')
    
    feat_df = pd.read_csv(feat_path)
    
    # Grab PG conn string from Environment variables configured in docker compose
    # e.g., postgresql+psycopg2://quant:quant_pass@postgres:5432/quantfolio
    pg_conn = os.getenv("AIRFLOW__DATABASE__SQL_ALCHEMY_CONN")
    if not pg_conn:
        # Fallback to direct construction for the DAG environment if missing
        pg_conn = "postgresql+psycopg2://quant:quant_pass@postgres:5432/quantfolio"
        
    engine = create_engine(pg_conn)
    
    # Split into ohlcv and features
    base_cols = ['ticker', 'trade_date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
    feat_cols = ['ticker', 'trade_date', 'log_return_1d', 'rolling_vol_21d', 'rsi_14', 'macd', 'macd_signal']
    
    ohlcv_df = feat_df[base_cols].copy()
    features_df = feat_df[feat_cols].copy()
    
    # Use pandas to_sql with if_exists='append'. In production, use upserts (ON CONFLICT DO UPDATE).
    # Since we are fetching 60 days overlapping, we need to handle duplicates.
    # Fastest way in pure pandas with postgres is to load to a temp table, but here
    # we just iterate or drop duplicates since this is an MVP. 
    # Proper UPSERT:
    with engine.begin() as conn:
        ohlcv_df.to_sql('daily_ohlcv_temp', conn, if_exists='replace', index=False)
        conn.execute('''
            INSERT INTO daily_ohlcv (ticker, trade_date, open, high, low, close, adj_close, volume)
            SELECT ticker, CAST(trade_date AS DATE), open, high, low, close, adj_close, volume 
            FROM daily_ohlcv_temp
            ON CONFLICT (ticker, trade_date) DO UPDATE SET
            open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low, 
            close=EXCLUDED.close, adj_close=EXCLUDED.adj_close, volume=EXCLUDED.volume;
        ''')
        
        features_df.to_sql('daily_features_temp', conn, if_exists='replace', index=False)
        conn.execute('''
            INSERT INTO daily_features (ticker, trade_date, log_return_1d, rolling_vol_21d, rsi_14, macd, macd_signal)
            SELECT ticker, CAST(trade_date AS DATE), log_return_1d, rolling_vol_21d, rsi_14, macd, macd_signal 
            FROM daily_features_temp
            ON CONFLICT (ticker, trade_date) DO UPDATE SET
            log_return_1d=EXCLUDED.log_return_1d, rolling_vol_21d=EXCLUDED.rolling_vol_21d, 
            rsi_14=EXCLUDED.rsi_14, macd=EXCLUDED.macd, macd_signal=EXCLUDED.macd_signal;
        ''')
        
def push_metrics(**kwargs):
    pipeline_success_metric.set_to_current_time()
    try:
        push_to_gateway('pushgateway:9091', job='airflow_ingestion_dag', registry=registry)
        print("Successfully pushed metrics to pushgateway.")
    except Exception as e:
        print(f"Failed to push metrics: {e}")

task1 = PythonOperator(
    task_id='extract_raw_data',
    python_callable=extract_and_upload,
    provide_context=True,
    dag=dag,
)

task2 = PythonOperator(
    task_id='transform_data',
    python_callable=transform_data,
    provide_context=True,
    dag=dag,
)

task3 = PythonOperator(
    task_id='load_to_postgres',
    python_callable=load_to_postgres,
    provide_context=True,
    dag=dag,
)

task4 = PythonOperator(
    task_id='push_success_metrics',
    python_callable=push_metrics,
    provide_context=True,
    dag=dag,
)

task1 >> task2 >> task3 >> task4
