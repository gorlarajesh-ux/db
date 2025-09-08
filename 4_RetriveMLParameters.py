# Databricks notebook source
from mlflow.tracking import MlflowClient

def list_nested_artifact(client, run_id: str, artifact_path: str, experiment_id: str):
    if artifact_path.is_dir:
        print(f"\nDirectory: {artifact_path.path}")
        
        nested_artifacts = client.list_artifacts(run_id, artifact_path.path)
        for nested_artifact in nested_artifacts:
            list_nested_artifact(client, run_id, nested_artifact, experiment_id)
    else:
        #print(f"File name: {artifact_path.path}")
        print(f"File path: dbfs:/databricks/mlflow-tracking/{experiment_id}/{run_id}/artifacts/{artifact_path.path}")
        #dbfs:/databricks/mlflow-tracking/3819138346157943/8e5190c219c74e12a68301b1404564ea/artifacts/images/reporting/adstock_decay.png
        #dbfs:/databricks/mlflow-tracking/3819138346157943/8e5190c219c74e12a68301b1404564ea/artifacts/images/reporting/adstock_decay.png



def getMLParameters(run_id:str)->None:
    client = MlflowClient()
    run_client = client.get_run(run_id)
    experiment_id = run_client.info.experiment_id

    print(f"Run ID: {run_id}")

    print("\nParameters:")
    for key, value in run_client.data.params.items():
        print(f"  {key} ->: {value}")

    print("\nMetrics:")
    for key, value in run_client.data.metrics.items():
        print(f"  {key} ->: {value}")

    #print("\nArtifacts:")
    artifact_paths = client.list_artifacts(run_id)
    #print('artifact_paths',artifact_paths)
    if artifact_paths:
        for artifact in artifact_paths:
            #print(artifact.path)
            if artifact.path in ['images']:
                artifact_nested = client.list_artifacts(run_id, artifact.path)
                #print('artifact_nested',artifact_nested)
                #print(type(artifact_nested))
                for nested_artifact in artifact_nested:
                    list_nested_artifact(client, run_id, nested_artifact,experiment_id)
                break

getMLParameters(run_id = "8e5190c219c74e12a68301b1404564ea")





# COMMAND ----------

import re
from datetime import datetime

from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit
from pyspark.sql.types import DoubleType, StringType, TimestampType

from utils.snowflake_utils import write_snowflake_table

TARGET_TABLE = "MMM.EVALUTATION_METRICS"

TYPES = {
    "RUN_ID": StringType(),
    "MAPE": DoubleType(),
    "R_SQUARED": DoubleType(),
    "WMAPE": DoubleType(),
    "TRAIN_TIME": DoubleType(),
    "ASA_APP": DoubleType(),
    "META_APP": DoubleType(),
    "LIFTOFF_APP": DoubleType(),
    "TAPTICA_APP": DoubleType(),
    "REDBOX_APP": DoubleType(),
    "GOOGLE_SEARCH": DoubleType(),
    "BING_SEARCH": DoubleType(),
    "GOOGLE_DISPLAY_OR_OLV": DoubleType(),
    "DV360_DISPLAY_OR_OLV": DoubleType(),
    "BING_DISPLAY": DoubleType(),
    "DV360_DISPLAY": DoubleType(),
    "GOOGLE_DISPLAY": DoubleType(),
    "COMMISSIONS_AFFILIATE": DoubleType(),
    "PLACEMENT_AFFILIATE": DoubleType(),
    "LINKEDIN_SOCIAL": DoubleType(),
    "TWITTER_SOCIAL": DoubleType(),
    "FACEBOOK_SOCIAL": DoubleType(),
    "SNAPCHAT_SOCIAL": DoubleType(),
    "REDDIT_SOCIAL": DoubleType(),
    "BRAND_SPEND": DoubleType(),
    "LOAD_TIMESTAMP": TimestampType(),
}

def _to_upper(name: str) -> str:
    n = re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_")
    return re.sub(r"_+", "_", n).upper()

def _coerce_num_or_str(v):
    if v is None:
        return None
    s = str(v).strip()
    if re.fullmatch(r"[-+]?\d+", s):
        try:
            return int(s)
        except:
            pass
    try:
        return float(s)
    except:
        return s

def _get_latest_run_across_experiments():
    client = MlflowClient()
    latest_ts = None
    latest_run = None
    latest_exp = None
    for exp in client.search_experiments(view_type=ViewType.ACTIVE_ONLY):
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["attribute.start_time desc"],
            max_results=1,
        )
        if runs:
            r = runs[0]
            if latest_ts is None or r.info.start_time > latest_ts:
                latest_ts = r.info.start_time
                latest_run = r
                latest_exp = exp
    if latest_run is None:
        raise ValueError("No MLflow runs found in any active experiment.")
    return latest_run, latest_exp


run, exp = _get_latest_run_across_experiments()

row = {
    "RUN_ID": str(run.info.run_id),
    "LOAD_TIMESTAMP": datetime.utcnow(),
}
row.update({_to_upper(k): _coerce_num_or_str(v) for k, v in (run.data.params or {}).items()})
row.update({_to_upper(k): _coerce_num_or_str(v) for k, v in (run.data.metrics or {}).items()})

spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
df_wide = spark.createDataFrame([row])

df1 = df_wide
for c, dtype in TYPES.items():
    if c not in df1.columns:
        df1 = df1.withColumn(c, lit(None).cast(dtype))
    else:
        df1 = df1.withColumn(c, col(c).cast(dtype))
TARGET_COLS = list(TYPES.keys())
df_out = df1.select(*TARGET_COLS)

write_snowflake_table(df_out, TARGET_TABLE, mode="append")
print(f"Wrote 1 row to {TARGET_TABLE} with columns: {', '.join(TARGET_COLS)}")
