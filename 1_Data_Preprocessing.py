# Databricks notebook source
# DBTITLE 1,Importing Modules and Setting Up Experiment in MLFlow
import os
import pandas as pd
import mlflow
import json
from pyspark.sql import SparkSession
from utils.snowflake_utils import read_sf_table
from pyspark.sql import functions as F

df_read_sf = read_sf_table("MODEL_INPUT_DATA")

# COMMAND ----------

# DBTITLE 1,Setting Up Snowflake Data Model Stage Input Data
max_source_file = df_read_sf.agg(F.max("source_file")).collect()[0][0]

df_delta = df_read_sf.filter(df_read_sf["source_file"] == max_source_file)

df_delta.write.format("delta").mode("overwrite").saveAsTable("djml_poc.data_model.stage_input_data")

# COMMAND ----------

experiment_name = "/Workspace/Users/mlstudy444@gmail.com/MMM_EXPERIMENT"
mlflow.set_experiment(experiment_name=experiment_name)

with open(
    "/Workspace/Repos/amitb090425@gmail.com/dj-mlpoc/databricks/config.json", "r"
) as f:
    config = json.load(f)

# COMMAND ----------

# DBTITLE 1,Media and Time Columns Setup for Data Analysis
kpi_type = "non_revenue"

default_kpi_column = "orders"

default_time_column = "week_start"

media_cols = [
    "asa_app",
    "bing_display",
    "bing_search",
    "dv360_display",
    "dv360_display_or_olv",
    "facebook_social",
    "google_display",
    "google_display_or_olv",
    "google_search",
    "linkedin_social",
    "meta_app",
    "reddit_social",
    "snapchat_social",
    "taptica_app",
    "twitter_social",
    "commissions_affiliate",
    "placement_affiliate",
    "brand_spend",
    "redbox_app",
    "liftoff_app",
    ]

media_spend_cols = [f"{col}_spend" for col in media_cols]

non_media_treatment_cols = [
    "sale_flag",
    "wsj_emails_total",
    ]

control_cols = [
    "news_anomaly",
    "holiday_flag",
]

df = spark.table("djml_poc.data_model.stage_input_data").toPandas()
df.columns = df.columns.str.lower()

df = df.sort_values("week_start",ascending=True)
df["week_start"] = df["week_start"].astype(str)

cols = {
    "default_time_column":  default_time_column,
    "media_cols": media_cols,
    "media_spend_cols": media_spend_cols,
    "non_media_treatment_cols": non_media_treatment_cols,
    "control_cols": control_cols,
    "kpi_type": kpi_type,
    "default_kpi_column": default_kpi_column,
}

# COMMAND ----------

# DBTITLE 1,Spend Calculations
with mlflow.start_run() as run:
    run_id = run.info.run_id
    weights = config["weights"]
    normalized_weights = {col: weight/len(weights) for col, weight in weights.items()}
    mlflow.log_params(normalized_weights)
    mlflow.log_dict(cols, "cols.json")

df["brand_spend"] = df["brand_spend"].astype(float)

for col in media_cols:
  df[f"{col}_spend"] = df["brand_spend"]*normalized_weights[col]

# COMMAND ----------

# DBTITLE 1,Process Data Splitting and MLFlow Logging
df = df[[default_time_column]+media_cols+media_spend_cols+non_media_treatment_cols+control_cols+[default_kpi_column]]

df_train = df.head(int(len(df)*0.8))
df_train.to_csv("train.csv", index=False)
df_test = df.drop(df_train.index)
df_test.to_csv("test.csv", index=False)

with mlflow.start_run(run_id=run_id) as run:
    mlflow.log_artifact("train.csv", artifact_path="processed_data")
    mlflow.log_artifact("test.csv", artifact_path="processed_data")

os.remove("test.csv")
os.remove("train.csv")

# COMMAND ----------

# DBTITLE 1,Setting Run ID in Databricks Job Values
dbutils.jobs.taskValues.set(key = "run_id", value = run_id)