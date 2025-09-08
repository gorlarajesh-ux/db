# Databricks notebook source
# DBTITLE 1,Filter latest data based on max source file
from utils.snowflake_utils import read_sf_table,  write_snowflake_table

from pyspark.sql import functions as F

df = read_sf_table("snowflake_db_catalog.mmm.model_input_data")
# df.show(5, truncate=False)

max_source_file = df.agg(F.max("source_file")).collect()[0][0]

df_latest = df.filter(df["source_file"] == max_source_file)

df_latest.show(5, truncate=False)



# COMMAND ----------

write_snowflake_table(df_latest, "MMM.DBWRITE_MODEL_INPUT", mode="overwrite")