# Databricks notebook source
# MAGIC %md
# MAGIC **Pick the latest file from S3 and save as a table**

# COMMAND ----------

# DBTITLE 1,Retrieve Latest File from S3 and Create Table
-- Pick the latest file from S3 and save as a table


folders = [f.name.strip("/") for f in dbutils.fs.ls("s3a://dj-poc/") if f.isDir()]
latest_folder = max(folders) if folders else None

if not latest_folder:
    raise RuntimeError("No dated folders found under s3a://dj-poc/")

latest_path = f"s3a://dj-poc/{latest_folder}/poc_dataset_{latest_folder}.csv"
print("Latest file:", latest_path)

df_cloud = spark.read.csv(latest_path, header=True, inferSchema=True)
df_cloud.show(5)


# COMMAND ----------

# DBTITLE 1,Save Delta Table as Input Data in WSJ Model
df_cloud.write.format("delta").mode("overwrite").saveAsTable("mmm_model.wsj_model.input_data")
# df_cloud.write.format("delta").mode("append").saveAsTable("mmm_model.wsj_model.input_data")


# COMMAND ----------

# DBTITLE 1,Examine All Data in Input Data Table
# MAGIC %sql
# MAGIC
# MAGIC -- select count(*) from mmm_model.wsj_model.input_data;
# MAGIC select * from mmm_model.wsj_model.input_data;

# COMMAND ----------

# MAGIC %md
# MAGIC **Write data to Delta table**

# COMMAND ----------

# DBTITLE 1,Save DataFrame to S3 as CSV
# df_cloud.write.csv(
#     "s3a://dj-poc/test.csv",
#     header=True,
#     mode="overwrite"
# )


# COMMAND ----------

# MAGIC %md
# MAGIC **Upload a file into DBFS and save as a table**

# COMMAND ----------

# DBTITLE 1,Upload CSV File to DBFS and Create Delta Table
-- Upload a file into DBFS and save as a table

df = spark.read.csv("/Volumes/mmm_model/wsj_model/input_data/poc_dataset.csv", header=True, inferSchema=True)

df.write.format("delta").mode("overwrite").saveAsTable("mmm_model.wsj_model.input_data")