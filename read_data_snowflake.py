# Databricks notebook source
# MAGIC %md
# MAGIC **Read data from Snowflake (direct read)**

# COMMAND ----------

# DBTITLE 1,sfoptions
sfOptions = {
  "sfURL": "mtc78136.us-east-1.snowflakecomputing.com",
  "sfDatabase": "DATA_MODEL_POC",    
  "sfSchema": "MMM",        
  "sfWarehouse": "DATA_MODEL_WH",  
  "sfRole": "SYSADMIN",           
  "sfUser": "AMITB090425",
  "sfPassword": "Jeepcompass2025"
}


# COMMAND ----------

# DBTITLE 1,Load data from Snowflake for model input
df = (spark.read
      .format("snowflake")
      .options(**sfOptions)
      .option("dbtable", "MODEL_INPUT_DATA")
      .load())
df.show()


# COMMAND ----------

# DBTITLE 1,Save DataFrame to Snowflake as 'test' table
(df.write
   .format("snowflake")
   .options(**sfOptions)
   .option("dbtable", "test")
   .mode("overwrite")                      
   .save())


# COMMAND ----------

# MAGIC %md
# MAGIC **Read data from a Snowflake Catalog**

# COMMAND ----------

# DBTITLE 1,Snowflake Data Load and Save Operations

df_direct = spark.read.table("snowflake_db_catalog.mmm.model_input_data")

df_direct.show(10, truncate=False)
df_direct.printSchema()

(df_direct.write
   .format("snowflake")
   .options(**sfOptions)
   .option("dbtable", "test1")
   .mode("overwrite")                      
   .save())


# COMMAND ----------

# DBTITLE 1,Load Snowflake model input data
# MAGIC %sql
# MAGIC
# MAGIC select * from snowflake_db_catalog.mmm.model_input_data;