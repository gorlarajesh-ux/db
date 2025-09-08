# Databricks notebook source
# MAGIC %sql
# MAGIC
# MAGIC -- Create catalog
# MAGIC CREATE CATALOG IF NOT EXISTS djml_poc
# MAGIC MANAGED LOCATION 's3://databricks-2a6ctezney9sfcvcjxjk5a-cloud-storage-bucket-poc/unity-catalog/263400637890400/catalogs/djml_poc/';
# MAGIC
# MAGIC -- Create schema
# MAGIC CREATE SCHEMA IF NOT EXISTS djml_poc.data_model;
# MAGIC
# MAGIC -- Create table
# MAGIC CREATE TABLE IF NOT EXISTS djml_poc.data_model.stage_input_data (
# MAGIC     WEEK_START              DATE,
# MAGIC     ASA_APP                 DECIMAL(18,2),
# MAGIC     BING_DISPLAY            DECIMAL(18,2),
# MAGIC     BING_SEARCH             DECIMAL(18,2),
# MAGIC     DV360_DISPLAY           DECIMAL(18,2),
# MAGIC     DV360_DISPLAY_OR_OLV    DECIMAL(18,2),
# MAGIC     FACEBOOK_SOCIAL         DECIMAL(18,2),
# MAGIC     GOOGLE_DISPLAY          DECIMAL(18,2),
# MAGIC     GOOGLE_DISPLAY_OR_OLV   DECIMAL(18,2),
# MAGIC     GOOGLE_SEARCH           DECIMAL(18,2),
# MAGIC     LINKEDIN_SOCIAL         DECIMAL(18,2),
# MAGIC     META_APP                DECIMAL(18,2),
# MAGIC     REDDIT_SOCIAL           DECIMAL(18,2),
# MAGIC     SNAPCHAT_SOCIAL         DECIMAL(18,2),
# MAGIC     TAPTICA_APP             DECIMAL(18,2),
# MAGIC     TWITTER_SOCIAL          DECIMAL(18,2),
# MAGIC     COMMISSIONS_AFFILIATE   DECIMAL(18,2),
# MAGIC     PLACEMENT_AFFILIATE     DECIMAL(18,2),
# MAGIC     BRAND_SPEND             DECIMAL(18,2),
# MAGIC     REDBOX_APP              DECIMAL(18,2),
# MAGIC     LIFTOFF_APP             DECIMAL(18,2),
# MAGIC     ORDERS                  DECIMAL(38,0),
# MAGIC     NEWS_ANOMALY            DECIMAL(38,0),
# MAGIC     HOLIDAY_FLAG            DECIMAL(38,0),
# MAGIC     SALE_FLAG               DECIMAL(38,0),
# MAGIC     WSJ_EMAILS_TOTAL        DECIMAL(18,2),
# MAGIC     LOAD_DT                 TIMESTAMP,
# MAGIC     SOURCE_FILE             STRING
# MAGIC )
# MAGIC USING delta;
# MAGIC