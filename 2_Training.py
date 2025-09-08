# Databricks notebook source
# DBTITLE 1,MLflow Experiment Setup and Configuration
import os
import mlflow
import time
import json
from mlflow.tracking import MlflowClient
from meridian import constants
from meridian.data import data_frame_input_data_builder
from meridian.data import test_utils
from meridian.model import model
from meridian.model import prior_distribution
from meridian.model import spec
import numpy as np
import pandas as pd
from psutil import virtual_memory
import tensorflow as tf
import tensorflow_probability as tfp

experiment_name = "/Users/mlstudy444@gmail.com/MMM_EXPERIMENT"
mlflow.set_experiment(experiment_name=experiment_name)
dbutils.widgets.text("mlflow_run_id", "","MLflow Run ID")
run_id = dbutils.jobs.taskValues.get(taskKey = "data_preprocessing", key = "run_id")
client = MlflowClient()

with open("/Workspace/Repos/amitb090425@gmail.com/dj-mlpoc/databricks/config.json","r") as f:
    config = json.load(f)

# COMMAND ----------

# DBTITLE 1,Check Available System Resources
ram_gb = virtual_memory().total / 1e9
print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))
print(
    'Num GPUs Available: ',
    len(tf.config.experimental.list_physical_devices('GPU')),
)
print(
    'Num CPUs Available: ',
    len(tf.config.experimental.list_physical_devices('CPU')),
)

# COMMAND ----------

# DBTITLE 1,Load Artifact Columns and Training Data
artifact_uri = mlflow.get_run(run_id=run_id).info.artifact_uri
cols = mlflow.artifacts.load_dict(artifact_uri + "/cols.json")

train_path = client.download_artifacts(run_id, "processed_data/train.csv")
df = pd.read_csv(train_path)

# COMMAND ----------

# DBTITLE 1,Control setup
builder = data_frame_input_data_builder.DataFrameInputDataBuilder(
    kpi_type = cols["kpi_type"],
    default_kpi_column = cols["default_kpi_column"],
    default_time_column = cols["default_time_column"],
    default_media_time_column = cols["default_time_column"],
    )
    
builder = (
    builder
    .with_kpi(
        df = df,
        kpi_col = cols["default_kpi_column"],
        time_col = cols["default_time_column"],
    )
    .with_media(
        df = df,
        media_cols = cols["media_cols"],
        media_spend_cols = cols["media_spend_cols"],
        media_channels = cols["media_cols"],
        )
    .with_non_media_treatments(
        df = df,
        non_media_treatment_cols = cols["non_media_treatment_cols"],
        )
    .with_controls(
        df = df,
        control_cols = cols["control_cols"],
        )
)

data = builder.build()

# COMMAND ----------

# DBTITLE 1,Model Prior Distribution Initialization
model_params = config["model_params"]

roi_mu = model_params["roi_mu"]
roi_sigma = model_params["roi_sigma"]
prior = prior_distribution.PriorDistribution(
    roi_m=tfp.distributions.LogNormal(roi_mu, roi_sigma, name=constants.ROI_M)
)
model_spec = spec.ModelSpec(prior=prior)

mmm = model.Meridian(input_data=data, model_spec=model_spec)

# COMMAND ----------

# DBTITLE 1,Run MCMC Sampling Process
start_time = time.perf_counter()
mmm.sample_prior(model_params["sample_prior"])
mmm.sample_posterior(
    n_chains=model_params["n_chains"],
    n_adapt=model_params["n_adapt"],
    n_burnin=model_params["n_burnin"],
    n_keep=model_params["n_keep"],
    seed=model_params["seed"],
)
end_time = time.perf_counter()

# COMMAND ----------

# DBTITLE 1,Log Model Parameters, Metrics, and Artifact in MLflow
with mlflow.start_run(run_id=run_id) as run:
    mlflow.log_params(model_params)
    mlflow.log_metric("train_time", (end_time - start_time)//60)
    file_path = "./trained_mmm.pkl"
    model.save_mmm(mmm, file_path)
    mlflow.log_artifact(file_path)
os.remove(file_path)