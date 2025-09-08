# Databricks notebook source
# DBTITLE 1,MLFlow Experiment Setup and Configuration
import os
import mlflow
import time
import json
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession
from meridian.analysis import summarizer
from meridian.analysis import visualizer
from meridian.model import model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cairosvg
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

experiment_name = "/Users/mlstudy444@gmail.com/MMM_EXPERIMENT"
mlflow.set_experiment(experiment_name=experiment_name)
run_id = dbutils.jobs.taskValues.get(taskKey = "data_preprocessing", key = "run_id")
client = MlflowClient()

with open("/Workspace/Repos/amitb090425@gmail.com/dj-mlpoc/databricks/config.json","r") as f:
    config = json.load(f)

# COMMAND ----------

# DBTITLE 1,Download and load trained MMM model
file_path = client.download_artifacts(run_id=run_id, path="trained_mmm.pkl")
mmm = model.load_mmm(file_path)

# COMMAND ----------

# DBTITLE 1,Model Diagnostic and Fit Visualizations
model_diagnostics = visualizer.ModelDiagnostics(mmm)
model_fit = visualizer.ModelFit(mmm)

# COMMAND ----------

# DBTITLE 1,Visualizations of Model Diagnostics and Fit
plots = {
    "rhat_boxplot": model_diagnostics.plot_rhat_boxplot(),
    "model_fit": model_fit.plot_model_fit(),
}

# COMMAND ----------

# DBTITLE 1,Log MLFlow Evaluation Images
with mlflow.start_run(run_id=run_id) as run:
    for name, plot in plots.items():
        plot.save("temp.svg")
        with open("temp.svg", "r") as f:
            img = Image.open(BytesIO(cairosvg.svg2png(f.read())))
            fig, ax = plt.subplots()
            ax.imshow(img)
        mlflow.log_figure(fig, f"images/evaluation/{name}.png")
os.remove("temp.svg")

# COMMAND ----------

# DBTITLE 1,Log Predictive Accuracy Metrics with MLFlow
with mlflow.start_run(run_id=run_id) as run:
    metrics = json.loads(model_diagnostics.predictive_accuracy_table().to_json())
    metrics = {key: value for key, value in zip(metrics["metric"].values(),metrics["value"].values())}
    mlflow.log_metrics(metrics)