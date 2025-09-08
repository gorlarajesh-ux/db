# Databricks notebook source
# DBTITLE 1,MLflow Experiment Setup with Configuration Loading
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
experiment_name = "/Users/mlstudy444@gmail.com/MMM_EXPERIMENT"
import cairosvg
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

mlflow.set_experiment(experiment_name=experiment_name)
run_id = dbutils.jobs.taskValues.get(taskKey = "data_preprocessing", key = "run_id")
client = MlflowClient()

with open("/Workspace/Repos/amitb090425@gmail.com/dj-mlpoc/databricks/config.json","r") as f:
    config = json.load(f)

# COMMAND ----------

# DBTITLE 1,Download and load MMM model and training data
file_path = client.download_artifacts(run_id=run_id, path="trained_mmm.pkl")
mmm = model.load_mmm(file_path)

train_path = client.download_artifacts(run_id, "processed_data/train.csv")
df = pd.read_csv(train_path)

# COMMAND ----------

# DBTITLE 1,Media Mix Modeling Visualizations
mmm_summarizer = summarizer.Summarizer(mmm)
media_summary = visualizer.MediaSummary(meridian=mmm)
media_effects = visualizer.MediaEffects(meridian=mmm)

# COMMAND ----------

# DBTITLE 1,MLflow Model Results Summary Generation
with mlflow.start_run(run_id=run_id) as run:
    start_date = df["week_start"].min()
    end_date = df["week_start"].max()
    mmm_summarizer.output_model_results_summary('temp.html', "./", start_date, end_date)
    mlflow.log_artifact("temp.html", artifact_path="html")
os.remove("temp.html")

# COMMAND ----------

# DBTITLE 1,Media Summary Visualization Plots
plots = {
    "channel_contribution_area_chart":   media_summary.plot_channel_contribution_area_chart(),
    "channel_contribution_bump_chart": media_summary.plot_channel_contribution_bump_chart(),
    "contribution_waterfall_chart": media_summary.plot_contribution_waterfall_chart(),
    "contribution_waterfall_chart": media_summary.plot_contribution_waterfall_chart(),
    "contribution_pie_chart": media_summary.plot_contribution_pie_chart(),
    "spend_vs_contribution": media_summary.plot_spend_vs_contribution(),
    "roi_bar_chart": media_summary.plot_roi_bar_chart(),
    "cpik": media_summary.plot_cpik(),
    "roi_vs_effectiveness": media_summary.plot_roi_vs_effectiveness(disable_size=True),
    "roi_vs_mroi": media_summary.plot_roi_vs_mroi(),
    "response_curves": media_effects.plot_response_curves(),
    "adstock_decay": media_effects.plot_adstock_decay(),
}

# COMMAND ----------

# DBTITLE 1,MLflow Image Logging in Reporting Folder
with mlflow.start_run(run_id=run_id) as run:
    for name, plot in plots.items():
        plot.save("temp.svg")
        with open("temp.svg", "r") as f:
            img = Image.open(BytesIO(cairosvg.svg2png(f.read())))
            fig, ax = plt.subplots()
            ax.imshow(img)
        mlflow.log_figure(fig, f"images/reporting/{name}.png")
os.remove("temp.svg")