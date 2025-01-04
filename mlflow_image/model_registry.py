import sys

from google.cloud import storage
import os
import argparse
import pandas as pd
import numpy as np
import md_utils
# from tempfile import TemporaryFile
from mlflow.tracking.client import MlflowClient
from collections import namedtuple
import mlflow
import joblib
import json
import shutil
from plotly.offline import plot
import plotnine 
import argparse 
import yaml
import os
import re
import urllib.parse

def validate_gcs_path(gcs_path):
    # Regular expression pattern for GCS path validation
    pattern = r'^gs://md-training-pipeline-bucket-nonprod+/[a-z0-9._-]+$'
    gcs_path = storage.blob.parse_gcs_url(gcs_path)
    gcs_path = gcs_path.strip()
    gcs_path = os.path.basename(gcs_path)

    gcs_path = urllib.parse.urlparse(gcs_path).path  # strip away domain if any
    gcs_path = gcs_path.replace("../", "")

    parsed_url = urllib.parse.urlparse(gcs_path)

    # Check if the scheme and netloc are present (this means it's an absolute URL)
    if not parsed_url.scheme or not parsed_url.netloc:
        raise ValueError("Invalid URL: URLs must be absolute")

    if not gcs_path.startswith('//'):
        raise ValueError("Invalid URL: URLs must be absolute")

    if gcs_path.startswith("gs://md-training-pipeline-bucket-nonprod") and re.match(pattern, gcs_path):
        return gcs_path
    else:
        raise ValueError("Invalid GCS path provided")

def validate_mlflow_exp_name(exp_name):
    # Validate the MLflow experiment name to prevent SSRF
    pattern = r'^[a-zA-Z0-9_-]+$'
    exp_name = exp_name.strip()
    return re.match(pattern, exp_name) is not None

def get_args():
    # Import arguments to local variables
    parser = argparse.ArgumentParser()
    parser.add_argument("--PIPELINE_ROOT", required=True, type=str)
    parser.add_argument("--ENV", required=True, type=str)
    parser.add_argument("--GCS_MODEL_PATH", required=True, type=str)
    parser.add_argument("--LATEST_MODEL_VERSION_PATH", required=True, type=str)
    parser.add_argument("--LATEST_MD_METRICS_PATH", required=True, type=str)
    parser.add_argument("--MODEL_REGISTRY_NAME", required=True, type=str)
    parser.add_argument("--MLFLOW_EXP_NAME", required=True, type=str)
    # parser.add_argument("--PREDICTION_INPUT", required=True, type=str)
    # parser.add_argument("--MERGED_PRED_CAT_INPUT", required=True, type=str)
    # parser.add_argument("--TXT_INPUT", required=True, type=str)
    # parser.add_argument("--RUN_CONFIGURATIONS_INPUT", required=True, type=str)
    parser.add_argument("--PARAM_TUNING", required=True, type=str)
    # parser.add_argument("--DATA_FRACTION", required=True, type=str)
    # parser.add_argument("--COVARIATES", required=True, type=str)
    # parser.add_argument("--RESPONSE", required=True, type=float)
    # parser.add_argument("--ROUNDS", required=True, type=str)
    # parser.add_argument("--TOLERANCE", required=True, type=str)
    # parser.add_argument("--ENCODE_FEATURES", required=True, type=str)
    # parser.add_argument("--TRAIN_PARAMS", required=True, type=str)
    args = parser.parse_args()

    # Validate GCS paths
    try:
        args.PIPELINE_ROOT = validate_gcs_path(args.PIPELINE_ROOT)
        args.GCS_MODEL_PATH = validate_gcs_path(args.GCS_MODEL_PATH)
        args.LATEST_MODEL_VERSION_PATH = validate_gcs_path(args.LATEST_MODEL_VERSION_PATH)
        args.LATEST_MD_METRICS_PATH = validate_gcs_path(args.LATEST_MD_METRICS_PATH)
        args.MODEL_REGISTRY_NAME = validate_mlflow_exp_name(args.MODEL_REGISTRY_NAME)
        args.MLFLOW_EXP_NAME = validate_mlflow_exp_name(args.MLFLOW_EXP_NAME)
    except Exception as e:
        sys.exit(1)

    return args


args = get_args()
pipeline_root = args.PIPELINE_ROOT
pipeline_root = validate_gcs_path(pipeline_root)

# pipeline_root = os.path.basename(pipeline_root)
env = args.ENV
gcs_model_path = args.GCS_MODEL_PATH
gcs_model_path = validate_gcs_path(gcs_model_path)
# gcs_model_path = os.path.basename(gcs_model_path)
latest_model_version_path = args.LATEST_MODEL_VERSION_PATH
latest_model_version_path = os.path.basename(latest_model_version_path)
latest_model_version_path = validate_gcs_path(latest_model_version_path)

latest_md_metrics_path = args.LATEST_MD_METRICS_PATH
latest_md_metrics_path = validate_gcs_path(latest_md_metrics_path)
# latest_md_metrics_path = os.path.basename(latest_md_metrics_path)
model_registry_name = args.MODEL_REGISTRY_NAME 
mlflow_exp_name = args.MLFLOW_EXP_NAME 
# prediction_input = args.PREDICTION_INPUT
# merged_pred_cat_input = args.MERGED_PRED_CAT_INPUT
# txt_input = args.TXT_INPUT
# run_configurations_input = yaml.safe_load(args.RUN_CONFIGURATIONS_INPUT)
param_tuning = args.PARAM_TUNING
# data_fraction = float(args.DATA_FRACTION)
# covariates = list(args.COVARIATES.replace(' ', '').split(','))
# response = list(args.RESPONSE.replace(' ', '').split(',')) 
# rounds = int(args.ROUNDS)
# tolerance = float(args.TOLERANCE)
# encode_features = list(args.encode_features.replace(' ', '').split(',')) 
# train_params = yaml.safe_load(args.TRAIN_PARAMS)

last_model_ver = int(pd.read_csv(latest_model_version_path)["latest_model_version"])
current_version = last_model_ver + 1

# pdf_prediction = pd.read_parquet(f"{pipeline_root}/{env}/latest_test_prediction.gzip")
# pdf_merged_pred_cat = pd.read_parquet(f"{pipeline_root}/{env}/latest_merged_pred_cat.gzip")

# shutil.copy2(txt_input.path+".txt", "post_analysis.txt")

#------------------------
# Parameters
#------------------------
params = {
    "version": current_version,
    "param_tuning": param_tuning,
    # "data_fraction": data_fraction,
    # "train_period": run_configurations_input[dynamic_config_key]["train_period"],
    # "test_period": run_configurations_input[dynamic_config_key]["test_period"],
    # "response": response,
    # "rounds": rounds,
    # "tolerance": tolerance,
    # "encode_features": encode_features,
    # "train_params": train_params,
}
#------------------------
# model
#------------------------
blob = storage.blob.Blob.from_string(gcs_model_path, client=storage.Client())
blob.download_to_filename("latest_model_output")
with open("latest_model_output", "rb") as file:
    markdown_model = joblib.load(file)
# #------------------------
# # Plots
# #------------------------
# analysis_obj = md_utils.PostAnalysis()
# # feature importance
# feature_importance = pd.DataFrame()
# feature_importance["features"] = covariates
# feature_importance["importance"] = markdown_model.steps[1][1].feature_importances_
# feat_imp_fig = analysis_obj.feat_imp_ggplot(data=feature_importance, x_val="features", y_val="importance")
# # Median Absolute Percentage Error
# median_ape = round(np.median(pdf_merged_pred_cat["ape"]), 2)
# text = f'Median APE: {median_ape}'
# ape_cdf_fig = analysis_obj.ape_cdf_ggplot(data=pdf_merged_pred_cat, text=text, x_val="ape")
# # Median Absolute Percentage Error Per Category
# cat_ape_cdf_fig = analysis_obj.cat_ape_cdf_ggplot(data=pdf_merged_pred_cat, x_val="ape")
# # Units Error Density
# bias = round(np.mean(pdf_prediction["units_error"]),1)
# text = f'Bias: {bias}'
# units_error_fig = analysis_obj.units_error_ggplot(data=pdf_prediction, text=text, x_min_lim=-250, x_max_lim=250, x_axis=500, y_axis=0.02, x_val="units_error")
# # Units Error Limited Axis Density
# bias = round(np.mean(pdf_prediction.query("units_error > -100 & units_error < 100")["units_error"]),1)
# text = f'Bias: {bias}'
# units_error_lim_fig = analysis_obj.units_error_ggplot(data=pdf_prediction, text=text, x_min_lim=-100, x_max_lim=100, x_axis=50, y_axis=0.02, x_val="units_error")
# # Units Error vs Cumulated Sales
# error_vs_true_fig = analysis_obj.error_true_pred_ggplot(data=pdf_prediction, text='', x_val="cum_sale", y_val="units_error")
# # Units Error vs Predicted Sales
# error_vs_pred_fig = analysis_obj.error_true_pred_ggplot(data=pdf_prediction, text='', x_val="predicted_sale", y_val="units_error")
# # Predicted Sales vs Cumulated Sales
# pred_vs_true_fig = analysis_obj.error_true_pred_ggplot(data=pdf_prediction, text='', x_val="cum_sale", y_val="predicted_sale")

# Initialize client
c = MlflowClient()

mlflow.set_experiment(mlflow_exp_name)

experiment_id = c.get_experiment_by_name(mlflow_exp_name).experiment_id

# Launching Multiple Runs in One Program.This is easy to do because the ActiveRun object returned by mlflow.start_run() is a
# Python context manager. You can “scope” each run to just one block of code as follows:
with mlflow.start_run(experiment_id=experiment_id) as run:
    # Get run id
    run_id = run.info.run_uuid

    mlflow.sklearn.log_model(
        sk_model=markdown_model,
        artifact_path="md_model",
        registered_model_name=model_registry_name,
    )

    # current_version = int(c.get_latest_versions(model_registry_name, stages=["None"])[0].version)

    c.set_tag(
        run_id,
        "mlflow.note.content",
        "This is experiment for testing"
    )

    # Define customer tag
    tags = {
        "Application": "Markdown",
        "tags_model_version": f"{str(current_version)}",
        "tags_run_id": f"{run_id}"
    }

    # Set Tag
    mlflow.set_tags(tags)

    # Log python re details
    # mlflow.log_artifact("post_analysis.txt")

    # logging params
    mlflow.log_param("run_id", run_id)
    mlflow.log_params(params)

    # log metrics
    blob = storage.blob.Blob.from_string(latest_md_metrics_path, client=storage.Client())
    md_metrics_input = json.loads(blob.download_as_string())
    mlflow.log_metrics(md_metrics_input)

    # log artifact
    # pd.DataFrame.from_dict(category_error_input).to_csv("category_error.csv", index=False),
    # mlflow.log_artifact("category_error.csv")

    # with open(markdown_model_output.path, "wb") as file:
    #     joblib.dump(markdown_model, file)

    # mlflow.log_figure(feat_imp_fig.draw(), "feat_imp.png")
    # mlflow.log_figure(ape_cdf_fig.draw(), "ape_cdf.png")
    # mlflow.log_figure(cat_ape_cdf_fig.draw(), "cat_ape_cdf.png")
    # mlflow.log_figure(units_error_fig.draw(), "units_error.png")
    # mlflow.log_figure(units_error_lim_fig.draw(), "units_error_dist_lim.png")
    # mlflow.log_figure(error_vs_true_fig.draw(), "error_vs_true.png")
    # mlflow.log_figure(error_vs_pred_fig.draw(), "error_vs_pred.png")
    # mlflow.log_figure(pred_vs_true_fig.draw(), "pred_vs_true.png")

