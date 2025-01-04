import sys

from google.cloud import storage
import md_utils
# import pickle
from mlflow.tracking.client import MlflowClient
import pandas as pd
import numpy as np
import pytz
import datetime
import argparse
import os
import re
from urllib.parse import urlparse
import urllib
from google.cloud import storage
import os

def validate_gcs_path(gcs_path):
    # Regular expression pattern for GCS path validation
    pattern = r'^gs://md-training-pipeline-bucket-nonprod+/[a-z0-9._-]+$'
    gcs_path = storage.blob.parse_gcs_url(gcs_path)
    gcs_path = gcs_path.strip()
    gcs_path = urllib.parse.urlparse(gcs_path).path  # strip away domain if any
    gcs_path = gcs_path.replace("../", "")
    # parsed_url = urllib.parse.urlparse(gcs_path)

    # Check if the scheme and netloc are present (this means it's an absolute URL)
    if not gcs_path.scheme or not gcs_path.netloc:
        raise ValueError("Invalid URL: URLs must be absolute")

    if gcs_path.startswith('//'):
        raise ValueError("Invalid URL: URLs must be absolute")

    if gcs_path.startswith("gs://md-training-pipeline-bucket-nonprod") and re.match(pattern, gcs_path):
        gcs_path = os.path.basename(gcs_path)
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
    parser.add_argument("--GCS_MODEL_PATH", required=True, type=str)
    parser.add_argument("--MODEL_REGISTRY_NAME", required=True, type=str)
    parser.add_argument("--MLFLOW_EXP_NAME", required=True, type=str)
    parser.add_argument("--LATEST_MODEL_VERSION_PATH", required=True, type=str)
    args = parser.parse_args()

    # # Validate and sanitize the user-provided inpu:
    try:
        args.PIPELINE_ROOT = validate_gcs_path(args.PIPELINE_ROOT)
        args.GCS_MODEL_PATH = validate_gcs_path(args.GCS_MODEL_PATH)
        args.MODEL_REGISTRY_NAME = validate_mlflow_exp_name(args.MODEL_REGISTRY_NAME)
        args.MLFLOW_EXP_NAME = validate_mlflow_exp_name(args.MLFLOW_EXP_NAME)
        args.LATEST_MODEL_VERSION_PATH = validate_gcs_path(args.LATEST_MODEL_VERSION_PATH)
    except Exception as e:
        sys.exit(1)
    return args

args = get_args()
pipeline_root = args.PIPELINE_ROOT
gcs_model_path = args.GCS_MODEL_PATH
model_registry_name = args.MODEL_REGISTRY_NAME 
mlflow_exp_name = args.MLFLOW_EXP_NAME 
latest_model_version_path = args.LATEST_MODEL_VERSION_PATH

# Initialize client
c = MlflowClient()
latest_model_version_path = os.path.basename(latest_model_version_path)
latest_model_version_path = validate_gcs_path(latest_model_version_path)
last_model_ver = int(pd.read_csv(latest_model_version_path)["latest_model_version"])
current_version = last_model_ver + 1
model_versions_data_path = f"{pipeline_root}/markdown_data/model_versions.csv"
model_versions_data_path = validate_gcs_path(model_versions_data_path)
model_versions_data_path = os.path.basename(model_versions_data_path)
model_versioning_feedback_data = pd.read_csv(model_versions_data_path)

# Get the metrics from
now_time = datetime.datetime.now(pytz.timezone('US/Central'))
model_ver_dict = {
    "filename": [f"mlflow_md_v{str(current_version)}"],
    "use_model": [0],
    "version": [current_version],
    "upload_date": [now_time.date().strftime("%Y-%m-%d")],
    "upload_time": [now_time.time().strftime("%H:%M:%S")],
    "end_date": np.nan,
    "end_time": np.nan,
    "model_stage": np.nan,
    "comments": "Model registered in Element MLFlow"
}
model_ver = pd.DataFrame(model_ver_dict)

pdf_updated_version = pd.concat([model_ver, model_versioning_feedback_data], axis=0, ignore_index=True).reset_index(drop=True)

model_versions_path = f"{pipeline_root}/markdown_data/model_versions.csv"
model_versions_path = validate_gcs_path(model_versions_path)
pdf_updated_version.to_csv(model_versions_path, index=False)
c.transition_model_version_stage(model_registry_name, str(current_version), "Production")
