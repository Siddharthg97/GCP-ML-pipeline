import argparse
import pandas as pd
import os
import mlflow
from mlflow.tracking.client import MlflowClient
import md_utils
import re

def validate_gcs_path(path):
    # Validate the GCS path to prevent SSRF
    pattern = r'^gs://[a-z0-9._-]+/[a-z0-9._-]+$'
    return re.match(pattern, path) is not None

def validate_mlflow_exp_name(exp_name):
    # Validate the MLflow experiment name to prevent SSRF
    pattern = r'^[a-zA-Z0-9_-]+$'
    return re.match(pattern, exp_name) is not None
def get_args():
    # Import arguments to local variables
    parser = argparse.ArgumentParser()
    parser.add_argument("--GCS_MODEL_PATH", required=True, type=str) 
    parser.add_argument("--MODEL_REGISTRY_NAME", required=True, type=str)
    parser.add_argument("--MLFLOW_EXP_NAME", required=True, type=str)
    parser.add_argument("--LATEST_MODEL_VERSION_PATH", required=True, type=str)
    args = parser.parse_args()

    args.GCS_MODEL_PATH = validate_gcs_path(args.GCS_MODEL_PATH)
    args.LATEST_MODEL_VERSION_PATH = validate_gcs_path(args.LATEST_MODEL_VERSION_PATH)
    args.MLFLOW_EXP_NAME = validate_mlflow_exp_name(args.MLFLOW_EXP_NAME)
    args.MODEL_REGISTRY_NAME = validate_mlflow_exp_name(args.MODEL_REGISTRY_NAME)

    return args

args = get_args()
GCS_MODEL_PATH = args.GCS_MODEL_PATH
MODEL_REGISTRY_NAME = args.MODEL_REGISTRY_NAME 
MLFLOW_EXP_NAME = args.MLFLOW_EXP_NAME 
LATEST_MODEL_VERSION_PATH = args.LATEST_MODEL_VERSION_PATH

# Initialize client
c = MlflowClient()

mlflow.set_experiment(MLFLOW_EXP_NAME)

try:
    last_model_version = c.get_latest_versions(MODEL_REGISTRY_NAME, stages=["Production"])[0].version
except:
    print("last model version of Production model stage is not found")
    last_model_version = 0

print("last_model_version:", last_model_version)

blob = last_model_version.to(LATEST_MODEL_VERSION_PATH, client=storage.Client())
blob.upload_from_filename(LATEST_MODEL_VERSION_PATH)
df_last_model_version = pd.DataFrame([last_model_version], columns=['latest_model_version'])
df_last_model_version.to_csv(LATEST_MODEL_VERSION_PATH)

