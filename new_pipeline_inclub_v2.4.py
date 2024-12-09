import sys
import os
from pathlib import Path
from pipeline_utils import pipeline_utils
import argparse
import datetime
import pprint
# Imports for vertex pipeline
from google.cloud import aiplatform
# import google_cloud_pipeline_components
# from google.cloud import aiplatform as gcc_aip
from google_cloud_pipeline_components.v1.custom_job import create_custom_training_job_from_component

from kfp.v2 import compiler
from kfp.v2.dsl import (
    Artifact,
    component,
    Condition,
    pipeline,
    Input,
    Output,
    Metrics,
    Model,
    Dataset,
    InputPath,
    OutputPath,
)
import kfp.components as comp
import kfp.dsl as dsl
from typing import NamedTuple, Dict, List, Tuple

import warnings

warnings.filterwarnings("ignore")


try:
    args = pipeline_utils.MarkdownArgs().get_args()
except:
    parser = argparse.ArgumentParser()
    parser.add_argument("--COMMIT_ID", required=True, type=str)
    parser.add_argument("--BRANCH", required=True, type=str)
    parser.add_argument("--is_prod", required=False, type=lambda x: (str(x).lower() == 'true'))
    sys.args = [
        "--COMMIT_ID", "1234",
        "--BRANCH", "dev",
        "--is_prod", False,
    ]
    args = parser.parse_args(sys.args)

PARAMS = pipeline_utils.YamlImport("settings.yml").yaml_import()

# Env flag for indentifying what env is used. valid values are: "dev" "stage" "prod"
BRANCH_ID = args.BRANCH
is_prod = args.is_prod

if BRANCH_ID == "stage" and is_prod == True:
    BRANCH_ID = "prod"

ENV = BRANCH_ID

IC_PARAM = PARAMS["envs"][ENV]["inclub"]
PARAM = PARAMS["envs"][ENV]
############################
# Model Condition Parameters
# ###########################
MODE = IC_PARAM["MODE"]
DYNAMIC_CONFIG = IC_PARAM["DYNAMIC_CONFIG"]
DATA_FRACTION = float(IC_PARAM["DATA_FRACTION"])
PRODUCTION_RUN = IC_PARAM["PRODUCTION_RUN"]
RUN_FREQUENCY = IC_PARAM["RUN_FREQUENCY"]
RUN_MLFLOW_EXP = IC_PARAM["RUN_MLFLOW_EXP"]
CATEGORY_UNIVERSE = IC_PARAM['CATEGORY_UNIVERSE']

NETWORK = PARAM['VPC_NETWORK']
# GCP Project id, service account, region, and docker images.
PROJECT_ID = PARAM['PROJECT_ID']
SERVICE_ACCOUNT = PARAM['SERVICE_ACCOUNT']
REGION = PARAM['REGION']
DATA_STORAGE_GCS_URI=PARAM['DATA_STORAGE_GCS_URI']

# Docker images
BASE_IMAGE = PARAM['BASE_IMAGE']
MLFLOW_IMAGE = PARAM['MLFLOW_IMAGE']

# Training Pipeline.
PIPELINE_ROOT = PARAM['PIPELINE_ROOT']
PIPELINE_NAME = IC_PARAM['PIPELINE_NAME']
PIPELINE_JSON = IC_PARAM['PIPELINE_JSON']
TMP_PIPELINE_JSON = os.path.join("/tmp", PIPELINE_JSON)

TRAIN_TABLE_NAME = IC_PARAM['TRAIN_TABLE_NAME']
VAL_TABLE_NAME = IC_PARAM['VAL_TABLE_NAME']
TEST_TABLE_NAME = IC_PARAM['TEST_TABLE_NAME']
LOGS_EVAL_TABLE_NAME = IC_PARAM['LOGS_EVAL_TABLE_NAME']
MANUAL_EVAL_TABLE_NAME = IC_PARAM['MANUAL_EVAL_TABLE_NAME']


#Elasticity table name
ELASTICITY_OUTPUT_PATH=IC_PARAM["ELASTICITY_OUTPUT_PATH"]
        
#Eval output logs
EVAL_PREDICTION_OUTPUT_PATH_LOGS = IC_PARAM['EVAL_PREDICTION_OUTPUT_PATH_LOGS']
EVAL_CAT_OUTPUT_PATH_LOGS = IC_PARAM['EVAL_CAT_OUTPUT_PATH_LOGS']
EVAL_NUM_WEEK_OUTPUT_PATH_LOGS = IC_PARAM['EVAL_NUM_WEEK_OUTPUT_PATH_LOGS']
EVAL_OVERALL_OUTPUT_PATH_LOGS = IC_PARAM['EVAL_OVERALL_OUTPUT_PATH_LOGS']

#Eval output manual
EVAL_PREDICTION_OUTPUT_PATH_MANUAL = IC_PARAM['EVAL_PREDICTION_OUTPUT_PATH_MANUAL']
EVAL_CAT_OUTPUT_PATH_MANUAL = IC_PARAM['EVAL_CAT_OUTPUT_PATH_MANUAL']
EVAL_NUM_WEEK_OUTPUT_PATH_MANUAL = IC_PARAM['EVAL_NUM_WEEK_OUTPUT_PATH_MANUAL']
EVAL_OVERALL_OUTPUT_PATH_MANUAL = IC_PARAM['EVAL_OVERALL_OUTPUT_PATH_MANUAL']

MARKDOWN_MODEL_NAME = IC_PARAM["MARKDOWN_MODEL_NAME"]
MODEL_REGISTRY_NAME = IC_PARAM['MODEL_REGISTRY_NAME']
MLFLOW_EXP_NAME = IC_PARAM["MLFLOW_EXP_NAME"]

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

LATEST_MD_PARAMS_PATH = IC_PARAM['LATEST_MD_PARAMS_PATH']
LATEST_MD_MODEL_PATH = IC_PARAM['LATEST_MD_MODEL_PATH']
LATEST_PIPELINE_PATH = IC_PARAM['LATEST_PIPELINE_PATH']
LATEST_MD_METRICS_PATH = IC_PARAM['LATEST_MD_METRICS_PATH']
LATEST_MD_METRICS_PATH_EVAL_LOGS = IC_PARAM['LATEST_MD_METRICS_PATH_EVAL_LOGS']
LATEST_MD_METRICS_PATH_EVAL_MANUAL = IC_PARAM['LATEST_MD_METRICS_PATH_EVAL_MANUAL']
RUN_PIPELINE = IC_PARAM['RUN_PIPELINE']

print(f"""
====**** Pipeline Parameters ****====
\nENV: {ENV}, 
\nNETWORK: {NETWORK},
\nPROJECT_ID: {PROJECT_ID}, 
\nBASE_IMAGE: {BASE_IMAGE}, 
\nMLFLOW_IMAGE: {MLFLOW_IMAGE},
\nPIPELINE_ROOT: {PIPELINE_ROOT}, 
\nPIPELINE_NAME: {PIPELINE_NAME}, 
\nPIPELINE_JSON: {PIPELINE_JSON},
\nMARKDOWN_MODEL_NAME: {MARKDOWN_MODEL_NAME},
\nMODEL_REGISTRY_NAME: {MODEL_REGISTRY_NAME},
\nMLFLOW_EXP_NAME: {MLFLOW_EXP_NAME},
\nLATEST_MD_PARAMS_PATH: {LATEST_MD_PARAMS_PATH},
\nLATEST_MD_METRICS_PATH: {LATEST_MD_METRICS_PATH},
""")

print(f""" train table name is {TRAIN_TABLE_NAME} """)
print(f""" output table name is {ELASTICITY_OUTPUT_PATH} """)

# Added Category "42, 44, 57" on Jan 24, Based on Manda's Email on Jan 19
CATEGORY_FEATURES = IC_PARAM['CATEGORY_FEATURES']
############################
# Model Parameters
# ###########################
# constrained_features
CONSTRAINED_FEATURES = tuple(IC_PARAM['CONSTRAINED_FEATURES'])
# covariates
COVARIATES = IC_PARAM['COVARIATES']
# encode_features
ENCODE_FEATURES = IC_PARAM['ENCODE_FEATURES']
# response
RESPONSE = IC_PARAM['RESPONSE']

########################################################
# Hyperparameters for XGboost - Manual Training
# #######################################################
# train_params
TRAIN_PARAMS = IC_PARAM['TRAIN_PARAMS']

print(f"Train Hyper-Parameters: {TRAIN_PARAMS}")

# Early Stopping
# tolerance
TOLERANCE = IC_PARAM['TOLERANCE']
# rounds
ROUNDS = IC_PARAM['ROUNDS']
# param_tuning
PARAM_TUNING = IC_PARAM['PARAM_TUNING']  # ['manual', 'auto']
# max_evals
MAX_EVALS = IC_PARAM['MAX_EVALS']
# model verbose
MODEL_VERBOSE = IC_PARAM['MODEL_VERBOSE']  # True

CONFIG_HASHMAP = {
    "MODE": MODE,
    "DYNAMIC_CONFIG": DYNAMIC_CONFIG,
    "DATA_FRACTION": DATA_FRACTION,
    "PRODUCTION_RUN": PRODUCTION_RUN,
    "RUN_FREQUENCY": RUN_FREQUENCY,
    "RUN_MLFLOW_EXP": RUN_MLFLOW_EXP,
    "CATEGORY_UNIVERSE": CATEGORY_UNIVERSE,
}

print(f"""
====**** Pipeline Parameters ****====
\nCATEGORY_FEATURES: {CATEGORY_FEATURES}, 
\nCONSTRAINED_FEATURES: {CONSTRAINED_FEATURES},
\nCOVARIATES: {COVARIATES}, 
\nENCODE_FEATURES: {ENCODE_FEATURES}, 
\nRESPONSE: {RESPONSE},
\nTRAIN_PARAMS: {TRAIN_PARAMS}, 
\nTOLERANCE: {TOLERANCE}, 
\nROUNDS: {ROUNDS},
\nPARAM_TUNING: {PARAM_TUNING},
\nMAX_EVALS: {MAX_EVALS},
\nMODEL_VERBOSE: {MODEL_VERBOSE},
\nCONFIG_HASHMAP: {CONFIG_HASHMAP},
""")

# -

###########################
# Model Versioning Feedback
# ##########################
@component(base_image=MLFLOW_IMAGE)
def model_versioning_feedback(
    model_registry_name: str, #model_ver_sqlquery_input
    mlflow_exp_name: str, # mlflow_exp_name
    # pipeline_root: str,
) -> int:
    import pandas as pd
    import mlflow
    from mlflow.tracking.client import MlflowClient
    import md_utils
    
    # Initialize client
    c = MlflowClient()
    
    mlflow.set_experiment(mlflow_exp_name)
    
    try:
        last_model_version = c.get_latest_versions(model_registry_name, stages=["Production"])[0].version
    except:
        print("last model version of Production model stage is not found")
        last_model_version = 0
        
    print("last_model_version:", last_model_version)
    return int(last_model_version)


###########################
# Get features data as training data
# ##########################
@component(base_image=BASE_IMAGE)
def get_prepared_data(
    train_table_name_input: str,
    val_table_name_input: str,
    test_table_name_input: str,
    ds_eval_table_name_input: str,
    manual_eval_table_name_input: str,
    project_id: str,
    env: str,
    pipeline_root: str,
    data_storage_gcs_uri : str,
    train_data_output: Output[Dataset],
    val_data_output: Output[Dataset],
    test_data_output: Output[Dataset],
    ds_eval_data_output: Output[Dataset],
    manual_eval_data_output: Output[Dataset]
) -> NamedTuple("Outputs", [("min_date", str),
                            ("max_date", str),
                            ]):
    import pandas as pd
    import numpy as np
    from google.cloud import bigquery
    import md_utils
    import gcsfs
    from pyarrow import parquet
    
    bq_project_id =train_table_name_input.split('.')[0]
    bq_dataset_id =train_table_name_input.split('.')[1]
    bq_train_table_id = train_table_name_input.split('.')[2]
    bq_val_table_id = val_table_name_input.split('.')[2]
    bq_test_table_id = test_table_name_input.split('.')[2]
    bq_ds_eval_table_id = ds_eval_table_name_input.split('.')[2]
    bq_manual_eval_table_id = manual_eval_table_name_input.split('.')[2]


    train_destination_uri = f'{data_storage_gcs_uri}/md_train_inclub/*.parquet'
    val_destination_uri = f'{data_storage_gcs_uri}/md_val_inclub/*.parquet'
    test_destination_uri = f'{data_storage_gcs_uri}/md_test_inclub/*.parquet'
    ds_eval_destination_uri = f'{data_storage_gcs_uri}/md_ds_eval_inclub/*.parquet'
    manual_eval_destination_uri = f'{data_storage_gcs_uri}/md_manual_eval_inclub/*.parquet'

    #delete old files
    fs = gcsfs.GCSFileSystem()
    train_files = ["gs://" + path for path in fs.glob(train_destination_uri)]
    val_files = ["gs://" + path for path in fs.glob(val_destination_uri)]
    test_files = ["gs://" + path for path in fs.glob(test_destination_uri)]
    ds_eval_files = ["gs://" + path for path in fs.glob(ds_eval_destination_uri)]
    manual_eval_files = ["gs://" + path for path in fs.glob(manual_eval_destination_uri)]

    if train_files:
        fs.rm(train_files)
    if val_files:
        fs.rm(val_files)
    if test_files:
        fs.rm(test_files)
    if ds_eval_files:
        fs.rm(ds_eval_files)
    if manual_eval_files:
        fs.rm(manual_eval_files)
    
    dataset_ref = bigquery.DatasetReference(bq_project_id, bq_dataset_id)
    train_table_ref = dataset_ref.table(bq_train_table_id)
    val_table_ref = dataset_ref.table(bq_val_table_id)
    test_table_ref = dataset_ref.table(bq_test_table_id)
    ds_eval_table_ref = dataset_ref.table(bq_ds_eval_table_id)
    manual_eval_table_ref = dataset_ref.table(bq_manual_eval_table_id)

    job_config = bigquery.job.ExtractJobConfig()
    job_config.destination_format = bigquery.DestinationFormat.PARQUET
    client = bigquery.Client(project=project_id)
    train_extract_job = client.extract_table(
        train_table_ref,
        train_destination_uri,
        location="US",
        job_config=job_config,
    )  
    train_extract_job.result()
    val_extract_job = client.extract_table(
        val_table_ref,
        val_destination_uri,
        location="US",
        job_config=job_config,
    )
    val_extract_job.result()
    test_extract_job = client.extract_table(
        test_table_ref,
        test_destination_uri,
        location="US",
        job_config=job_config,
    )
    test_extract_job.result()
    ds_eval_extract_job = client.extract_table(
        ds_eval_table_ref,
        ds_eval_destination_uri,
        location="US",
        job_config=job_config,
    )
    ds_eval_extract_job.result()
    manual_eval_extract_job = client.extract_table(
        manual_eval_table_ref,
        manual_eval_destination_uri,
        location="US",
        job_config=job_config,
    )
    manual_eval_extract_job.result()


    print("Writing data to GCS bucket is completed")


    train_ds = parquet.ParquetDataset(train_files, filesystem=fs)
    train_data = train_ds.read().to_pandas()
    print(train_data.columns)
    print('training_data load completed')

    val_ds = parquet.ParquetDataset(val_files, filesystem=fs)
    val_data = val_ds.read().to_pandas()
    print('val_data load completed')

    test_ds = parquet.ParquetDataset(test_files, filesystem=fs)
    test_data = test_ds.read().to_pandas()
    print('test_data load completed')

    ds_eval_ds = parquet.ParquetDataset(ds_eval_files, filesystem=fs)
    ds_eval_data = ds_eval_ds.read().to_pandas()
    print('Eval_data load completed')

    manual_eval_ds = parquet.ParquetDataset(manual_eval_files, filesystem=fs)
    manual_eval_data = manual_eval_ds.read().to_pandas()
    print('Eval_data load completed')


    min_date = np.min(train_data['date'])
    max_date = np.max(train_data['date'])
    print(f'Train Min Week Start Date {min_date}')
    print(f'Train Max Week Start Date {max_date}')
    print("Train features_data size:", train_data.shape)

    min_date = np.min(val_data['date'])
    max_date = np.max(val_data['date'])
    print(f'val Min Week Start Date {min_date}')
    print(f'val Max Week Start Date {max_date}')
    print("val features_data size:", val_data.shape)

    min_date = np.min(test_data['date'])
    max_date = np.max(test_data['date'])
    print(f'Test Min Week Start Date {min_date}')
    print(f'Test Max Week Start Date {max_date}')
    print("Test features_data size:", test_data.shape)

    min_date = np.min(ds_eval_data['date'])
    max_date = np.max(ds_eval_data['date'])
    print(f'DS Eval Min Week Start Date {min_date}')
    print(f'DS Eval Max Week Start Date {max_date}')
    print("DS Eval features_data size:", ds_eval_data.shape)

    min_date = np.min(manual_eval_data['date'])
    max_date = np.max(manual_eval_data['date'])
    print(f'Manual Eval Min Week Start Date {min_date}')
    print(f'Manual Eval Max Week Start Date {max_date}')
    print("Manual Eval features_data size:", manual_eval_data.shape)

    #test code vn54vvu
    ds_eval_data['department_nbr'] = ds_eval_data['department_nbr'].fillna(-1)
    manual_eval_data['department_nbr'] = manual_eval_data['department_nbr'].fillna(-1)


    train_data.to_parquet(train_data_output.path + ".gzip", index=False, compression="gzip")
    print("train_data write finished")
    val_data.to_parquet(val_data_output.path + ".gzip", index=False, compression="gzip")
    print("val_data write finished")
    test_data.to_parquet(test_data_output.path + ".gzip", index=False, compression="gzip")
    print("test_data write finished")
    ds_eval_data.to_parquet(ds_eval_data_output.path + ".gzip", index=False, compression="gzip")
    print("ds_eval_data write finished")
    manual_eval_data.to_parquet(manual_eval_data_output.path + ".gzip", index=False, compression="gzip")
    print("manual_eval_data_output write finished")

    return str(min_date), str(max_date)

# custom_training_job = create_custom_training_job_from_component(
#     get_prepared_data,
#     display_name = 'get_prepared_data',
#     machine_type = 'n2-highmem-80',
#     replica_count= 4
# )

###########################
# Run configurations 
# ##########################
@component(base_image=BASE_IMAGE)
def run_configurations(
    config_hashmap: Dict, 
    last_md_ver_input: int,
    category_universe_value_input: str, 
) -> NamedTuple("Outputs", [("run_config_hmap_output", Dict)]
):
    import datetime
    import md_utils
    from collections import namedtuple

    last_model_ver = last_md_ver_input
    
    config_obj = md_utils.Config()
    config_hashmap = config_obj.run_config(last_md_ver=last_model_ver, config_hmap=config_hashmap)
    
    run_config_output = namedtuple("Outputs", ["run_config_hmap_output"])
    return run_config_output(config_hashmap)


###########################
# Model Preprocessiong
# ##########################
@component(base_image=BASE_IMAGE)
def premodeling_processing(
        covariates: List[str],  # COVARIATES
        constrained_features: List[str],  # CONSTRAINED_FEATURES
) -> List[int]:
    
    return [1 if col in constrained_features else 0 for col in covariates]


###########################
# Encode category
# ##########################
@component(base_image=BASE_IMAGE)
def category_encoding(
    train_input: Input[Dataset], 
    val_input: Input[Dataset], 
    covariates: List[str], #COVARIATES
    response: List[str], #RESPONSE
    encode_features: List[str], #ENCODE_FEATURES
    
    train_cbe_output: Output[Dataset], 
    val_cbe_output: Output[Dataset], 
):
    import pandas as pd
    import md_utils
    
    train = pd.read_parquet(train_input.path+".gzip")
    val = pd.read_parquet(val_input.path+".gzip")
    
    premodeling_obj = md_utils.MarkdownPreModeling()
    train_cbe, val_cbe = premodeling_obj.category_encoded_data(
        train=train, 
        val=val, 
        covariates=covariates,
        response=response,
        category_cols=encode_features,
    ) # Used for Hyperparameter Tuning Only
    
    train_cbe.to_parquet(train_cbe_output.path+".gzip", index=False, compression="gzip")
    val_cbe.to_parquet(val_cbe_output.path+".gzip", index=False, compression="gzip")


###########################
# Tune hyperparameters
# ##########################
@component(base_image=BASE_IMAGE)
def hyperparam_tuning(
    train_cbe_input: Input[Dataset], 
    val_cbe_input: Input[Dataset], 
    covariates: List[str], #COVARIATES
    response: List[str], #RESPONSE
    rounds: int, #ROUNDS
    tolerance: float, #TOLERANCE
    constraints: List[int],
    max_evals: int, #MAX_EVALS
    model_verbose: bool,
    latest_md_params_path_input: str,
    train_cbe_output: Output[Dataset], 
    val_cbe_output: Output[Dataset], 
) -> NamedTuple("Outputs", [("best_params", Dict)]):
    import pandas as pd
    import md_utils
    import pprint
    from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
    from google.cloud import storage
    import json
    from collections import namedtuple
    
    train_cbe = pd.read_parquet(train_cbe_input.path+".gzip")
    val_cbe = pd.read_parquet(val_cbe_input.path+".gzip")
    
    training_obj = md_utils.ModelTraining()
    
    best_params = training_obj.hyperparam_tuning(
        train=train_cbe, 
        val=val_cbe, 
        covariates=covariates, 
        response=response, 
        rounds=rounds, 
        tolerance=tolerance, 
        monotone_constraints=tuple(constraints),
        model_metric=mean_absolute_error,
        max_evals=max_evals,
        model_verbose=model_verbose,
    )
    
    print("Best Params Obtained from Tuning Process")
    pprint.pprint(best_params)
    
    print("Saved the md_params")
    blob = storage.blob.Blob.from_string(latest_md_params_path_input, client=storage.Client())
    blob.upload_from_string(data=json.dumps(best_params, indent=4), content_type="application/json")
    
    train_cbe.to_parquet(train_cbe_output.path+".gzip", index=False, compression="gzip")
    val_cbe.to_parquet(val_cbe_output.path+".gzip", index=False, compression="gzip")
    
    hyperparam_tuning_output = namedtuple("Outputs", ["best_params"])
    return hyperparam_tuning_output(best_params)

# custom_training_job_tuning = create_custom_training_job_from_component(
#     hyperparam_tuning,
#     display_name = 'hyperparam_tuning',
#     machine_type = 'n2-highmem-80',
#     replica_count= 4
# )
###########################
# Auto parameter model training
# ##########################
@component(base_image=BASE_IMAGE)
def auto_param_train_eval_markdown_model(
    param_flag: str, #PARAM_TUNING
    manual_params: Dict, #TRAIN_PARAMS
    train_input: Input[Dataset],
    val_input: Input[Dataset],
    train_cbe_input: Input[Dataset], 
    val_cbe_input: Input[Dataset], 
    covariates: List[str], #COVARIATES
    response: List[str], #RESPONSE
    rounds: int, #ROUNDS
    tolerance: float, #TOLERANCE
    constraints: List[int],
    encode_features: List[str],
    latest_md_params_path_input: str,
    markdown_model_output: Output[Model],
) -> NamedTuple("Outputs", [("train_val_metrics", Dict)]):
    import pandas as pd
    from google.cloud import storage
    import json
    from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
    import md_utils
    import joblib
    from collections import namedtuple
    
    train = pd.read_parquet(train_input.path+".gzip")
    val = pd.read_parquet(val_input.path+".gzip")
    
    config_obj = md_utils.Config()
    params = config_obj.model_params(param_flag=param_flag,  manual_params=manual_params, latest_md_params_path=latest_md_params_path_input)
    
    training_obj = md_utils.ModelTraining()
    try:
        pipeline = training_obj.fit_model(
            train=train, 
            val=val, 
            covariates=covariates, 
            response=response, 
            rounds=rounds, 
            tolerance=tolerance, 
            monotone_constraints=constraints,
            category_cols=encode_features,
            model_params=params,
        )
    except:
        print(f"param {params} is not defined")
    
    with open(markdown_model_output.path, "wb") as file:  
        joblib.dump(pipeline, file)

    
    train_val_metrics = training_obj.train_val_metrics(
        pipeline=pipeline, 
        train=train, 
        val=val, 
        covariates=covariates, 
        response=response
    )
    
    train_eval_markdown_model_output = namedtuple("Outputs", ["train_val_metrics"])
    return train_eval_markdown_model_output(train_val_metrics)


###########################
# Manual parameter model training
# ##########################
@component(base_image=BASE_IMAGE)
def train_eval_markdown_model(
    param_flag: str, #PARAM_TUNING
    manual_params: Dict, #TRAIN_PARAMS
    train_input: Input[Dataset],
    val_input: Input[Dataset],
    covariates: List[str], #COVARIATES
    response: List[str], #RESPONSE
    rounds: int, #ROUNDS
    tolerance: float, #TOLERANCE
    constraints: List[int],
    encode_features: List[str],
    latest_md_params_path_input: str,
    markdown_model_output: Output[Model],
) -> NamedTuple("Outputs", [("train_val_metrics", Dict)]):
    import pandas as pd
    from google.cloud import storage
    import json
    from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
    import md_utils
    import joblib
    from collections import namedtuple
    
    train = pd.read_parquet(train_input.path+".gzip")
    val = pd.read_parquet(val_input.path+".gzip")
    
    config_obj = md_utils.Config()
    params = config_obj.model_params(param_flag=param_flag,  manual_params=manual_params, latest_md_params_path=latest_md_params_path_input)
    
    training_obj = md_utils.ModelTraining()
    
    try:
        pipeline = training_obj.fit_model(
            train=train, 
            val=val, 
            covariates=covariates, 
            response=response, 
            rounds=rounds, 
            tolerance=tolerance, 
            monotone_constraints=constraints,
            category_cols=encode_features,
            model_params=params
        )
    except Exception as e: print(e)
    
    with open(markdown_model_output.path, "wb") as file:  
        joblib.dump(pipeline, file)

    
    train_val_metrics = training_obj.train_val_metrics(
        pipeline=pipeline, 
        train=train, 
        val=val, 
        covariates=covariates, 
        response=response
    )
    
    train_eval_markdown_model_output = namedtuple("Outputs", ["train_val_metrics"])
    return train_eval_markdown_model_output(train_val_metrics)


###########################
# model evaluation
# ##########################
@component(base_image=BASE_IMAGE)
def model_test(
    pipeline_root: str,
    test_data_input: Input[Dataset], 
    markdown_model_input: Input[Model],
    covariates: List[str], #COVARIATES
    train_val_metrics: Dict,
    latest_md_metrics_path_input: str,
    md_metrics_output: Output[Metrics],
    test_prediction_output: Output[Dataset],
    merged_pred_cat_output: Output[Dataset],
    category_error_output: Output[Dataset]
) -> NamedTuple("Outputs", [("md_metrics", Dict), ("category_error", Dict)]
):
    import pandas as pd
    import md_utils
    import joblib
    from google.cloud import storage
    from collections import namedtuple
    import json

    
    test_data = pd.read_parquet(test_data_input.path+".gzip")
    print("test_data size:", test_data.shape)
    # preprocessing_obj = md_utils.Preprocessing()
    # list_col_dt_convers = ["date", "md_overall_start_date", "md_overall_end_date"]
    # test_data = preprocessing_obj.typecast_datetime(data=test_data, cols=list_col_dt_convers)
    # float_col_list = ["selling_price", "pre_md_selling_price"] + [f"week_{i}_discount" for i in range(1, 9)]
    # test_data = preprocessing_obj.typecast_float(test_data, float_col_list)
    # 
    # test_data.rename(columns = {"item_nbr":"mds_fam_id"}, inplace = True)
    # 
    eval_obj = md_utils.EvaluationLayer()
    # full_md_test_data = eval_obj.full_markdown_data(test_data) # Fetch Markdown Data
    # print("full_md_test_data size:", full_md_test_data.shape)
    # modified_md_test_data = preprocessing_obj.change_dtypes(full_md_test_data, ['department_nbr', 'sub_dept_nbr']) # full_md_test_data
    # print("modified_md_test_data size:", modified_md_test_data.shape)
    # features_md_test_data = preprocessing_obj.add_features(modified_md_test_data) # modified_md_test_data
    # print("features_md_test_data size:", features_md_test_data.shape)
    # bucketed_md_test_data = preprocessing_obj.bucketing_avg_sales(features_md_test_data) # features_md_test_data
    # print("bucketed_md_test_data size:", bucketed_md_test_data.shape)
    
    with open(markdown_model_input.path, "rb") as file:  
        pipeline = joblib.load(file)
        
    test_prediction = eval_obj.sale_prediction(data=test_data, pipeline=pipeline, covariates=covariates) # bucketed_md_test_data
    test_metrics, test_prediction = eval_obj.test_metrics(data=test_prediction)
    
    print("test_prediction size:", test_prediction.shape)
    print("test_metrics:", test_metrics)
    
    md_metrics = eval_obj.combine_hmaps(train_val_metrics, test_metrics)
    print("md_metrics:", md_metrics) 
    
    print("Saved the md_metrics")
    blob = storage.blob.Blob.from_string(latest_md_metrics_path_input, client=storage.Client())
    blob.upload_from_string(data=json.dumps(md_metrics, indent=4), content_type="application/json")
    
    for metric_key, metric_val in md_metrics.items():
        md_metrics_output.log_metric(metric_key, str(metric_val))
    
    test_prediction["dept_nbr"] = test_prediction["department_nbr"].astype(int)
    test_prediction.to_parquet(test_prediction_output.path+".gzip", index=False, compression="gzip")
    
    cat_description = pd.read_csv(f"{pipeline_root}/markdown_data/cat_nbr_description_map.csv")
    cat_description=cat_description.rename({'category_nbr':'dept_nbr'},axis='columns')
    pdf_merged_pred_cat = test_prediction.merge(cat_description, on="dept_nbr", how="left")
                                           
    category_error=eval_obj.compute_category_error(test_prediction).merge(cat_description, on="dept_nbr", how="left")                  
    pdf_merged_pred_cat.to_parquet(merged_pred_cat_output.path+".gzip", index=False, compression="gzip")                             
    category_error.to_parquet(category_error_output.path+".gzip", index=False, compression="gzip")
    category_error['eval_set_start_date']=pd.to_datetime(category_error['eval_set_start_date'], format = '%Y-%m-%d').dt.strftime('%Y-%m-%d')
    category_error_dict =category_error.to_dict()
    model_test_output = namedtuple("Outputs", ["md_metrics", "category_error"])
    return model_test_output(md_metrics, category_error_dict)


###########################
# model evaluation
# ##########################
@component(base_image=BASE_IMAGE)
def model_evaluation(
        pipeline_root: str,
        project_id: str,
        test_data_input: Input[Dataset],
        markdown_model_input: Input[Model],
        covariates: List[str],  # COVARIATES
        train_val_test_metrics: Dict,
        latest_md_metrics_path_input: str,
        eval_prediction_output_path: str,
        eval_cat_output_path: str,
        eval_num_week_output_path: str,
        eval_overall_output_path: str,
        md_metrics_output: Output[Metrics],
        test_prediction_output: Output[Dataset],
        category_error_output: Output[Dataset],
        num_week_error_output: Output[Dataset],
        overall_error_output: Output[Dataset],
) -> NamedTuple("Outputs", [("md_metrics", Dict), ("category_error", Dict)]
                ):

    
    import pandas as pd
    import md_utils
    import joblib
    from google.cloud import storage
    from collections import namedtuple
    import json
    import datetime
    from datetime import date
    import numpy as np
    from google.cloud import bigquery
    from google.cloud import storage
    
    eval_pred_cols = [
        'subclass_nbr',
        'mds_fam_id',
        'club_nbr',
        'dept_nbr',
        'item_nbr',
        'date',
        'unit_price_amt',
        'markdown_start_date_dt',
        'oos_date',
        'num_weeks',
        'target',
        'predicted_sales',
        'run_date'
    ]
      
    cat_results_cols = [
        'n_samples',
        'rmse',
        'wmape',
        'smape',
        'oep_sample',
        'oe_value',
        'oep_total',
        'dept_nbr',
        'metric_target',
        'eval_set_start_date',
        'description',
        'run_date'
    ]
    num_week_results_cols = [
        'n_samples',
        'wmape',
        'wmape_weekly',
        'num_weeks',
        'metric_target',
        'weekly_bias',
        'eval_set_start_date',
        'run_date'
    ]
    
    overall_results_cols = [
        'n_samples',
        'rmse',
        'wmape',
        'smape',
        'oep_sample',
        'oe_value',
        'oep_total',
        'metric_target',
        'eval_set_start_date',
        'run_date'
    ]
        
    
    
    today_dt = str(date.today())

    test_data = pd.read_parquet(test_data_input.path + ".gzip")
    print("test_data size:", test_data.shape)

    eval_obj = md_utils.EvaluationLayer()

    with open(markdown_model_input.path, "rb") as file:
        pipeline = joblib.load(file)

    test_prediction = eval_obj.sale_prediction(data=test_data, pipeline=pipeline,
                                               covariates=covariates)  # bucketed_md_test_data    
    test_metrics, test_prediction = eval_obj.eval_metrics(data=test_prediction)
    
    print("test_prediction size:", test_prediction.shape)
    print("test_metrics:", test_metrics)
    
    md_metrics = eval_obj.combine_hmaps(train_val_test_metrics, test_metrics)
    
    test_prediction['target_last_week'] = test_prediction.sort_values(['mds_fam_id', 'club_nbr', 'date', 'num_weeks']).groupby(['mds_fam_id', 'club_nbr', 'date'])['target'].shift(1)
    test_prediction['target_last_week'] = test_prediction['target_last_week'].fillna(0)
    test_prediction['target_weekly'] = test_prediction['target'] - test_prediction['target_last_week']
    
    test_prediction['pred_last_week'] = test_prediction.sort_values(['mds_fam_id', 'club_nbr', 'date', 'num_weeks']).groupby(['mds_fam_id', 'club_nbr', 'date'])['predicted_sales'].shift(1)
    test_prediction['pred_last_week'] = test_prediction['pred_last_week'].fillna(0)
    test_prediction['predicted_sales_weekly'] = test_prediction['predicted_sales'] - test_prediction['pred_last_week']
    
    week_test_metrics, test_prediction = eval_obj.eval_metrics_weekly(data=test_prediction)
    
    md_metrics = eval_obj.combine_hmaps(md_metrics, week_test_metrics)
    
    print("md_metrics:", md_metrics)
    
    blob = storage.blob.Blob.from_string(latest_md_metrics_path_input, client=storage.Client())
    blob.upload_from_string(data=json.dumps(md_metrics, indent=4), content_type="application/json")
    
    for metric_key, metric_val in md_metrics.items():
        md_metrics_output.log_metric(metric_key, str(metric_val))
    
    test_prediction["dept_nbr"] = test_prediction["department_nbr"].astype(int)
    test_prediction['run_date'] = today_dt
#     test_prediction['target'] = np.where(test_prediction['target'] == 0, 1, test_prediction['target'])
    test_prediction.to_parquet(test_prediction_output.path + ".gzip", index=False, compression="gzip")
    test_prediction_bq = test_prediction[eval_pred_cols]
    
    
    eval_pred_table_schema = [
        bigquery.SchemaField("subclass_nbr", "INTEGER", mode="NULLABLE"),
        bigquery.SchemaField("mds_fam_id", "INTEGER", mode="NULLABLE"),
        bigquery.SchemaField("date", "DATE", mode="NULLABLE"),
        bigquery.SchemaField("dept_nbr", "INTEGER", mode="NULLABLE"),
        bigquery.SchemaField("item_nbr", "INTEGER", mode="NULLABLE"),
        bigquery.SchemaField("club_nbr", "INTEGER", mode="NULLABLE"),
        bigquery.SchemaField("unit_price_amt", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("markdown_start_date_dt", "DATE", mode="NULLABLE"),
        bigquery.SchemaField("oos_date", "DATE", mode="NULLABLE"),
        bigquery.SchemaField("num_weeks", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("target", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("predicted_sales", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("run_date", "STRING", mode="NULLABLE"),
        ]
    # create bq client
    client = bigquery.Client(project_id)
    # job config
    job_config = bigquery.LoadJobConfig(
        schema=eval_pred_table_schema,
        write_disposition="WRITE_APPEND",
    )  
    job = client.load_table_from_dataframe(test_prediction_bq, eval_prediction_output_path, job_config=job_config)
    job.result()

    cat_description = pd.read_csv(f"{pipeline_root}/markdown_data/cat_nbr_description_map.csv")
    cat_description = cat_description.rename({'category_nbr': 'dept_nbr'}, axis='columns')
    pdf_merged_pred_cat = test_prediction.merge(cat_description, on="dept_nbr", how="left")

    category_error = eval_obj.compute_category_error(test_prediction).merge(cat_description, on="dept_nbr", how="left")
    category_error['run_date'] = today_dt
    
    category_results_bq = category_error[cat_results_cols]
    category_error['eval_set_start_date'] = pd.to_datetime(category_error['eval_set_start_date'],
                                                           format='%Y-%m-%d').dt.strftime('%Y-%m-%d')
    category_error.to_parquet(category_error_output.path + ".gzip", index=False, compression="gzip")
    category_error_dict = category_error.to_dict()
    model_evaluation_output = namedtuple("Outputs", ["md_metrics", "category_error"])
    
    
    eval_cat_results_schema = [
        bigquery.SchemaField("n_samples", "INTEGER", mode="NULLABLE"),
        bigquery.SchemaField("rmse", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("wmape", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("smape", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("oep_sample", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("oe_value", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("oep_total", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("dept_nbr", "INTEGER", mode="NULLABLE"),
        bigquery.SchemaField("metric_target", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("eval_set_start_date", "DATE", mode="NULLABLE"),
        bigquery.SchemaField("description", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("run_date", "STRING", mode="NULLABLE"),
        ]
    # create bq client
    client = bigquery.Client(project_id)
    # job config
    job_config = bigquery.LoadJobConfig(
        schema=eval_cat_results_schema,
        write_disposition="WRITE_APPEND",
    )  
    job = client.load_table_from_dataframe(category_results_bq, eval_cat_output_path, job_config=job_config)
    job.result()

    num_week_error = eval_obj.compute_num_week_error(test_prediction)
    num_week_error['run_date'] = today_dt

    
    
    num_week_error_bq = num_week_error[num_week_results_cols]
    num_week_error['eval_set_start_date'] = pd.to_datetime(num_week_error['eval_set_start_date'],
                                                           format='%Y-%m-%d').dt.strftime('%Y-%m-%d')    
    num_week_error.to_parquet(num_week_error_output.path + ".gzip", index=False, compression="gzip")
    num_week_error_dict = num_week_error.to_dict()
    
    
    eval_num_week_results_schema = [
        bigquery.SchemaField("n_samples", "INTEGER", mode="NULLABLE"),
        bigquery.SchemaField("wmape", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("wmape_weekly", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("num_weeks", "INTEGER", mode="NULLABLE"),
        bigquery.SchemaField("metric_target", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("weekly_bias", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("eval_set_start_date", "DATE", mode="NULLABLE"),
        bigquery.SchemaField("run_date", "STRING", mode="NULLABLE"),
        ]
    # create bq client
    client = bigquery.Client(project_id)
    # job config
    job_config = bigquery.LoadJobConfig(
        schema=eval_num_week_results_schema,
        write_disposition="WRITE_APPEND",
    )  
    job = client.load_table_from_dataframe(num_week_error_bq,eval_num_week_output_path , job_config=job_config)
    job.result()
    
    test_prediction_overall = test_prediction.copy()
    test_prediction_overall['dept_nbr'] = 0
    overall_test_results = eval_obj.compute_category_error(test_prediction_overall)
    overall_test_results['run_date'] = today_dt
    # overall_test_results['eval_set_start_date'] = pd.to_datetime(overall_test_results['eval_set_start_date'],
    #                                                        format='%Y-%m-%d').dt.strftime('%Y-%m-%d')
    overall_test_results.to_parquet(overall_error_output.path + ".gzip", index=False, compression="gzip")
    overall_results_bq = overall_test_results[overall_results_cols]

    eval_overall_results_schema = [
        bigquery.SchemaField("n_samples", "INTEGER", mode="NULLABLE"),
        bigquery.SchemaField("rmse", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("wmape", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("smape", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("oep_sample", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("oe_value", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("oep_total", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("metric_target", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("eval_set_start_date", "DATE", mode="NULLABLE"),
        bigquery.SchemaField("run_date", "STRING", mode="NULLABLE"),
        ]
    # create bq client
    client = bigquery.Client(project_id)
    # job config
    job_config = bigquery.LoadJobConfig(
        schema=eval_overall_results_schema,
        write_disposition="WRITE_APPEND",
    )  
    job = client.load_table_from_dataframe(overall_results_bq, eval_overall_output_path, job_config=job_config)
    job.result()
    
    return model_evaluation_output(md_metrics, category_error_dict)


###########################
# Post Analysis
# ##########################
@component(base_image=BASE_IMAGE)
def post_analysis(
    project_id: str,
    pipeline_root: str, 
    test_data_input: Input[Dataset],
    markdown_model_input: Input[Model],
    elasticity_output_path: str,
    output_importance_image: OutputPath(),
    output_curve_image: OutputPath()
):
    """Register the NewItem model to MLFlow.
    Args:
        markdown_model_input: Markdown model saved in GCS bucket.
        covariates: 
    Returns:
    """
    from google.cloud import storage
    import os
    import pandas as pd
    import numpy as np
    import argparse
    import md_utils
    import pickle
    from collections import namedtuple
    import joblib
    import json
    from google.cloud import bigquery
    
    client = bigquery.Client(project=project_id)
    test_data = pd.read_parquet(test_data_input.path + ".gzip")
    elasticity=md_utils.ElasticityTest(num_weeks=1,specified_input_pd=test_data, inventory_modifier=1)
    elasticity.xgb_model_loader(markdown_model_input.path)
    elasticity.plot_model_importance(output_importance_image)
    elasticity_data=elasticity.create_elasticity_matrix()
    predicted_data=elasticity.sale_prediction(elasticity_data,"xgb")
    elasticity.plot_elasticity_curve(predicted_data,output_curve_image)
    elasticity.plot_metrics("xgb")
    
    client = bigquery.Client(project_id)
    # Define table name, in format dataset.table_name
    job_config = bigquery.job.LoadJobConfig()
    job_config.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE

    # Load data to BQ
    job = client.load_table_from_dataframe(predicted_data, elasticity_output_path, job_config)



# +
# # +
###########################
# Register model in Element MLFlow 
# ##########################
@component(base_image=MLFLOW_IMAGE)
def element_model_registry(
    pipeline_name:str,
    dynamic_config_key:str,
    run_configurations_input: Dict,
    param_tuning: str, #PARAM_TUNING
    data_fraction: float, #CONFIG_HASHMAP["DATA_FRACTION"]
    covariates: List[str], #COVARIATES
    response: List[str], #RESPONSE
    rounds: int, #ROUNDS
    tolerance: float, #TOLERANCE
    # constraints: List[int],
    encode_features: List[str],
    train_params: Dict, #TRAIN_PARAMS
    # txt_input: Input[Artifact],
    # prediction_input: Input[Dataset],
    # merged_pred_cat_input: Input[Dataset],
    markdown_model_input: Input[Model],
    md_metrics_input_logs: Dict,
    md_metrics_input_manual: Dict,
    category_error_input: Dict,
    model_registry_name: str,
    mlflow_exp_name: str,
    markdown_model_output: Output[Model],
) -> NamedTuple("Outputs", [("model_registry_name", str), 
                            ("mlflow_exp_name", str),
                            ("current_version", int)
]):
    """Register the NewItem model to MLFlow.
    Args:
        newitem_model_input: NewItem model saved in GCS bucket.
        model_registry_name: Model registry name for MLFlow Model.
        mlflow_exp_name: Model experiment name for MLFlow Experiemnt.
        current_auc_score: AUC score from trained model.
    Returns:
        model_registry_name: Model registry name for MLFlow Model as output.
        mlflow_exp_name: Model experiment name for MLFlow Experiemnt as output.
        current_version: Model version of model as output.
        current_auc_score: AUC score from trained model as output
    """
    from google.cloud import storage
    import os
    import argparse
    import pandas as pd
    import numpy as np
    import md_utils
    import pickle
    from tempfile import TemporaryFile
    from mlflow.tracking.client import MlflowClient
    from collections import namedtuple
    import mlflow
    import joblib
    import json
    import shutil
    from plotly.offline import plot
    import plotnine 
    
    # pdf_prediction = pd.read_parquet(prediction_input.path+".gzip")
    # pdf_merged_pred_cat = pd.read_parquet(merged_pred_cat_input.path+".gzip")
    # shutil.copy2(txt_input.path+".txt", "post_analysis.txt")
    md_metrics_input_logs_updated={}
    md_metrics_input_manual_updated={}
    for i,j in zip(list(md_metrics_input_logs.keys()),md_metrics_input_logs.values()):
        if "eval" in i:
            md_metrics_input_logs_updated[f"{i}_logs"]=j
        else :
            md_metrics_input_logs_updated[i]=j
    for m,n in zip(list(md_metrics_input_manual.keys()),md_metrics_input_manual.values()):
        if "eval" in m:
            md_metrics_input_manual_updated[f"{m}_manual"]=n
        else :
            md_metrics_input_manual_updated[m]=n
    #------------------------
    # Parameters   
    #------------------------
    params = {
        "version": run_configurations_input["run_version"], 
        "param_tuning": param_tuning,
        "data_fraction": data_fraction,
        "train_period": run_configurations_input[dynamic_config_key]["train_period"],
        "test_period": run_configurations_input[dynamic_config_key]["test_period"],
        # "category_universe": run_configurations_input["category_universe"],
        "covariates": covariates,
        "response": response,
        "rounds": rounds,
        "tolerance": tolerance,
        "encode_features": encode_features,
        "train_params": train_params,
    }
    #------------------------
    # model
    #------------------------
    with open(markdown_model_input.path, "rb") as file:
        markdown_model = joblib.load(file)
    #------------------------
    # Plots
    #------------------------
    analysis_obj = md_utils.PostAnalysis()
    # feature importance
    feature_importance = pd.DataFrame()
    feature_importance["features"] = covariates
    feature_importance["importance"] = markdown_model.steps[1][1].feature_importances_
    feat_imp_fig = analysis_obj.feat_imp_ggplot(data=feature_importance, x_val="features", y_val="importance")
    # Median Absolute Percentage Error
    # median_ape = round(np.median(pdf_merged_pred_cat["ape"]), 2)
    # text = f'Median APE: {median_ape}'
    # ape_cdf_fig = analysis_obj.ape_cdf_ggplot(data=pdf_merged_pred_cat, text=text, x_val="ape")
    # Median Absolute Percentage Error Per Category
    # cat_ape_cdf_fig = analysis_obj.cat_ape_cdf_ggplot(data=pdf_merged_pred_cat, x_val="ape")
    # Units Error Density
    # bias = round(np.mean(pdf_prediction["units_error"]),1)
    # text = f'Bias: {bias}'
    # units_error_fig = analysis_obj.units_error_ggplot(data=pdf_prediction, text=text, x_min_lim=-250, x_max_lim=250, x_axis=500, y_axis=0.02, x_val="units_error")
    # Units Error Limited Axis Density
    # bias = round(np.mean(pdf_prediction.query("units_error > -100 & units_error < 100")["units_error"]),1)
    # text = f'Bias: {bias}'
    # units_error_lim_fig = analysis_obj.units_error_ggplot(data=pdf_prediction, text=text, x_min_lim=-100, x_max_lim=100, x_axis=50, y_axis=0.02, x_val="units_error")
    # Units Error vs Cumulated Sales
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
    with mlflow.start_run(experiment_id=experiment_id,run_name = pipeline_name) as run:
        # Get run id 
        run_id = run.info.run_uuid
        
        mlflow.sklearn.log_model(
            sk_model=markdown_model, 
            artifact_path="md_model",
            registered_model_name=model_registry_name,
        )
        
        current_version = int(c.get_latest_versions(model_registry_name, stages=["None"])[0].version)
        print('current_version')
        
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
        #mlflow.log_params(params)
        # log metrics
        print(md_metrics_input_logs_updated)
        mlflow.log_metrics(md_metrics_input_logs_updated)

        print(md_metrics_input_manual_updated)
        mlflow.log_metrics(md_metrics_input_manual_updated)
        
        # log artifact
        pd.DataFrame.from_dict(category_error_input).to_csv("category_error.csv", index=False),
        mlflow.log_artifact("category_error.csv")
        
#         with open(markdown_model_output.path, "wb") as file:  
#             joblib.dump(markdown_model, file)
        
        # mlflow.log_figure(feat_imp_fig.draw(), "feat_imp.png")
        # mlflow.log_figure(ape_cdf_fig.draw(), "ape_cdf.png")
        # mlflow.log_figure(cat_ape_cdf_fig.draw(), "cat_ape_cdf.png")
        # mlflow.log_figure(units_error_fig.draw(), "units_error.png")
        # mlflow.log_figure(units_error_lim_fig.draw(), "units_error_dist_lim.png")
        # mlflow.log_figure(error_vs_true_fig.draw(), "error_vs_true.png")
        # mlflow.log_figure(error_vs_pred_fig.draw(), "error_vs_pred.png")
        # mlflow.log_figure(pred_vs_true_fig.draw(), "pred_vs_true.png")
        
    element_model_registry_output = namedtuple("Outputs", ["model_registry_name", "mlflow_exp_name", "current_version"])
    return element_model_registry_output(model_registry_name, mlflow_exp_name, current_version)

# -

###########################
# Version Transition
# ##########################
@component(base_image=MLFLOW_IMAGE)
def version_transition(
    pipeline_root: str, 
    model_registry_name: str,
    mlflow_exp_name: str,
    current_version: int,
):
    from google.cloud import storage
    import md_utils
    import pickle
    from mlflow.tracking.client import MlflowClient
    import pandas as pd
    import numpy as np
    import pytz
    import datetime
    
    # Initialize client
    c = MlflowClient()
    model_versioning_feedback_data = pd.read_csv(f"{pipeline_root}/markdown_data/model_versions.csv")
    
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
    pdf_updated_version.to_csv(f"{pipeline_root}/markdown_data/model_versions.csv", index=False),
    
    c.transition_model_version_stage(model_registry_name, str(current_version), "Production")


@dsl.pipeline(pipeline_root=PIPELINE_ROOT, name=PIPELINE_NAME)
def pipeline():    
    
    last_model_version = model_versioning_feedback(
        model_registry_name=MODEL_REGISTRY_NAME,
        mlflow_exp_name=MLFLOW_EXP_NAME, 
    )
    
    configurations = run_configurations(
        config_hashmap=CONFIG_HASHMAP,
        last_md_ver_input=last_model_version.outputs["Output"],
        category_universe_value_input=CATEGORY_UNIVERSE,
    )
    
    dynamic_config_key = ""
    if CONFIG_HASHMAP["DYNAMIC_CONFIG"] == True:
        dynamic_config_key = "dynamic_config"
    else:
        dynamic_config_key = "non_dynamic_config"
    
    # prepared_data =custom_training_job(
    #     project='wmt-mlp-p-price-npd-pricing',
    #     train_table_name_input=TRAIN_TABLE_NAME,
    #     val_table_name_input=VAL_TABLE_NAME,
    #     test_table_name_input=TEST_TABLE_NAME,
    #     eval_table_name_input=EVAL_TABLE_NAME,
    #     project_id=PROJECT_ID,
    #     env=ENV,
    #     pipeline_root=PIPELINE_ROOT
    # )
    prepared_data = (
    get_prepared_data(
        train_table_name_input=TRAIN_TABLE_NAME,
        val_table_name_input=VAL_TABLE_NAME,
        test_table_name_input=TEST_TABLE_NAME,
        ds_eval_table_name_input=LOGS_EVAL_TABLE_NAME,
        manual_eval_table_name_input = MANUAL_EVAL_TABLE_NAME,
        project_id=PROJECT_ID,
        env=ENV,
        pipeline_root=PIPELINE_ROOT,
        data_storage_gcs_uri=DATA_STORAGE_GCS_URI
    ).set_cpu_limit("96")
    .set_memory_limit("624G")
    )

    premodel = (
        premodeling_processing(
            covariates=COVARIATES,
            constrained_features=list(CONSTRAINED_FEATURES),
        ).set_cpu_limit("32")
        .set_memory_limit("32G")
    )
    
    if PARAM_TUNING == "auto":
        endcoded_category = (
            category_encoding(
                train_input=prepared_data.outputs["train_data_output"],
                val_input=prepared_data.outputs["val_data_output"],
                covariates=COVARIATES,
                response=RESPONSE,
                encode_features=ENCODE_FEATURES,
            ).set_cpu_limit("96")
            .set_memory_limit("624G")
        )
        
        tuned_hyperparameters = (
            hyperparam_tuning(
                train_cbe_input=endcoded_category.outputs["train_cbe_output"], 
                val_cbe_input=endcoded_category.outputs["val_cbe_output"], 
                covariates=COVARIATES, 
                response=RESPONSE, 
                rounds=ROUNDS,
                tolerance=TOLERANCE,
                constraints=premodel.outputs["Output"],
                max_evals=MAX_EVALS,
                model_verbose=MODEL_VERBOSE,
                latest_md_params_path_input=LATEST_MD_PARAMS_PATH,
            ).set_cpu_limit("96")
             .set_memory_limit("624G")
        )
#         tuned_hyperparameters = (custom_training_job_tuning(
#                 project='wmt-mlp-p-price-npd-pricing',
#                 train_cbe_input=endcoded_category.outputs["train_cbe_output"], 
#                 val_cbe_input=endcoded_category.outputs["val_cbe_output"], 
#                 covariates=COVARIATES, 
#                 response=RESPONSE, 
#                 rounds=ROUNDS,
#                 tolerance=TOLERANCE,
#                 constraints=premodel.outputs["Output"],
#                 max_evals=MAX_EVALS,
#                 model_verbose=MODEL_VERBOSE,
#                 latest_md_params_path_input=LATEST_MD_PARAMS_PATH,
#             ))
        
        model_training = (
            auto_param_train_eval_markdown_model(
                param_flag=PARAM_TUNING,
                manual_params=tuned_hyperparameters.outputs["best_params"],
                train_input=prepared_data.outputs["train_data_output"],
                val_input=prepared_data.outputs["val_data_output"],
                train_cbe_input=tuned_hyperparameters.outputs["train_cbe_output"],
                val_cbe_input=tuned_hyperparameters.outputs["val_cbe_output"],
                covariates=COVARIATES,
                response=RESPONSE,
                rounds=ROUNDS,
                tolerance=TOLERANCE,
                constraints=premodel.outputs["Output"],
                encode_features=ENCODE_FEATURES,
                latest_md_params_path_input=LATEST_MD_PARAMS_PATH,
            ).set_cpu_limit("96")
            .set_memory_limit("624G")
        )
        
    elif PARAM_TUNING == "manual":
        model_training = (
            train_eval_markdown_model(
                param_flag=PARAM_TUNING,
                manual_params=TRAIN_PARAMS,
                train_input=prepared_data.outputs["train_data_output"],
                val_input=prepared_data.outputs["val_data_output"],
                covariates=COVARIATES,
                response=RESPONSE,
                rounds=ROUNDS,
                tolerance=TOLERANCE,
                constraints=premodel.outputs["Output"],
                encode_features=ENCODE_FEATURES,
                latest_md_params_path_input=LATEST_MD_PARAMS_PATH,
            ).set_cpu_limit("96")
            .set_memory_limit("624G")
        )
    # The following part needs to be rewritten
    evaluated_model = (
        model_test(
            pipeline_root=PIPELINE_ROOT,
            test_data_input=prepared_data.outputs["test_data_output"],
            markdown_model_input=model_training.outputs["markdown_model_output"],
            covariates=COVARIATES,
            train_val_metrics=model_training.outputs["train_val_metrics"],
            latest_md_metrics_path_input=LATEST_MD_METRICS_PATH,
        ).set_cpu_limit("96")
         .set_memory_limit("624G")
    )
    
    evaluated_model_eval_data_logs = (
        model_evaluation(
            pipeline_root=PIPELINE_ROOT,
            project_id=PROJECT_ID,
            test_data_input=prepared_data.outputs["ds_eval_data_output"],
            markdown_model_input=model_training.outputs["markdown_model_output"],
            train_val_test_metrics=evaluated_model.outputs["md_metrics"],
            covariates=COVARIATES,
            latest_md_metrics_path_input = LATEST_MD_METRICS_PATH_EVAL_LOGS,
            eval_prediction_output_path = EVAL_PREDICTION_OUTPUT_PATH_LOGS,
            eval_cat_output_path = EVAL_CAT_OUTPUT_PATH_LOGS,
            eval_num_week_output_path = EVAL_NUM_WEEK_OUTPUT_PATH_LOGS,
            eval_overall_output_path = EVAL_OVERALL_OUTPUT_PATH_LOGS
        ).set_cpu_limit("32")
         .set_memory_limit("64G")
    )

    evaluated_model_eval_data_manual = (
        model_evaluation(
            pipeline_root=PIPELINE_ROOT,
            project_id=PROJECT_ID,
            test_data_input=prepared_data.outputs["manual_eval_data_output"],
            markdown_model_input=model_training.outputs["markdown_model_output"],
            train_val_test_metrics=evaluated_model.outputs["md_metrics"],
            covariates=COVARIATES,
            latest_md_metrics_path_input = LATEST_MD_METRICS_PATH_EVAL_MANUAL,
            eval_prediction_output_path = EVAL_PREDICTION_OUTPUT_PATH_MANUAL,
            eval_cat_output_path = EVAL_CAT_OUTPUT_PATH_MANUAL,
            eval_num_week_output_path = EVAL_NUM_WEEK_OUTPUT_PATH_MANUAL,
            eval_overall_output_path = EVAL_OVERALL_OUTPUT_PATH_MANUAL
        ).set_cpu_limit("32")
         .set_memory_limit("64G")
    )

    post_analysis_job = (
        post_analysis(
            project_id=PROJECT_ID,
            pipeline_root=PIPELINE_ROOT,
            test_data_input=prepared_data.outputs["ds_eval_data_output"],
            markdown_model_input=model_training.outputs["markdown_model_output"],
            elasticity_output_path=ELASTICITY_OUTPUT_PATH
            
        ).set_cpu_limit("64")
         .set_memory_limit("256G")
    )
    

    mlflow_job = (
        element_model_registry(
            pipeline_name=PIPELINE_NAME,
            dynamic_config_key=dynamic_config_key,
            run_configurations_input=configurations.outputs["run_config_hmap_output"],
            param_tuning=PARAM_TUNING,
            data_fraction=CONFIG_HASHMAP["DATA_FRACTION"],
            covariates=COVARIATES,
            response=RESPONSE,
            rounds=ROUNDS,
            tolerance=TOLERANCE,
            encode_features=ENCODE_FEATURES,
            train_params=TRAIN_PARAMS,
            markdown_model_input=model_training.outputs["markdown_model_output"],
            md_metrics_input_logs=evaluated_model_eval_data_logs.outputs["md_metrics"],
            md_metrics_input_manual=evaluated_model_eval_data_manual.outputs["md_metrics"],
            category_error_input=evaluated_model.outputs["category_error"],
            model_registry_name=MODEL_REGISTRY_NAME,
            mlflow_exp_name=MLFLOW_EXP_NAME,
        ).set_cpu_limit("32")
         .set_memory_limit("64G")
    )
    #
    # version_trans = (
    #     version_transition(
    #         pipeline_root=PIPELINE_ROOT,
    #         model_registry_name=MODEL_REGISTRY_NAME,
    #         mlflow_exp_name=MLFLOW_EXP_NAME,
    #         current_version=mlflow_job.outputs["current_version"],
    #     ).set_cpu_limit("4")
    #      .set_memory_limit("16G")
    # )

if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=pipeline, 
        package_path=TMP_PIPELINE_JSON
    )

    pipeline_job = aiplatform.PipelineJob(
        display_name=f"{PARAM_TUNING}-{PIPELINE_NAME}-{TIMESTAMP}",
        template_path=TMP_PIPELINE_JSON,
        pipeline_root=PIPELINE_ROOT,
        parameter_values={},
        enable_caching=False,
    )
    
    pipeline_utils.PipelineUtils(
        storage_path=LATEST_PIPELINE_PATH,
        file_name=TMP_PIPELINE_JSON
    ).store_pipeline()
    
    pipeline_job.submit(service_account=SERVICE_ACCOUNT, network=NETWORK)





