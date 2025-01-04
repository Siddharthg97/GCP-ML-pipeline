
01/03/2025

08:18:49 PM
Auto-refresh


25
Press shift + / for Shortcuts

deferred

failed

queued

removed

restarting

running

scheduled

shutdown

skipped

success

up_for_reschedule

up_for_retry

upstream_failed

no_status

DAG
md-feature-pipeline.first_level_features
/

Task
engineer

Details

Graph

Gantt

Code

Audit Log
Parsed at: 2025-01-04, 07:20:09 UTC
import re
import yaml
import uuid
import json
from airflow import DAG
from datetime import datetime, timedelta
from airflow.utils.dates import days_ago
from airflow.providers.google.cloud.operators.dataproc import DataprocCreateClusterOperator, \
    DataprocDeleteClusterOperator, DataprocSubmitJobOperator, ClusterGenerator, DataprocCreateBatchOperator
from airflow.operators.dagrun_operator import TriggerDagRunOperator
from airflow.providers.google.cloud.hooks.gcs import GCSHook
from airflow.models import Variable


def get_config(dict, def_val, *keys):
    for key in keys:
        try:
            dict = dict[key]
        except KeyError:
            dict = def_val
            break
    return dict



GCP_PROJECT = Variable.get("GCP_PROJECT_ID") # "dev-sams-ds-featurestore"
DAG_START_DATE = Variable.get("DAG_START_DATE")
MATERIALIZATION_DATASET = Variable.get("GCP_MATERIALIZATION_DS")  # "materialization_dataset"
FS_SOURCE_DATASET = Variable.get("FS_SOURCE_DATASET")
FP_BUCKET = Variable.get("GCP_FP_BUCKET") # "ds-feature-pipelines"
TEMP_BUCKET = Variable.get("GCS_TEMP_BUCKET")
CONN_ID = "google_cloud_default" # "ds_featurestore_sa_key"
REGION = "us-central1"
CUSTOM_CONTAINER = "us-central1-docker.pkg.dev/dev-sams-ds-featurestore/dev-sams-ds-fs-docker-registry/fs-materializer:stage"
SPARK_BIGQUERY_JAR_FILE = 'gs://spark-lib/bigquery/spark-bigquery-with-dependencies_2.12-0.24.2.jar'
env = Variable.get("ENV")
env_args = {
    "GCP_PROJECT": GCP_PROJECT,
    "MATERIALIZATION_DATASET": MATERIALIZATION_DATASET,
    "FS_SOURCE_DATASET": FS_SOURCE_DATASET,
    "GCS_TEMP_BUCKET": TEMP_BUCKET
}

# Add args and Dag
default_args = {
    'owner': 'dsarch',
    'depends_on_past': False, 
    'start_date': DAG_START_DATE,
    'email': ['Sams-MLOps@wal-mart.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1)
}

hook = GCSHook(CONN_ID)
# Get the dag config
feature_files = hook.list(FP_BUCKET, delimiter="config.yml")
if len(feature_files) >= 1:
    for config_file in feature_files:
        # Get the dag config
        str_config = hook.download(FP_BUCKET, config_file)
        config = yaml.safe_load(str_config)
        team_id = config["team_id"]
        for config_name, config in config["pipelines"].items():
            dag_id = f"{team_id}.{config_name}"
            cluster_name = re.sub('[^0-9a-zA-Z]+', "-", dag_id).lower()
            schedule = get_config(config, None, "airflow", "schedule")
            config_args = get_config(config, [], "dataproc", "jobs", "args")
            config_args_dict = {"JOB_ARGS": config_args}
            job_args_json = [json.dumps({**env_args, **config_args_dict})]
            master_machine_type = get_config(config, "n1-standard-4", "dataproc", "cluster", "master_machine_type")
            worker_machine_type = get_config(config, "n1-standard-4", "dataproc", "cluster", "worker_machine_type")
            subnet_uri = get_config(config,"projects/shared-vpc-admin/regions/{reg}/subnetworks/prod-{reg}-01".format(reg=REGION),"dataproc", "cluster", "subnet_uri")
            num_workers = get_config(config, 5, "dataproc", "cluster", "num_workers")
            pip_packages = get_config(config, "great-expectations==0.16.8", "dataproc", "cluster", "pip_packages")
            materialization = get_config(config, False, "materialization", "online")
            feature_view = get_config(config, "feature_v1", "materialization", "feature_view")
            lookback_days = get_config(config, 20, "materialization", "lookback_days")
            dependent_dags = get_config(config, False, "dependent_features", "features")

            dag = DAG(
                dag_id,
                default_args=default_args,
                description='Dynamic DAG creation',
                schedule_interval=schedule,
                catchup=False, # if catchup is True - verify start date
                max_active_runs=1,
                concurrency=6,
            )

            CLUSTER_CONFIG = ClusterGenerator(
                gcp_conn_id=CONN_ID,
                project_id=GCP_PROJECT,
                region=REGION,
                cluster_name=cluster_name,
                service_account=f"svc-deploy-mgmt@{GCP_PROJECT}.iam.gserviceaccount.com",
                subnetwork_uri=subnet_uri,
                internal_ip_only=True,
                custom_image_project_id="wmt-pcloud-trusted-images",
                custom_image_family="wmt-dataproc-custom-2-1",
                num_workers=num_workers,
                num_masters=1,
                master_machine_type=master_machine_type,
                master_disk_type="pd-standard",
                master_disk_size=1024,
                worker_machine_type=worker_machine_type,
                worker_disk_type="pd-standard",
                worker_disk_size=1024,
                properties={},
                autoscaling_policy=None,
                idle_delete_ttl=7200,
                metadata={"gcs-connector-version": '2.2.9',
                          "spark-bigquery-connector-version": "0.32.2",
                          "DATAPROC_VERSION": '2.0',
                          "PIP_PACKAGES": pip_packages
                          },
                init_actions_uris=["gs://mle-dataproc-artifacts/mle-dataproc-connector.sh",
                                   "gs://mle-dataproc-artifacts/pip-install.sh"]
            ).make()

            unique_id = str(uuid.uuid4()).split("-")[4]

            INGRESS_PYSPARK_JOB = {
                "reference": {"project_id": GCP_PROJECT,
                              "job_id": f"{config_name}_ingress_{unique_id}"},
                "placement": {"cluster_name": cluster_name},
                "pyspark_job": {"main_python_file_uri": f"gs://{FP_BUCKET}/{team_id}/{config_name}/ingress.py",
                                "args": job_args_json
                                },
            }

            ENGINEER_PYSPARK_JOB = {
                "reference": {"project_id": GCP_PROJECT,
                              "job_id": f"{config_name}_engineer_{unique_id}"},
                "placement": {"cluster_name": cluster_name},
                "pyspark_job": {"main_python_file_uri": f"gs://{FP_BUCKET}/{team_id}/{config_name}/engineer.py",
                                "args": job_args_json
                                },
            }

            EGRESS_PYSPARK_JOB = {
                "reference": {"project_id": GCP_PROJECT,
                              "job_id": f"{config_name}_egress_{unique_id}"},
                "placement": {"cluster_name": cluster_name},
                "pyspark_job": {"main_python_file_uri": f"gs://{FP_BUCKET}/{team_id}/{config_name}/egress.py",
                                "args": job_args_json
                                },
            }

            create_cluster = DataprocCreateClusterOperator(
                task_id="create_cluster",
                project_id=GCP_PROJECT,
                cluster_config=CLUSTER_CONFIG,
                region=REGION,
                cluster_name=cluster_name,
                gcp_conn_id=CONN_ID,
                labels=dict(env=env,team=team_id,name=config_name),
                dag=dag
            )

            ingress_task = DataprocSubmitJobOperator(
                task_id="ingress",
                job=INGRESS_PYSPARK_JOB,
                region=REGION,
                project_id=GCP_PROJECT,
                dag=dag
            )

            engineer_task = DataprocSubmitJobOperator(
                task_id="engineer",
                job=ENGINEER_PYSPARK_JOB,
                region=REGION,
                project_id=GCP_PROJECT,
                dag=dag
            )

            egress_task = DataprocSubmitJobOperator(
                task_id="egress",
                job=EGRESS_PYSPARK_JOB,
                region=REGION,
                project_id=GCP_PROJECT,
                dag=dag
            )

            delete_cluster = DataprocDeleteClusterOperator(
                task_id="delete_cluster",
                project_id=GCP_PROJECT,
                cluster_name=cluster_name,
                gcp_conn_id=CONN_ID,
                region=REGION,
                dag=dag
            )

            if dependent_dags:
                for feature in dependent_dags:
                    fp_trigger = TriggerDagRunOperator(dag=dag,
                                                       task_id=f'fp_trigger.{feature}',
                                                       trigger_dag_id=f"{team_id}.{feature}",
                                                       conf={"feature_name": config_name})
                    delete_cluster.set_downstream(fp_trigger)

            if materialization:
                fs_args = {
                    "GCP_PROJECT" : GCP_PROJECT,
                    "FEATURE_VIEW": feature_view,
                    # todo Check if should fetch start and end dates from user
                    "LOOKBACK_DAYS": lookback_days,
                    "MATERIALIZATION_DATASET": MATERIALIZATION_DATASET,
                    "GCS_TEMP_BUCKET": TEMP_BUCKET
                }
                fs_args = [json.dumps(fs_args)]
                FS_JOB = {
                    "reference": {"project_id": GCP_PROJECT,
                                  "job_id": f"{config_name}_materialization_{unique_id}"},
                    "placement": {"cluster_name": cluster_name},
                    "pyspark_job": {"main_python_file_uri": f"gs://dsarch-neptune-dsarch-{env}-dprocjobs/materialize-features.py",
                                    "args": fs_args
                                    },
                }

                fs_materialization = DataprocSubmitJobOperator(
                    task_id="fs_materialization",
                    job=FS_JOB,
                    region=REGION,
                    project_id=GCP_PROJECT,
                    dag=dag
                )

                create_cluster >> ingress_task >> engineer_task >> egress_task >> fs_materialization >> delete_cluster

            else:
                create_cluster >> ingress_task >> engineer_task >> egress_task >> delete_cluster

            # DAG is created among the global objects
            globals()[dag_id] = dag
