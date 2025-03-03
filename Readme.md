### Steps to create GCP ML pipeline


1) Settings.yml file - contains environment variables , model configuration & data location
2) ML pipeline scripts - to run these scripts we use kubeflow   
3) Utils module  - contains function definitions
4) base flow image 
5) ML flow image
6) Pipeline_utils
7) Decryption
Q How to start creating the ML pipeline ?
Kubeflow components - decorators & pipline. To call each decorator defined we need an python application already running, so we run the docker files to create docker images and use them to create containerized application.

To create ML flow pipeline we need containerized applications containing -python configurations, 


1) Import require libraries
2) create object of argument parser to define input variables 
3) Import all required variables from settings.yml configuration
4) Start defining decorators as wrappers
5) Each decorator requires python application for which we are using containerized application
6) Pipeline definition - create decorator for pipeline, calling all the functions defined in other decorator.
7) execution if __name__== main
8) we need to define the compiler where we provide pipeline name & json containing pipeline definition
9) create object of pipline job using AI pipeline platform is defined where we pass display name , pipeline root containing pipeline artifacts,
10) pipline utils is used to pass the pipline json to a location in gcs bucket
11) run the pipline created.

We have project utils/md utils file to call the required functions, within each decorator

Now we have pipeline utils to access the settings .yml , save  pipeline json to gcs bucket

To work on - json specificatio file, model registry.py
PIPELINE_JSON: "inclub-md-pipeline-dev.json"
TMP_PIPELINE_JSON = os.path.join("/tmp", PIPELINE_JSON)
### Kubeflow components
from kfp.v2 import compiler 

The kfp.v2.compiler module is part of the Kubeflow Pipelines SDK (KFP) and is used to compile Python-based pipeline definitions into pipeline job specifications in YAML format. These specifications can then be submitted to run on a Kubeflow Pipelines environment, such as Google Cloud's AI Platform Pipelines.

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
------------------------------------------------------------------------------------
step-2 Extracting all the variables/parameters from settings.yml file specifically for - model condition / run configuration parameters , images,\
training pipeline parameters  ,train & output data parameters,  training parameters & hyper patameter tuning

------------------------------------------------------------------------------------
Typical Usage of kfp.v2.compiler
Here’s how to use it in a basic workflow:

Installation
Ensure you have the Kubeflow Pipelines SDK installed:

bash
Copy code
pip install kfp
Pipeline Definition
Define your pipeline using the @dsl.pipeline decorator from kfp.v2.dsl:

python
Copy code
from kfp.v2 import dsl

@dsl.pipeline(
    name="my-sample-pipeline",
    description="A sample pipeline to demonstrate KFP compilation."
)
def my_pipeline(
    input_data: str,
    num_steps: int
):
    # Pipeline tasks can be added here
    pass
Compilation
Use kfp.v2.compiler.Compiler to compile the pipeline into a YAML job specification:

python
Copy code
from kfp.v2 import compiler

# Define the path to save the compiled pipeline
pipeline_job_path = "my_pipeline_job.json"

# Compile the pipeline
compiler.Compiler().compile(
    pipeline_func=my_pipeline,
    package_path=pipeline_job_path
)

print(f"Pipeline job spec saved to: {pipeline_job_path}")
Key Parameters of Compiler.compile()
pipeline_func:

The Python function representing the pipeline.
Decorated with @dsl.pipeline.
package_path:

The file path where the compiled YAML/JSON specification will be saved.
Typically ends in .json.
type_check (optional):

Set to True or False to enable/disable type-checking for pipeline inputs and outputs.
Running the Compiled Pipeline
Once you have the compiled pipeline specification (e.g., my_pipeline_job.json), you can submit it to a Kubeflow Pipelines-enabled environment using the kfp.Client or Google Cloud's AI Platform.

Example with Google Cloud AI Platform:

python
Copy code
from google.cloud import aiplatform

aiplatform.init(project="my-project", location="us-central1")

pipeline_job = aiplatform.PipelineJob(
    display_name="my-sample-pipeline",
    template_path="my_pipeline_job.json",
    parameter_values={
        "input_data": "gs://my-bucket/data.csv",
        "num_steps": 10
    },
    pipeline_root="gs://my-bucket/pipeline-root/"
)

pipeline_job.run()
-----------------------------------------------------------------
**Notes**
# step-2 
Extracting all the variables/parameters from settings.yml file specifically for -  model condition /  ,
run configuration parameters - mentioned below
training pipeline parameters  - pipeline root, pipeline json,gcs uri, 
train & output data parameters - paths of eval datasets, input train , val datasets configurations
mlflow parameters - MLFlow Model Registry name and experiment name ,images,
training parameters & hyper patameter tuning

CONFIG_HASHMAP = {
    "MODE": MODE,
    "DYNAMIC_CONFIG": DYNAMIC_CONFIG,
    "DATA_FRACTION": DATA_FRACTION,
    "PRODUCTION_RUN": PRODUCTION_RUN,
    "RUN_FREQUENCY": RUN_FREQUENCY,
    "RUN_MLFLOW_EXP": RUN_MLFLOW_EXP,
    "CATEGORY_UNIVERSE": CATEGORY_UNIVERSE,
}

class Config:   - latest markdown version, test horizon, run frequency, 
    def __init__(self):
        pass
    def run_config(self, last_md_ver, config_hmap):
        """Conditionally run the configurations.
        Args:
            last_md_ver: Markdown model version.
            config_hmap: configuration hashmap. 
        Returns:
            Configuration hashmap. 
        """
        import datetime
        from collections import namedtuple
        from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
        
        # Set Training and Testing Dates Dynamically Based on the Run Date
        if config_hmap["DYNAMIC_CONFIG"]: 
            # Run Version
            run_version = str(last_md_ver + 1)
            description = f'base model version {run_version}'
            
            today = datetime.date.today()
            if config_hmap["RUN_FREQUENCY"] == 'weekly':
                # Keeing Status Quo for Weekly Runs
                train_horizon = 78 # in weeks
                test_horizon = 26
                last_monday = today - datetime.timedelta(today.weekday())
                # Adding 14 Days Buffer to Allow Tables to Get Updated
                test_start_date = last_monday - datetime.timedelta(days=14) - datetime.timedelta(weeks=test_horizon) 
            elif config_hmap["RUN_FREQUENCY"] == 'monthly':
                train_horizon = 78 # 1.5 years of Historical Data
                test_horizon = 10
                last_monday = today - datetime.timedelta(today.weekday()) 
                # If Day Number of the Last Monday is in First Half of Month take Test Data till 1st
                if last_monday.day < 15: 
                    test_start_date = last_monday.replace(day=1) - datetime.timedelta(weeks=test_horizon)
                else:
                    test_start_date = last_monday.replace(day=15) - datetime.timedelta(weeks=test_horizon)
                    
            train_end_date = test_start_date - datetime.timedelta(days=1)
            
            # Data Parameters
            train_period = {
                'end_date': (str(train_end_date),), 
                'horizon': (train_horizon,) # Lookback for train data (in weeks)
            }
            test_period = {
                'start_date': (str(test_start_date),),
                'horizon': (test_horizon,) # Lookforward for test data (in weeks)
            }
            print(f'Train Period - {train_period}')
            print(f'Test Period - {test_period}')
        else:
            run_version = str(last_md_ver + 1)
            description = f'base model version {run_version}'
            # Data Parameters
            train_period = {
                'end_date': ('2022-09-30',), 
                'horizon': (26,) # Lookback for train data (in weeks)
            }
            test_period = {
                'start_date': ('2022-10-01',),
                'horizon': (8,) # Lookforward for test data (in weeks)
            }
            print(f'Train Period - {train_period}')
            
        category_universe = tuple(MarkdownUtils(config_hmap["CATEGORY_UNIVERSE"]).category_universe)
        
        
        print(f'Current Run Version - {run_version}')
        
        run_config = {
            "run_version": run_version,
            "description": description,
            "category_universe": category_universe,
        }
        
        if config_hmap["DYNAMIC_CONFIG"]:
            run_config["dynamic_config"] = {
                "today": today.strftime('%Y-%m-%d'), 
                "run_frequency": config_hmap["RUN_FREQUENCY"], 
                "train_horizon": train_horizon, 
                "test_horizon": test_horizon, 
                "last_monday": last_monday.strftime('%Y-%m-%d'), 
                "test_start_date": test_start_date.strftime('%Y-%m-%d'),
                "train_end_date": train_end_date.strftime('%Y-%m-%d'),
                "train_period": train_period,
                "test_period": test_period,
            }
        else:
            run_config["non_dynamic_config"] = {
                "train_period": train_period,
                "test_period": test_period,
            }
            
        return run_config 

## Environment variables
https://www.datacamp.com/tutorial/python-environment-variables 

These are key value pairs containing variable name and their path respectively. The path can belong to any executable file like exe file for conda or pip or pyspark or java. Otherwise env varibles can be API keys or login id with passwords to access several applications like cloud applications while running the python scripts.
Another benefit of environment variables is configurability. You can easily adjust settings (database URLs, file paths) by modifying environment variables without changing your code. This feature is especially helpful if you use the same setting in multiple parts of your project.
These login id and passwords or API keys are not safe to be stored in github repositories as can be accessed by hackers or any malicious activity

We can create our own argument parser
It is basucally a container that contains all the argument we want to pass from commmand line.
https://docs.python.org/3/library/argparse.html
https://www.youtube.com/watch?v=FsAPt_9Bf3U
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

***mlflowclient***
c = MlflowClient()

Instantiates an MlflowClient object.
This client provides an interface to MLflow's tracking service, enabling operations such as creating experiments, logging parameters, or retrieving runs.
mlflow.set_experiment(mlflow_exp_name)

Sets the active MLflow experiment by its name (mlflow_exp_name).
If the experiment does not already exist, it will be created.
Future tracking API calls, like logging metrics or parameters, will associate with this experiment.
c.get_experiment_by_name(mlflow_exp_name)

Retrieves the experiment object by its name.
Returns an Experiment object if it exists or None if it doesn’t.
.experiment_id

Extracts the experiment_id property from the returned Experiment object.
The experiment_id is used to identify the experiment in MLflow's backend.
Example in Context
Here's an example of how this snippet might fit into a broader MLflow workflow:

python
Copy code
import mlflow
from mlflow.tracking import MlflowClient






# Define the experiment name
mlflow_exp_name = "My Experiment"

# Initialize the MLflow client
c = MlflowClient()

# Set the experiment (creates it if it doesn't exist)
mlflow.set_experiment(mlflow_exp_name)

# Retrieve the experiment ID
experiment_id = c.get_experiment_by_name(mlflow_exp_name).experiment_id

print(f"Experiment '{mlflow_exp_name}' is set with ID: {experiment_id}")

# Example: Log a parameter to the experiment
with mlflow.start_run(experiment_id=experiment_id):
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_metric("accuracy", 0.95)
Key Notes
Dependencies:

Ensure mlflow is installed:
bash
Copy code
pip install mlflow
MLflow Tracking URI:

If you’re using a custom MLflow tracking server (not the default local one), set the tracking URI:
python
Copy code
mlflow.set_tracking_uri("http://your-tracking-server")
Error Handling:

Handle the case where the experiment does not exist gracefully:
python
Copy code
experiment = c.get_experiment_by_name(mlflow_exp_name)
if experiment is None:
    raise ValueError(f"Experiment '{mlflow_exp_name}' does not exist.")
experiment_id = experiment.experiment_id


------------------------------------------------------------------------------



You said:
run_id = run.info.run_uuid
ChatGPT said:
ChatGPT
In MLflow, run_id = run.info.run_uuid is used to extract the unique identifier of a run. However, the run_uuid property has been deprecated and replaced with run_id in recent versions of MLflow. You should use run.info.run_id instead for better compatibility and to avoid warnings.

Explanation
run:

Represents an MLflow run object. This is typically obtained when you start a new run using mlflow.start_run() or retrieve an existing run using the MlflowClient.
run.info:

Provides metadata about the run, including its ID, experiment ID, start time, and status.
run_uuid (Deprecated):

A unique identifier for the run. This was used in older versions of MLflow.
It has been replaced with run_id, which serves the same purpose.
run_id (Preferred):

The modern and recommended way to get the unique identifier of a run.
Correct Usage
Here’s how you should retrieve the run ID:

python
Copy code
import mlflow

# Start an MLflow run
with mlflow.start_run() as run:
    # Get the run ID
    run_id = run.info.run_id  # Preferred
    print(f"Run ID: {run_id}")


# Set tag
c.set_tag(run_id, "mlflow.note.content", "This is experiment for testing"):
c: An instance of MlflowClient.
set_tag: A method to add or update a tag for a specific MLflow run. Tags are key-value pairs that provide metadata about the run.
run_id: The unique identifier of the run to which the tag is being added.
"mlflow.note.content": A special tag used in MLflow to store user-provided notes about a run.
"This is experiment for testing": The value of the tag, in this case, a note describing the purpose of the experiment.


Purpose of mlflow.note.content:

Tags prefixed with mlflow. are reserved for MLflow’s internal use.
The mlflow.note.content tag is specifically designed for adding free-form notes to a run, which are viewable in the MLflow UI.
Custom Tags:

You can add custom tags for better metadata organization, e.g.:
python
Copy code
c.set_tag(run_id, "project_stage", "testing")
c.set_tag(run_id, "team", "data-science")
Viewing Tags in the UI:

The tags, including mlflow.note.content, can be viewed in the MLflow Tracking UI under the "Tags" section of the run.
Tag Overwriting:

If you call set_tag with the same key multiple times for the same run, the value will be updated.
Error Handling:

Ensure that run_id is valid and the run exists in your MLflow tracking server or backend.
Installing and Using MLflow
Install MLflow if not already installed:
bash
Copy code
pip install mlflow
Set the tracking URI if you are using a custom tracking server:
python
Copy code
mlflow.set_tracking_uri("http://your-tracking-server")

---------------------------

pipeline_root:

Specifies the storage location (e.g., a GCS bucket or local directory) where artifacts produced by the pipeline will be stored.
Example: pipeline_root="gs://my-bucket/pipelines/".

-------------------------------------
The provided code snippet creates an instance of PipelineJob from the Google Cloud aiplatform library to configure and submit a Vertex AI pipeline job. Here's a detailed breakdown of the code:

Explanation
Parameters:
display_name:

A human-readable name for the pipeline job.
Combines several variables (PARAM_TUNING, PIPELINE_NAME, TIMESTAMP) for unique identification.
Example: "tuning-job-ml-pipeline-20231227T123456"
template_path:

The path to the pipeline template JSON file, often generated using tools like Kubeflow or TFX.
Example: "gs://your-bucket-name/path/to/pipeline.json"


 
pipeline_root:

Specifies the GCS path where the pipeline stores artifacts and outputs.
Example: "gs://your-pipeline-root/path/"

parameter_values:
A dictionary of parameters to customize the pipeline run.
Example: {"learning_rate": 0.01, "num_epochs": 5}.
Empty {} means the pipeline uses default parameters defined in the template.

enable_caching:
Determines whether caching is enabled for the pipeline steps.
False disables caching, ensuring all steps are re-executed even if the inputs and logic remain the same.
Submitting the Pipeline Job
After creating the PipelineJob object, you typically call the run method to submit it for execution:

python
Copy code
from google.cloud import aiplatform

# Initialize Vertex AI client (ensure you've set GOOGLE_APPLICATION_CREDENTIALS)
aiplatform.init(project="your-gcp-project-id", location="us-central1")

# Define pipeline job
pipeline_job = aiplatform.PipelineJob(
    display_name=f"{PARAM_TUNING}-{PIPELINE_NAME}-{TIMESTAMP}",
    template_path=TMP_PIPELINE_JSON,
    pipeline_root=PIPELINE_ROOT,
    parameter_values={},
    enable_caching=False,
)


# Submit the pipeline job
pipeline_job.run(sync=True)  # Use sync=False to run asynchronously
print(f"Pipeline job {pipeline_job.display_name} submitted.")


