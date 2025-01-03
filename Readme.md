### Steps to create GCP ML pipeline

1) Settings.yml file - contains model configuration & data location
2) ML pipeline scripts - 
3) Utils module  - contains function definitions
4) base flow image 
5) ML flow image
6) Pipeline_utils
7) Decryption


To create ML flow pipeline we need containerized applications conatining -python configurations, 




**Notes** 
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

