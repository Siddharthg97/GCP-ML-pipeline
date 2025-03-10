***System arguments***
if __name__ == '__main__':
    args = json.loads(sys.argv[1]) 
    project_id = args["GCP_PROJECT"] 
    bq_dataset = args["MATERIALIZATION_DATASET"] 
    ds_temp_bucket = args["GCS_TEMP_BUCKET"]
    source_dataset = args["FS_SOURCE_DATASET"]


***Get or Create the Logger***

logging.getLogger('py4j'): Creates or retrieves a logger named 'py4j'.
2)Set Logging Level:
  logger.setLevel(logging.INFO): Configures the logger to handle messages at the INFO level and above (INFO, WARNING, ERROR, CRITICAL).

3)Create a Console Handler:
  ch = logging.StreamHandler(): Creates a handler that outputs log messages to the console.
  ch.setLevel(logging.INFO): Sets the console handler to only process messages at the INFO level or above.

4)Create a Formatter:
  logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'): Specifies the format for log messages, including:
  %(asctime)s: Timestamp of the log entry.
  %(name)s: Name of the logger ('py4j' in this case).
  %(levelname)s: Log level (INFO, ERROR, etc.).
  %(message)s: The actual log message.
  Attach Formatter to the Handler:

5)ch.setFormatter(formatter): Adds the formatter to the console handler.
6)Add Handler to the Logger:
  logger.addHandler(ch): Adds the console handler to the logger, enabling the formatted log messages to be displayed on the console.

Input code
  logger.info("This is an info message.")
  logger.warning("This is a warning message.")
  logger.error("This is an error message.")


Output code
2024-12-24 12:00:00 - py4j - INFO - This is an info message.
2024-12-24 12:00:01 - py4j - WARNING - This is a warning message.
2024-12-24 12:00:02 - py4j - ERROR - This is an error message.

  ***Input arguments for python notebook to get access to all resources***
  Generally how things work is we have have vpc that is virtual Private Cloud, which is a private cloud computing environment that's contained within a public cloud. VPCs 
  allow organizations to create and control their own virtual networks, which are isolated from other public cloud tenants.
  So within this we can create our own project having all the functionalities like biq query instance api , Vertex AI api etc.
  For all of these functionalities we define the region. All the users would require an account to access the projects having all
  functionalities which can be service account with iam access, admin account. These account should have ADID group access to these projects 

  
project_id = 'wmt-mlp-p-price-npd-pricing'
REGION = 'us-central1'
bq_dataset='markdown_pipeline'
ds_temp_bucket = 'markdown_pipeline'
source_dataset = 'markdown_pipeline'


A service account JSON file is required in many contexts when working with cloud services, such as Google Cloud Platform 
(GCP).It is a secure way to authenticate and authorize applications or scripts to access cloud resources programmatically.
service_json = json.loads(''' 
{
"type": "service_account",
  "project_id": "wmt--p-price-npd-pricing",
  "private_key_id": "",
  "private_key": "-----BEGIN PRIVATE KEY-----KEY-----\\n",
  "client_email": "svc-price--pricing-admin@wmt-mlp-p-price-npd-pricing.iam.gserviceaccount.com",
  "client_id": "",
  "auth_uri": "https://.google.com/o/oauth2/auth",
  "token_uri": "https://.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www..com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www..com/robot/v1/metadata/x509/svc-price-npd-pricing-admin%40wmt-mlp-p-price-npd-pricing.iam.gserviceaccount.com",
  "universe_domain": ".com"
}
''')
credentials = service_account.Credentials.from_service_account_info(service_json)
bq = bigquery.Client(credentials=credentials, project=credentials.project_id)

What is a Service Account JSON?
Definition:

A service account JSON file contains the credentials for a service account in GCP.
It allows applications to authenticate and interact with GCP services without requiring user intervention.
Contents:

The JSON file includes:
The service account's email address.
A unique ID for the service account.
A private key for secure authentication.
Scopes or permissions associated with the service account.

credentials = service_account.Credentials.from_service_account_info(service_json)
bq = bigquery.Client(credentials=credentials, project=credentials.project_id)


When is it Required?
Interacting with GCP APIs programmatically.
Running CI/CD pipelines that interact with GCP resources.
Using Google Cloud SDK or libraries outside of the GCP environment.
Automating workflows on resources like BigQuery, Cloud Storage, or Compute Engine.

***GCloud cli command to spawn data proc cluster***
gcloud dataproc clusters create $DP_CLUSTER_NAME \
 --region $GCP_REGION \
 --subnet projects/shared-vpc-admin/regions/$GCP_REGION/subnetworks/$GCP_NETWORK \
 --enable-component-gateway \
 --optional-components JUPYTER,ZEPPELIN \
 --no-address \
 --zone us-central1-a \
 --service-account $DP_SERVICE_ACCOUNT \
 --master-machine-type e2-standard-32\
 --master-boot-disk-size 1TB \
 --num-masters 1 \
 --num-workers 4 \
 --worker-machine-type e2-standard-32 \
 --worker-boot-disk-size 1TB \
 --project $GCP_PROJECT \
 --image-version=2.1-debian11 \
 --max-idle 3600s \
 --scopes 'https://www.googleapis.com/auth/cloud-platform' \
 --initialization-actions gs://mle-dataproc-artifacts/mle-dataproc-connector.sh \
 --properties=^#^dataproc:pip.packages='ibis-framework[bigquery]==7.1.0'#dataproc:pip.packages='great-expectations==0.16.8' \
 --metadata spark-bigquery-connector-version=0.32.2

gcloud dataproc jobs submit pyspark --cluster=markdown --region=us-central1 gs://markdown_pipeline/feature_test/engineer.py 
