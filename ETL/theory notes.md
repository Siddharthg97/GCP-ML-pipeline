****Airflow***
https://www.youtube.com/watch?v=YgodScEIbOc&t=823s
https://medium.com/thefork/a-guide-to-mlops-with-airflow-and-mlflow-e19a82901f88
https://airflow.apache.org/docs/apache-airflow/stable/index.html
https://www.youtube.com/watch?v=d4cu_rzv4A8


1) It is a common tool for managing data workflows. Built on Apache airflow is open source and needs to be installed on existing
infrastructure

3) Airflow is a platform for building and running workflows, represented as a DAG (a Directed Acyclic Graph), and contains individual pieces of work called Tasks, arranged with dependencies and data flows 
taken into account to say how they should be executed.
Since every Data Science project needs its own configuration, essentially metadata and model versions, we can wrap up 
everything inside a single DAG, build and schedule it in the enterprise ETL.

4) It defines multiple tasks and dictates in which order they have to run and which tasks depend in what others. 
The DAG is only concerned with how to execute the tasks, the order to run them in, in which frequency,
how many times to retry them and if they have timeouts, and so on.

***How to create DAGs***
1) using google cloud composer  - create it's environment which are self contained airflow deployments running on top of Kubernates cluster
So, in the configurations of google cloud composer, we specify the type whether it's auto scalable or configurable with airflow-1 or -2
In configuration we need to specify the image version for composer along with airflow type and service account that has access go the project
in which the composer resides.
NOTE : We can have several composer environment created within a project in location suppprted by project.
2) We can provide architechture for composer environment and provide vpc network setting as well.
3) Once composer environment is created we can see that against this environment we have folders like Dag, airflow webserver
being created. Now the dag folder stores the gcs bucket in which python script is present containing the dag script to be executed  
Airflow webserver is the is the UI in which all dags are present irrespective of whether active , paused or failed etc.
NOTE : We would need to define the operator to build a DAG. These can be bash operator , python operator etc.


Within composer environment hyper link we have option of PYPI packages where we would need to define the package & version 
to be installed to run dags.
