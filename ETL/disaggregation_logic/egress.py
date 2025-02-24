import pyspark
from pyspark.sql import SparkSession
from collections import defaultdict
import great_expectations as gx
import pandas as pd
import numpy as np
import datetime
import sys
import json


def get_bq_spark_session(project_id, bq_dataset, ds_temp_bucket):
    return (
        SparkSession.builder.config("viewsEnabled", "true")
        .config("materializationProject", project_id)
        .config("materializationDataset", bq_dataset)
        .config("temporaryGcsBucket", ds_temp_bucket)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )

def pdf_feature_quality(json_quality):
    result_quality = defaultdict(str)
    result_quality["utc_time"] = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    result_quality["column"] = json_quality["expectation_config"]["_kwargs"]["column"]
    result_quality["expectation_type"] = json_quality["expectation_config"]["_expectation_type"]
    result_quality["success"] = json_quality["success"]
    result_quality["result"] = [str(key) + ':' + str(val) for key, val in json_quality["result"].items()]
    return pd.DataFrame.from_dict([result_quality])

def generate_table_name(project_id, bq_dataset, table_name):

    return str(project_id+"."+"markdown"+"."+table_name)


if __name__ == '__main__':
    
    args = json.loads(sys.argv[1]) 
    project_id = args["GCP_PROJECT"] 
    bq_dataset = args["MATERIALIZATION_DATASET"] 
    ds_temp_bucket = args["GCS_TEMP_BUCKET"]
    project_id_disaggregation = 'wmt-mlp-p-price-npd-pricing'
    
    spark = get_bq_spark_session(project_id_disaggregation, bq_dataset, ds_temp_bucket)
    spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")

    input_table = generate_table_name(project_id_disaggregation, bq_dataset, "disaggregation_weights")
    output_table = generate_table_name(project_id_disaggregation, bq_dataset, "disaggregation_weights_egress_check")

    egress_features = f"""
        SELECT parent_item_nbr, child_item_nbr, club_nbr, scenario, final_weights
        FROM {input_table}
        LIMIT 100
    """

    egress_df = spark.read.format("bigquery").option("query", egress_features).load()
    json_gx_egress = gx.dataset.SparkDFDataset(egress_df)

    # Non-null Check
    nonnull_check_list = [pdf_feature_quality(json_gx_egress.expect_column_values_to_not_be_null(column=col)) for col in ["item_nbr","mds_fam_id",'club_nbr']]
    # Type Check
    type_check_list = [pdf_feature_quality(json_gx_egress.expect_column_values_to_be_of_type(column="parent_item_nbr", type_="IntegerType")),
                       pdf_feature_quality(json_gx_egress.expect_column_values_to_be_of_type(column="child_item_nbr", type_="IntegerType")),
                       pdf_feature_quality(json_gx_egress.expect_column_values_to_be_of_type(column="club_nbr", type_="IntegerType"))]

    pdf_egress_quality = pd.concat(nonnull_check_list + type_check_list, ignore_index=True)
    df_egress_quality = spark.createDataFrame(pdf_egress_quality)
    # Save the spark dataframe in BigQuery
    df_egress_quality.write.mode("append").format("bigquery").option("intermediateFormat", "orc").\
        option("table", output_table).save()