import pyspark
from pyspark.sql import SparkSession
from collections import defaultdict
import great_expectations as gx
import pandas as pd
import numpy as np
import datetime
import sys
import json
from datetime import date, timedelta


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
    
    spark = get_bq_spark_session(project_id, bq_dataset, ds_temp_bucket)
    spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")

    CUT_OFF_DATE_END = str(date.today() - timedelta(days=2))
    CUT_OFF_DATE_START = (datetime.datetime.today() - datetime.timedelta(days=8)).strftime("%Y-%m-%d")
    
    output_table = generate_table_name(project_id, bq_dataset, "disaggregation_logic_ingress_check")
    MDSE_INV_DLY = "prod-sams-cdp.US_SAMS_PRODUCT360_CDP_VM.MDSE_INVENTORY_DLY"
    SCAN = "wmt-edw-prod.US_WC_MB_VM.SCANX"


    ingress_features = f"""
        SELECT scan_id, unit_qty,visit_nbr,visit_date
        FROM {SCAN}
        WHERE visit_date>='{CUT_OFF_DATE_START}' AND visit_date<='{CUT_OFF_DATE_END}'
        LIMIT 100
    """

    ingress_df = spark.read.format("bigquery").option("query", ingress_features).load()
    json_gx_ingress = gx.dataset.SparkDFDataset(ingress_df)

    # Non-null Check
    nonnull_check_list = [pdf_feature_quality(json_gx_ingress.expect_column_values_to_not_be_null(column=col)) for col in ['scan_id']]
    # Type Check
    type_check_list = [pdf_feature_quality(json_gx_ingress.expect_column_values_to_be_of_type(column="scan_id", type_="IntegerType")),
                       pdf_feature_quality(json_gx_ingress.expect_column_values_to_be_of_type(column="visit_nbr", type_="IntegerType"))]

    pdf_ingress_quality = pd.concat(nonnull_check_list + type_check_list, ignore_index=True)
    df_ingress_quality = spark.createDataFrame(pdf_ingress_quality)

    # Save the spark dataframe in BigQuery
    df_ingress_quality.write.mode("append").format("bigquery").option("intermediateFormat", "orc").\
        option("table", output_table).save()