# Part of codes are from Tyler and Khushboo's work
# Sampling  
import argparse
import calendar
import pyspark
from pyspark.sql import SparkSession
from typing import Tuple
import datetime
from datetime import date
from datetime import timedelta
from pyspark.sql import functions as F
from pyspark.sql.functions import *
from pyspark.sql import DataFrame
PySparkDF = pyspark.sql.dataframe.DataFrame
from pyspark.sql.window import Window
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import sys
import json
import time
from functools import reduce


def get_bq_spark_session(project_id, bq_dataset, ds_temp_bucket):
    return (
        SparkSession.builder.config("viewsEnabled", "true")
        .config("materializationProject", project_id)
        .config("materializationDataset", bq_dataset)
        .config("temporaryGcsBucket", ds_temp_bucket)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )


def read_bq_table_to_df(spark: SparkSession, sql_query: str) -> PySparkDF:
    return spark.read \
        .format('bigquery') \
        .option('query', sql_query) \
        .load()


def write_df_to_bq_table(df: PySparkDF, write_mode: str, bq_table: str) -> None:
    df.write \
        .mode(write_mode) \
        .format('bigquery') \
        .option('intermediateFormat', 'orc') \
        .option('table', bq_table) \
        .save()


def read_second_level_features() -> PySparkDF: # read second level features
    sql_query = f"""
        SELECT 
            *
        FROM
            {SECOND_LEVEL}
    """
    return sql_query
def read_second_level_features_filtered() -> PySparkDF:
    sql_query = f"""
        SELECT 
            *
        FROM
            {SECOND_LEVEL_FILTERED}
    """
    return sql_query
def read_second_level_eval():
    sql_query = f"""
           SELECT 
               *
           FROM
               {SECOND_LEVEL_EVAL}
           WHERE CAST(date as date) >= '{MAX_EVAL_START_DATE}'
       """
    return sql_query
def read_data_samples() -> PySparkDF:
    sql_query = f"""
        SELECT 
            *
        FROM
            {DATA_SAMPLES}
    """
    return sql_query
    
    # markdown start date is given by business, the idea behined start start and end date of markdown is that if during the start of season we have sufficinet inventoryour model may not recommend any markdown but possibly during th end of season it still
    # we hold sufficient inventory markdown our model might give sufficient markdwon before we reach to date of liquidation.
s
def get_wingman(): # contains manual markdowns - decided by markdown management as per their expertise  for atmost 1 year from todays date (MAX_EVAL_START_DATE)
    sql_query = f"""
        SELECT
            item_nbr,
            club_nbr,
            cast(md_start_date AS date) as markdown_start_date_dt,
            cast(md_end_date AS date) as markdown_end_date_dt,
            sp as markdown_price_amt
        FROM {WINGMAN_EXECUTION}
        WHERE club_nbr != 6279
        AND cast(md_start_date AS date) >= '{MAX_EVAL_START_DATE}'
        and  retailreasoncode not like '%DS%'   #??
        and retailreasoncode like '%EOL%'              #??
    """
    return sql_query


#Mostly markdown managers/planning team decide the markdown start and end dates (end may be optional) and oos_date. We recommend markdown session dates and price
def get_logs():    # contains details of data science executed markdowns given by DS teams and they are fairly different than manual markdowns becuase the markdown were given by DS teams
    sql_query = f"""  # it is at item club date level for a tenure not more than 1 year from todays date
        SELECT DISTINCT
            item_nbr,
            club_nbr,
            liquidation_value as liquidation_price,
            predicted_md_units as expected_sale_units, # Q2 predicted target ? not meaningfulf for us since we don't use it
            current_retail as current_retail_price,
            current_units as current_inventory,
            markdown_retail as recommended_markdown_price, # Q3 the markdown given input is by whom - markdown managers? #not it is given per optimization chose best markdown from DS modelsted markdwon price in past (markdown managers)
            cast(cast(oos_date as datetime) as date) as oos_date,
            cast(cast(markdown_start_date as datetime) as date) as markdown_start_date_dt,
            cast(cast(created_ts as datetime) as date) as date,
            cast(plan_item_club_details.plan_id as integer) as plan_id
        FROM {PLAN_ITEM_CLUB_DETAILS} AS plan_item_club_details #   Q1 how are we getting/generating this table  - from executed markdowns tables created
        INNER JOIN {PLAN_DETAILS} as plan_details       
        ON plan_item_club_details.plan_id = plan_details.plan_id
        WHERE plan_details.plan_status_code in (10,11)
        AND  cast(cast(created_ts as datetime) as date) >= '{MAX_EVAL_START_DATE}'
        and plan_item_club_details.markdown_reason_code like '%Data%'
        # and trim(plan_item_club_details.markdown_reason_code) = 'Data science' # markdowns that were planned to be used by DS models
    """
    return sql_query

def get_item_dim() -> PySparkDF:
    sql_query = f"""
        SELECT 
          DISTINCT
          MDS_FAM_ID as system_item_nbr,
          item_nbr,
          dept_nbr,
          subclass_nbr
        FROM
          {ITEM_DIM}
        WHERE current_ind = 'Y' ?? Q3 all these filters 
        AND country_code = 'US'
        AND base_div_nbr = 18
        AND subclass_nbr not in (61, 89, 91, 97)
    """
    return sql_query



def get_data_samples_eval_plan(day_sales_base: PySparkDF, plan_logs: PySparkDF) -> Tuple[PySparkDF, PySparkDF]: # sample eval data creatipon for DS executed markdown
    day_sales_base=day_sales_base.withColumn("date", F.to_timestamp(col("date"))).withColumn("week", date_format(col("date"), "w")) #week number
    day_sales_base=day_sales_base.withColumn("date", F.to_timestamp(col("date"))).withColumn("year", date_format(col("date"), "Y"))  # Q6 to_timestamp and date_format
    day_sales_base=day_sales_base.withColumn("date", F.to_timestamp(col("date"))).withColumn("month", date_format(col("date"), "M"))

    day_sales_base = day_sales_base.join(plan_logs, on=['item_nbr','club_nbr','date'],how='inner') # join second level eval table and ds excuted markdown table
    day_sales_base = day_sales_base.filter(F.col('markdown_start_date_dt').isNotNull()) 
# markdown_start_date is must have feature ,apart from that also evaluation would be for model performance based on dates on which markdown happened . This sample data creation is not for training  
    day_sales_base = day_sales_base.filter(F.col("avg_price_1_week_back_amt") != 0) ##Q7 is this useful ?
    day_sales_base = day_sales_base.withColumn("avg_price_1_week_back_amt", least(day_sales_base["avg_price_1_week_back_amt"], day_sales_base["current_retail_price"]))# comparing todays price and avg weekly price min in past 4 weeks, then choosing the min
    for i in range(1, 9):
        day_sales_base = day_sales_base.withColumn(f'discount_{i}_week_next_nbr', 1 - F.col(f'avg_price_charged_{i}_week_next_amt')/F.col('avg_price_1_week_back_amt')) # again computing the discount using based on new avg weekly price 1 week back


    # Split each sample to eight samples with 1 week duration, 2-week duration,..., 8-week duration
    data_samples = day_sales_base.withColumn('target', F.col(f'unit_sold_8_week_next_cnt')).withColumn('num_weeks',F.lit(8)).withColumn('week_target', F.col(f'unit_sold_8_week_next_cnt')-F.col(f'unit_sold_7_week_next_cnt'))
    for i in range(7, 0, -1):
        day_sales_base = day_sales_base.withColumn('target', F.col(f'unit_sold_{i}_week_next_cnt'))
        day_sales_base = day_sales_base.withColumn('num_weeks', F.lit(i))
        day_sales_base = day_sales_base.withColumn(f'discount_{i+1}_week_next_nbr', F.lit(-1.00)) 
        try:
            day_sales_base = day_sales_base.withColumn('week_target', F.col(f'unit_sold_{i}_week_next_cnt')-F.col(f'unit_sold_{i-1}_week_next_cnt'))
        except:
            day_sales_base = day_sales_base.withColumn('week_target', F.col(f'unit_sold_{i}_week_next_cnt'))

        data_samples = data_samples.unionAll(day_sales_base) # here we shall be creating new df i.e. data samples appending the same row for 8 times but with diff target and num weeks.

    data_samples = data_samples.withColumn('week_inventory_expected_to_last_cnt', # Q7
                                           F.col('inventory_1_month_last_weekly_sale_nbr') / F.col('num_weeks'))
    data_samples = data_samples.withColumn('ratio_target', F.col('target') / F.col('inventory_qty'))    

    
    # Get the over_sell_ratio= weekly sales at the discounted price/ weekly sales of the previous four weeks
    # Get the under_sell_ratio= weekly sales of the previous four weeks/weekly sales at the discounted price
    # Remove the samples that has extrem over-sell or under-sell
    data_samples = data_samples.withColumn('over_sell_ratio',
                                           F.col('target') / F.col('num_weeks') / F.col('avg_weekly_unit_sold_1_month_back_cnt')) \# this ratio identifies the peaks/anomalies 
        .withColumn('under_sell_ratio', F.col('avg_weekly_unit_sold_1_month_back_cnt') / F.col('target') * F.col('num_weeks'))
    # Q8 Where is the real significance for these ratios
    w=Window.partitionBy('item_nbr','club_nbr')
    data_samples=data_samples.withColumn('max_oos_date',F.max(F.col('oos_date')).over(w)).filter(F.col('date')<=F.col('max_oos_date')) # filtering only the data such that, present date should be less than max oos date, for sample collection for an item club
    data_samples = data_samples.filter(F.col('markdown_start_date_dt').isNotNull()) # repeated # Q what about the case where oos is less than present date , but max oos date is ahead of present date.

    eval_data = data_samples.filter(
        F.datediff(F.col('oos_date'), F.col('date')) / 7 >= F.col('num_weeks')).filter(  # 
        F.col('date') < F.col('oos_date')).filter(F.datediff(F.lit(TEST_END_DATE), F.col('date')) / 7 >= F.col('num_weeks')) #Q test end date ??
#     eval_data = eval_data.withColumn('target', when(F.col('target') == 0, 1).otherwise(F.col('target')))
    w = Window.partitionBy('mds_fam_id', 'club_nbr', 'date','target').orderBy(F.asc("num_weeks"))
    eval_data = eval_data.withColumn("rn_num_weeks", F.row_number().over(w)).filter(F.col('rn_num_weeks') == 1)
    # Filter out rows having value as 0 in all 'units_sum_next_{i}_weeks' columns
    columns_to_check = [f'unit_sold_{i}_week_next_cnt' for i in range(1, 9)]
    eval_data = eval_data.filter(~((F.col(columns_to_check[0]) == 0) &
                                             (F.col(columns_to_check[1]) == 0) &
                                             (F.col(columns_to_check[2]) == 0) &
                                             (F.col(columns_to_check[3]) == 0) &
                                             (F.col(columns_to_check[4]) == 0) &
                                             (F.col(columns_to_check[5]) == 0) &
                                             (F.col(columns_to_check[6]) == 0) &
                                             (F.col(columns_to_check[7]) == 0)))
    eval_data_plan = eval_data.filter(
        (F.col('date') >= EVAL_START_DATE) & (F.col('date') <= TEST_END_DATE)) 
    eval_data_plan_complete = eval_data.filter(
        (F.col('date') >= MAX_EVAL_START_DATE) & (F.col('date') <= TEST_END_DATE))
    eval_data_plan_complete = eval_data_plan_complete.select('mds_fam_id','item_nbr', 'club_nbr','date', 'num_weeks', 'markdown_start_date_dt', 'oos_date', 'target', 'week_target')
                                                     
    return eval_data_plan, eval_data_plan_complete


def get_data_samples_eval_wingman(day_sales_base: PySparkDF, wingman: PySparkDF) -> PySparkDF: # eval data sample creation for excuted manual markdowns
    
    day_sales_base=day_sales_base.withColumn("date", F.to_timestamp(col("date"))).withColumn("week", date_format(col("date"), "w")) #week number
    day_sales_base=day_sales_base.withColumn("date", F.to_timestamp(col("date"))).withColumn("year", date_format(col("date"), "Y"))
    day_sales_base=day_sales_base.withColumn("date", F.to_timestamp(col("date"))).withColumn("month", date_format(col("date"), "M"))

    wingman = wingman.withColumn('item_nbr', F.col('item_nbr').cast('Integer'))
    wingman = wingman.withColumn('club_nbr', F.col('club_nbr').cast('Integer'))
    wingman = wingman.withColumn('date', F.col('markdown_start_date_dt'))
    day_sales_base = day_sales_base.withColumn('item_nbr', F.col('item_nbr').cast('Integer'))
    day_sales_base = day_sales_base.join(wingman, on = ['item_nbr', 'club_nbr', 'date'], how = 'inner')
    day_sales_base = day_sales_base.filter(F.col('markdown_start_date_dt').isNotNull()) # only those item club dates on which markdown happened
    # day_sales_base = day_sales_base.filter(F.col("avg_price_1_week_back_amt") != 0)
    # day_sales_base = day_sales_base.withColumn("avg_price_1_week_back_amt", least(day_sales_base["avg_price_1_week_back_amt"], day_sales_base["current_retail_price"]))
    # for i in range(1, 9):
    #     day_sales_base = day_sales_base.withColumn(f'discount_{i}_week_next_nbr', 1 - F.col(f'avg_price_charged_{i}_week_next_amt')/F.col('avg_price_1_week_back_amt'))


    # Split each sample to eight samples with 1 week duration, 2-week duration,..., 8-week duration
    data_samples = day_sales_base.withColumn('target', F.col(f'unit_sold_8_week_next_cnt')).withColumn('num_weeks',
                                                                                                      F.lit(8))
    for i in range(7, 0, -1):
        day_sales_base = day_sales_base.withColumn('target', F.col(f'unit_sold_{i}_week_next_cnt'))
        day_sales_base = day_sales_base.withColumn('num_weeks', F.lit(i))
        day_sales_base = day_sales_base.withColumn(f'discount_{i+1}_week_next_nbr', F.lit(-1.00))

        data_samples = data_samples.unionAll(day_sales_base)

    data_samples = data_samples.withColumn('week_inventory_expected_to_last_cnt',
                                           F.col('inventory_1_month_last_weekly_sale_nbr') / F.col('num_weeks'))
    data_samples = data_samples.withColumn('ratio_target', F.col('target') / F.col('inventory_qty'))    

    
    # Get the over_sell_ratio= weekly sales at the discounted price/ weekly sales of the previous four weeks
    # Get the under_sell_ratio= weekly sales of the previous four weeks/weekly sales at the discounted price
    # Remove the samples that has extrem over-sell or under-sell
    data_samples = data_samples.withColumn('over_sell_ratio',
                                           F.col('target') / F.col('num_weeks') / F.col('avg_weekly_unit_sold_1_month_back_cnt')) \
        .withColumn('under_sell_ratio', F.col('avg_weekly_unit_sold_1_month_back_cnt') / F.col('target') * F.col('num_weeks'))
    
    w=Window.partitionBy('item_nbr') # throughout past 35 months
    data_samples=data_samples.withColumn('oos_date',F.max(F.col('markdown_end_date_dt')).over(w))
    data_samples = data_samples.filter(F.col('markdown_start_date_dt').isNotNull()) 
    eval_data = data_samples.filter(
        (F.col('date') >= EVAL_START_DATE) & (F.col('date') <= TEST_END_DATE)) 
    eval_data = eval_data.filter(
        F.datediff(F.col('oos_date'), F.col('date')) / 7 >= F.col('num_weeks')).filter(
        F.col('date') < F.col('oos_date')).filter(F.datediff(F.lit(TEST_END_DATE), F.col('date')) / 7 >= F.col('num_weeks'))
#     eval_data = eval_data.withColumn('target', when(F.col('target') == 0, 1).otherwise(F.col('target')))
    w = Window.partitionBy('mds_fam_id', 'club_nbr', 'date','target').orderBy(F.asc("num_weeks"))
    eval_data = eval_data.withColumn("rn_num_weeks", F.row_number().over(w)).filter(F.col('rn_num_weeks') == 1)
    # Filter out rows having value as 0 in all 'units_sum_next_{i}_weeks' columns
    columns_to_check = [f'unit_sold_{i}_week_next_cnt' for i in range(1, 9)]
    eval_data_wingman = eval_data.filter(~((F.col(columns_to_check[0]) == 0) &
                                             (F.col(columns_to_check[1]) == 0) &
                                             (F.col(columns_to_check[2]) == 0) &
                                             (F.col(columns_to_check[3]) == 0) &
                                             (F.col(columns_to_check[4]) == 0) &
                                             (F.col(columns_to_check[5]) == 0) &
                                             (F.col(columns_to_check[6]) == 0) &
                                             (F.col(columns_to_check[7]) == 0)))
                                                     
    return eval_data_wingman




def column_add(a, b):
    return a.__add__(b)

def get_initial_filter(day_sales_base: PySparkDF) -> PySparkDF:
    """
    Apply filters and transformations to the `day_sales_base` DataFrame.
    Args:
        day_sales_base (PySparkDF): The input DataFrame.
    Returns:
        PySparkDF: The filtered and transformed DataFrame.
    """

    day_sales_base=day_sales_base.withColumn("date", F.to_timestamp(col("date"))).withColumn("week", date_format(col("date"), "w")) #week number
    day_sales_base=day_sales_base.withColumn("date", F.to_timestamp(col("date"))).withColumn("year", date_format(col("date"), "Y"))
    day_sales_base=day_sales_base.withColumn("date", F.to_timestamp(col("date"))).withColumn("month", date_format(col("date"), "M"))

    # Apply filter to remove rows with negative values and missing values in week_1_discount to week_8_discount columns
    for col_name in [f'discount_{i}_week_next_nbr' for i in range(1, 9)]:
        day_sales_base = day_sales_base.withColumn(col_name, F.coalesce(col_name, 'max_weekly_discount_4_week_back_amt')).filter(
            F.col(col_name) >= 0) #Q9 where do we define this max_weekly_discount_4_week_back_amt ?

    # Filter out rows having value as 0 in all 'units_sum_next_{i}_weeks' columns
    columns_to_check = [f'unit_sold_{i}_week_next_cnt' for i in range(1, 9)] # Suppose for an item club date if units sold 3 weeks ahead is zero , then should remove row for the same
    day_sales_base = day_sales_base.filter(~((F.col(columns_to_check[0]) == 0) &
                                             (F.col(columns_to_check[1]) == 0) &
                                             (F.col(columns_to_check[2]) == 0) &
                                             (F.col(columns_to_check[3]) == 0) &
                                             (F.col(columns_to_check[4]) == 0) &
                                             (F.col(columns_to_check[5]) == 0) &
                                             (F.col(columns_to_check[6]) == 0) &
                                             (F.col(columns_to_check[7]) == 0)))

    # Calculate the sum of weekly discount as weights and apply filters
    day_sales_base = day_sales_base.withColumn('weights', F.expr('+'.join([f'discount_{i}_week_next_nbr' for i in range(1, 9)]))) \
        .filter(F.col('weights') >= 0) \
        .withColumn('weights', F.abs('weights')) \
        .withColumn('weights', F.round(F.col('weights') + 1, 2))

    # Apply window function to get the row number and filter out rows with rn != 1
    w = Window.partitionBy('mds_fam_id', 'club_nbr', 'month', 'year').orderBy(F.desc('weights'))
    day_sales_base = day_sales_base.withColumn("rn", F.row_number().over(w)).filter(F.col('rn') == 1)

    return day_sales_base

def expand_to_8_weeks(day_sales_base: PySparkDF) -> PySparkDF:
    """
    Expands the given PySpark DataFrame to include 8 weeks of data.
    Args:
        day_sales_base (PySparkDF): The base DataFrame containing day sales data.
    Returns:
        PySparkDF: The expanded DataFrame with additional columns and filters applied.
    """
    data_samples = day_sales_base.withColumn('target', F.col(f'unit_sold_8_week_next_cnt')).withColumn('num_weeks', F.lit(8))
    for i in range(7, 0, -1):
        day_sales_base = day_sales_base.withColumn('target', F.col(f'unit_sold_{i}_week_next_cnt'))
        day_sales_base = day_sales_base.withColumn('num_weeks', F.lit(i))
        day_sales_base = day_sales_base.withColumn(f'discount_{i+1}_week_next_nbr', F.lit(-1.00))
        data_samples = data_samples.unionAll(day_sales_base)

    data_samples = data_samples.withColumn('week_inventory_expected_to_last_cnt', F.col('inventory_1_month_last_weekly_sale_nbr') / F.col('num_weeks'))
    data_samples = data_samples.filter(F.col('inventory_qty') > 3) # Q10 ?
    data_samples = data_samples.withColumn('ratio_target', F.col('target') / F.col('inventory_qty'))
    data_samples = data_samples.filter(F.col('inventory_qty') >= F.col('target'))
    data_samples = data_samples.filter(F.col('date') > CUT_OFF_DATE2) # cut off date ?


    data_samples = data_samples.withColumn('over_sell_ratio', F.col('target') / F.col('num_weeks') / F.col('avg_weekly_unit_sold_1_month_back_cnt')) \
        .withColumn('under_sell_ratio', F.col('avg_weekly_unit_sold_1_month_back_cnt') / F.col('target') * F.col('num_weeks'))

    return data_samples


def split_train_val_test(data_samples: PySparkDF) -> PySparkDF:
    train = data_samples.filter( #Q  we generate train data from sample data , that removes non markdown start date ??
        (F.col('date') >= TRAIN_START_DATE) & (F.col('date') <= TRAIN_END_DATE))
    val = data_samples.filter((F.col('date') >= VAL_START_DATE) & (F.col('date') <= VAL_END_DATE))
    test = data_samples.filter(
        (F.col('date') >= TEST_START_DATE) & (F.col('date') <= TEST_END_DATE))

    train = train.filter(F.datediff(F.lit(TRAIN_END_DATE), F.col('date')) / 7 >= F.col('num_weeks'))
    # Use test_end_date for val so that we can get long-period samples
    val = val.filter(F.datediff(F.lit(TEST_END_DATE), F.col('date')) / 7 >= F.col('num_weeks'))
    test = test.filter(F.datediff(F.lit(TEST_END_DATE), F.col('date')) / 7 >= F.col('num_weeks'))
    # Since the oos date may not be accurate, we remove all the samples with zero sales in that week
    w = Window.partitionBy('mds_fam_id', 'club_nbr', 'date','target').orderBy(F.asc("num_weeks"))
    train= train.withColumn("rn_num_weeks", F.row_number().over(w)).filter(F.col('rn_num_weeks') == 1)
    test= test.withColumn("rn_num_weeks", F.row_number().over(w)).filter(F.col('rn_num_weeks') == 1)
    val= val.withColumn("rn_num_weeks", F.row_number().over(w)).filter(F.col('rn_num_weeks') == 1)

    return train, val, test


def generate_table_name(project_id, bq_dataset , table_name):

    return str(project_id+"."+bq_dataset+"."+table_name)

def drop_columns(df: DataFrame, columns: list) -> DataFrame:
    existing_columns = set(df.columns)
    columns_to_drop = [col for col in columns if col in existing_columns]
    return df.drop(*columns_to_drop)

if __name__ == '__main__':
    args = json.loads(sys.argv[1]) 
    project_id = args["GCP_PROJECT"] 
    bq_dataset = args["MATERIALIZATION_DATASET"] 
    ds_temp_bucket = args["GCS_TEMP_BUCKET"]
    source_dataset = args["FS_SOURCE_DATASET"]
    print('Log part0 done')


    
    spark = get_bq_spark_session(project_id, bq_dataset, ds_temp_bucket)
    spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")

    SECOND_LEVEL = generate_table_name(project_id, source_dataset, "club_item_daily_v1")
    WINGMAN_EXECUTION = generate_table_name(project_id, "markdown", "md_info")
    PLAN_ITEM_CLUB_DETAILS = 'prod-sams-cdp.prod_sams_merch_markdown_optimization_markdown_optimization.plan_item_club_details'
    PLAN_DETAILS = 'prod-sams-cdp.prod_sams_merch_markdown_optimization_markdown_optimization.plan_details'
    ITEM_DIM = "wmt-edw-prod.WW_CORE_DIM_VM.ITEM_DIM"


    SECOND_LEVEL_FILTERED=generate_table_name(project_id, "markdown", "inclub_second_level_features_filtered")
    SECOND_LEVEL_EVAL=generate_table_name(project_id, "markdown", "inclub_second_level_features_eval")
    TODAY = str((date.today() - timedelta(days=1)).strftime("%Y-%m-%d"))

    # cut_off_date: only use transaction data after this date (600-day period).
    # cut_off_date2: since the model needs the past one month sales, all features should be after this date.
    TODAY_DATE = datetime.datetime.today()
    CUT_OFF_DATE = (datetime.datetime.today() - datetime.timedelta(days=30 * 35)).strftime("%Y-%m-%d")
    CUT_OFF_DATE2 = (datetime.datetime.strptime(CUT_OFF_DATE, "%Y-%m-%d") + datetime.timedelta(days=30)).strftime(
        "%Y-%m-%d")
    TRAIN_START_DATE = (datetime.datetime.strptime((datetime.datetime.today()-datetime.timedelta(days=30*20)).strftime("%Y-%m-%d"), "%Y-%m-%d")+datetime.timedelta(days=30)).strftime("%Y-%m-%d")
    TRAIN_END_DATE = (datetime.datetime.today() - datetime.timedelta(days=30 * 4)).strftime("%Y-%m-%d")
    VAL_START_DATE = (datetime.datetime.today() - datetime.timedelta(days=30 * 3 - 1)).strftime("%Y-%m-%d")
    VAL_END_DATE = (datetime.datetime.today() - datetime.timedelta(days=30 * 2)).strftime("%Y-%m-%d")
    TEST_START_DATE = (datetime.datetime.today() - datetime.timedelta(days=30 * 1 - 1)).strftime("%Y-%m-%d")
    TEST_END_DATE = (datetime.datetime.today() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    EVAL_START_DATE = (datetime.datetime.today() - datetime.timedelta(days=30*4)).strftime("%Y-%m-%d")
    MAX_EVAL_START_DATE = (datetime.datetime.today() - datetime.timedelta(days=365)).strftime("%Y-%m-%d")

    # Output tables
    output_table_train = generate_table_name(project_id, "markdown", "inclub_train_features")
    output_table_val = generate_table_name(project_id, "markdown", "inclub_val_features")
    output_table_test = generate_table_name(project_id, "markdown", "inclub_test_features")

    #inference_table = generate_table_name(project_id, bq_dataset, "inclub_today_features")
    eval_table_plan = generate_table_name(project_id, "markdown", "DS_logs_eval")
    eval_table_wingman = generate_table_name(project_id, "markdown", "DS_manual_eval")
    eval_table_plan_complete = generate_table_name(project_id, "markdown", "DS_logs_eval_complete")
    DATA_SAMPLES = generate_table_name(project_id, "markdown", "inclub_data_samples")

    cols_to_drop = ['unix_time','week_start_date_ly','week_ly','year_ly','month_ly','scnbr','inbr','customer_item_nbr','customer_club_nbr', 'day', 'week_day']

    second_level_features_sql=read_second_level_features()
    second_level_features=read_bq_table_to_df(spark,second_level_features_sql)

    print('Log part1 done') # reading second_level_features

    logs_sql = get_logs()
    plan_logs=read_bq_table_to_df(spark,logs_sql) 

    wingman_sql = get_wingman()
    wingman=read_bq_table_to_df(spark,wingman_sql)

    item_dim_sql = get_item_dim()
    item_dim=read_bq_table_to_df(spark,item_dim_sql)

    second_level_features_filtered = get_initial_filter(second_level_features) # Q What are we filtering in second level features ? 
    second_level_features_filtered = drop_columns(second_level_features_filtered,cols_to_drop)
    write_df_to_bq_table(
        df=second_level_features_filtered,
        write_mode='overwrite',
        bq_table=SECOND_LEVEL_FILTERED
    )
    print('Log part2 done') # we get final filtered second_level_features from here

    # second_level_features_eval=second_level_features.filter(F.col('markdown_start_date_dt').isNotNull())
    second_level_features_eval = drop_columns(second_level_features,cols_to_drop) # Q what are the columns to drop refer to ?
    write_df_to_bq_table(
        df=second_level_features_eval,
        write_mode='overwrite',
        bq_table=SECOND_LEVEL_EVAL
    )

    print('Log part3 done')
    covariates = ['discount_1_week_next_nbr', 'discount_2_week_next_nbr', 'discount_3_week_next_nbr', 'discount_4_week_next_nbr', 'discount_5_week_next_nbr',
                  'discount_6_week_next_nbr', 'discount_7_week_next_nbr', 'discount_8_week_next_nbr', 'club_nbr', 'department_nbr', 'subclass_nbr',
                  'median_price_6_month_last_amt', 'price_1_week_back_median_price_6_month_last_nbr', 'price_2_week_back_median_price_6_month_last_nbr',
                  'price_3_week_back_median_price_6_month_last_nbr', 'price_4_week_back_median_price_6_month_last_nbr', 'avg_weekly_unit_sold_1_month_back_cnt', 'week_inventory_expected_to_last_cnt',
                  'day_on_shelf_cnt', 'num_weeks', 'unit_sold_1_week_back_cnt', 'unit_sold_2_week_back_cnt',
                  'unit_sold_3_week_back_cnt', 'unit_sold_4_week_back_cnt', 'month', 'week', 'avg_unit_sold_subcategory_52_week_back_cnt', 'change_unit_sold_subcategory_same_week_1_year_back_cnt',
                  'avg_unit_sold_dept_52_week_back_cnt', 'avg_unit_sold_52_week_back_cnt', 'change_unit_sold_1_2_week_back_cnt', 'change_unit_sold_2_3_week_back_cnt', 'change_unit_sold_3_4_week_back_cnt', 'subclass_unit_sold_same_week_1_year_back_nbr']


    plan_eval, plan_eval_complete = get_data_samples_eval_plan(read_bq_table_to_df(spark,read_second_level_eval()), plan_logs)# reading second level eval refers to picking data from table SECOND_LEVEL_EVAL but date is greater than markdown eval start date
    plan_eval = drop_columns(plan_eval,cols_to_drop)  # 
    write_df_to_bq_table(
        df=plan_eval,
        write_mode='overwrite',
        bq_table=eval_table_plan
    )
    write_df_to_bq_table(
        df=plan_eval_complete,
        write_mode='overwrite',
        bq_table=eval_table_plan_complete
    )
    
    print('Log part4 done')
    wingman_eval = get_data_samples_eval_wingman(read_bq_table_to_df(spark,read_second_level_eval()), wingman)
    wingman_eval = drop_columns(wingman_eval,cols_to_drop)
    write_df_to_bq_table(
        df=wingman_eval,
        write_mode='overwrite',
        bq_table=eval_table_wingman
    )
    print('Log part5 done')
    #
    data_samples=expand_to_8_weeks(read_bq_table_to_df(spark,read_second_level_features_filtered()))
    data_samples = drop_columns(data_samples,cols_to_drop)
    write_df_to_bq_table(
        df=data_samples,
        write_mode='overwrite',
        bq_table=DATA_SAMPLES
    )
    print('Log part6 done')
    train, val, test = split_train_val_test(read_bq_table_to_df(spark, read_data_samples()))

    # train = drop_columns(train,cols_to_drop)
    select_cols = covariates+['target','inventory_qty','item_nbr', 'mds_fam_id','date']
    train=train.select(select_cols)
    write_df_to_bq_table(
        df=train,
        write_mode='overwrite',
        bq_table=output_table_train
    )
    print('Log part7 done')
    # val = drop_columns(val,cols_to_drop)
    val=val.select(select_cols)
    write_df_to_bq_table(
        df=val,
        write_mode='overwrite',
        bq_table=output_table_val
    )
    print('Log part8 done')
    # test = drop_columns(test,cols_to_drop)
    test=test.select(select_cols)
    write_df_to_bq_table(
        df=test,
        write_mode='overwrite',
        bq_table=output_table_test
    )
    print('Log part9 done')