# Part of codes are from Tyler and Khushboo's work
import argparse
import calendar
import pyspark
from pyspark.sql import SparkSession
from typing import Tuple
import datetime
from datetime import date, timedelta
from pyspark.sql import functions as F
from pyspark.sql.functions import *
PySparkDF = pyspark.sql.dataframe.DataFrame
from pyspark.sql.window import Window
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import sys
import json
import logging

# Create or get the logger
logger = logging.getLogger('py4j')

# Set level
logger.setLevel(logging.INFO)

# Create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add formatter to ch
ch.setFormatter(formatter)

# Add ch to logger
logger.addHandler(ch)

def get_bq_spark_session(project_id, bq_dataset, ds_temp_bucket):
    return (
        SparkSession.builder.config("viewsEnabled", "true")
        .config("materializationProject", project_id)
        .config("materializationDataset", bq_dataset)
        .config("temporaryGcsBucket", ds_temp_bucket)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )

def get_date_range() -> Tuple[str, str]:
    begin_iso = CUT_OFF_DATE_START
    end_iso = CUT_OFF_DATE_END
    return (begin_iso, end_iso)


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


def get_scan() -> PySparkDF:                   # sql to get df for order's table
    begin_iso, end_iso = get_date_range()
    sql_query = f"""
        SELECT
            s.store_nbr,
            s.scan_id,
            s.visit_date,
            s.unit_qty,
            s.retail_price
        FROM
            {SCAN} s,
            {STORE_INFO} si,
            (SELECT * FROM {VISIT} WHERE visit_date >= '{begin_iso}') v
        WHERE
            ------- JOINS -------
            s.store_nbr = si.store_nbr     #  inner join b/w s and si followed by visit table based on store_nbr , visit_nbr , visit_date
            AND s.store_nbr = v.store_nbr   
            AND s.visit_nbr = v.visit_nbr  # ?? visit_nbr
            AND s.visit_date = v.visit_date
            ------- FILTERS -------
            AND v.visit_subtype_code != 198   # visit for pickup after booking online
            AND s.SCAN_TYPE = 0           # Q1 scan type =0   , executed scan - manual errors while scanning to remove multiple scans happening for an item
            and s.visit_date >= '{begin_iso}'
            and s.visit_date <= '{end_iso}'
            AND si.store_type NOT IN ('G', 'W')   # G , W ??   W- warehouse, but we are concerned with retail stores   
    """
    return sql_query


def get_inventory() -> PySparkDF:   # sql to get inventory data
    begin_iso, end_iso = get_date_range()

    sql_query = f"""
        SELECT 
            INV.system_item_nbr,   #system_item_nbr and item_nbr are same
            INV.snapshot_date,      # date of observation
            INV.onsite_onhand_qty,   # day level obs for onsite_onhand_qty
            INV.on_order_qty,        # quantity ordered by store
            INV.item_on_shelf_date_dt,  # date at which item was placed on shelf
            INV.club_nbr,
            ID.item_nbr,
            ID.dept_nbr,
            ID.subclass_nbr
        FROM
        (SELECT 
            cast(item_nbr as INT) as system_item_nbr,
            snapshot_date,
            onsite_onhand_qty,
            on_order_qty,
            item_on_shelf_date as item_on_shelf_date_dt,
            club_nbr
        FROM {MDSE_INV_DLY}
        WHERE SNAPSHOT_DATE >= "{begin_iso}"
            and SNAPSHOT_DATE <="{end_iso}"
            and STATUS not in ('R', 'L', 'D')   # return , defective , L- left around ??
        ) as INV
        INNER JOIN         
        (SELECT                      #Selecting only the dept_nbr present in CATS_SQL and joining by 
            DISTINCT         
            MDS_FAM_ID as system_item_nbr,
            item_nbr,
            dept_nbr,
            subclass_nbr
        FROM
            {ITEM_DIM}
        WHERE dept_nbr in {CATS_SQL}) as ID
        ON INV.system_item_nbr = ID.system_item_nbr 
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
        WHERE dept_nbr in {CATS_SQL}
    """
    return sql_query

def get_wingman():
    sql_query = f"""
        SELECT
            item_nbr,
            club_nbr,
            cast(md_start_date AS date) as markdown_start_date_dt,
            cast(md_end_date AS date) as markdown_end_date_dt,
            sp as markdown_price_amt
        FROM {WINGMAN_EXECUTION}
        WHERE club_nbr != 6279
    """
    return sql_query


def get_logs():
    sql_query = f"""
        SELECT DISTINCT
            item_nbr,
            club_nbr,
            liquidation_value as liquidation_price,
            predicted_md_units as expected_sale_units,
            current_retail as current_retail_price,
            current_units as current_inventory,
            markdown_retail as recommended_markdown_price,
            cast(cast(oos_date as datetime) as date) as oos_date,
            cast(cast(markdown_start_date as datetime) as date) as markdown_start_date_dt,
            cast(cast(created_ts as datetime) as date) as date,
            cast(plan_item_club_details.plan_id as integer) as plan_id
        FROM {PLAN_ITEM_CLUB_DETAILS} AS plan_item_club_details
        INNER JOIN {PLAN_DETAILS} as plan_details
        ON plan_item_club_details.plan_id = plan_details.plan_id
        where plan_details.plan_status_code in (10,11)
        # and trim(plan_item_club_details.markdown_reason_code) = 'Data science'
    """
    return sql_query

def get_inv_dept(inv: PySparkDF) -> PySparkDF:
    inv_dept = inv.withColumn('club_nbr', F.col('club_nbr').cast('Integer')) \
        .withColumn("date", F.date_format(F.col("snapshot_date"), "yyyy-MM-dd")) \ # snapshot date is date column
        .withColumn('inventory_qty',  # new col defining
                    F.when((F.col("dept_nbr").isin([42, 44, 57])), F.col('onsite_onhand_qty') + F.col('on_order_qty')) \  # for slow moving items we add it ??
                    .otherwise(F.col('onsite_onhand_qty'))) \ 
        .withColumn('inventory_qty', F.col('inventory_qty').cast('Integer'))
    return inv_dept

def get_first_level_features(raw_sales: PySparkDF, inv_dept: PySparkDF, item_dim: PySparkDF) -> PySparkDF:                  # Q2 difference b/w CATS_SQL & CATS - stores same thing 
    item_cat = item_dim.select('system_item_nbr', 'dept_nbr', 'subclass_nbr', 'item_nbr').distinct()
    item_onshelf = inv_dept.filter(F.col('dept_nbr').isin(CATS)).groupBy('system_item_nbr', 'club_nbr', 'item_on_shelf_date_dt').agg(F.max('snapshot_date').alias('max_snapshot_date'))  #Q3 max snapshot_date ??
    item_onshelf = item_onshelf.withColumnRenamed('system_item_nbr', 'mds_fam_id').withColumnRenamed('club_nbr', 'store_nbr')  
    day_sales_base = raw_sales.filter(F.col('unit_qty') > 0) \                      
        .groupBy('store_nbr', 'scan_id', 'visit_date') \                                                    
        .agg(F.sum('unit_qty').cast('decimal(18,2)').alias('unit_sold_cnt'), 
             F.sum('retail_price').cast('decimal(18,2)').alias('sale_amt'),
             (F.sum('retail_price')/F.sum('unit_qty')).cast('decimal(18,2)').alias('unit_price_amt')) \# creating column unit price amt
        .withColumnRenamed('store_nbr', 'club_nbr') \
        .withColumnRenamed('visit_date', 'date') \
        .withColumnRenamed('scan_id', 'system_item_nbr') \  ## ?? is scan  id == system item nbr == mds fam id  one - one mapping item br 
        .join(item_cat, on=['system_item_nbr'], how='inner') \   # picking those categories that are part of CAT_SQL
        .join(inv_dept.drop('dept_nbr', 'subclass_nbr', 'item_nbr', 'item_on_shelf_date_dt'), on=['system_item_nbr', 'date', 'club_nbr'], how='left') \ #possibly to provide on_order_qty onsite_onhand_qty
        .withColumn('unit_sold_cnt', F.coalesce('unit_sold_cnt', F.lit(0))) \# ??
        .withColumn('inventory_qty', F.coalesce('inventory_qty', F.lit(0)))   #?? 
    
    join_cond = [day_sales_base.system_item_nbr == item_onshelf.mds_fam_id, day_sales_base.club_nbr == item_onshelf.store_nbr, day_sales_base.date >= item_onshelf.item_on_shelf_date_dt, day_sales_base.date <= item_onshelf.max_snapshot_date]
    day_sales_base = day_sales_base.join(item_onshelf, how='left', on=join_cond) # join cond  
    day_sales_base = day_sales_base.drop('mds_fam_id', 'store_nbr')     # 
        
    day_sales_base = day_sales_base.withColumnRenamed('dept_nbr', 'department_nbr')  
    day_sales_base = day_sales_base.withColumnRenamed('system_item_nbr', 'mds_fam_id')
    day_sales_base = day_sales_base.withColumnRenamed('onsite_onhand_qty', 'inventory_on_site_qty')
    day_sales_base = day_sales_base.withColumnRenamed('on_order_qty', 'inventory_on_order_qty')
    #day_sales_base = day_sales_base.withColumnRenamed('units', 'mds_fam_id')

    logs = day_sales_base.filter(F.col('date') >= str(date.today() - timedelta(days=485))).select(   # Q4 we have chosen past 485  days from today - no such strong reason to adopt 485 only
        'mds_fam_id', 'club_nbr', 'department_nbr', 'subclass_nbr', 'item_nbr').distinct()
    logs = logs.withColumn('date', F.lit(TODAY).cast('string'))
    day_sales_base = day_sales_base.join(logs, on=['mds_fam_id', 'club_nbr', 'date', 'department_nbr',
                                                   'subclass_nbr', 'item_nbr'], how='outer')     #Q5 why are we using logs table  - recommendation on item club -  atleast one unit sold in last 485 days when sys_item_nbr is within CATS_SQL

    day_sales_base = day_sales_base.withColumn('unit_sold_cnt', F.coalesce('unit_sold_cnt', F.lit(0)))
    day_sales_base = day_sales_base.withColumn('on_shelf_date',
                                               F.date_format(F.col("item_on_shelf_date_dt"), "yyyy-MM-dd"))
    day_sales_base = day_sales_base.withColumn('day_on_shelf_cnt',
                                               F.datediff(F.col('date'), F.col('item_on_shelf_date_dt')))
    # item_club_filter=plan_logs.select('item_nbr','club_nbr').distinct()
    # day_sales_base = day_sales_base.join(plan_logs, on=['item_nbr','club_nbr','date'],how='outer')

    day_sales_base = day_sales_base.withColumn('unix_time', F.unix_timestamp('date', 'yyyy-MM-dd')) #unix_time fix


    # Get the past 4 weeks price/sales  & next 8 weeks sale  ( creating features )
    for i in range(12):
        if i <= 3:
            col_name_suffix = str(4 - i) + '_week_back_cnt' # unit count sold
            price_col_name_suffix = str(4 - i) + '_week_back_amt'  # amount generated
        else:
            col_name_suffix = str(i - 3) + '_week_next_cnt'
            price_col_name_suffix = 'charged_' + str(i - 3) + '_week_next_amt'

        if i <= 3:
            units_winSpec = Window.partitionBy('club_nbr', 'mds_fam_id').orderBy('unix_time').rangeBetween(
                7 * (i - 4) * 86400, 7 * (i - 3) * 86400 - 86400)
        else:
            units_winSpec = Window.partitionBy('club_nbr', 'mds_fam_id').orderBy('unix_time').rangeBetween(86400,
                                                                                                                7 * (
                                                                                                                        i - 3) * 86400)
        day_sales_base = day_sales_base.withColumn('unit_sold_' + col_name_suffix, F.sum('unit_sold_cnt').over(units_winSpec))
        price_winSpec = Window.partitionBy('club_nbr', 'mds_fam_id').orderBy('unix_time').rangeBetween(
            7 * (i - 4) * 86400, 7 * (i - 3) * 86400 - 86400)
        day_sales_base = day_sales_base.withColumn('avg_price_' + price_col_name_suffix,
                                                   F.avg('unit_price_amt').over(price_winSpec))
    lag52_Spec = Window.partitionBy('club_nbr','mds_fam_id').orderBy('unix_time').rangeBetween(7*(-52)*86400, 7*(-51)*86400-86400)
    day_sales_base = day_sales_base.withColumn('unit_sold_52_week_back_cnt', F.sum('unit_sold_cnt').over(lag52_Spec))
    # day_sales_base=day_sales_base.na.drop(how='all', subset=[f'unit_sold_{i}_week_next_cnt' for i in range(1,9)])

    #day_sales_base = day_sales_base.na.drop(how='all', subset=[f'units_sum_next_{i}_weeks' for i in range(1, 9)]) -> moved to sampling

    return day_sales_base

def generate_table_name(project_id, bq_dataset, table_name):

    return str(project_id+"."+"markdown"+"."+table_name)

if __name__ == '__main__':
    
    args = json.loads(sys.argv[1]) 
    project_id = args["GCP_PROJECT"] 
    bq_dataset = args["MATERIALIZATION_DATASET"] 
    ds_temp_bucket = args["GCS_TEMP_BUCKET"]
    source_dataset = args["FS_SOURCE_DATASET"]


    spark = get_bq_spark_session(project_id, bq_dataset, ds_temp_bucket)
    spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--s", help="CUT_OFF_DATE_START, e.g. 2023-09-01 ", type=str,required=True)
    # parser.add_argument("--e", help="CUT_OFF_DATE_END, e.g., 2023-09-30", type=str,required=True)
    #
    # args = parser.parse_args()
    # CUT_OFF_DATE_START = args.s
    # CUT_OFF_DATE_END=args.e

    TODAY = str(date.today() - timedelta(days=1))
    CUT_OFF_DATE_START = (datetime.datetime.today() - datetime.timedelta(days=30*35)).strftime("%Y-%m-%d")
    CUT_OFF_DATE_END=date.today().isoformat()
    
    # Input tables
    ITEM_DIM = "wmt-edw-prod.US_WC_VM.ITEM_DIM_CUR"
    MDSE_INV_DLY = "prod-sams-cdp.US_SAMS_PRODUCT360_CDP_VM.MDSE_INVENTORY_DLY"
    VISIT_MEMBER = 'wmt-edw-prod.US_WC_MB_VM.VISIT_MEMBER'
    PLAN_ITEM_CLUB_DETAILS = 'prod-sams-cdp.prod_sams_merch_markdown_optimization_markdown_optimization.plan_item_club_details'
    PLAN_DETAILS = 'prod-sams-cdp.prod_sams_merch_markdown_optimization_markdown_optimization.plan_details'
    
    SCAN = "wmt-edw-prod.US_WC_MB_VM.SCANX"
    STORE_INFO = "wmt-edw-prod.US_WC_VM.STORE_INFO"
    VISIT = "wmt-edw-prod.US_WC_MB_VM.VISIT"
    WINGMAN_EXECUTION = generate_table_name(project_id, bq_dataset, "md_info")
    CALENDAR_TABLE = "wmt-edw-prod.US_CORE_DIM_VM.CALENDAR_DIM"
    #PRICE_LOG = "prod-sams-cdp.prod_pricing_wingman_pricing.current_retail_action_log"
    #PROMO_TABLE =  "prod-sams-cdp.US_SAMS_PRODUCT360_CDP_VM.CLUB_ITEM_PROMO"
    CATS_SQL = (22,23,33,34,66,67,68,95,10,11,14,15,17,21,32,36,50,60,92,97,7,9,12,16,18,51,89,2,4,8,13,47,54,94,98,39,96,3,5,6,20,29,31,64,69,70,71,74,80,81,83,85,86,42,44,57,1, 40,48,52,58,78,41,43,46,49, 61,53,88)
    CATS = [22,23,33,34,66,67,68,95,10,11,14,15,17,21,32,36,50,60,92,97,7,9,12,16,18,51,89,2,4,8,13,47,54,94,98,39,96,3,5,6,20,29,31,64,69,70,71,74,80,81,83,85,86,42,44,57,1, 40,48,52,58,78,41,43,46,49, 61,53,88]

    logger.info('-- 1 --generate_table_name start --')
    # Output tables
    output_table = generate_table_name(project_id, bq_dataset, "inclub_first_level_features")
    logger.info('-- 2 -- get_scan start  --')

    #read scan table
    raw_sales_sql = get_scan()
    logger.info('-- 3 -- read_bq_table_to_df start --')

    raw_sales=read_bq_table_to_df(spark,raw_sales_sql)
    #read inventory table
    logger.info('-- 4 -- get_inventory start --')

    inv_sql = get_inventory()
    logger.info('-- 5 -- read_bq_table_to_df start spark,inv_sql--')

    inv=read_bq_table_to_df(spark,inv_sql)
    #creating first level features
    item_dim_sql = get_item_dim()
    item_dim=read_bq_table_to_df(spark,item_dim_sql)
    logger.info('-- 6 -- get_inv_dept start --')

    inv_dept = get_inv_dept(inv)
    logger.info('-- 7 -- get_wingman start --')

    #wingman table
    # wingman_sql = get_wingman()
    # logger.info('-- 9 -- read_bq_table_to_df start spark,wingman_sql --')

    # wingman=read_bq_table_to_df(spark,wingman_sql)
    
    #logs table
    logs_sql = get_logs()
    logger.info('-- 13 -- read_bq_table_to_df start spark,logs_sql --')

    # plan_logs=read_bq_table_to_df(spark,logs_sql)
    # logger.info('-- 14 -- get_logs  --')

    first_level_features = get_first_level_features(raw_sales, inv_dept, item_dim)

    #change the datatype of columns from bignumeric to float
    new_data_types = {"unit_price_amt": FloatType(), "avg_price_charged_8_week_next_amt": FloatType(), "avg_price_charged_7_week_next_amt": FloatType() ,"avg_price_charged_6_week_next_amt": FloatType(),"avg_price_charged_5_week_next_amt": FloatType(),"avg_price_charged_4_week_next_amt": FloatType(),"avg_price_charged_3_week_next_amt": FloatType(), "avg_price_charged_2_week_next_amt": FloatType(),"avg_price_charged_1_week_next_amt": FloatType(), "avg_price_4_week_back_amt": FloatType(),"avg_price_3_week_back_amt": FloatType(),"avg_price_2_week_back_amt": FloatType(),"avg_price_1_week_back_amt": FloatType()}
    for col_name, new_data_type in new_data_types.items():
        first_level_features = first_level_features.withColumn(col_name, col(col_name).cast(new_data_type))
    #storing the data
    logger.info('-- 15 -- write_df_to_bq_table start  --')

    write_df_to_bq_table(
        df=first_level_features,
        write_mode='overwrite',
        bq_table=output_table
    )# 35 months - 20 in intrain and 12~15 months val + test.