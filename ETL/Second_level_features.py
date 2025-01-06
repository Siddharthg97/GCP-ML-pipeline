# Part of codes are from Tyler and Khushboo's work
# Second level
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

def get_col_types():    # column and their datatypes
    col_types = {
        "avg_price_1_week_back_amt": "FLOAT64",
        "avg_price_2_week_back_amt": "FLOAT64",
        "avg_price_3_week_back_amt": "FLOAT64",
        "avg_price_4_week_back_amt": "FLOAT64",
        "avg_price_charged_1_week_next_amt": "FLOAT64",
        "avg_price_charged_2_week_next_amt": "FLOAT64",
        "avg_price_charged_3_week_next_amt": "FLOAT64",
        "avg_price_charged_4_week_next_amt": "FLOAT64",
        "avg_price_charged_5_week_next_amt": "FLOAT64",
        "avg_price_charged_6_week_next_amt": "FLOAT64",
        "avg_price_charged_7_week_next_amt": "FLOAT64",
        "avg_price_charged_8_week_next_amt": "FLOAT64",
        "avg_unit_sold_52_week_back_cnt": "FLOAT64",
        "avg_unit_sold_dept_52_week_back_cnt": "FLOAT64",
        "avg_unit_sold_dept_club_52_week_back_cnt": "FLOAT64",
        "avg_unit_sold_subcategory_52_week_back_cnt": "FLOAT64",
        "avg_weekly_unit_sold_1_month_back_cnt": "FLOAT64",
        "change_unit_sold_1_2_week_back_cnt": "FLOAT64",
        "change_unit_sold_2_3_week_back_cnt": "FLOAT64",
        "change_unit_sold_3_4_week_back_cnt": "FLOAT64",
        "change_unit_sold_same_week_1_year_back_cnt": "FLOAT64",
        "change_unit_sold_subcategory_same_week_1_year_back_cnt": "FLOAT64",
        "club_nbr": "INTEGER",
        "date": "DATE",
        "day_on_shelf_cnt": "INTEGER",
        "department_nbr": "INTEGER",
        "discount_1_week_next_nbr": "FLOAT64",
        "discount_2_week_next_nbr": "FLOAT64",
        "discount_3_week_next_nbr": "FLOAT64",
        "discount_4_week_next_nbr": "FLOAT64",
        "discount_5_week_next_nbr": "FLOAT64",
        "discount_6_week_next_nbr": "FLOAT64",
        "discount_7_week_next_nbr": "FLOAT64",
        "discount_8_week_next_nbr": "FLOAT64",
        "inventory_1_month_last_weekly_sale_nbr": "FLOAT64",
        "inventory_on_order_qty": "FLOAT64",
        "inventory_on_site_qty": "FLOAT64",
        "inventory_qty": "FLOAT64",
        "inventory_unit_1_week_back_pct": "FLOAT64",
        "inventory_unit_2_week_back_pct": "FLOAT64",
        "inventory_unit_3_week_back_pct": "FLOAT64",
        "inventory_unit_4_week_back_pct": "FLOAT64",
        "item_nbr": "INTEGER",
        "item_on_shelf_date_dt":"DATE",
        "max_price_6_month_last_amt": "FLOAT64",
        "max_unit_6_month_last_nbr": "FLOAT64",
        "max_weekly_discount_4_week_back_amt": "FLOAT64",
        "mds_fam_id": "INTEGER",
        "median_price_6_month_last_amt": "FLOAT64",
        "median_unit_6_month_last_nbr": "FLOAT64",
        "min_price_6_month_last_amt": "FLOAT64",
        "min_unit_6_month_last_nbr": "FLOAT64",
        "min_weekly_price_4_week_back_amt": "FLOAT64",
        "min_weekly_unit_sold_4_week_back_val": "FLOAT64",
        "price_1_week_back_median_price_6_month_last_nbr": "FLOAT64",
        "price_2_week_back_median_price_6_month_last_nbr": "FLOAT64",
        "price_3_week_back_median_price_6_month_last_nbr": "FLOAT64",
        "price_4_week_back_median_price_6_month_last_nbr": "FLOAT64",
        "sale_amt": "FLOAT64",
        "subclass_nbr": "INTEGER",
        "subclass_unit_sold_same_week_1_year_back_nbr": "FLOAT64",
        "unit_price_amt": "FLOAT64",
        "unit_sold_1_week_back_cnt": "FLOAT64",
        "unit_sold_1_week_next_cnt": "FLOAT64",
        "unit_sold_2_week_back_cnt": "FLOAT64",
        "unit_sold_2_week_next_cnt": "FLOAT64",
        "unit_sold_3_week_back_cnt": "FLOAT64",
        "unit_sold_3_week_next_cnt": "FLOAT64",
        "unit_sold_4_week_back_cnt": "FLOAT64",
        "unit_sold_4_week_next_cnt": "FLOAT64",
        "unit_sold_5_week_next_cnt": "FLOAT64",
        "unit_sold_52_week_back_cnt": "FLOAT64",
        "unit_sold_6_week_next_cnt": "FLOAT64",
        "unit_sold_7_week_next_cnt": "FLOAT64",
        "unit_sold_8_week_next_cnt": "FLOAT64",
        "unit_sold_cnt": "FLOAT64"
    }
    return col_types

def update_column_types(df, column_types):  # this is requied since we have migrated from ADB to GCP, Q1 while we create feature store ??
    expressions = [
        col(column).cast("int").alias(column) if dtype.upper() == "INTEGER" else
        round(col(column).cast("double"),2).alias(column) if dtype.upper() == "FLOAT64" else
        col(column)
        for column, dtype in column_types.items() if column in df.columns 
    ]

    return df.select(*expressions + [col(c) for c in df.columns if c not in column_types])   # columns created and existing.


def read_first_level_features() -> PySparkDF:
    sql_query = f"""
        SELECT 
            *,
            CAST(DATE as timestamp) as event_timestamp,   # featurestore requires time stamp
            current_timestamp as created_timestamp      # present date renamed as created_timestamp
        FROM
            {FIRST_LEVEL}
    """
    return sql_query


def get_weekly_discount(day_sales_base: PySparkDF) -> PySparkDF:  

    for i in range(8):3 # addition of features - discount 1 to 8 weeks ahead nbr 
        day_sales_base = day_sales_base.withColumn(f'discount_{i+1}_week_next_nbr',    1 - F.col(f'avg_price_charged_{i+1}_week_next_amt') / F.col(  # the unit level price is likely to change becoz of markdown applied, although MRO remains same                                        
                                                       'avg_price_1_week_back_amt'))    # discount for ith week can be positive or negative ?? Q1 Ideally it cannot be negative ,
    return day_sales_base

def get_time_index(day_sales_base: PySparkDF) -> PySparkDF: # for how many days is the item present on shelf wrt to present date.
    # Get on_shelf days
    day_sales_base = day_sales_base.withColumn('on_shelf_date',
                                               F.date_format(F.col("item_on_shelf_date_dt"), "yyyy-MM-dd")) \
        .withColumn('day_on_shelf_cnt', F.datediff(F.col('date'), F.col('item_on_shelf_date_dt')))   # difference b/w present date  and date on whivch item was placed on shelf

    for i in range(8): # not necessary
        day_sales_base = day_sales_base.withColumn(f'discount_{i+1}_week_next_nbr',    # repeated
                                                   1 - F.col(f'avg_price_charged_{i+1}_week_next_amt') / F.col(
                                                       'avg_price_1_week_back_amt'))      
    # Get the year, week no. ,week_day for present date
    day_sales_base = day_sales_base.withColumn('year', F.year('date')) \
        .withColumn('week', F.weekofyear('date')) \
        .withColumn('week_day', F.date_format('date', 'u'))
    return day_sales_base


def get_past_half_year_features(day_sales_base: PySparkDF) -> PySparkDF: # 6 months features -( max , min, median) count for units sold , (max, min,median) unit price at day level
    # Past half year features ~ almost 24 weeks 
    long_item_winSpec = Window.partitionBy('club_nbr', 'mds_fam_id').orderBy('unix_time').rangeBetween(
        -6 * 86400 + 7 * (-23) * 86400, 0)  #Q4 Why is the duration mentioned like this ?
    day_sales_base = day_sales_base.withColumn('max_price_6_month_last_amt', F.max('unit_price_amt').over(long_item_winSpec)) \
        .withColumn('max_unit_6_month_last_nbr', F.max('unit_sold_cnt').over(long_item_winSpec)) \
        .withColumn('min_price_6_month_last_amt', F.min('unit_price_amt').over(long_item_winSpec)) \
        .withColumn('min_unit_6_month_last_nbr', F.min('unit_sold_cnt').over(long_item_winSpec)) \
        .withColumn('median_price_6_month_last_amt',
                    F.expr('percentile_approx(unit_price_amt, 0.5)').over(long_item_winSpec)) \
        .withColumn('median_unit_6_month_last_nbr',
                    F.expr('percentile_approx(unit_sold_cnt, 0.5)').over(long_item_winSpec))
    return day_sales_base


def imputing_null_values(day_sales_base: PySparkDF) -> PySparkDF: mins sales, max discount , min price
    # If there is no sale in the week, fill the unit_price_amt with the min_price seen in the past 4 weeks. Q6 becoz we are observing till last 4 weeks from present date , becuase we are keepingonly past 4 weeks data for retrospection
    # If there is no sale in all the past 4 weeks, the unit_price_amt will be inf. Q7 this infinity is not required becuase  min unit price is considering past 1 weekunit price that has present date as well. 

    min_sales = F.least(*[
        F.when(F.col(c).isNull(), float("inf")).otherwise(F.col(c))
        for c in [f'unit_sold_{i}_week_back_cnt' for i in range(1, 5)]
    ]).alias("min_weekly_unit_sold_4_week_back_val")
    max_discount = F.greatest(*[
        F.when(F.col(c).isNull(), float("-inf")).otherwise(F.col(c))
        for c in [f'discount_{i}_week_next_nbr' for i in range(1, 9)]
    ]).alias("max_weekly_discount_8_week_next_amt")    # Q5 are these columns created 
        
    min_price = F.least(*[
        F.when(F.col(c).isNull(), float("inf")).otherwise(F.col(c))
        for c in [f'avg_price_{i}_week_back_amt' for i in range(1, 5)]
    ]).alias("min_weekly_price_4_week_back_amt")

    day_sales_base = day_sales_base.select('*', min_sales, max_discount, min_price)

    for col_name in [f'unit_sold_{i}_week_back_cnt' for i in range(1, 5)]:
        day_sales_base = day_sales_base.withColumn(col_name, F.coalesce(col_name, F.lit(0)))   # we don't use min sales or min_weekly_unit_sold_4_week_back_val for imputation rather 0 is used
    #for col_name in [f'week_{i}_discount' for i in range(1, 9)]:
    #    day_sales_base = day_sales_base.withColumn(col_name, F.coalesce(col_name, 'max_discount')).filter(
    #        F.col(col_name) >= 0) -> moved to sampling
    for col_name in [f'avg_price_{i}_week_back_amt' for i in range(1, 5)]:
        day_sales_base = day_sales_base.withColumn(col_name, F.coalesce(col_name, 'min_weekly_price_4_week_back_amt'))  # imputing avg unit price weekly by min of past weeks 4 which includes present unity price as well.
    day_sales_base = day_sales_base.fillna(0, [f'unit_sold_{i}_week_next_cnt' for i in range(1, 9)])# imputing null with 0 for units sold 1 o 9 weeks ahead
    day_sales_base=day_sales_base.withColumn('unit_sold_52_week_back_cnt',F.coalesce('unit_sold_52_week_back_cnt',F.lit(0)))  # imputing with 0 for past 52nd week
    return day_sales_base


def get_price_units_ratio(day_sales_base: PySparkDF) -> PySparkDF:
    # Get the average weekly sales of past one month becuase imputation has been done 
    day_sales_base = day_sales_base.withColumn('avg_weekly_unit_sold_1_month_back_cnt', (
            F.col('unit_sold_1_week_back_cnt') + F.col('unit_sold_2_week_back_cnt') + F.col(
        'unit_sold_3_week_back_cnt') + F.col('unit_sold_4_week_back_cnt')) / 4).withColumn('avg_weekly_unit_sold_1_month_back_cnt',
                                                                                   F.greatest(*[
                                                                                       'avg_weekly_unit_sold_1_month_back_cnt',
                                                                                       F.lit(0.1)]))  ## Q2 why have we chosen 0.1 as value, since it shall be used in denominator , we kept least count as 0.1
    # Get the price comparison unit_price_amt ratio=unit_price_amt of the week in past 1 to 4 /past_half_year_median_price
    for i in range(1, 5):
        day_sales_base = day_sales_base.withColumn(f'price_{i}_week_back_median_price_6_month_last_nbr',
                                                   F.col(f'avg_price_{i}_week_back_amt') / F.col(    # for comparing 1 week back price with 6 months back median sale - this done for certain item clum that are not sold very frequently hence and we require the aggregated units solds for them 
                                                       'median_price_6_month_last_amt'))
    for i in range(1, 5): # inventory picture as per present date inventory quantity and units sold in past 4 weeks and avg weekly sales throughout a month
        day_sales_base = day_sales_base.withColumn(f'inventory_unit_{i}_week_back_pct', F.col('inventory_qty') / F.greatest( #
            *[f'unit_sold_{i}_week_back_cnt', F.lit(0.025)]))  # we imputed units sold in past weeks with 0 
    day_sales_base = day_sales_base.withColumn('inventory_1_month_last_weekly_sale_nbr',    # 
                                               F.col('inventory_qty') / F.col("avg_weekly_unit_sold_1_month_back_cnt"))
    return day_sales_base

def holiday_features(data_etl): # data_tel is day_sales_base calculated in above function
    # 52 week back unit_sold_cnt with respective to different hirearachy 
    data_etl=data_etl.withColumn("date", F.to_timestamp(col("date"))).withColumn("week", date_format(col("date"), "w")) #week number
    data_etl=data_etl.withColumn("date", F.to_timestamp(col("date"))).withColumn("year", date_format(col("date"), "Y"))
    data_etl=data_etl.withColumn("date", F.to_timestamp(col("date"))).withColumn("month", date_format(col("date"), "M"))
    
    #ly temporal columns - creating 1 yer back week start date , month and year to capture seasonality
    data_etl = data_etl.withColumn('week_start_date_ly', F.date_sub(data_etl['date'], 365))## 1 year back week start date  F.date_sub
    data_etl=data_etl.withColumn("week_start_date_ly", F.to_timestamp(col("week_start_date_ly"))).withColumn("week_ly", date_format(col("week_start_date_ly"), "w")) # 1 year back week
    data_etl=data_etl.withColumn("week_start_date_ly", F.to_timestamp(col("week_start_date_ly"))).withColumn("year_ly", date_format(col("week_start_date_ly"), "Y")) # 1 year back  year
    data_etl=data_etl.withColumn("week_start_date_ly", F.to_timestamp(col("week_start_date_ly"))).withColumn("month_ly", date_format(col("week_start_date_ly"), "M")) # 1 year back  month

    #casting into integer for joining 
    data_etl = data_etl.withColumn("year",col("year").cast("int"))
    data_etl = data_etl.withColumn("week",col("week").cast("int"))
    data_etl = data_etl.withColumn("month",col("month").cast("int"))
    data_etl = data_etl.withColumn("year_ly",col("year_ly").cast("int"))
    data_etl = data_etl.withColumn("week_ly",col("week_ly").cast("int"))
    data_etl = data_etl.withColumn("month_ly",col("month_ly").cast("int"))
    
    #csc_change calculation at subclass level , we have data of past 35 months .NOTE -  we have already imputed with 0 for 52 week back units sold, so if we go date 12 months back obtain 52 week back units sold it shall be 0 now.
    club_subclass = data_etl.groupBy('subclass_nbr','club_nbr', 'week', 'year', 'week_ly', 'year_ly').agg(mean("unit_sold_52_week_back_cnt")) #for aggregation becoz some item club are quite similiar within same grp


    club_subclass = club_subclass\# present date club_subclass
                    .withColumnRenamed("avg(unit_sold_52_week_back_cnt)", "target_52_csc_this_year")\
                    .groupBy('subclass_nbr','club_nbr', 'week', 'year')\# grp by sub class item week year  of present date
                          .agg(F.first('week_ly').alias('week_ly'), 
                               F.first('year_ly').alias('year_ly'), 
                               F.mean('target_52_csc_this_year').alias('target_52_csc_this_year'))

    club_subclass_ly = club_subclass\ # 1 year back club_subclass
                        .selectExpr('subclass_nbr as sub_category_nbr_ly', # renaming sub_category col and club no. col
                                    'club_nbr as club_nbr_ly',   # renaming club_nbr 
                                            'week_ly', 
                                            'year_ly', 
                                            'target_52_csc_this_year as target_52_csc_last_year')\
                        .groupBy('sub_category_nbr_ly','club_nbr_ly', 'week_ly', 'year_ly')\ # groupby sub_class,club, 1 year back week and year 
                          .agg(F.mean('target_52_csc_last_year').alias('target_52_csc_last_year'))

    club_subclass = club_subclass.selectExpr('subclass_nbr','club_nbr', 'week', 'year', 'target_52_csc_this_year').join(club_subclass_ly,  # left join  present class_subclass & 1 year back class_subclass
                                   on=(club_subclass.subclass_nbr == club_subclass_ly.sub_category_nbr_ly) & 
                                      (club_subclass.club_nbr == club_subclass_ly.club_nbr_ly) &
                                      (club_subclass.week == club_subclass_ly.week_ly) & 
                                      (club_subclass.year == club_subclass_ly.year_ly), 
                                   how='left')

    club_subclass = club_subclass.withColumn('csc_change', (club_subclass['target_52_csc_this_year'] / club_subclass['target_52_csc_last_year']) - 1)

    club_subclass = club_subclass.selectExpr('subclass_nbr as scnbr', 
                                            'club_nbr as cnbr',
                                            'week as w', 
                                            'year as y',
                                            'target_52_csc_this_year as avg_unit_sold_subcategory_52_week_back_cnt', # we expect that the subcategories sold 1 year back should be ideally be our target
                                            'csc_change as change_unit_sold_subcategory_same_week_1_year_back_cnt')


    data_etl = data_etl.join(club_subclass, (data_etl.subclass_nbr == club_subclass.scnbr)&(data_etl.club_nbr == club_subclass.cnbr)&(data_etl.year == club_subclass.y)&(data_etl.week == club_subclass.w), "left").drop(*['w','y'])




    #52 weeks lag unit_sold_cnt sold after agg based on department_nbr, club_nbr # the above process is repeated based on dept_nbr
    dep_club = data_etl.groupBy('department_nbr', 'club_nbr', 'week', 'year').agg(mean("unit_sold_52_week_back_cnt"))
    dep_club = dep_club.withColumnRenamed("avg(unit_sold_52_week_back_cnt)", "avg_unit_sold_dept_club_52_week_back_cnt")
   
    dep_club = dep_club.select('department_nbr', 'club_nbr', 'week', 'year', 'avg_unit_sold_dept_club_52_week_back_cnt')
    dep_club = dep_club.toDF(*['dnbr', 'cnbr', 'w', 'y', 'avg_unit_sold_dept_club_52_week_back_cnt']).drop(*['dep_club.department_nbr', 'dep_club.club_nbr', 'dep_club.year', 'dep_club.week'])

    data_etl = data_etl.join(dep_club, (data_etl.department_nbr == dep_club.dnbr)&(data_etl.club_nbr == dep_club.cnbr)&(data_etl.year == dep_club.y)&(data_etl.week == dep_club.w), "left").drop(*['w','y','cnbr'])
    
    
    #52 weeks lag unit_sold_cnt sold with respective to department_nbr & not club  here  not considering the club , can you give ann scenario where this agg is meaningful
    dep = data_etl.groupBy('department_nbr', 'week', 'year').agg(mean("unit_sold_52_week_back_cnt"))
    dep = dep.withColumnRenamed("avg(unit_sold_52_week_back_cnt)", "avg_unit_sold_dept_52_week_back_cnt")
   
    dep = dep.select('department_nbr', 'week', 'year', 'avg_unit_sold_dept_52_week_back_cnt')
    dep = dep.toDF(*['dnbr','w', 'y', 'avg_unit_sold_dept_52_week_back_cnt']).drop(*['dep.department_nbr', 'dep.year', 'dep.week'])

    data_etl = data_etl.join(dep, (data_etl.department_nbr == dep.dnbr)&(data_etl.year == dep.y)&(data_etl.week == dep.w), "left").drop(*['w','y','cnbr'])

    data_etl = data_etl.drop(*['dnbr'])
  

    #item_change calculation
    item = data_etl.groupBy('mds_fam_id', 'week', 'year', 'week_ly', 'year_ly').agg(mean("unit_sold_52_week_back_cnt"))


    item = item\
                    .withColumnRenamed("avg(unit_sold_52_week_back_cnt)", "target_52_item_this_year")\
                    .groupBy('mds_fam_id', 'week', 'year')\
                          .agg(F.first('week_ly').alias('week_ly'), 
                               F.first('year_ly').alias('year_ly'), 
                               F.mean('target_52_item_this_year').alias('target_52_item_this_year'))

    item_ly = item\
                        .selectExpr('mds_fam_id as system_item_nbr_ly', 
                                            'week_ly', 
                                            'year_ly', 
                                            'target_52_item_this_year as target_52_item_last_year')\
                        .groupBy('system_item_nbr_ly', 'week_ly', 'year_ly')\
                          .agg(F.mean('target_52_item_last_year').alias('target_52_item_last_year'))

    item = item.selectExpr('mds_fam_id', 'week', 'year', 'target_52_item_this_year').join(item_ly, 
                                   on=(item.mds_fam_id == item_ly.system_item_nbr_ly) & 
                                      (item.week == item_ly.week_ly) & 
                                      (item.year == item_ly.year_ly), 
                                   how='left')

    item = item.withColumn('item_change', (item['target_52_item_this_year'] / item['target_52_item_last_year']) - 1)

    item = item.selectExpr('mds_fam_id as inbr', 
                            'week as w', 
                            'year as y',
                            'target_52_item_this_year as avg_unit_sold_52_week_back_cnt',
                            'item_change as change_unit_sold_same_week_1_year_back_cnt')


    data_etl = data_etl.join(item, (data_etl.mds_fam_id == item.inbr)&(data_etl.year == item.y)&(data_etl.week == item.w), "left").drop(*['w','y'])
    
    
    
    #unit_sold_cnt change in 1,2,3,4 weeks # change in units sold b/w 51 
    data_etl=data_etl.withColumn("change_unit_sold_1_2_week_back_cnt", (col("unit_sold_1_week_back_cnt")-col("unit_sold_2_week_back_cnt"))/col("unit_sold_1_week_back_cnt"))
    data_etl=data_etl.withColumn("change_unit_sold_2_3_week_back_cnt", (col("unit_sold_2_week_back_cnt")-col("unit_sold_3_week_back_cnt"))/col("unit_sold_2_week_back_cnt"))
    data_etl=data_etl.withColumn("change_unit_sold_3_4_week_back_cnt", (col("unit_sold_3_week_back_cnt")-col("unit_sold_4_week_back_cnt"))/col("unit_sold_3_week_back_cnt"))

    # Calculate sc_weights
    sc_weights = (
        data_etl.groupBy("subclass_nbr", "year", "month")
        .agg(F.sum("unit_sold_52_week_back_cnt").alias("target"))
    # .withColumn("year", F.col("year") + 1)
        )

    # Calculate sc_weights_max
    sc_weights_max = (
        sc_weights.groupBy("year", "month")
        .agg(F.sum("target").alias("target_max"))
        )

    # Join sc_weights with sc_weights_max
    sc_weights = (
        sc_weights.join(sc_weights_max, on=["year", "month"])
        .withColumn("subclass_unit_sold_same_week_1_year_back_nbr", F.col("target") / F.col("target_max"))
        .select("subclass_nbr", "year", "month", "subclass_unit_sold_same_week_1_year_back_nbr")
        )

    # Join train with sc_weights
    data_etl = data_etl.join(sc_weights, on=["subclass_nbr", "year", "month"], how="left")

    new_data_types = {"avg_unit_sold_subcategory_52_week_back_cnt": FloatType(), "change_unit_sold_subcategory_same_week_1_year_back_cnt": FloatType() ,"avg_unit_sold_dept_club_52_week_back_cnt": FloatType(),"avg_unit_sold_dept_52_week_back_cnt": FloatType(),"avg_unit_sold_52_week_back_cnt": FloatType(), "change_unit_sold_same_week_1_year_back_cnt": FloatType(),"change_unit_sold_1_2_week_back_cnt": FloatType(), "change_unit_sold_2_3_week_back_cnt": FloatType(),"change_unit_sold_3_4_week_back_cnt": FloatType(),"subclass_unit_sold_same_week_1_year_back_nbr": FloatType()}
    for col_name, new_data_type in new_data_types.items():
        data_etl = data_etl.withColumn(col_name, col(col_name).cast(new_data_type))

    return data_etl

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



    spark = get_bq_spark_session(project_id, bq_dataset, ds_temp_bucket)
    spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")

    cols_to_drop = ['unix_time','week_start_date_ly','week_ly','year_ly','month_ly','scnbr','inbr','customer_item_nbr','customer_club_nbr','year','month', 'week', 'day', 'week_day', 'on_shelf_date', 'snapshot_date']

    TODAY = str((date.today() - timedelta(days=1)).strftime("%Y-%m-%d"))


    FIRST_LEVEL = generate_table_name(project_id, 'markdown', "inclub_first_level_features")
    output_table = generate_table_name(project_id, source_dataset, "club_item_daily_v1")
    #inference_table = generate_table_name(project_id, "dev_fs_source_tables", "inclub_today_features")
    logger.info('-- 1 --read_first_level_features start --')

    first_level_features_sql = read_first_level_features()
    logger.info('-- 2 --read_bq_table_to_df start spark, first_level_features_sql--')

    first_level_features = read_bq_table_to_df(spark, first_level_features_sql)
    logger.info('-- 3 --get_weekly_discount start --')
    features_with_discount = get_weekly_discount(first_level_features)  # this first_level_features become day_sales_base when we define function

    logger.info('-- 5 --get_time_index start --')
    features_with_time_index = get_time_index(features_with_discount)
    logger.info('-- 6 --get_past_half_year_features start --')
    features_with_past_half_year = get_past_half_year_features(features_with_time_index)
    logger.info('-- 7 --imputing_null_values start --')
    features_imputed = imputing_null_values(features_with_past_half_year)
    logger.info('-- 8 --get_price_units_ratio start --')
    features_with_ratio = get_price_units_ratio(features_imputed)
    logger.info('-- 9 --holiday_features start --')
    data_samples= holiday_features(features_with_ratio)
    logger.info('-- 10 --drop_columns start --')

    features_with_holiday_reduced = drop_columns(data_samples, cols_to_drop)
    logger.info('-- 11 --get_col_types start --')

    column_types = get_col_types()
    logger.info('-- 12 --update_column_types start --')

    features_with_holiday_dt_updated = update_column_types(features_with_holiday_reduced, column_types)

    #today_features = features_with_holiday_reduced.filter(F.col('date') == TODAY)
    logger.info('-- 13 --write_df_to_bq_table start --')

    write_df_to_bq_table(
        df=features_with_holiday_dt_updated,
        write_mode='overwrite',
        bq_table=output_table
    )

    #write_df_to_bq_table(
    #    df=today_features,
    #    write_mode='overwrite',
    #    bq_table=inference_table
    #)