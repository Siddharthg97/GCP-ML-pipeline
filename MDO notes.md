BASICS

Objective - The objective of MDO is to predict optmized markdown prices among Sam's Club such that revenue is recovered the most or recovery rate is highest.

Problem Statement -So generally what happens is when inventory get's reduntant on clubs & we are about to reach liquidation date or in case of seasonal that remain unsold after season getting over,

Approach - Markdown are applied where item prices are reduced by certain fraction, to make sure that inventories are efficiently sold well on time before liquidation date. These nmarkdownare applied in scenarios

1) atmost 8 weeks before dilution

2) before is season is about over for seasonal items

They are applied specifically applied for item club from certain dates called markdown start date & for 8 weeks  at predicted rates. This is done with help of optimization model that identifies most optimum discount rates for markdown based on maximizing recovery.

rate. These are done for two channels - online markets (dotcom) and offline markets (inclub)  .

Note : It's quite similar to predicting unit elasticity but shall also involve no. of weeks. We try to simulate unit elasticity and by using the optimization model we obtain the most optimum markdown price & num weeks for any markdown session ( item, club & markdown start date). 

Terminology

1) Recovery rate -  The amount of revenue recovered as part of markdown including the liquidation v/s the cost associated in purchasing at start of markdown session.

(Markdown price * items sold +  items left * liquidation price)/(cost price * inventory present at the start of session)

2) Sell through rate - 

3) Unit elasticity  - 




Process / project pipeline /

ETL pipeline - to create features, train, val & test data.DS & Manual  Executed markdown data 

features -
Discount features for next 8 weeks obtained by comparing avg weekly price for next 8 weeks w.r.t avg weekly price for past 1 week. 
Price features - avg weekly price for past 4 weeks
Price ratios - avg weekly price for 1 to 4 weeks v/s median price 6 months back ( min , max are not useful and mean price can get skewed ) 
units sold weekly in past 4 weeks
avg weekly units sold past 1 month
change in units sold from 1-2, 2-3 & 3-4 
avg units solds in 52nd week in past 
 avg units solds for a dept in 52nd week in past
avg units solds in sub cat 52nd week in past
change in units sold for dept in 52nd week
change in units sold for sub-cat in 52nd week
Train - from 36 month back start date till 4th month. We would need to explode till 8 weeks  and sum up units sold 1 to 8 weeks as target.
Val - for 3rd month past
Test - for 1st month past
Executed markdown table - In past 4 months for DS & Manual markdowns, we filter the features data for only markdown start date followed by exploding  for all 8 weeks & filter for weeks using start & end dates. Filter num weeks for start date & present date and for past 4 months.  

Notes : 

ML pipeline

what are resources we require are 

1) settings.yml file - contains all data location and model configurations

2) inclub training pipeline - python script 

3) base creator image

4) ml creator image




ETL

We created features in three stages - 1st level features followed by second level features and them final features being used for train , val  , test & executed markdown data




first level features comprised of taking input source table after certain operations like filtering them for inclub data only , setting tenure of past 35 months for certain department no. for which sale was observed in past 1 year, from yesterday's date for both order's data and inventory data .

Merging the inventory table and orders table with granularity at item , club date representing that total units sold throught the day in a particular club on certain date. Hence every record representing the sale of any item no. we can see units sold, retail price, inventory quantity , item on shelf date etc. Now we start feature engineering, where we generate feature at initial level referring to - units sold in next 8 weeks and past 4 weeks ,weekly avg prices in past 4 weeks and next 8 weeks.




Second level features

Here we create new features like discount features for next 8 weeks, weekly avg price 1 month back , units sold weekly 1 month back, price ratios - median price 6 months backs v/s avg weekly price 4 weeks back, avg weekly units sold 365 days back, inventory ratios ,change in units sold 1 to 2 weeks & so on, aggregated units sold at department level  sub category level.







Q Why are we using these features - feature selection has been done based on feature importance and unnecessary features have been removed




data sampling

exploding it to 8 weeks while creating new columns - num_week & target - unit sold 1 week ahead , 2 weeks ahead & so on. 




executed markdowns details - 




Features

Since it is time series forecasting model generally for that we consider time dependent features ( lag or lead features), exogenous variables and standard time features like week ,month & year. Meanwhile these features like lag /lead or exogenous variables can be used to create seasonal features or trends features. 

CATEGORY_FEATURES:
        - 'club_nbr'
        - 'department_nbr'
        - 'subclass_nbr'
        - 'week'
        - 'month'
      CONSTRAINED_FEATURES:
        - 'num_weeks'
        - 'discount_1_week_next_nbr'
        - 'discount_2_week_next_nbr'
        - 'discount_3_week_next_nbr'
        - 'discount_4_week_next_nbr'
        - 'discount_5_week_next_nbr'
        - 'discount_6_week_next_nbr'
        - 'discount_7_week_next_nbr'
        - 'discount_8_week_next_nbr'




COVARIATES:
        - 'discount_1_week_next_nbr'      - discount features
        - 'discount_2_week_next_nbr'
        - 'discount_3_week_next_nbr'
        - 'discount_4_week_next_nbr'
        - 'discount_5_week_next_nbr'
        - 'discount_6_week_next_nbr'
        - 'discount_7_week_next_nbr'
        - 'discount_8_week_next_nbr'
        - 'club_nbr'                                     categorical features
        - 'department_nbr'
        - 'subclass_nbr'
        - 'median_price_6_month_last_amt' - 6 month back feature 
        - 'price_1_week_back_median_price_6_month_last_nbr' - price ratio features 
        - 'price_2_week_back_median_price_6_month_last_nbr'
        - 'price_3_week_back_median_price_6_month_last_nbr'
        - 'price_4_week_back_median_price_6_month_last_nbr'
        - 'avg_weekly_unit_sold_1_month_back_cnt'  - units sold feature
        - 'day_on_shelf_cnt'    - date reference feature
        - 'num_weeks'   
        - 'unit_sold_1_week_back_cnt' - units sold feature
        - 'unit_sold_2_week_back_cnt'
        - 'unit_sold_3_week_back_cnt'
        - 'unit_sold_4_week_back_cnt'
        - 'month'      categorical feature
        - 'week'        categorical feature
        - 'avg_unit_sold_subcategory_52_week_back_cnt'                   1 year back feature -  avg unit sold features subcategory aggregated 52 week back count 
        - 'change_unit_sold_subcategory_same_week_1_year_back_cnt'    change in units sold for the same week at subategory level 52 weeks back - aggregation of avg units sold is done at sub category , club , week , month , year ( for this year and last year ) is done. Now further aggregation is done separately for this year , month and week and last year , month & week to get avg units sold atsub category level. Now corresponding  to last year matching present week, month with calculated last year , week and month , we compare avg unit sold as ratio( traget this year/target last year -1)
        - 'avg_unit_sold_dept_52_week_back_cnt'            1 year back features - avg  unit sold features category/dept aggregated 52 week back count 
        - 'avg_unit_sold_52_week_back_cnt'                     1 year back -  avg units sold feature 52 week back count
        - 'change_unit_sold_1_2_week_back_cnt'              change in units sold feature from first to 2nd week
        - 'change_unit_sold_2_3_week_back_cnt'              
        - 'change_unit_sold_3_4_week_back_cnt'
        - 'change_unit_sold_same_week_1_year_back_cnt'   change in units sold for the same week 52 week back at item level .so again aggregation at item level for same week, month , year, week 1 year back ,  month 1 year back an year 1 year back. Now aggregating again as done above for this year and last year and taking ratio 
        - 'week_inventory_expected_to_last_cnt'            inventory ratio - (inventory quantity / avg weekly units sold 1 month back /ith week 
      ENCODE_FEATURES:
        - 'club_nbr'
        - 'department_nbr'
        - 'subclass_nbr'
        - 'week'
        - 'month'




Data pre-processing 

1) encoding - catboost encoder 

Q When to use catboost encoder ?

Suitable for large datasets:

Due to its efficient handling of categorical data, CatBoost can perform well on large datasets with many categorical features. 

When to use a CatBoost encoder:

When you have a dataset with a significant number of categorical features. 
When you want to avoid the complexity of manually encoding categorical variables. 
When you need a model that can handle high-cardinality categorical features effectively. 


            
        NOTE: CatBooster encoder is target-based categorical encoder such as a supervised encoder that encodes categorical columns according to the target value, supporting both binomial and continuous targets. 
        encoding folmula := (TargetSum + Prior)/(FeatureCount + a)
        where a = 1, TargetSum = sum of target values based on category, Prior = sum of target values in entire dataset / total data rows, FeatureCount = number of counted specific category value in the column. 
        e.g.
        train = pd.DataFrame({
            'color': ["red", "blue", "blue", "green", "red",
                      "red", "black", "black", "blue", "green"],
                      
            'grade': [1, 2, 3, 2, 3, 1, 4, 4, 2, 3], })
        
        # Define train and target
        target = train[['grade']]
        train = train.drop('grade', axis = 1)
        
        # Define catboost encoder
        cbe_encoder = ce.cat_boost.CatBoostEncoder()
        
        # Fit encoder and transform the features
        cbe_encoder.fit(train, target)
        train_cbe = cbe_encoder.transform(train)
        
        Prior = 25/10 = 2.5; For `red` category, TargetCount = 1+3+1 = 5; FeatureCount = 3 -> (5 + 2.5)/(3 + 1) = 1.875
         

2) imputation - already done - units sold features in past 4 weeks imputed with 0 value , discount feature imputed with max discount ( ideally we do not consider any discount unless provided), hence we consider the worst case scenario that max discount has been provided.

price feature to be used in price ratios is imputed with min price in past 4 weeks. 

3) train val test - for creating train , val & test dataset,  we explode every record corresponding to unique combination of item ,club & date with target as cumulative units sold in next ith week & num week. Already done the objective val & test dataset was to look into model performance while actual executed markdown data was not available, hence to see whether model is underfitting or overfitting, we use the dataset especially to perform hyperparameter tuning.

Now the datasets have gap of 1 month in b/w to avoid data leakage. so overall we use past 20 months data out of which we have 16 months data in train and 4 months data for val & test, but to actually compute these features we need past 35 months data as we go 12 month back for 

20th month in past. Once we came out with champion model we don't need to use just one moth to validate since we have past 4 months data, on the other hand we would need to incorporate computational efforts if we use all 4 months , but since we have exploded to 8 weeks for every item club and we know that the rows shall have quite similar information if dates are very close. So for model performance analysis we consider only those records on which markdown started so that there can be significant difference in dates and records don't share same information.




Model training

what ml models have been used 

base line model ensemble - XG boost , DNN , light gbm , rf

https://www.analyticsvidhya.com/blog/2024/01/xgboost-for-time-series-forecasting/

XG boost for time series forecasting 

To assess the performance of the XGBoost model, one must partition the time-series data into training and testing sets. The training set facilitates model training, and the testing set enables the evaluation of its performance on unseen data. Preserving the temporal order of observations is crucial when splitting the data.




Handle seasonality and trends  : XGBoost can effectively handle seasonality and trends in time-series data. Seasonal features can be incorporated into the model to capture periodic patterns, while trend features can capture long-term upward or downward trends. By considering seasonality and trends, XGBoost can provide more accurate forecasts.

    : Non-stationary data, where the statistical properties change over time, can pose challenges for time-series forecasting. XGBoost can handle non-stationary data by incorporating differencing techniques or by using advanced models such as ARIMA-XGBoost hybrids. These techniques help in capturing the underlying patterns in non-stationary data.

Incorporating External factors/Exogenous variables: In some time-series forecasting tasks, external factors can significantly influence the target variable. XGBoost allows for the incorporation of external factors as additional predictors, enhancing the model’s predictive power. For example, in energy demand forecasting, weather data can be included as an external factor to capture its impact on energy consumption.

Feature selection/importance: Feature selection plays a vital role in time-series forecasting with XGBoost. It is important to identify the most relevant features that contribute to accurate predictions. XGBoost provides feature importance scores, which can guide the selection of the most influential features.

Utility: Dealing with Irregular and Sparse Data

XGBoost performs best when the time-series data is regular and dense. Irregular or sparse data, where there are missing observations or long gaps between observations, can pose challenges for XGBoost. In such cases, data imputation or interpolation techniques may be required to fill in the missing values or create a denser time series.

Drawbacks of XGBoost model: XGBoost may struggle to capture long-term dependencies in time-series data. If the target variable depends on events or patterns that occurred far in the past, XGBoost’s performance may be limited. In such cases, advanced models like recurrent neural networks (RNNs) or long short-term memory (LSTM) networks may be more suitable.







TIDE/RNN/LSTM

TIDE" in time series forecasting stands for "Time-series Dense Encoder," which is a deep learning model based on a Multi-Layer Perceptron (MLP) architecture designed specifically for long-term time series forecasting tasks; it encodes past time series data along with any relevant covariates using dense MLP layers, then decodes this encoded representation to generate future predictions. 

Key points about TIDE:

MLP-based:

Unlike some other advanced time series models that use transformers, TIDE relies on the simplicity of MLPs, which can lead to faster training times while still achieving high performance on long-term forecasting problems. 

Encoder-Decoder structure:

The model consists of an encoder that maps the past time series and covariates into a dense representation, and a decoder that uses this representation to generate future predictions. 

Handling covariates:

TIDE can effectively incorporate external variables (covariates) that might influence the time series, allowing for more accurate predictions. 

Advantages:

Fast training: Due to its MLP architecture, TIDE typically trains much faster than complex transformer-based models. 
Long-horizon forecasting: Particularly well-suited for predicting far into the future compared to other time series models. 
Flexibility: Can be adapted to various time series datasets with different feature sets. 




Time series forecasting 

check results on val test - overfitting / underfitting

Hyper para meter tuning - 

what metrics are we using - cumulative & week level 

mape , wmape , median ape ,   rmse , mae

rmse,mae , mse, are scale dependent have no reference points but mape , wmape , median ape are scale independent & follow lower the better type characteristics. But among them also mape,wmape is not outlier resistant where as median ape outlier resistant, but we want the metric to be sensitive to outlier. Now among mape & wmape, we use wmape becoz it takes into consideration total because in mape we compute absolute % sum them up and divide by total data points, hence abs % are taken equally important but actually they are not equally important, hence we would need to give weightage and thus compute weightage average , but how to give weightage so based on units sold actually , higher the units sold for any item club more is weighatge given ti it's abs % error. 

DS executed markdown evaluation results 

Metrics

	

scenario 1

	

scenario 2


week_eval_weighted_mape	0.5281	0.5158
cumul_eval_weighted_mape	0.4581	0.4421
week_eval_rmse	32.7952	31.9175
cumul_eval_rmse	42.1681	40.5697
week_eval_mae	18.6893	18.251
cumul_eval_mae	23.6464	

22.8165




cumul_bias

	-0.3329	-0.2918


week_bias

	-0.3059	-0.2611


cumul_eval_mape

	73650305904612.08	76791073139650.47
week_eval_mape	73650305904612.47	76791073139650.84







feature that are important

xgboost importance - plot ggplot with default criteria as gain. there can be other crterias as well - split frequency/weight , cover. SHAP Can be also used  




application UI - fast api to predict model 




ITEM MAPPING CHANGE

Problem Statement - The granularity at which source table like orders/scan table , inventory table get created is going to change and these table shall become more granular by taking into consideration suppliers/vendor replenishing them and based product specification like color, size or any minute change. So earlier where source table would have only same id i.e. item_nbr representing the product irrespective of vendor or product specifications, now there shall be unique supplier id i.e. supplier_item_nbr for each vendor and product_id sunch that for every unique  such that for every item club there exist one-one mapping b/w product_id and supplier_item_nbr and for this unique combination we shall generate new_item_nbr.

So several new_item_nbr shall be falling under same item_nbr. This means that ETL pipelines needs to be updated but since our existing model takes old data structure, to avoid any disruption in model predictions in prod environment, the ETL would need to transition from new item structure to old item structure and ml pipeline shall run as it is.











