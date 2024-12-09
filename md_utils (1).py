import pandas as pd
import numpy as np
import datetime
import category_encoders as ce
import xgboost
import optuna
from sklearn.pipeline import Pipeline
from plotly.offline import plot
import plotnine as pn
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
from typing import List, Dict, Union, Optional
import joblib
import gcsfs

def bias(y_true, y_pred):
    return np.sum(y_pred-y_true)/np.sum(y_true)

def smape(a, f):
    return 1 / len(a) * np.sum(2 * np.abs(f - a) / (np.abs(a) + np.abs(f) + 1) * 100)

def weighted_mape(y_true, y_pred):
    return np.sum(np.abs(y_pred-y_true))/np.sum(y_true)

def median_ape(y_true, y_pred):
    errors = np.abs((y_true - y_pred) / y_true)
    median_ape = np.median(errors)
    return median_ape


def oep(true, pred, weights=[]):
    if len(weights)==0:
        weights=true
    error = pred - true
    idx = (error > 0)
    oep_sample = np.sum(idx) / pred.shape[0]

    total_oe_units = np.sum(pred[idx] - true[idx])
    total_oe_percent = total_oe_units / np.sum(true)

#     error_percent = error / true
    oe_p = error[idx] / np.sum(weights[idx])
    # oe_p=oe_p[oe_p<20]

    mean_oe_percent = np.sum(oe_p) * oep_sample
    return oep_sample, total_oe_units, total_oe_percent, mean_oe_percent



def uep(true, pred, weights=[]):
    if len(weights)==0:
        weights=true
    error = pred - true
    idx = (error < 0)
    uep_sample = np.sum(idx) / pred.shape[0]

    total_ue_units = np.sum(pred[idx] - true[idx])
    total_ue_percent = total_ue_units / np.sum(true)

#     error_percent = error / true
    ue_p = error[idx] / np.sum(weights[idx])
    # oe_p=oe_p[oe_p<20]

    mean_ue_percent = np.sum(ue_p) * uep_sample
    return uep_sample, total_ue_units, total_ue_percent, mean_ue_percent



def wmape(y_true, y_pred):
    return np.sum(np.abs(y_pred-y_true))/np.sum(y_true)

class ElasticityTest:
    def __init__(self, num_weeks: int = 1, data_gcs_uri: Optional[str] = None, specified_input_pd: Optional[pd.DataFrame] = None, inventory_modifier: float = 1) -> None:
        """
        Initializes the ElasticityTest class.
        Args:
            num_weeks (int): Number of weeks to filter data.
            data_gcs_uri (Optional[str]): GCS URI where the data is stored.
            specified_input_pd (Optional[pd.DataFrame]): Pre-loaded DataFrame to use directly.
        Returns:
            None
        """
        self.num_weeks: int = num_weeks
        self.data_gcs_uri: Optional[str] = data_gcs_uri
        self.fs = gcsfs.GCSFileSystem()
        self.specified_input_pd: Optional[pd.DataFrame] = specified_input_pd
        if specified_input_pd is not None:
            self.df: pd.DataFrame = specified_input_pd
        else:
            self.df: pd.DataFrame = self._load_parquet_files()
        self.df = self.df[self.df['num_weeks'] == self.num_weeks]
        self.df['inventory_qty'] = self.df['inventory_qty']*inventory_modifier
        
    def _load_parquet_files(self) -> pd.DataFrame:
        """
        Loads parquet files from Google Cloud Storage.
        Returns:
            pd.DataFrame: A DataFrame containing the concatenated data from all parquet files.
        """
        parquet_files: List[str] = [f"gs://{path}" for path in self.fs.glob(f"{self.data_gcs_uri}")]
        df_list: List[pd.DataFrame] = [pd.read_parquet(file) for file in parquet_files]
        return pd.concat(df_list)
    def _assign_week_discount(self, row: pd.Series, discount: float) -> pd.Series:
        """
        Assigns a discount to a specific week in the row.
        Args:
            row (pd.Series): A row from the DataFrame.
            discount (float): Discount value to be assigned.
        Returns:
            pd.Series: The modified row with the discount applied.
        """
        for i in range(1,(int(row['num_weeks'])+1)):
            row[f'discount_{i}_week_next_nbr']=np.round(discount,2)
        return row
    def create_elasticity_matrix(self, min_discount: float = 0, max_discount: float = 0.81, step_length: float = 0.1) -> pd.DataFrame:
        """
        Creates an elasticity matrix by applying different discount rates.
        Args:
            min_discount (float): The minimum discount rate.
            max_discount (float): The maximum discount rate.
            step_length (float): The increment step between discount rates.
        Returns:
            pd.DataFrame: A DataFrame containing the elasticity matrix.
        """
        elasticity_matrix: List[pd.DataFrame] = []
        input_data: pd.DataFrame = self.df.copy()
        for i in range(1,9):
            input_data[f'discount_{i}_week_next_nbr']=-1.0
        for discount in np.arange(min_discount, max_discount, step_length):
            input_data_one_discount: pd.DataFrame = input_data.copy()
            input_data_one_discount=input_data_one_discount.apply(self._assign_week_discount, discount=discount, axis=1)
            elasticity_matrix.append(input_data_one_discount)
        elasticity_matrix_pd: pd.DataFrame = pd.concat(elasticity_matrix, axis=0)
        print(elasticity_matrix_pd.shape)
        return elasticity_matrix_pd
    def xgb_model_loader(self, model_path: str) -> None:
        """
        Loads an XGBoost model from a specified path.
        Args:
            model_path (str): The path to the XGBoost model file.
        Returns:
            None
        """
        try:
            with open(model_path, "rb") as file:
                self.pipeline = joblib.load(file)
        except:
            with self.fs.open(model_path) as file:
                self.pipeline = joblib.load(file)
        self.covariates: List[str] = self.pipeline[-1].get_booster().feature_names
    def dnn_model_loader(self, model_path: str, covariates: List) -> None:
        """
        Loads a deep neural network model from a specified path.
        Args:
            model_path (str): The path to the DNN model file.
        Returns:
            None
        """
        # device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(f"Current device: {device}")
        # checkpoint: Dict[str, Union[dict, torch.Tensor]] = torch.load(model_path, map_location=device)
        # self.pipeline = HybridModel(**checkpoint['model_params']).to(device)
        # self.pipeline.load_state_dict(checkpoint['state_dict'])
        # self.pipeline.eval()
        # self.covariates=covariates
    def plot_model_importance(self, img_path) -> None:
        """
        Plots feature importances from the model.
        Returns:
            None
        """
        plt.figure(figsize=(8, 6))
        importances: np.ndarray = self.pipeline.steps[1][1].feature_importances_
        plt.bar(self.covariates, importances, orientation='vertical')
        plt.xticks(rotation=90)
        plt.xlabel('Feature')
        plt.ylabel('Importance')
        plt.title('Feature Importances')
        plt.show()
        plt.savefig(img_path)
    def sale_prediction(self, data: pd.DataFrame, model_type: str) -> Union[np.ndarray, pd.Series]:
        """
        Predicts sales using the loaded model.
        Args:
            data (pd.DataFrame): Input data for prediction.
            model_type (str): Type of the model ('xgb' or 'dnn').
        Returns:
            Union[np.ndarray, pd.Series]: Predicted sales.
        """
        
        if model_type=="xgb":
            data[self.covariates]=data[self.covariates].astype("float")
            data['predicted_sale'] = self.pipeline.predict(data[self.covariates])
        if model_type=="dnn":
            with torch.no_grad():
                data['predicted_sale'] = self.pipeline(torch.tensor(data[self.covariates].fillna(0).values.astype(float),dtype=torch.float32).unsqueeze(2)).numpy()
        data.loc[data['predicted_sale'] <= 0, 'predicted_sale'] = 0
        data['cap_predicted_sale'] = np.round(data[['predicted_sale', 'inventory_qty']].min(axis=1),2)
        return data
    def plot_elasticity_curve(self, predicted_sale: pd.DataFrame, img_path) -> None:
        """
        Plots the elasticity curve based on predicted sales.
        Args:
            predicted_sale (pd.DataFrame): DataFrame containing the predicted sales.
        Returns:
            None
        """
        predicted_sale['max_predicted_sale']=predicted_sale.groupby(['club_nbr','item_nbr','date','num_weeks'])['predicted_sale'].transform('max')
        predicted_sale['min_predicted_sale']=predicted_sale.groupby(['club_nbr','item_nbr','date','num_weeks'])['predicted_sale'].transform('min')
        predicted_sale['max_sale_lift']=(predicted_sale['max_predicted_sale']+1)/(predicted_sale['min_predicted_sale']+1)
        predicted_sale['sale_lift']=(predicted_sale['predicted_sale']+1)/(predicted_sale['min_predicted_sale']+1)
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=predicted_sale, x=f'discount_{self.num_weeks}_week_next_nbr', y='sale_lift', showfliers=False)
        plt.xlabel('Discount')
        plt.ylabel('Sale Lift')
        plt.title('Elasticity Curve')
        plt.show()
        plt.savefig(img_path)
    def plot_metrics(self,model_type):
        data_pred=self.sale_prediction(self.df,model_type)
        print(f"overall wmape for {self.num_weeks} weeks",wmape(data_pred['target'],data_pred['cap_predicted_sale']))
        print(f"smape for {self.num_weeks} weeks",smape(data_pred['target'],data_pred['cap_predicted_sale']))
        print(f"bias for {self.num_weeks} weeks",bias(data_pred['target'],data_pred['cap_predicted_sale']))
class Config:
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
    
    def model_params(self, param_flag, manual_params, latest_md_params_path):
        """Conditionally outputs parameters.
        Args:
            param_flag: Markdown model auto/manual configuration flag.
            manual_params: manual params if given. 
            latest_md_params_path: Markdown latest parameter saving path. 
        Returns:
            parameters as hashmap either 'auto' or 'manual'. 
        """
        from google.cloud import storage
        from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
        import json
        
        params = None
        if param_flag == "auto":
            # Get params from GCS bucket
            # params_path = "gs://dev-markdown-training-pipeline-bucket-nonprod/best_params"
            blob = storage.blob.Blob.from_string(latest_md_params_path, client=storage.Client())
            params = json.loads(blob.download_as_string())
            if "eval_metric" not in params.keys():
                params["eval_metric"] = mean_absolute_error
            #blob.upload_from_string(data=json.dumps(best_params, indent=4), content_type="application/json")
        elif param_flag == "manual":
            # Get params
            params = manual_params
            params["eval_metric"] = mean_absolute_error
            
        return params

##################################
# Category Utils Class for category universe
# #################################
class MarkdownUtils:
    
    def __init__(self, category_universe):
        # Category Universe
        self.category_universe = list(set(map(int, category_universe.replace(" ", "").split(','))))

                

##################################
# Data Processing Class
# #################################
class Preprocessing:
    def __init__(self):
        pass 

    def change_dtypes(self, data, cols):
        """Convert the data type of given column to string.
        Args:
            data: Pandas dataframe
            cols: Converting column to string.
        Returns:
            data: Pandas dataframe with converted column.
        """
        # Convert Category and Sub Category to String
        data[cols] = data[cols].astype(str)
        
        return data
    
    def typecast_datetime(self, data, cols):
        """Typecast datetime for multiple columns.
        Args:
            data: Pandas dataframe. 
            cols: multiple columns to change the column value type to datetime. 
        Returns:
            Pandas dataframe with column of datetime type. 
        """
        data[cols] = data[cols].astype("datetime64[ns]")
        return data
    
    def typecast_float(self, data, cols):
        """Typecast float for multiple columns.
        Args:
            data: Pandas dataframe. 
            cols: multiple columns to change the column value type to float. 
        Returns:
            Pandas dataframe with column of float type.
        """
        data[cols] = data[cols].apply(np.float64)
        return data
        

##################################
# Pre Modelling Processing Class
# #################################

class MarkdownPreModeling:
    def __init__(self):
        pass

    def monotonic_constraints(self, train, covariates, constrained_features):
        """Monotonically set up boolean values based on contrained columns.
        1 if col is in constrained_features and 0 else. 
        Args:
            train: train data 
            covariates: covariate feature columns.
            constrained_features: contrained feature columns
        Returns:
            tuple of constraints in boolean.
        """
        # Return Boolean List of Features with Constraints
        x_train = train[covariates]
        bool_constraints = [1 if col in constrained_features else 0 for col in x_train.columns]
        print(f'Monotone Constraints for the Model: {bool_constraints}')

        return tuple(bool_constraints)

    def category_encoded_data(self, train, val, covariates, response, category_cols=['region_name', 'cat_subcat']):
        """Category encoding based on `region_name` and `cat_subcat` for the targeting column `cum_sale`.
        Args:
            train: train Pandas dataframe. 
            val: validation Pandas dataframe. 
            covariates: covariate feature columns.
            response: response feature columns for category encoding. 
            category_cols: (default ['region_name', 'cat_subcat'])
        Returns:
            train and validation data with catagory encoding based on target (y_train: data for response e.g. `cum_sale`)
            
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
        """
        # Initialize Encoder
        cbe_encoder = ce.cat_boost.CatBoostEncoder(cols=category_cols)
        
        # Fit encoder and Transform the Data
        x_train = train[covariates]
        y_train = train[response]
        x_val = val[covariates]
        cbe_encoder.fit(x_train, y_train)
        train_cbe = cbe_encoder.transform(x_train)
        test_cbe = cbe_encoder.transform(x_val)
        
        # Combine X and Y
        train = pd.concat([train_cbe, y_train], axis=1)
        val = pd.concat([test_cbe, val[response]], axis=1)
        
        return train, val

##################################
# Model Training Class
# #################################
class ModelTraining:
    def __init__(self):
        pass
    
    def hyperparam_tuning(
        self, 
        train, 
        val, 
        covariates, 
        response, 
        rounds, 
        tolerance, 
        monotone_constraints, 
        model_metric, 
        max_evals, 
        model_verbose=False
    ):
        """Tune hyper-parameters with Optuna for eXtreme Gradient Boosting regressor. 
        Args:
            train: train Pandas dataframe. 
            val: validation Pandas dataframe. 
            covariates: covariate feature columns.
            response: response feature columns for category encoding.  
            rounds: rounds for early stopping in XGboost. 
            tolerance: minimum delta for early stopping in XGboost.
            monotone_constraints: monotonic contstraints XG Boost Regressor. 1 is for increase, -1 is for decrease, and 0 is monotonic
            model_metric: evaluation metrics for tunning. 
            max_evals: number of trials for optimization. 
        Returns:
            estimate the best score for hyper-parameter tuning.
        
        NOTE: ** monototic_contraints ** 
            week_1_discount : 1
            week_2_discount : 1
            week_3_discount : 1
            week_4_discount : 1
            week_5_discount : 1
            week_6_discount : 1
            week_7_discount : 1
            week_8_discount : 1
            md_week_count : 1
            week_of_year : 0
            frac_pre_units : 0
            region_name : 0
            cat_subcat : 0
            md_start_inventory : 0
            sales_1_week_back : 0
            sales_2_week_back : 0
            sales_3_week_back : 0
            sales_4_week_back : 0
            pre_md_selling_price : 0
        """
        # Objective Function to Minimize
        def objective(trial):

            params = {
              'n_estimators': 600, 
              'learning_rate' : trial.suggest_discrete_uniform('learning_rate', 0.001, 0.051, 0.01), 
              'max_depth': trial.suggest_int('max_depth', 14, 38, 2),
              'colsample_bytree': trial.suggest_discrete_uniform('colsample_bytree', 0.6, 0.9, 0.1),
              'subsample': trial.suggest_discrete_uniform('subsample', 0.6, 0.9, 0.1),
              'lambda': trial.suggest_discrete_uniform('lambda', 3.1, 10.1, 1)
            }
            
            print(f'Model Parameters ....{str(params)}....')

            model = xgboost.XGBRegressor(
                **params,
                n_jobs=80,
                monotone_constraints=tuple(monotone_constraints),                                          
                eval_metric=model_metric,
                random_state=123
            )

            early_stop = xgboost.callback.EarlyStopping(rounds=rounds, min_delta=tolerance)

            model.fit(
                train[covariates],
                train[response],
                eval_set=[(val[covariates], val[response])],
                verbose=False,
                callbacks=[early_stop]
            )

            estimate = model.best_score
            print(f'Error on Validation Set is ....{str(estimate)}....')

            return estimate

        study_cat = optuna.create_study(direction='minimize')
        
        study_cat.optimize(
            objective,
            n_trials=max_evals,
            n_jobs=4,
            show_progress_bar=True,
            gc_after_trial=True
            
        )
        best_param = study_cat.best_params

        return best_param

    def fit_model(
        self, 
        train, 
        val, 
        covariates, 
        response, 
        rounds, 
        tolerance, 
        monotone_constraints, 
        category_cols, 
        model_params
    ):
        """Train the model. 
        Args:
            train: train Pandas dataframe. 
            val: validation Pandas dataframe. 
            covariates: covariate feature columns.
            response: response feature columns for category encoding.
            rounds: rounds for early stopping in XGboost. 
            tolerance: minimum delta for early stopping in XGboost.
            monotone_constraints: monotonic contstraints XG Boost Regressor. 1 is for increase, -1 is for decrease, and 0 is monotonic
            category_cols: catetory columns for encoding. e.g. ['cat_subcat', 'region_name']
            model_params: model parameters. 
        Returns:
            model pipeline object. 
        """
        # Create Pipeline
        estimators = [
            ('encoding', ce.cat_boost.CatBoostEncoder(cols=category_cols)),
            ('xgb_model', xgboost.XGBRegressor(
                **model_params,
                monotone_constraints=tuple(monotone_constraints),
                random_state=123)
            ),
        ]
        pipeline = Pipeline(steps=estimators)

        # Early Stopping
        early_stop = xgboost.callback.EarlyStopping(rounds=rounds, min_delta=tolerance)

        # Fit Pipeline
        x_train = train[covariates]
        y_train = train[response]
        x_val = val[covariates]
        y_val = val[response]
        pipeline.fit(
            x_train,
            y_train,
            xgb_model__verbose=True,
            xgb_model__callbacks=[early_stop],
            xgb_model__eval_set=[(pipeline.steps[:-1][0][1].fit(x_train, y_train).transform(x_val), y_val)]
        )

        return pipeline

    def train_val_metrics(
        self,
        pipeline, 
        train, 
        val, 
        covariates, 
        response
    ):
        """Calculate model evaluation metrics.
        Args:
            pipeline: model pipeline.
            train: train Pandas dataframe. 
            val: validation Pandas dataframe. 
            covariates: covariate feature columns.
            response: response feature columns for category encoding.  
            model_metric: model evaluation metrics. 
        Returns:
            Evaluation metrics as output. 
        """
        train_pred = pipeline.predict(train[covariates])
        val_pred = pipeline.predict(val[covariates])
        y_train = train[response]
        y_val = val[response]

        y_train = y_train.values.reshape((-1, 1))
        train_pred = train_pred.reshape((-1, 1))

        y_val = y_val.values.reshape((-1, 1))
        val_pred = val_pred.reshape((-1, 1))

        # Metrics
        metrics = {
            "train_mae": round(mean_absolute_error(y_train, train_pred), 4),
            "train_rmse": round(np.sqrt(mean_squared_error(y_train, train_pred)) ,4),
            "train_mape": round(mean_absolute_percentage_error(y_train, train_pred) ,4),
            "train_smape":  round(smape(y_train, train_pred) ,4),
            "train_weighted_mape":  round(weighted_mape(y_train, train_pred) ,4),
            "train_median_ape":round(median_ape(y_train, train_pred) ,4),
            "train_overestimation_sample_pct": round(oep(y_train, train_pred)[0], 4),
            "train_overestimation_total_units": round(oep(y_train, train_pred)[1], 4),
            "train_overestimation_units_pct": round(oep(y_train, train_pred)[2], 4),
            "train_overestimation_mean_pct": round(oep(y_train, train_pred)[3], 4),



            "val_mae": round(mean_absolute_error(y_val, val_pred), 4),
            "val_rmse": round(np.sqrt(mean_squared_error(y_val, val_pred)), 4),
            "val_mape": round(mean_absolute_percentage_error(y_val, val_pred), 4),
            "val_smape": round(smape(y_val, val_pred), 4),
            "val_weighted_mape": round(weighted_mape(y_val, val_pred), 4),
            "val_median_ape": round(median_ape(y_val, val_pred), 4),
            "val_overestimation_sample_pct": round(oep(y_val, val_pred)[0], 4),
            "val_overestimation_total_units": round(oep(y_val, val_pred)[1], 4),
            "val_overestimation_units_pct": round(oep(y_val, val_pred)[2], 4),
            "val_overestimation_mean_pct": round(oep(y_val, val_pred)[3], 4),


        }
        
        return metrics

##################################
# Evaluation Layer class
# #################################
class EvaluationLayer:
    def __init__(self):
        pass
    
    def full_markdown_data(self, data):
        """Filter the full data with the condition of `md_week_no` == 1 after adding feature `md_week_no` based on sorting `date` and groupbying (`club_nbr`, `system_item_nbr`, `n_md`) and cummulating it.
        
        Args:
            data: Pandas dataframe. 
        Returns:
            data: Filtered data. 
        """
        # Return Week when Markdown Ends
        data = data.copy(deep=False)
        data['md_week_no'] = (
            data.sort_values(['date'], ascending=[True])
                .groupby(['club_nbr', 'system_item_nbr', 'n_md'])
                .cumcount()+1
        )
        data = data[data['md_week_no'] == 1]
        print(f'No of Data Points Evaluated {len(data)}')
        
        return data

    def full_session_data(self, data):
        """Filter the full data with the condition of `md_session_no` == 1 after adding feature `md_week_no` based on sorting `date` and groupbying (`club_nbr`, `system_item_nbr`, `n_md`, `selling_price`) and cummulating it.
        Args:
            data: Pandas dataframe. 
        Returns:
            data: Filtered data.
        """
        # Return Week when Session Ends
        data = data.copy(deep=False)
        data['md_session_no'] = (
            data.sort_values(['date'], ascending=[True])
                .groupby(['club_nbr', 'system_item_nbr', 'n_md', 'selling_price'])
                .cumcount()+1
        )
        data = data[data['md_session_no'] == 1]
        print(f'No of Data Points Evaluated {len(data)}')

        return data    

    def sale_prediction(self, data, pipeline, covariates):
        """Conditionally mark the predicted result as 1 if `predicted_sale` <= 0 after getting the model prediction. After then, get the min value between column `predicted_sale` and `md_start_inventory`.
        Args:
            data: feeding data to the model.
            pipeline: model pipeline.
            covariates: covariate feature columns for feeding data. 
        """
        # Predict Sale
        data = data.copy(deep=False)    
        data['predicted_sales'] = pipeline.predict(data[covariates])
        
        # Rectify Predicted Sale and Cap by Available Inventory
        data['predicted_sales'] = round(data['predicted_sales'])
        data.loc[data['predicted_sales'] <= 0, 'predicted_sales'] = 0
        if ("inv" in data.columns) and ("inventory" not in data.columns):
            data['inventory'] = data['inv']
        elif ("fc_inventory" in data.columns) and ("inventory" not in data.columns):
            data['inventory'] = data["fc_inventory"]
        elif ("inventory_qty" in data.columns) and ("inventory" not in data.columns):
            data['inventory'] = data["inventory_qty"]
        
        if "inventory" in data.columns:
            data['predicted_sales'] = data[['predicted_sales', 'inventory']].min(axis=1)
            
        
        return data

    def cap_prediction(self, data, capping_map):
        """Based on given column, left-join the data on column `bucket` and filtered data > 80 quantiled. 
        Args:
            data: Pandas dataframe. 
            capping_map: mapping for joining. 
        """
        # Cap the Prediction Based in Mapping Table
        data = data.copy(deep=False)
        data = data.merge(
            capping_map,
            on=['bucket'],
            how='left'
        )
        capby = round(data['quantile80']*data['md_week_count'])
        data.loc[(data['predicted_sale'] > capby), 'predicted_sale'] = capby
        
        return data
    
    def test_metrics(self, data): # markdown_metrics
        """Get the metrics of the model performance for mean_ape, median_ape, rmse, and mae after adding feature `units_error`, `pe`, `ape`.
        Args:
            data: predicted result
        Returns:
            metrics: metric values as hashmap
            data: predicted result
        """
        data = data.copy(deep=False)
        test_pred = data['predicted_sales'].values
        y_test = data['target'].values

        metrics = {
            "test_mae": round(mean_absolute_error(y_test, test_pred), 2),
            "test_rmse": round(np.sqrt(mean_squared_error(y_test, test_pred)) ,2),
            "test_mape": round(mean_absolute_percentage_error(y_test, test_pred) ,2),
            "test_smape":  round(smape(y_test, test_pred) ,2),
            "test_weighted_mape":  round(weighted_mape(y_test, test_pred) ,2),
            "test_median_ape":round(median_ape(y_test, test_pred) ,2),
            "test_overestimation_sample_pct": round(oep(y_test, test_pred)[0], 2),
            "test_overestimation_total_units": round(oep(y_test, test_pred)[1], 2),
            "test_overestimation_units_pct": round(oep(y_test, test_pred)[2], 2),
            "test_overestimation_mean_pct": round(oep(y_test, test_pred)[3], 2),
        }
        
        return metrics, data

    def eval_metrics(self, data): # markdown_metrics
        """Get the metrics of the model performance for mean_ape, median_ape, rmse, and mae after adding feature `units_error`, `pe`, `ape`.
        Args:
            data: predicted result
        Returns:
            metrics: metric values as hashmap
            data: predicted result
        """
        data = data.copy(deep=False)
        test_pred = data['predicted_sales'].values
        y_test = data['target'].values
        
        num_week_y_test = {}
        num_week_test_pred = {}
        
        for num_week in sorted(data.num_weeks.unique()):
            filtered_data = data[data['num_weeks'] == num_week]
            
            y_test_nw = list(filtered_data['target'])
            num_week_y_test['y_test_nw_{}'.format(num_week)] = y_test_nw
            
            test_pred_nw = list(filtered_data['predicted_sales'])
            num_week_test_pred['test_pred_nw_{}'.format(num_week)] = test_pred_nw
            
        metrics = {
            "cumul_eval_bias": round(bias(y_test, test_pred), 4),
            "cumul_eval_mae": round(mean_absolute_error(y_test, test_pred), 4),
            "cumul_eval_rmse": round(np.sqrt(mean_squared_error(y_test, test_pred)) ,4),
            "cumul_eval_mape": round(mean_absolute_percentage_error(y_test, test_pred) ,4),
            "cumul_eval_smape":  round(smape(y_test, test_pred) ,4),
            "cumul_eval_weighted_mape":  round(weighted_mape(y_test, test_pred) ,4),
            "cumul_eval_median_ape":round(median_ape(y_test, test_pred) ,4),
            "cumul_eval_overestimation_sample_pct": round(oep(y_test, test_pred)[0], 4),
            "cumul_eval_overestimation_total_units": round(oep(y_test, test_pred)[1], 4),
            "cumul_eval_overestimation_units_pct": round(oep(y_test, test_pred)[2], 4),
            "cumul_eval_overestimation_mean_pct": round(oep(y_test, test_pred)[3], 4),
            "cumul_eval_underestimation_sample_pct": round(uep(y_test, test_pred)[0], 4),
            "cumul_eval_underestimation_total_units": round(uep(y_test, test_pred)[1], 4),
            "cumul_eval_underestimation_units_pct": round(uep(y_test, test_pred)[2], 4),
            "cumul_eval_underestimation_mean_pct": round(uep(y_test, test_pred)[3], 4),
        }
        for i in range(1, 9):
            if f'y_test_nw_{i}' in num_week_y_test:
                metrics[f"cumul_eval_wmape_nw_{i}"] = round(weighted_mape(np.array(num_week_y_test[f'y_test_nw_{i}']), np.array(num_week_test_pred[f'test_pred_nw_{i}'])) ,4)
        
        return metrics, data

    def eval_metrics_weekly(self, data): # markdown_metrics
        """Get the metrics of the model performance for mean_ape, median_ape, rmse, and mae after adding feature `units_error`, `pe`, `ape`.
        Args:
            data: predicted result
        Returns:
            metrics: metric values as hashmap
            data: predicted result
        """
        data = data.copy(deep=False)
        test_pred = data['predicted_sales_weekly'].values
        y_test = data['target_weekly'].values
        
        num_week_y_test = {}
        num_week_test_pred = {}
        
        for num_week in sorted(data.num_weeks.unique()):
            filtered_data = data[data['num_weeks'] == num_week]
            
            y_test_nw = list(filtered_data['target_weekly'])
            num_week_y_test['y_test_nw_{}'.format(num_week)] = y_test_nw
            
            test_pred_nw = list(filtered_data['predicted_sales_weekly'])
            num_week_test_pred['test_pred_nw_{}'.format(num_week)] = test_pred_nw

        metrics = {
            "week_eval_bias": round(bias(y_test, test_pred), 4),
            "week_eval_mae": round(mean_absolute_error(y_test, test_pred), 4),
            "week_eval_rmse": round(np.sqrt(mean_squared_error(y_test, test_pred)) ,4),
            "week_eval_mape": round(mean_absolute_percentage_error(y_test, test_pred) ,4),
            "week_eval_smape":  round(smape(y_test, test_pred) ,4),
            "week_eval_weighted_mape":  round(weighted_mape(y_test, test_pred) ,4),
            "week_eval_median_ape":round(median_ape(y_test, test_pred) ,4),
            "week_eval_overestimation_sample_pct": round(oep(y_test, test_pred)[0], 4),
            "week_eval_overestimation_total_units": round(oep(y_test, test_pred)[1], 4),
            "week_eval_overestimation_units_pct": round(oep(y_test, test_pred)[2], 4),
            "week_eval_overestimation_mean_pct": round(oep(y_test, test_pred)[3], 4),
            "week_eval_underestimation_sample_pct": round(uep(y_test, test_pred)[0], 4),
            "week_eval_underestimation_total_units": round(uep(y_test, test_pred)[1], 4),
            "week_eval_underestimation_units_pct": round(uep(y_test, test_pred)[2], 4),
            "week_eval_underestimation_mean_pct": round(uep(y_test, test_pred)[3], 4),
        }
        for i in range(1, 9):
            if f'y_test_nw_{i}' in num_week_y_test:
                metrics[f"weekly_eval_wmape_nw_{i}"] = round(weighted_mape(np.array(num_week_y_test[f'y_test_nw_{i}']), np.array(num_week_test_pred[f'test_pred_nw_{i}'])) ,4)
                
                metrics[f"weekly_eval_bias_nw_{i}"] =round(bias(np.array(num_week_y_test[f'y_test_nw_{i}']), np.array(num_week_test_pred[f'test_pred_nw_{i}'])), 4)
                
                metrics[f"weekly_eval_sample_count_nw_{i}"] =len(num_week_y_test[f'y_test_nw_{i}'])
            
        return metrics, data

    def compute_category_error(self,data):
        """Compute Category Error.
        Args:
            data: Pandas dataframe of merged_pred_cat.
        Return 
            errors per category.
        """
        results = []
        for cat in sorted(data.dept_nbr.unique()):
            df_cat = data[data['dept_nbr'] == cat]

            y_test =df_cat['target'].values.reshape((-1, 1))
            test_pred = df_cat['predicted_sales'].values.reshape((-1, 1))

            m_mae = mean_absolute_error(y_test, test_pred)
            m_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

            m_wmape = weighted_mape(y_test, test_pred)
            m_median_ape = median_ape(y_test, test_pred)
            m_smape = smape(y_test, test_pred)

            m_oep_sample, m_total_oe_units, m_oep_total, m_woep = oep(y_test, test_pred, y_test)
            n_samples = y_test.shape[0]

            cat_result = pd.DataFrame(data=[
                [n_samples, m_mae, m_rmse, m_wmape, m_median_ape, m_smape, m_oep_sample, m_total_oe_units, m_oep_total,
                 m_woep]], columns=['n_samples', 'mae', 'rmse', 'wmape', 'median_ape', 'smape', 'oep_sample',
                                    'oe_value', 'oep_total', 'oep_weighted'])
            cat_result['dept_nbr'] = cat
            cat_result['metric_target'] = 'units'
            cat_result['eval_set_start_date'] = data.date.min()
            results.append(cat_result)
        return pd.concat(results, axis=0)
    
    def compute_num_week_error(self,data):
        """Compute Num_week Error.
        Args:
            data: Pandas dataframe of merged_pred_cat.
        Return 
            errors per category.
        """
        results = []
        for num_week in sorted(data.num_weeks.unique()):
            df_num_week = data[data['num_weeks'] == num_week]

            y_test =df_num_week['target'].values.reshape((-1, 1))
            test_pred = df_num_week['predicted_sales'].values.reshape((-1, 1))
            
            y_test_weekly =df_num_week['target_weekly'].values.reshape((-1, 1))
            test_pred_weekly = df_num_week['predicted_sales_weekly'].values.reshape((-1, 1))


            m_wmape = weighted_mape(y_test, test_pred)
            m_weekly_bias=round(bias(y_test, test_pred), 4)
            m_wmape_weekly =round( weighted_mape(y_test_weekly, test_pred_weekly),4)

            n_samples = y_test.shape[0]

            num_week_result = pd.DataFrame(data=[
                [n_samples, m_wmape, m_wmape_weekly]], columns=['n_samples', 'wmape', 'wmape_weekly'])
            num_week_result['num_weeks'] = num_week
            num_week_result['metric_target'] = 'units'
            num_week_result['weekly_bias'] = m_weekly_bias
            num_week_result['eval_set_start_date'] = data.date.min()
            results.append(num_week_result)
        return pd.concat(results, axis=0)
    
    def combine_hmaps(self, hmap1, hmap2):
        """Combine two hmaps for having one full metrics. 
        Args:
            hmap1: first hash map
            hmap2: second hash map
        Returns:
            combined hash map.
        """
        hmap1.update(hmap2)
        return hmap1

class PostAnalysis:
    def __init__(self):
        pass
    

    def mark_monotonicity(self, predicted_sales):
        """Mark Monotonic patterns
        Args: 
            predicted_sales: Pandas dataframe.
        Returns:
            boolean as monotonic or not. 
        """
        monotonic = 0
        max_val = np.inf
        for idx in range(0, len(predicted_sales)):
            # Update Max Value Iteratively
            if predicted_sales[idx] < max_val:
                max_val = predicted_sales[idx]
            else:
                monotonic = 1
                break
        return monotonic
    
    def check_monotonicity(self, data):
        """Check the array in column of 'week_1_discount' in data is monotonic or not after adding 'monotonic' column.
        Args: 
            data: Pandas dataframe.
        Returns:
            boolean as monotonic or not. 
        """
        monotonicity = (
            data.sort_values("week_1_discount")
                .groupby(["club_nbr", "system_item_nbr"])["predicted_sales"].apply(list)
                .reset_index()
        )
        monotonicity["monotonic"] = monotonicity["predicted_sales"].apply(self.mark_monotonicity)
        
        if np.sum(monotonicity["monotonic"]) == len(monotonicity):
            statement = "All Item Clubs Show Monotonic Behavior"
            print(statement)
            return statement
        else:
            statement = "Item Clubs Do Not Show Monotonic Behavior"
            return statement
            # raise Exception(statement)
    
    def check_elasticity(self, data):
        """Check elasticity of given data.
        Args: 
            data: Pandas dataframe.
        Returns:
            String statement of proportion of inelasticity.
        """
        # Computing Elasticity
        elasticity = data.loc[(data.week_1_discount == 0.1) | (data.week_1_discount == 0.6)]
        elasticity["price"] = elasticity.apply(lambda x: round(x['past_half_year_median_price']*x["price_ratio_1_week_back"] * (1 - x["week_1_discount"]), 2), axis=1)
        
        elasticity[['sales_10', 'price_10']] = (
            elasticity.groupby(['club_nbr', 'system_item_nbr'])[['predicted_sales', 'price']]
                      .shift(1)
        )
        elasticity.rename(columns={'predicted_sales': 'sales_60', 'price': 'price_60'}, inplace=True)
        
        elasticity = elasticity[~elasticity['sales_10'].isna()]
        
        elasticity['change_in_sales'] = (
            (elasticity['sales_60'] - elasticity['sales_10'])
            /
            (elasticity['sales_60'] + elasticity['sales_10'])
        )
        elasticity['change_in_price'] = (
            (elasticity['price_10'] - elasticity['price_60'])
            /
            (elasticity['price_10'] + elasticity['price_60'])
        )
        elasticity['elasticity'] = elasticity['change_in_sales'] / elasticity['change_in_price']
        
        # Sale at 90% Discount
        sale_90 = data.loc[(data.week_1_discount == 0.9)][['club_nbr', 'system_item_nbr', 'predicted_sales']]
        sale_90.rename(columns={'predicted_sales': 'sale_90'}, inplace=True)

        # Check Elasticity Condition
        elasticity = elasticity.merge(
            sale_90,
            on=['club_nbr', 'system_item_nbr'],
            how='inner'
        )
        elasticity['inelastic'] = 0
        elasticity.loc[elasticity['elasticity'] < 0.10, 'inelastic'] = 1
        elasticity.loc[elasticity['sale_90'] >= elasticity['inventory']*0.75, 'inelastic'] = 0
        
        no_inelastic_items = np.sum(elasticity['inelastic'])
        statement = f"No of Inelastic Items {no_inelastic_items} \nProportion of Inelastic Items {no_inelastic_items/len(elasticity)}"
        print(statement)
        return statement

        
    def feat_imp_ggplot(self, data, x_val, y_val):
        """Analyze/plot the important features 
        Args: 
            data: Pandas dataframe.
            x_val: x feature
            y_val: y feature
        Returns:
            bar plot of important features.
        """
        feat_imp_plot = (
            pn.ggplot(data)
            + pn.geom_col(pn.aes(x=x_val, y=y_val))
            + pn.theme(figure_size=(20, 6), axis_text_x=pn.element_text(angle=45))
        )
        return feat_imp_plot
        
    def ape_cdf_ggplot(self, data, text, x_val, y_val=None):
        """Plot the cdf for absolute percentage error. 
        Args: 
            data: Pandas dataframe.
            text: annotation text.
            x_val: x feature
            y_val: y feature
        Returns:
            cumulative dstributed function plot for absolute percentage error.
        """
        ape_cdf_plot = (pn.ggplot(data, pn.aes(x=x_val))
                + pn.stat_ecdf()
                + pn.xlim(0, 2)
                + pn.scale_x_continuous(breaks=np.arange(0, 2.25, 0.25), limits=[0, 2])        
                + pn.geom_abline(intercept=0,slope=1/2,linetype='dotted')
                + pn.labs(x='Absolute Percentage Error', y='Probability')
                + pn.annotate('text', x=1, y=0.75, label=text)
               )
        return ape_cdf_plot
    
    def cat_ape_cdf_ggplot(self, data, x_val, y_val=None):
        """Plot a cdf for absolute percentage error per category.
        Args: 
            data: Pandas dataframe.
            x_val: x feature
            y_val: y feature
        Returns:
            cumulative dstributed function plot for absolute percentage error per category.
        """
        cat_ape_cdf_plot = (
            pn.ggplot(data, pn.aes(x=x_val))
            + pn.stat_ecdf()
            + pn.facet_wrap('description')
            + pn.xlim(0, 2)        
            + pn.theme(figure_size=(18, 12))
            + pn.geom_vline(xintercept=[0.1, 0.5])
            + pn.geom_abline(intercept=0,slope=1/2,linetype='dotted')
            + pn.labs(x='Absolute Percentage Error', y='Probability')
        )
        return cat_ape_cdf_plot
    
    def units_error_ggplot(self, data, text, x_min_lim, x_max_lim, x_axis, y_axis, x_val, y_val=None):
        """Plot units error density.
        Args: 
            data: Pandas dataframe.
            text: annotation text.
            x_min_lim: min x-axis limitation
            x_max_lim: max x-axis limitation
            x_axis: annotation x_axis
            y_axis: annotation y_axis
            x_val: x feature
            y_val: y feature
        Returns:
            Plot units error density.
        """
        units_error_plot = (
            pn.ggplot(data, pn.aes(x=x_val))
            + pn.geom_density()
            + pn.geom_rug()
            + pn.xlim(x_min_lim, x_max_lim)
            + pn.annotate('text', x=x_axis, y=y_axis, label=text)
        )
        return units_error_plot
    
    def error_true_pred_ggplot(self, data, text, x_val, y_val):
        """Plot based on error, cum_sale (true), and predicted sale. 
        Args: 
            data: Pandas dataframe.
            text: annotation text.
            x_val: x feature
            y_val: y feature
        Returns:
            Plot one vs another.
        """
        error_wrt_true_plot = (
            pn.ggplot(data, pn.aes(x=x_val, y=y_val))
            + pn.geom_point(pn.aes(color='md_start_inventory'))
            + pn.geom_smooth(color='cyan')
        )
        return error_wrt_true_plot
    

##################################
# Slack Integration class
# #################################
class SlackIntegration:
    def __init__(self):
        pass
    
    def slack_client(self):
        """Get the Slack client via slack token. 
        NOTE: NEED TO GT THE TOKEN AND SAVE IT IN GOOGLE SECRET MANAGERS.
        Args: 
            None
        Returns:
            client: Slack client
            api_response: api_test() output from Slack client. 
        """
        HUBOT_SLACK_TOKEN = dbutils.secrets.get(scope="slack",key="hubot")
        client = WebClient(token=HUBOT_SLACK_TOKEN)
        api_response = client.api_test()

        return client, api_response
