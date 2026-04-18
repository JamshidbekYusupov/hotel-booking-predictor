from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
import tabulate
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler, StandardScaler

log_path = r'C:\hotel_booking\Logging\basline_pipe.log'
logging.basicConfig(
    filename = log_path,
    filemode ='a',
    level = logging.INFO,
    format = '%(asctime)s-%(levelname)s-%(message)s'
)


class Basline(BaseEstimator):
    def __init__(self, X_train, X_test, y_train, y_test, algorithm, model_name):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_pred = None
        self.algorithm = algorithm
        self.model_name = model_name
        self.metrics = {}
        self.model = None
        self.preprocessor = None
    
    def pipeline_building(self):
        try:
            num_cols = self.X_train.select_dtypes(include = [np.number]).columns.to_list()
            cat_cols = self.X_train.select_dtypes(exclude = [np.number]).columns.to_list()

            cat_pipe = Pipeline([
                ('impute', SimpleImputer(strategy='most_frequent')),
                ('encode', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))

            ])  
            num_pipe = Pipeline([
                ('impute',SimpleImputer(strategy='mean')),
                ('scaler', MinMaxScaler())
            ])

            self.preprocessor = ColumnTransformer(
                transformers=[
                ('cat', cat_pipe, cat_cols),
                ('num', num_pipe, num_cols),
                ]
            )
            logging.info(f'Pipeline for categorical and numerical features is built')
            return self
        except Exception as e:
            logging.error(f'Error while building pipeline for categorical and numerical features')
            raise 

    def pipeline_fit(self):
        try:
            if self.algorithm is None:
                raise ValueError(f'Model is not identified, pls assign the algorithm')
            
            self.model = Pipeline([
                ('preprocessor', self.preprocessor),
                ('model', self.algorithm)
            ])
            self.model.fit(self.X_train, self.y_train)
            logging.info(f'Model is trained with {self.model_name}')
            return self
        except Exception as e:
            logging.error(f'Error while training model with {self.model_name} algorithm, Error: {e}')
            raise
    
    def prediction(self):
        try:
            self.y_pred = self.model.predict(self.X_test)
            logging.info(f'Prediction is done with {self.model_name}')
            return self
        except Exception as e:
            logging.error(f'Error while predcting with {self.model_name}')
            raise
    def model_evaluvation(self):
        try:
            self.metrics = {
                'Model': self.model_name,
                'Accuracy Score': accuracy_score(self.y_test, self.y_pred),
                'Precison Score': precision_score(self.y_test, self.y_pred, average='weighted'),
                'F1 Score': f1_score(self.y_test, self.y_pred, average='weighted'),
                'Recall Score': recall_score(self.y_test, self.y_pred,average='weighted'),
            }
            # metrics_df = pd.DataFrame([self.metrics])

            # metrics_path = r'C:\hotel_booking\Metrics\Baseline_Evaluvation'

            # os.makedirs(metrics_path, exist_ok=True)
            # out_path = os.path.join(metrics_path, 'baseline_evaluvation.txt')
            # table = tabulate.tabulate(metrics_df, headers = 'keys',tablefmt='grid', showindex=False)

            # with open(out_path, 'a') as f:
            #     f.write(f"Evaluation Results of {self.model_name}\n")
            #     f.write(table)
            #     f.write('\n')
            logging.info(f'Results are saved at {self.model_name}')

            return self.metrics
        except Exception as e:
            logging.error(f'Error while saving results: {e}')
            raise
        
    def model_saving(self):
        try:
            out_dir = r'C:\hotel_booking\Models\Baseline'
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f'{self.model_name}.joblib')
            joblib.dump(self.model, out_path)
            logging.info(f'{self.model_name} is saved at {out_dir}')
            return self
        except Exception as e:
            logging.error(f'Error while saving {self.model_name}')
            raise
