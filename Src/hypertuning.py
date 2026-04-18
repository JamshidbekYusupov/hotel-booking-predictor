from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import logging
import os, sys
import joblib
import numpy as np
from skopt import BayesSearchCV

sys.path.append(r"C:\hotel_booking")
from Src.feature_engineering import HotelFeatureEngineering


log_path = r'C:\hotel_booking\Logging\hyperparamter.log'

logging.basicConfig(
    filename = log_path,
    filemode ='a',
    level = logging.INFO,
    format = '%(asctime)s-%(levelname)s-%(message)s'
)

class HyperTuning(BaseEstimator):
    def __init__(self, X_train, X_test, y_train, y_test, algorithm, algorithm_name):

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = algorithm
        self.model_name = algorithm_name
        self.metrics = {}
        self.preprocessor = None
        self.y_pred = None
    
    def build_pipe(self):
##==================== Buliding pipeline, for categorical and numerical columns=============##
        try:

            fe = HotelFeatureEngineering()
            X_train_fe = fe.fit_transform(self.X_train)
            num_cols = X_train_fe.select_dtypes(include = [np.number]).columns.to_list()
            cat_cols = X_train_fe.select_dtypes(exclude = [np.number]).columns.to_list()
            
            #### Numeric pipeline
            num_pipe = Pipeline([
                ('impute', KNNImputer(n_neighbors=3)),
                ('scale', StandardScaler())
            ])
            ## Categorical Pipeline
            cat_pipe = Pipeline([
                ('impute', SimpleImputer(strategy='most_frequent')),
                ('encode', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
            ])
##=============================Column transforming========================##
            self.preprocessor = ColumnTransformer([
                    ('num', num_pipe, num_cols),
                    ('cat', cat_pipe, cat_cols)
            ])
            logging.info(f'Pipeline is built for {self.model_name} successfully')
            return self
        except Exception as e:
            logging.error(f'Error while building pipeline for {self.model_name}')
            raise
##======================== Hyperparameter tuning==============================##
    def hyperparameter_tuning(self, params):
        try:
             logging.info(f"Starting Bayesian optimization for {self.model_name}")
             
             full_pipeline = Pipeline([
                 ('engineering', HotelFeatureEngineering()),
                 ('preprocessor', self.preprocessor),
                 ('model', self.model)
             ])

             bayesian_search = BayesSearchCV(
                 estimator=full_pipeline,
                 search_spaces=params,
                 cv = 3,
                 n_jobs=-1,
                 random_state=42,
                 n_iter=6,
                 scoring='recall', ## here recall is more important as we should not miss 0s
                 )
             
             bayesian_search.fit(self.X_train, self.y_train)
             
             self.model = bayesian_search.best_estimator_

             logging.info(f'Bayesion optimization is done complately for {self.model_name}')
             logging.info(f'Best params:{bayesian_search.best_params_}')

             return self
        except Exception as e:
            logging.error(f'Error while hyperparametr tuning for {self.model_name}')
            raise
## ======================= Saving the model after hyperparameter tuning====================##
    def saving_model(self):
        try:
            model_path = r'C:\hotel_booking\Models\Improved'
            os.makedirs(model_path, exist_ok=True)
            out_path = os.path.join(model_path, f'{self.model_name}.joblib')
            joblib.dump(self.model, out_path)
            logging.info(f'{self.model_name} is saved at {model_path}')
            return self
        
        except Exception as e:
            logging.error(f'Error while saving {self.model_name}')
            raise
##========================Prediction with X_test set=======================##
    def prediction(self):

        try:
            self.y_pred = self.model.predict(self.X_test)
            logging.info(f'Prediction for the {self.model_name} is done')
            return self
        except Exception as e:
            logging.error(f'Error while doing prediction for {self.model_name}')
            raise
## ====================================Getting Metrics ==========================##
    def evaluation(self):
        try:
            # Binary classification — weighted = macro here, kept for consistency
                self.metrics={
                'Model': self.model_name,
                'Accuracy Score': accuracy_score(self.y_test, self.y_pred),
                'Precision Score': precision_score(self.y_test, self.y_pred, average='weighted'),
                'F1 Score': f1_score(self.y_test, self.y_pred, average='weighted'),
                'Recall Score': recall_score(self.y_test, self.y_pred,average='weighted'),
                'Roc Auc':roc_auc_score(self.y_test, self.y_pred,average='weighted')
                    }
                logging.info(f'Metrics for {self.model_name} is done successfully')
                return self.metrics
        except Exception as e:
            logging.error(f'Error while evaluvation of {self.model_name}')
            raise