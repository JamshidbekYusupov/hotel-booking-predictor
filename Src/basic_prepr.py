import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelBinarizer, MinMaxScaler, StandardScaler
import logging
from sklearn.model_selection import train_test_split
log_path = r'C:\hotel_booking\Logging\basic_processing.log'

logging.basicConfig(
    filename = log_path,
    filemode ='a',
    level = logging.INFO,
    format = '%(asctime)s-%(levelname)s-%(message)s'
)

class Basic_preprocessing:
    def __init__(self, df: pd.DataFrame, target:str):
        self.df = df
        self.X_train, self.X_test, self.y_train,self.y_test = train_test_split(self.df.drop(columns=target), 
                                            self.df[target], test_size=0.2, random_state=42)

    def imputation(self):

        self.X_train = self.X_train.copy()
        self.X_test = self.X_test.copy()
        try:
            fill_values = {}
            for col in self.X_train.columns:
                if self.X_train[col].dtype == 'object':
                    fill_values[col] = self.X_train[col].mode()[0]
                    # self.X_train[col] = self.X_train[col].fillna(fill_values[col])
                else:
                    fill_values[col] = self.X_train[col].mean()
                    # self.X_train[col] = self.X_train[col].fillna(fill_values[col])
            
            self.X_train[col] = self.X_train[col].fillna(fill_values[col])
            
            for col, value in fill_values.items():
                self.X_test[col] = self.X_test[col].fillna(value)
            
            logging.info(f'Basic imputation is completed')

            data_path = r'C:\hotel_booking\Data\Prep_Data\basic_prep'

            os.makedirs(data_path, exist_ok=True)
            out_path = os.path.join(data_path, 'X_train_imputed.csv')
            self.X_train.to_csv(out_path, index = False)
            ### X_Test saving
            out_path = os.path.join(data_path, 'X_test_imputed.csv')
            self.X_test.to_csv(out_path, index = False)
            logging.info(f'Imputed data is saved at {data_path}')
            return self
        except Exception as e:
            logging.error(f'Error while doing imputation:{e}')
            raise

    
    def encoding(self):
        try:
            one_hot_cols = []
            ord_enc_vals = {}

            for col in self.X_train.columns:
                
                if self.X_train[col].dtype == 'object':
                    if self.X_train[col].nunique() <= 3:
                        one_hot_cols.append(col)
                    else:
                        ord_enc = OrdinalEncoder(
                        handle_unknown='use_encoded_value',
                        unknown_value=-1
                    )
                        self.X_train[col] = ord_enc.fit_transform(self.X_train[[col]]).ravel()
                        ord_enc_vals[col] = ord_enc
            self.X_train = pd.get_dummies(self.X_train, columns=one_hot_cols, drop_first=True, dtype=int)
            for col, values in ord_enc_vals.items():
                self.X_test[col] = values.transform(self.X_test[[col]]).ravel()

            self.X_test = pd.get_dummies(self.X_test, columns=one_hot_cols, drop_first=True, dtype=int)
            self.X_test = self.X_test.reindex(columns=self.X_train.columns, fill_value=0)
            
            data_path = r'C:\hotel_booking\Data\Prep_Data\basic_prep'
            os.makedirs(data_path, exist_ok=True)

            ### X Train saving
            out_path = os.path.join(data_path, 'X_train_enc.csv')
            self.X_train.to_csv(out_path, index = False)
            ### X_Test saving
            out_path = os.path.join(data_path, 'X_test_enc.csv')
            self.X_test.to_csv(out_path, index = False)
            logging.info(f'Data is encoded and saved at {data_path}')
            return self
    
        except Exception as e:
            logging.error(f'Error while encoding: {e}')
            raise

    def scaling(self):
        try:

            num_cols = self.X_train.select_dtypes(include = [np.number])
            min_max = MinMaxScaler()

            for col in num_cols:
                self.X_train[col] = min_max.fit_transform(self.X_train[[col]])
                self.X_test[col] = min_max.transform(self.X_test[[col]])
            self.X_test[col] = min_max.transform(self.X_test[[col]])

            data_path = r'C:\hotel_booking\Data\Prep_Data\basic_prep'
            os.makedirs(data_path, exist_ok=True)

            ### X Train saving
            out_path = os.path.join(data_path, 'X_train_scal.csv')
            self.X_train.to_csv(out_path, index = False)
            ### X_Test saving
            out_path = os.path.join(data_path, 'X_test_scal.csv')
            self.X_test.to_csv(out_path, index = False)

            logging.info(f'Data is scaled and saved at {data_path}')
            return self
        except Exception as e:
            logging.error(f'Error while scaling data: {e}')
            raise        
