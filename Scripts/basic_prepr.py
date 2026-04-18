import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
import os, sys
sys.path.append(r"C:\hotel_booking")
from Src.basic_prepr import Basic_preprocessing

data = r'C:\hotel_booking\Data\Raw_Data\hotel_bookings_updated_2024.csv'
from sklearn.model_selection import train_test_split
df = pd.read_csv(data)
X = df.drop(columns='is_canceled')
y = df['is_canceled']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

basic_preprocessing = Basic_preprocessing(df = df, target='is_canceled')

basic_preprocessing.imputation().encoding().scaling()

