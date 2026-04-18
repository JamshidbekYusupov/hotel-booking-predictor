import os, sys
import pandas as pd
import numpy as np

sys.path.append(r"C:\hotel_booking")
X_test_tree = pd.read_csv(r'C:\hotel_booking\Data\Prep_Data\basic_prep\encoding\X_test_enc.csv')

from Src.feature_engineering import HotelFeatureEngineering

eng = HotelFeatureEngineering()

test_sample = X_test_tree.sample(5)

transformed_sample = eng.transform(test_sample)

# Checking the results
print("New Columns Check:")
print(transformed_sample[['total_guests', 'price_per_person']].head(3))

print("\nLog Transform Check (Values should be small):")
print(transformed_sample['adr'].head())

