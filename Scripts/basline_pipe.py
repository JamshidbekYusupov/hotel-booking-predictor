import os, sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split

sys.path.append(r"C:\hotel_booking")
from Src.basline_pipe import Basline

X_train_tree = pd.read_csv(r'C:\hotel_booking\Data\Prep_Data\basic_prep\encoding\X_train_enc.csv')
X_test_tree = pd.read_csv(r'C:\hotel_booking\Data\Prep_Data\basic_prep\encoding\X_test_enc.csv')

X_train_scaled = pd.read_csv(r'C:\hotel_booking\Data\Prep_Data\basic_prep\scaling\X_train_scal.csv')
X_test_scaled = pd.read_csv(r'C:\hotel_booking\Data\Prep_Data\basic_prep\scaling\X_test_scal.csv')

y_train = pd.read_csv(r'C:\hotel_booking\Data\Prep_Data\basic_prep\y_train.csv').values.ravel()
y_test = pd.read_csv(r'C:\hotel_booking\Data\Prep_Data\basic_prep\y_test.csv').values.ravel()

df = pd.read_csv(r'C:\hotel_booking\Data\Raw_Data\hotel_bookings_updated_2024.csv')

X = df.drop(columns = 'is_canceled')
y = df['is_canceled']

X_train, X_test, y_train, y_test = train_test_split(X,  y, test_size=0.2, random_state=42)



lr = LogisticRegression(max_iter=3000)
dt = DecisionTreeClassifier(random_state=42)
svm = SVC(probability=True, random_state=42)

models = {
    "Random_Forest":RandomForestClassifier(random_state=42),
    "XGBoost": xgb.XGBClassifier(random_state=42),

    "Hard_voting":VotingClassifier(
    estimators=[('lr', lr),('svm', svm)],   #--Base learners: lr, dt, svc
    voting='hard'),

    "Soft_voting":VotingClassifier(
    estimators=[('lr', lr), ('svm', svm)],   #--Base learners: lr, dt, svc
    voting='soft'
)
}

tree_models = ['Random_Forest','XGBoost']
all_results = []
for name, model in models.items():

    # if name in tree_models:
    #     X_train = X_train_tree
    #     X_test = X_test_tree
    # else:
    #     X_train = X_train_scaled
    #     X_test = X_test_scaled
    
    baseline = Basline(X_train=X_train, X_test = X_test, 
                      y_train = y_train, y_test = y_test, 
                      algorithm=model, model_name=name)
    baseline.pipeline_building()
    baseline.pipeline_fit()
    baseline.prediction()
    metrics = baseline.model_evaluvation()
    all_results.append(metrics)

    baseline.model_saving()
import pandas as pd
from tabulate import tabulate
import os

results_df = pd.DataFrame(all_results)

metrics_path = r'C:\hotel_booking\Metrics\Baseline_Evaluvation'
os.makedirs(metrics_path, exist_ok=True)

out_path = os.path.join(metrics_path, 'baseline_all_models.txt')

table = tabulate(results_df, headers='keys', tablefmt='grid', showindex=True)

with open(out_path, 'w') as f:   # <-- overwrite instead of append
    f.write("Final Model Evaluvation \n")
    f.write(table)
