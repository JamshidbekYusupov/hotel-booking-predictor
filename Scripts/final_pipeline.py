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
from tabulate import tabulate
from skopt.space import Real, Categorical, Integer
sys.path.append(r"C:\hotel_booking")
from Src.hypertuning import HyperTuning

## Giving raw data to our pipeline
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

    # "Hard_voting":VotingClassifier(
    # estimators=[('lr', lr),('dt', dt)],   #--Base learners: lr, dt, svc
    # voting='hard'),

    # "Soft_voting":VotingClassifier(
    # estimators=[('lr', lr), ('dt', dt)],   #--Base learners: lr, dt, svc
    # voting='soft'
# )
}

param_grids = {

    "Random_Forest":{
        'model__n_estimators': Integer(10, 30),
        'model__max_depth': Integer(3, 10),
        'model__min_samples_split': Integer(2, 10),
        'model__min_samples_leaf': Integer(1, 10),
        'model__max_features': Categorical(['sqrt', 'log2', None]),
        },

    "XGBoost":{
        'model__n_estimators': Integer(10, 30),
        'model__max_depth': Integer(3, 10),
        'model__learning_rate': Real(0.01, 0.3, prior='log-uniform'),
        'model__subsample': Real(0.6, 1.0),
        'model__reg_alpha': Real(0.01, 10, prior='log-uniform'),
        'model__reg_lambda': Real(0.01, 10, prior='log-uniform')
        },

    # For VotingClassifiers we tune the base learners only
    # Prefix: model__<estimator_name>__<param>

    # "Hard_voting":{
    #     'model__lr__C': Real(0.01, 10, prior='log-uniform'),
    #     'model__lr__max_iter': Integer(10, 30),
    #     'model__svm__C': Real(0.01, 10, prior='log-uniform'),
    #     'model__svm__kernel': Categorical(['linear', 'rbf', 'poly'])
    #     },

    # "Soft_voting":{
    #     'model__lr__C': Real(0.01, 30, prior='log-uniform'),
    #     'model__lr__max_iter': Integer(10, 30),
    #     'model__svm__C': Real(0.01, 10, prior='log-uniform')
    #     # kernel fixed to 'rbf' for soft voting
    #     },
}
all_results = []

for name, model in models.items():

    hyper_tuning = HyperTuning(X_train=X_train, X_test = X_test, 
                      y_train = y_train, y_test = y_test, 
                      algorithm=model, algorithm_name=name)
    hyper_tuning.build_pipe()
    hyper_tuning.hyperparameter_tuning(param_grids[name])
    hyper_tuning.saving_model()
    hyper_tuning.prediction()
    metrics = hyper_tuning.evaluation()
    all_results.append(metrics)

results_df = pd.DataFrame(all_results)

metrics_path = r'C:\hotel_booking\Metrics\Hyperparameter_Metrics'
os.makedirs(metrics_path, exist_ok=True)

out_path = os.path.join(metrics_path, 'Hyperparameter_Metrics.txt')

table = tabulate(results_df, headers='keys', tablefmt='grid', showindex=True)

with open(out_path, 'w') as f:   # <-- overwriting instead of appendding
    f.write("Final Model Evaluation \n")
    f.write(table)

## Pring best model metrics
best = results_df.loc[results_df['Recall Score'].idxmax(), 'Model']
print(f"Best model by Recall: {best}")