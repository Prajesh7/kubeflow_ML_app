import pandas as pd
import os
from mlFlowProject import logger
from sklearn.ensemble import RandomForestRegressor
import optuna
import yaml
import joblib
import numpy as np
from mlFlowProject.entity.config_entity import ModelTrainerConfig
from mlFlowProject.constants import *
from sklearn.model_selection import cross_val_score

# class ModelTrainer:
#     def __init__(self, config: ModelTrainerConfig):
#         self.config = config
        
#     def train(self):
#         train_data = pd.read_csv(self.config.train_data_path)
#         test_data = pd.read_csv(self.config.test_data_path)
        
#         x_train = np.array(train_data['Open']).reshape(-1, 1)
#         x_test = np.array(test_data['Open']).reshape(-1, 1)
#         y_train = np.array(train_data[[self.config.target_col]]).reshape(-1)
#         y_test = np.array(test_data[[self.config.target_col]]).reshape(-1)
        
#         # print(x_train.shape, y_train.shape)
        
#         rfr = RandomForestRegressor(
#             n_estimators= self.config.n_estimators,
#             max_depth=self.config.max_depth
#         )
        
#         rfr.fit(x_train, y_train)
        
#         joblib.dump(rfr, os.path.join(self.config.root_dir, self.config.model_name))

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
    
    def hp_tune(self, trial, xtrain, ytrain):
        # crit = trial.suggest_categorical("criterion", ["entropy"])
        n_est = trial.suggest_int("n_estimators", 2, 200, log = True)
        m_depth = trial.suggest_int("max_depth", 1, 100, log = True)
        rfr = RandomForestRegressor(n_estimators = n_est, max_depth = m_depth)
        
        score = cross_val_score(rfr, xtrain, ytrain, cv = 3)
        accuracy = score.mean()
        return accuracy

    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)
        
        x_train = np.array(train_data['Open']).reshape(-1, 1)
        x_test = np.array(test_data['Open']).reshape(-1, 1)
        y_train = np.array(train_data[[self.config.target_col]]).reshape(-1)
        y_test = np.array(test_data[[self.config.target_col]]).reshape(-1)
        
        print(x_train.shape, y_train.shape)
        study = optuna.create_study(direction="maximize")
        # study.optimize(self.hp_tune, (n_trials=25, x_train, y_train))
        study.optimize(lambda trial: self.hp_tune(trial, x_train, y_train), n_trials=25)
        n_estimators = study.best_params['n_estimators']
        max_depth = study.best_params['max_depth']
        rfr = RandomForestRegressor(
            n_estimators= n_estimators,
            max_depth= max_depth
        )
        
        rfr.fit(x_train, y_train)
        
        joblib.dump(rfr, os.path.join(self.config.root_dir, self.config.model_name))
        
        if self.config.n_estimators != n_estimators or self.config.max_depth != max_depth:
            with open(PARAMS_FILE_PATH, 'r') as f:
                tuned_params = yaml.safe_load(f)
            tuned_params["RandomForestRegressor"]["n_estimators"] = n_estimators
            tuned_params["RandomForestRegressor"]["max_depth"] = max_depth
            with open(PARAMS_FILE_PATH, 'w') as f:
                yaml.dump(tuned_params, f, default_flow_style=False)