import pandas as pd
import os
from mlFlowProject import logger
from sklearn.ensemble import RandomForestRegressor
# import optuna
import joblib
import numpy as np
from mlFlowProject.entity.config_entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        
    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)
        
        x_train = np.array(train_data['Open']).reshape(-1, 1)
        x_test = np.array(test_data['Open']).reshape(-1, 1)
        y_train = np.array(train_data[[self.config.target_col]]).reshape(-1)
        y_test = np.array(test_data[[self.config.target_col]]).reshape(-1)
        
        # print(x_train.shape, y_train.shape)
        
        rfr = RandomForestRegressor(
            n_estimators= self.config.n_estimators,
            max_depth=self.config.max_depth
        )
        
        rfr.fit(x_train, y_train)
        
        joblib.dump(rfr, os.path.join(self.config.root_dir, self.config.model_name))