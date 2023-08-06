import os
import pandas as pd
from sklearn.model_selection import train_test_split
from mlFlowProject import logger
from mlFlowProject.entity.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
    
    def train_test_split(self):
        data = pd.read_csv(self.config.data_path)
        data = data.dropna()
        # X = data['Open']
        # y = data['Close']
       
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
        
        # train = pd.DataFrame()
        # train['X'] = X_train
        # train['y'] = y_train
        # test = pd.DataFrame()
        # test['X'] = X_test
        # test['y'] = y_test
        train, test = train_test_split(data)
        
        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index = False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index = False)
        
        logger.info("Data split into train and test")
        logger.info(f"train data shape: {train.shape}")
        logger.info(f"test data shape: {test.shape}")
        
        # print(train.shape)
        # print(test.shape)