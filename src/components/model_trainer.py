import os
import sys
from dataclasses import dataclass
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "XGBoost": XGBClassifier(),
                "CatBoosting": CatBoostClassifier(verbose=False),
                "AdaBoost": AdaBoostClassifier(),
                "GradientBoost": GradientBoostingClassifier(),
                "Support Vector Classifier": SVC()
               
            }
            params = {
            "Decision Tree": {
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'max_depth': [None, 10, 20, 30, 40, 50],
            #'min_samples_split': [2, 5, 10],
            #'min_samples_leaf': [1, 2, 4]
            },
            "Random Forest": {
            'n_estimators': [8, 16, 32, 64],
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 20, 30, 40],
            #'min_samples_split': [2, 5, 10],
            #'min_samples_leaf': [1, 2, 4],
            #'max_features': ['auto', 'sqrt', 'log2']
            },
            "Gradient Boosting": {
            'n_estimators': [8, 16, 32, 64, 128],
            'learning_rate': [0.1, 0.01, 0.05, 0.001],
            #'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
            'max_depth': [3, 5, 7, 9],
            #'min_samples_split': [2, 5, 10],
            #'min_samples_leaf': [1, 2, 4]
            },
            
            "XGBoost": {
            'n_estimators': [8, 16, 32, 64, 128],
            'learning_rate': [0.1, 0.01, 0.05, 0.001],
            'max_depth': [3, 5, 7, 9],
            #'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
            #'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0]
            },
            "CatBoosting": {
            'iterations': [30, 50, 100],
            'depth': [6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1],
            #'l2_leaf_reg': [1, 3, 5, 7, 9]
            },
            "AdaBoost": {
            'n_estimators': [8, 16, 32, 64, 128],
            'learning_rate': [0.1, 0.01, 0.5, 0.001],
            'algorithm': ['SAMME']
            },
            "GradientBoost": {
            'n_estimators': [8, 16, 32, 64, 128],
            'learning_rate': [0.1, 0.01, 0.05, 0.001],
            #'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
            'max_depth': [3, 5, 7, 9],
            #'min_samples_split': [2, 5, 10],
            #'min_samples_leaf': [1, 2, 4]
            },
            "Support Vector Classifier": {
            'C': [0.001, 0.01, 0.1, 1, 10],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'degree': [2, 3, 4, 5],
            #'gamma': ['scale', 'auto']
            },
        }
            model_report = evaluate_models(X_train, y_train, X_test, y_test, models,params=params)

            ## To get best model score from dict
            best_model_accuracy = max(sorted(model_report.values()))

            ## To get best model name from dict
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_accuracy)]
            best_model = models[best_model_name]

            print(f'Best Model Found, Model Name: {best_model_name}, Accuracy: {best_model_accuracy}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found, Model Name: {best_model_name}, Accuracy: {best_model_accuracy}')

            if best_model_accuracy < 0.6:
                raise CustomException("No best model found")
            logging.info("Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            accuracy = accuracy_score(y_test, predicted)
            return accuracy


        except Exception as e:
            raise CustomException(e,sys)
