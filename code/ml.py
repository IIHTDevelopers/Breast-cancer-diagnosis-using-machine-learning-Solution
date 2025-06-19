import pandas as pd
import numpy as np
import pickle
import bisect
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_log_error, accuracy_score, precision_score, \
        recall_score, f1_score
import sys, os
sys.path.insert(0, os.getcwd())
from code import constants
 
class TargetEncoder:
    def __init__(self, cat_vars):
        self.cat_vars = cat_vars
        self.encoder = OneHotEncoder(sparse=False)
 
    def fit(self, X, y):
        self.encoder.fit(X[self.cat_vars])
 
    def transform(self, X):
        encoded = self.encoder.transform(X[self.cat_vars])
        X = X.drop(self.cat_vars, axis=1)
        X_encoded = pd.concat([X, pd.DataFrame(encoded, index=X.index)], axis=1)
        return X_encoded
 
class CostMetrics():
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
 
    def msle(self):
        return mean_squared_log_error(self.y_true, self.y_pred)
 
    def accuracy(self):
        return accuracy_score(self.y_true, self.y_pred)
 
    def precision(self):
        return precision_score(self.y_true, self.y_pred)
 
    def recall(self):
        return recall_score(self.y_true, self.y_pred)
 
    def f1_score(self, avg='binary'):
        return f1_score(self.y_true, self.y_pred, average=avg)
 
class Model():
    def __init__(self):
        self.data_file = constants.DATA_FOLDER + constants.DATA_FILE
        self.model_file = constants.MODEL_FOLDER + constants.MODEL_FILE
        self.graphs_folder = constants.GRAPHS_FOLDER
 
    def data_transformation(self, test_data=None, is_train=True):
        if not is_train or test_data is not None:
            test = test_data.copy()
 
        X = pd.read_csv(self.data_file)
        X[X.diagnosis == 'M'] = 1
        X[X.diagnosis == 'B'] = 0
        y = X['diagnosis']
        y = y.astype('int')
        X = X.drop(['diagnosis', 'id'], axis=1)
        numerical_vars = list(X._get_numeric_data().columns)
        cat_vars = list(set(X.columns) - set(numerical_vars))
 
        X_train, X_test, y_train, y_test = train_test_split(
            X[numerical_vars + cat_vars],  # predictors
            y,
            test_size=0.2,
            random_state=3,
        )
 
        if len(numerical_vars):
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            X_train[numerical_vars] = imputer.fit_transform(X_train[numerical_vars])
            X_test[numerical_vars] = imputer.transform(X_test[numerical_vars])
            if not is_train:
                test[numerical_vars] = imputer.transform(test[numerical_vars])
 
            transformer = MinMaxScaler()
            X_train[numerical_vars] = transformer.fit_transform(X_train[numerical_vars])
            X_test[numerical_vars] = transformer.transform(X_test[numerical_vars])
            if not is_train:
                test[numerical_vars] = transformer.transform(test[numerical_vars])
 
        if len(cat_vars):
            imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
            X_train[cat_vars] = imputer.fit_transform(X_train[cat_vars])
            X_test[cat_vars] = imputer.transform(X_test[cat_vars])
            if not is_train:
                test[cat_vars] = imputer.transform(test[cat_vars])
 
            transformer = TargetEncoder(cat_vars)
            transformer.fit(X_train[cat_vars], y_train)
            X_train = transformer.transform(X_train)
            X_test = transformer.transform(X_test)
            if not is_train:
                test = transformer.transform(test)
 
        if is_train:
            return X_train, X_test, y_train, y_test
        else:
            return test
 
    def model_fit(self, X_train, y_train):
        gb_clf = GradientBoostingClassifier(
            loss='log_loss',  # or 'exponential'
            random_state=10,
            n_estimators=50,
        )
        gb_clf.fit(X_train, y_train)
        pickle.dump(gb_clf, open(self.model_file, 'wb'))
 
    def model_predict(self, X_test):
        loaded_model = pickle.load(open(self.model_file, 'rb'))
        y_pred = loaded_model.predict(X_test)
        return y_pred
 
    def cost_metric(self, y_true, y_pred):
        cm = CostMetrics(y_true, y_pred)
        return cm.accuracy()
 
if __name__ == "__main__":
    if not os.path.exists(constants.MODEL_FOLDER):
        os.mkdir(constants.MODEL_FOLDER)
    model = Model()
    X_train, X_test, y_train, y_test = model.data_transformation()
    model.model_fit(X_train, y_train)