# -*- coding: utf-8 -*-
"""Decision trees data on versions

"""
import dvc.api
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from fast_ml.model_development import train_valid_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import plot_confusion_matrix

import mlflow
import mlflow.sklearn
import logging
import warnings

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


path = 'data/AdSmartABdata.csv'
repo = "../"
version = "'v3'"
# return to normal tag version and print in markdown

data_url = dvc.api.get_url(
    path=path,
    repo=repo,
)

mlflow.set_experiment('Ad Campaign Analysis')

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    np.random.seed(50)
    df = pd.read_csv('../data/AdSmartABdata.csv', index_col=0)
    mlflow.log_param('data_url', data_url)
    mlflow.log_param('data_version', version)
    mlflow.log_param('input_rows', df.shape[0])
    mlflow.log_param('input_cols', df.shape[1])
    mlflow.log_param('model_type','Decision Trees')

    len(df['device_make'].unique())

    def group_brand(brand):
        '''
        Groups several individual phones into one manufacturer group
        '''
        device_makes = df['device_make'].unique()
        for device_make in device_makes:
            if brand in device_make.lower():
                # print(device_make)
                # print(df['device_make'][df['device_make']== device_make] )
                df['device_make'][df['device_make']
                                  == device_make] = brand
        # print(len(df['device_make'].unique()))

    known_brands = ['lg', 'nokia', 'iphone', 'pixel', 'xiaomi', 'oneplus', 'htc', 'samsung', 'vog', 'vfd', 'pot', 'moto', 'mrd', 'MAR-LX1A',
                    'huawei', 'ANE-LX1', 'CLT-L09', 'ELE-L09', 'I3312']
    # group known brands together
    for known_brand in known_brands:
        group_brand(known_brand)

    df['device_make'].unique()
    phone_group = df.groupby('device_make')
    phone_group['device_make'].value_counts()

    for platform in df['device_make'].unique():
        print(platform)
        group_brand(platform)

    df['device_make'].unique()
    len(df['device_make'].unique())
    for grouped_platform in df['device_make'].unique():
        if grouped_platform not in known_brands:
            print(grouped_platform)
            df['device_make'][df['device_make']
                              == grouped_platform] = "Generic Smartphone"
        else:
            pass

    print(len(df['device_make'].unique()))

    df.head()

    ordinal_encoder = OrdinalEncoder()
    ordinal_encode_columns = ['experiment', 'date', 'device_make']
    df[ordinal_encode_columns] = ordinal_encoder.fit_transform(
        df[ordinal_encode_columns])
    df.head()

    X = df.drop("answer", axis=1)

    scaler = StandardScaler()
    scaler.fit(X)
    zz = scaler.transform(X)
    scaled_features_df = pd.DataFrame(zz, index=df.index, columns=X.columns)
    scaled_features_df.head()

    scaled_features_df = pd.concat(
        [scaled_features_df, df.answer], axis=1)

    X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(scaled_features_df, target='answer',
                                                                                method='sorted', sort_by_col='date',
                                                                                train_size=0.7, valid_size=0.1, test_size=0.2)

    clf = DecisionTreeClassifier(random_state=0)
    # cross_val_score(clf, X_train, y_train, cv=10)
    clf.fit(X_train, y_train)

    predicted_views = clf.predict(X_test)
    acc = accuracy_score(y_test, predicted_views)
    print(acc)
    mlflow.log_param('acc', acc)
    clf.get_depth()
    with open("dt_metrics.txt", 'w') as outfile:
        outfile.write("Accuracy: " + str(acc) + "\n")

    # Plot it
    disp = plot_confusion_matrix(
        clf, X_test, y_test, normalize='true', cmap=plt.cm.Blues)
    plt.savefig('dt_confusion_matrix.png')
