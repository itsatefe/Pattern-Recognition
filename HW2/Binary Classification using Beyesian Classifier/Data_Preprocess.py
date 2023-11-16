import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import jdc
from sklearn.model_selection import train_test_split


# 3.1 Data_Preprocess

class Data_Preprocess:
    def __init__(self, url, column_names_csv,threshold_nan=2000):
        self.url = url
        self.threshold_nan = threshold_nan
        self.column_names_csv = column_names_csv
        self.dataset = None


    def load_dataset(self):
        column_names_df = pd.read_csv(self.column_names_csv)
        column_names = column_names_df['Column_Names'].tolist()
        self.dataset = pd.read_csv(self.url, header=None, names=column_names)
        self.dataset.replace("?", np.nan, inplace=True)    


    def display_data_types(self):
        if self.dataset is not None:
            return self.dataset.dtypes
        else:
            return "Dataset not loaded."


    def display_unique_values(self):
        if self.dataset is not None:
            for column in self.dataset.columns:
                unique_values = self.dataset[column].unique()
                print(f"Unique values for {column}:\n{unique_values}\n")
        else:
            print("Dataset not loaded.")


    def encode_categorical_features(self):
        if self.dataset is not None:
            label_encoder = LabelEncoder()
            categorical_features = self.dataset.select_dtypes(include=['object']).columns
            for feature in categorical_features:
                self.dataset[feature] = label_encoder.fit_transform(self.dataset[feature])
        else:
            print("Dataset not loaded.")


    def drop_columns_with_high_nan(self, threshold=2000):
        if self.dataset is not None:
            nan_counts = self.dataset.isna().sum()
            columns_to_drop = nan_counts[nan_counts > threshold].index
            self.dataset = self.dataset.drop(columns=columns_to_drop)
            return nan_counts
        else:
            print("Dataset not loaded.")


