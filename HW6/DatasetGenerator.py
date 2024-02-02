from sklearn.datasets import load_iris, make_moons
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

class DatasetGenerator:
    def __init__(self):
        pass

    def get_linear_dataset(self):
        iris = load_iris()
        X = iris.data
        y = iris.target
        iris_df = pd.DataFrame(X, columns=iris.feature_names)
        iris_df['species'] = iris.target_names[y]
        filtered_df = iris_df[iris_df['species'].isin(['setosa', 'versicolor'])]
        linear_dataset = filtered_df[['petal length (cm)', 'petal width (cm)', 'species']]
        dataset = pd.DataFrame()
        dataset['species'] = LabelEncoder().fit_transform(linear_dataset['species'])
        scaler = StandardScaler()
        dataset[['petal length (cm)', 'petal width (cm)']] = scaler.fit_transform(linear_dataset[['petal length (cm)', 'petal width (cm)']])
        return dataset


    def get_nonlinear_dataset(self, n_samples=500, noise=0.15, random_state=42):
        X_non_linear, y_non_linear = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
        nonlinear_dataset = pd.DataFrame(X_non_linear, columns=['Feature1', 'Feature2'])
        nonlinear_dataset['Label'] = y_non_linear
        X_nonlinear = nonlinear_dataset[['Feature1', 'Feature2']]
        y_nonlinear = nonlinear_dataset['Label']
        scaler = StandardScaler()
        X_nonlinear_scaled = scaler.fit_transform(X_nonlinear)
        nonlinear_dataset_scaled = pd.DataFrame(X_nonlinear_scaled, columns=['Feature1', 'Feature2'])
        nonlinear_dataset_scaled['Label'] = y_nonlinear.reset_index(drop=True)
        return nonlinear_dataset_scaled
