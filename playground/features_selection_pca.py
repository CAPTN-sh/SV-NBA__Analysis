## DEPENDENCIES >>>
import os
import sys
from typing import List, Tuple, Dict, Any, Optional, Callable, Union
from pathlib import Path

import joblib
from functools import partial

# Add root directory to path for imports >
root_dir = Path.cwd().resolve().parent
if root_dir.exists():
    sys.path.append(str(root_dir))
else:
    raise FileNotFoundError('Root directory not found')

# import custom libraries >
from src.load import load_multiple_trajectoryCollection_parallel_pickle as lmtp
from src.load import load_datasets, load_df_to_dataset
from src.traj_dataloader import (TrajectoryDataset, 
                                 create_dataloader, 
                                 separate_files_by_season, 
                                 split_data, 
                                 get_files,
                                 AISDataset,
                                 )
from src.scaler import CustomMinMaxScaler, reduce_resolution

from datetime import datetime, timedelta

import dotsi
import itertools
import pickle

import numpy as np
import pandas as pd

# torch libraries >
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.io import read_image

# sklearn libraries >
import sklearn as sk
from sklearn.model_selection import (train_test_split, 
                                     GridSearchCV, 
                                     RandomizedSearchCV)#, HalvingGridSearchCV, HalvingRandomSearchCV)
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score 
# from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline, make_pipeline

# Features selection >
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import ( mutual_info_classif,
                                       SelectKBest,
                                       chi2,
                                       VarianceThreshold,
                                       RFE,
                                       )
from mlxtend.feature_selection import ExhaustiveFeatureSelector
from skfeature.function.similarity_based import fisher_score
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA

# Hyperopt >
import optuna

# Mask warnings
import warnings
warnings.filterwarnings("ignore")  # Mask all warnings
# from sklearn.exceptions import ConvergenceWarning
# warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Plot >
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots  # https://github.com/garrettj403/SciencePlots?tab=readme-ov-file
plt.style.use(['science', 'grid', 'notebook'])  # , 'ieee'

# Multiprocessing >
from concurrent.futures import ProcessPoolExecutor
from functools import partial

# Toy datasets >
from sklearn.datasets import load_iris  # Sample dataset


# %%
## FLAGS & GLOBAL VALUES >>>

# Down sample the resolution
DOWN_SAMPLE = False  # used with SCALE and SAVE_SCALE to save the scaled data: (if True) with down sampled resolution, or with (not False) not.
# Explore
EXPLORE = False
# Debug
DEBUG = False
# Develop
DEVELOP = False

# HYPERPARAMETER OPTIMISATION
HYPEROPT = True

if HYPEROPT:
    OPTUNA = False # Optimise using Optuna
    GRIDSEARCH = False  # Optimise using GridSearchCV
    RANDOMSEARCH = True  # Optimise using RandomizedSearchCV

# SAVE SELECTED FEATURES in root / models / selected_features
SAVE_SELECT_FEATURES = True

# WORKING SERVER
AVAILABLE_SERVERS = ['ZS', 'PLOEN', 'KIEL', 'WYK']
CURRENT_SERVER = AVAILABLE_SERVERS[0]

# seed
split_seed = 42

# If DOWN_SAMPLE, define the target time resolution
targeted_resolution_min = 1  # minute

# TODO: The following featues are corrupted by containing NaNs. Fix this. For now, these columns are dropped
corrupted_features = ["stopped", "abs_ccs", "curv"]


# Use up to 70% of the available cpu cores
n_jobs = joblib.cpu_count()
print("Number of CPUs available:", n_jobs)
if CURRENT_SERVER == 'ZS':
    n_jobs = int(0.6 * n_jobs)
else:
    n_jobs = int(0.7 * n_jobs)
print("Number of CPUs to use:", n_jobs)

# %%
## PATHS >>>
# data dir
data_dir = root_dir / 'data'
data_dir = data_dir.resolve()
if not data_dir.exists():
    raise FileNotFoundError('Data directory not found')

if CURRENT_SERVER == 'ZS':
    # assets dir  # TODO: Used temporarly during the features seletion process. Remove this!
    assets_dir = data_dir / 'assets'
    assets_dir = assets_dir.resolve()
    if not assets_dir.exists():
        raise FileNotFoundError(f'Assets directory in {CURRENT_SERVER} not found')
else:
    # aistraj dir
    assets_dir = data_dir / 'local' / 'aistraj'
    assets_dir = assets_dir.resolve()
    if not assets_dir.exists():
        raise FileNotFoundError('Assets directory not found')

    # train-validate-test (tvt) dir
    tvt_assets_dir = assets_dir / 'tvt_assets'
    tvt_assets_dir = tvt_assets_dir.resolve()
    if not tvt_assets_dir.exists():
        raise FileNotFoundError('Train-Validate-Test Assets directory not found')

    # tvt: extended pickle dir
    tvt_extended_dir = tvt_assets_dir / 'extended'
    tvt_extended_dir = tvt_extended_dir.resolve()
    if not tvt_extended_dir.exists():
        raise FileNotFoundError('TVT Extended Pickled Data directory not found')

    # tvt: scaled pickle dir
    tvt_scaled_dir = tvt_assets_dir / 'scaled'
    tvt_scaled_dir = tvt_scaled_dir.resolve()
    if not tvt_scaled_dir.exists():
        raise FileNotFoundError('TVT Scaled Pickled Data directory not found')

    # tvt: logs dir
    tvt_logs_dir = tvt_assets_dir / 'logs'
    tvt_logs_dir = tvt_logs_dir.resolve()
    if not tvt_logs_dir.exists():
        raise FileNotFoundError('TVT logs directory not found')
    
    
# models dir
models_dir = root_dir / 'models'
models_dir = models_dir.resolve()
if not models_dir.exists():
    raise FileNotFoundError('Models directory not found')
    
# Selected Features dir
selected_features_dir = models_dir / 'selected_features'
selected_features_dir = selected_features_dir.resolve()
if not selected_features_dir.exists():
    raise FileNotFoundError('selected features directory not found')

# %%
import_paths = {'train': None, 'validate': None, 'test': None}

if DOWN_SAMPLE:
    import_paths = {
                    'train': tvt_scaled_dir / 'scaled_cleaned_downsampled_extended_train_df.parquet',
                    'validate': tvt_scaled_dir / 'scaled_cleaned_downsampled_extended_validate_df.parquet',
                    'test': tvt_scaled_dir / 'scaled_cleaned_downsampled_extended_test_df.parquet'
                    }
else:  
    if CURRENT_SERVER != 'ZS':
        import_paths = {
                        'train': tvt_scaled_dir / 'scaled_cleaned_extended_train_df.parquet',
                        'validate': tvt_scaled_dir / 'scaled_cleaned_extended_validate_df.parquet',
                        'test': tvt_scaled_dir / 'scaled_cleaned_extended_test_df.parquet'
                        }
    else:
        import_paths = {
                        'train': assets_dir / 'scaled_cleaned_extended_train_df.parquet',
                        'validate': assets_dir / 'scaled_cleaned_extended_validate_df.parquet',
                        'test': assets_dir / 'scaled_cleaned_extended_test_df.parquet'
                        }
        
# Assets container >
train_df, validate_df, test_df = None, None, None
assets = {'train': train_df, 'validate': validate_df, 'test': test_df}

# %%
'''
Load the data
'''
def load_data():

    validate_df = load_df_to_dataset(import_paths['validate'], use_dask=False).data  # Load the validate dataset

    # %%
    if EXPLORE:
        columns = validate_df.columns
        print(f"Num. Cols: {len(columns)}: {columns}")
        print()
        print(f"Num. Samples: {validate_df.shape[0]}")

    # %%
    if EXPLORE:
        print(validate_df.describe())

    # %%
    if EXPLORE:
        validate_df.info()
    
    # Concatenate the datasets >
    asset_df = validate_df  # pd.concat([train_df, validate_df, test_df], axis=0)
    
    # %%
    # Remove the columns not to include in the feature selection
    cols_not_to_study = ['epoch', 'datetime', 'obj_id', 'traj_id', 'stopped', 'curv']

    # Check that the column in cols_not_to_study are in the dataset, otherwise remove them from the list >
    cols_not_to_study = [col for col in cols_not_to_study if col in asset_df.columns]

    print(f"Cols not to study: {cols_not_to_study}")
    df = asset_df.drop(columns=cols_not_to_study)
    return df 

# %%
'''
Feature Selction using PCA
'''
def feature_selection_pca(df: pd.DataFrame, n_components: int) -> Tuple[PCA, pd.DataFrame]:
    """
    Perform feature selection using PCA.

    Args:
        df (DataFrame): The input dataframe containing the features.
        n_components (int): The number of principal components to keep.

    Returns:
        tuple: A tuple containing two objects:
            - pca_model (PCA): The trained PCA model.
            - fs_df (DataFrame): DataFrame containing selected features and their importance scores.
    """
    
    # Instantiate a place holder for the variance threshold method (vtm) selected features >
    fs_df = pd.DataFrame(columns=['selected_features', 'feature_importance'])
    
    # Instantiate PCA >
    pca = PCA(n_components=n_components)
    
    # Standardized data >
    # scaler = StandardScaler()
    # df_standardized = scaler.fit_transform(df)
    
    # Fit the PCA model >
    pca.fit(df)  #(df_standardized)
    
    # Extract feature importance scores >
    feature_scores = pca.explained_variance_ratio_
    
    # Normalize reconstruction errors using Normalizer
    # normalizer = Normalizer()
    # normalized_scores = normalizer.fit_transform(feature_scores)
    feature_scores = feature_scores.flatten()
    feature_scores = (feature_scores - np.min(feature_scores)) / (np.max(feature_scores) - np.min(feature_scores))
    
    # Sort features based on importance scores >
    # sorted_features = sorted(zip(df.columns, normalized_scores), key=lambda x: x[1], reverse=True)
    sorted_features = sorted(zip(df.columns, feature_scores), key=lambda x: x[1], reverse=True)
    
    # Normalize the reconstruction errors to get scores that represent the importance of each feature. 
    selected_features = [feat for feat, _ in sorted_features]
    feature_scores = [score for _, score in sorted_features]
    
    # Create DataFrame to store selected features and scores
    fs_df['selected_features'] = selected_features
    fs_df['feature_importance'] = feature_scores
    
    # fs_df = pd.DataFrame(sorted_features, columns=['selected_features', 'feature_importance'])
    
    return pca, fs_df


# %%
'''
Optimse the number of principal components for PCA using GridSearchCV
'''
def optimise_pca_hyperparameters(df: pd.DataFrame, n_components_range: Tuple[int, int]) -> int:
    """
    Optimize the number of principal components for PCA using GridSearchCV.

    Args:
        df (DataFrame): The input DataFrame for training the PCA model.
        n_components_range (tuple): Range of the number of principal components to explore.

    Returns:
        int: The best number of principal components found by GridSearchCV.
    """
    # param_grid = {'pca__n_components': list(range(n_components_range[0], n_components_range[1]+1))}
    param_grid = {'n_components': list(range(n_components_range[0], n_components_range[1]+1))}
    
    # Constructing the pipeline
    pca = PCA()
    
    # n_samples = df.shape[0]
    # kmeans = KMeans(n_clusters=min(30, n_samples//2), random_state=42)
    # pipeline = Pipeline([('pca', pca), ('kmeans', kmeans)])
    
    # # Define a function to compute silhouette score
    # def silhouette_scorer(estimator, X):
    #     labels = estimator.predict(X)
    #     return silhouette_score(X, labels)
                    
    # grid_search = GridSearchCV(estimator=pipeline, 
    #                            param_grid=param_grid, 
    #                            cv=None, 
    #                            scoring=silhouette_scorer,  # 'neg_mean_squared_error',
    #                            n_jobs=n_jobs,
    #                            verbose=3)
    
    grid_search = GridSearchCV(estimator=pca, 
                               param_grid=param_grid, 
                               cv=None, 
                               scoring='neg_mean_squared_error',
                               n_jobs=n_jobs,
                               verbose=3)
    grid_search.fit(df)
    
    best_n_components = grid_search.best_params_['n_components']
    # best_n_components = grid_search.best_params_['pca__n_components']
    
    return best_n_components

# %%
def main():
    # load the data
    df = load_data()
    # Optimise the number of compnents for PCA
    n_components_range = (df.shape[1]-1, df.shape[1])  #min(len(df.columns), 10))
    best_n_components = optimise_pca_hyperparameters(df, n_components_range)

    print(f"Best number of principal components: {best_n_components}")

    # Select features using PCA
    print("Selecting features using PCA")
    _, fs_df = feature_selection_pca(df, best_n_components)

    # Free up memory
    del df

    # %%
    print(fs_df)
    print('=='*20)
    # Save the selected features to the models directory >
    if SAVE_SELECT_FEATURES:
        print("Saving selected features")
        fs_df.to_csv(selected_features_dir / 'selected_features_pca.csv', index=False)
        print("Selected Features saved to:", selected_features_dir / 'selected_features_pca.csv')
        
        
if __name__ == "__main__":
    main()
    print("Completed Successfully!")