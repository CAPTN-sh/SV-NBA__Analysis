
###################
## DEPENDENCIES >>>
###################
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

# Hyperopt >
import optuna
import warnings
warnings.filterwarnings("ignore")

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


############################
## FLAGS & GLOBAL VALUES >>>
############################

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
    GRIDSEARCH = True  # Optimise using GridSearchCV
    RANDOMSEARCH = False  # Optimise using RandomizedSearchCV

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
    n_jobs = int(0.4 * n_jobs)
else:
    n_jobs = int(0.7 * n_jobs)
print("Number of CPUs to use:", n_jobs)

############
## PATHS >>>
############
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

##################
## LOAD DATASET >>
##################
def load_data():
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
    # LOAD VALIDATION SET >>
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

    # %%
    # Concatenate the datasets >
    asset_df = validate_df  # pd.concat([train_df, validate_df, test_df], axis=0)

    # %%
    # COLUMNS NOT TO INCLUDE IN FEATURE SELECTION
    cols_not_to_study = ['epoch', 'datetime', 'obj_id', 'traj_id', 'stopped', 'curv']

    # Check that the column in cols_not_to_study are in the dataset, otherwise remove them from the list >
    cols_not_to_study = [col for col in cols_not_to_study if col in asset_df.columns]

    print(f"Cols not to study: {cols_not_to_study}")
    # Create a copy of the dataset and drop the columns not to study >
    df = asset_df.drop(columns=cols_not_to_study)
    
    return df

# %%
###############################
## VARIANCE THRESHOLD METHOD >>
###############################

## DEFINE A WRAPPER FOR THE VARIANCE THRESHOLD METHOD >>
def variance_threshold_feature_selection(data: pd.DataFrame, threshold: float) -> Tuple[VarianceThreshold, pd.DataFrame]:
    """
    Perform feature selection using variance threshold.
    Assign the feature_importance based on the normalised variance of the features. 
    The lower the variance, the less important the feature.
    

    Args:
        data (pd.DataFrame): The input DataFrame containing the features.
        threshold (float): The threshold value for variance.

    Returns:
        Union[callable, pd.DataFrame]:
            [callable]: is the fitted VarianceThreshold object.
            [pd.DataFrame]: is the selected features in descending order.
                            The DataFrame contains two columns:
                                - `selected_features`: The selected features.
                                - `feature_importance`: The corresponding feature importance values.
    """
    # Instantiate a place holder for the variance threshold method (vtm) selected features >
    fs_df = pd.DataFrame(columns=['selected_features', 'feature_importance'])

    selector = VarianceThreshold(threshold=threshold)
    selector.fit(data)
    selected_features = data.columns[selector.get_support()]
    feature_importance = selector.variances_
    # L2 normalisation
    # feature_importance = Normalizer().fit_transform(feature_importance.reshape(-1,1))
    feature_importance = feature_importance.flatten()
    feature_importance = (feature_importance - np.min(feature_importance)) / (np.max(feature_importance) - np.min(feature_importance))
    
    # put the data in fs_df >
    fs_df['selected_features'] = selected_features
    fs_df['feature_importance'] = feature_importance
    
    return selector, fs_df

# %%
## Define the Optuna objective function for the optimisation of ``threshold`` hyperparameter >>
def objective(trial, 
                      df: pd.DataFrame, 
                      cluster: Callable,
                      n_clusters: int, 
                      random_state: Optional[int]=42, 
                      score_metric: Optional[Callable]=silhouette_score,
                      steps: Optional[float]=0.1):
            """Optimization objective function for feature selection.

            This function takes a trial object, a DataFrame, and optional parameters for the number of clusters and random state.
            It performs feature selection using the VarianceThreshold method and trains a clustering model (e.g., KMeans) on the selected features.
            The silhouette score is then calculated and returned as the optimization objective.

            Args:
                trial (optuna.Trial): The trial object used for optimization.
                df (pd.DataFrame): The input DataFrame containing the features.
                cluster (Callable): The clustering algorithm to be used.
                n_clusters (int): The number of clusters for the clustering algorithm.
                random_state (int, optional): The random state for reproducibility. Defaults to 42.
                score_metric (Callable, optional): The scoring metric used to evaluate the clustering model. Defaults to sklearn.metrics.silhouette_score.
                steps (float, optional): The step size for the threshold search space. Defaults to 0.2.

            Returns:
                float: The silhouette score of the clustering model trained on the selected features.
            """
            # Print the current trial number
            print("Running Trial Number:", trial.number)
            
            # Define the search space for the threshold
            threshold = trial.suggest_discrete_uniform(name='threshold', low=0, high=1, q=steps)  # Limit to 5 values between 0 and 1
            
            # Instantiate the VarianceThreshold object with the suggested threshold
            selector, _ = variance_threshold_feature_selection(df, threshold)
            
            # Apply the selector to the data
            x_selected = selector.transform(df)
            
            # Train a clustering model (e.g., KMeans) on the selected features
            clusterer = cluster(n_clusters=n_clusters, random_state=random_state)
            clusters = clusterer.fit_predict(x_selected)
            
            # Calculate silhouette score
            silhouette = score_metric(x_selected, clusters)
            return silhouette  # Use Optuna for hyperparameter optimisation
        
# %%
def main():
    # load the data >
    df = load_data()


    ## DEFINE THE HYPERPARAMETER OPTIMISATION >>
    best_threshold = None
    if HYPEROPT:
            # Optimise >
            params = {'cluster': KMeans,
                    'n_clusters': 30,
                    'random_state': 42,
                    'metric': silhouette_score,
                    'n_iter': 100,
                    'step': 0.1,
                    'n_jobs': n_jobs}
            if OPTUNA:
                    study_params = {'direction': 'maximize'}

                    # Create a study object and optimize the objective function >
                    study = optuna.create_study(direction=study_params['direction'])

                    # Use the validation set only for optimisation >
                    study.optimize(partial(objective,
                                        df=df,
                                        cluster=params['cluster'],
                                        n_clusters=params['n_clusters'],
                                        random_state=params['random_state'],
                                        score_metric=params['metric'],
                                        steps=params['step']), 
                            n_trials=params['n_iter'],
                            n_jobs=params['n_jobs'])
                    # study.optimize(lambda trial: objective(trial, 
                    #                                        df=df, 
                    #                                        n_clusters=study_params.n_clusters, 
                    #                                        random_state=split_seed), 
                    #                n_trials=study_params.n_trials,
                    #                n_jobs=study_params.n_jobs)
                    # study.optimize(objective, n_trials=study_params.n_trials)

                    # Get the best threshold
                    best_threshold = study.best_params['threshold']
                    print("Best Threshold:", best_threshold)

                    # Free up memory >
                    del study

            if GRIDSEARCH:
                    # Define the parameter grid for RandomizedSearchCV
                    # param_grid = {'vt__threshold': np.arange(0, 1, params['step'])}
                    param_grid = {'threshold': np.arange(0, 1, params['step'])}
                    vtm = VarianceThreshold()

                    # # Initialize the pipeline with VarianceThreshold and KMeans clustering
                    # clusterer = params['cluster']
                    # pipeline = Pipeline([('vt', vtm),
                    #                     ('kmeans', clusterer(n_clusters=params['n_clusters'], 
                    #                                         random_state=params['random_state']))
                    #                     ])

                    # # Define a function to compute silhouette score
                    # def silhouette_scorer(estimator, X):
                    #         labels = estimator.predict(X)
                    #         return silhouette_score(X, labels)

                    # # Initialize RandomizedSearchCV
                    # grid_search = GridSearchCV(estimator=pipeline,
                    #                         param_grid=param_grid,
                    #                         scoring=silhouette_scorer,
                    #                         cv=None,
                    #                         n_jobs=params['n_jobs'],
                    #                         verbose=3)
                    
                    # Initialize RandomizedSearchCV
                    grid_search = GridSearchCV(estimator=vtm,
                                            param_grid=param_grid,
                                            scoring='neg_mean_squared_error',
                                            cv=None,
                                            n_jobs=params['n_jobs'],
                                            verbose=3)

                    # Fit RandomizedSearchCV
                    grid_search.fit(df)

                    # Print the best parameters and best score
                    # best_threshold = grid_search.best_params_['vt__threshold']
                    best_threshold = grid_search.best_params_['threshold']
                    
                    print("Best threshold:", grid_search.best_params_['threshold'])
                    print("Best silhouette score:", grid_search.best_score_)

                    # Free up memory >
                    del grid_search
                    
            if RANDOMSEARCH:
                    # Define the parameter grid for RandomizedSearchCV
                    param_grid = {'vt__threshold': np.arange(0, 1, params['step'])}

                    # Initialize the pipeline with VarianceThreshold and KMeans clustering
                    clusterer = params['cluster']
                    pipeline = Pipeline([('vt', VarianceThreshold()),
                                        ('kmeans', clusterer(n_clusters=params['n_clusters'], 
                                                            random_state=params['random_state']))
                                        ])

                    # Define a function to compute silhouette score
                    def silhouette_scorer(estimator, X):
                            labels = estimator.predict(X)
                            return silhouette_score(X, labels)

                    # Initialize RandomizedSearchCV
                    random_search = RandomizedSearchCV(estimator=pipeline,
                                                    param_distributions=param_grid,
                                                    scoring=silhouette_scorer,
                                                    n_iter=params['n_iter'],
                                                    cv=None,
                                                    random_state=params['random_state'],
                                                    n_jobs=params['n_jobs'],
                                                    verbose=1)

                    # Fit RandomizedSearchCV
                    random_search.fit(df)

                    # Print the best parameters and best score
                    best_threshold = random_search.best_params_['threshold']
                    
                    print("Best threshold:", random_search.best_params_['threshold'])
                    print("Best silhouette score:", random_search.best_score_)

                    # Free up memory >
                    del randomized_search

    # If HYPEROPT, then use the optimised threshold, otherwise use the default threshold >
    threshold = None

    if HYPEROPT:
        threshold = best_threshold
    else:
        threshold = 0.1
        
    # Selecte features and return scores >
    selector, fs_df = variance_threshold_feature_selection(df, threshold)
    # features_selected = selector.transform(df)
    print("Selection completed >>")
    print(fs_df)
    print("=="*20)
    # Free up memory
    del df

    # %%
    ## SAVE >>
    if SAVE_SELECT_FEATURES:
        fs_df.to_csv(selected_features_dir / 'selected_features_vtm.csv', index=False)
        print("Saving the selected features >>")
        print("Selected Features saved to:", selected_features_dir / 'selected_features_vtm.csv')


if __name__ == "__main__":
    main()
    print("Completed Successfully!")