#########################
## IMPORT DEPENDENCIES >>
#########################
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Callable, Union

import joblib

# Add a the root directory
root_dir = Path.cwd().resolve().parent
if root_dir.exists():
    sys.path.append(str(root_dir))
else:
    raise FileNotFoundError('Root directory not found.')

# Numpy and Pandas
import numpy as np
import pandas as pd
from pandas import DataFrame

# Cluster and sklearn
import hdbscan
import sklearn as sk
from sklearn.utils import shuffle
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# torch
import torch
import shap

# Plot
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
small_plt_type = ['science', 'grid', 'notebook']
big_plt_type = ['science', 'grid', 'notebook', 'ieee']
plt.style.use(big_plt_type)


#############
## Globals >>
#############





def sample_features_and_indices(features, indices, sample_size, random_state=42):
    """
    Samples a subset of features and indices based on the specified sample size.
    If the number of features is greater than the sample size, it randomly
    samples 'sample_size' features and corresponding indices.
    Otherwise, it returns the original features and indices.

    Parameters:
    - features: Array-like or list of features.
    - indices: Array-like or list of indices corresponding to the features.
    - sample_size: The size of the sample to return.
    - random_state: An integer seed for reproducibility of the random shuffle.

    Returns:
    - sampled_features: A subset of the original features.
    - sampled_indices: A subset of the original indices corresponding to the sampled features.
    """
    if len(features) > sample_size:
        shuffled_features, shuffled_indices = shuffle(features, indices, random_state=random_state)
        sampled_features = shuffled_features[:sample_size]
        sampled_indices = shuffled_indices[:sample_size]
    else:
        sampled_features = features
        sampled_indices = indices

    return sampled_features, sampled_indices


def cluster(df: pd.DataFrame,
            n_clusters: int = 5, 
            cluster_method: str = 'kmeans', 
            distance_metric: str = 'euclidean',
            n_jobs: Optional[int] = None,
            append_to_df: bool = False,
            save_model_path: str = './',
            save_model_name: Optional[str] = None,
            verbose: Optional[int] = 0) -> Tuple[pd.DataFrame, dict, Callable]:
    """
    Trains a cluster model (K-Means, HDBScan, or GMM) on the features, calculates the silhouette score,
    the Calinski-Harabasz score, and the Davies-Bouldin score. 
    For HDBSCAN, also calculates the noise point ratio.
    
    Parameters:
    - df: DataFrame containing the features for clustering.
    - n_clusters: Number of clusters.
    - cluster_method: Clustering method ('kmeans' or 'gmm').
    - model_save_path: Path to save the trained model.
    
    Returns:
    - cluster_labels: Labels of clusters for each data point.
    - model: Trained model.
    """
    # Create the cluster object
    model = None
    if cluster_method == 'kmeans':
        batch_size = 1024 if n_jobs is None else 1024*n_jobs  # set the batch_size greater than 256 * number of cores to enable parallelism on all cores.
        model = MiniBatchKMeans(n_clusters=n_clusters,
                                max_iter= max(300, int(df.shape[0]/(10*n_clusters))),
                                batch_size=batch_size,
                                random_state=0, 
                                n_init='auto', 
                                verbose=verbose)
    elif cluster_method == 'gmm':
        model = GaussianMixture(n_components=n_clusters,
                                covariance_type='full',
                                max_iter=100,
                                n_init=10,
                                random_state=0)
    elif cluster_method == 'hdbscan':
        model = hdbscan.HDBSCAN(min_cluster_size=3*n_clusters, 
                                min_samples=None,
                                metric=distance_metric,
                                n_jobs=n_jobs,
                                store_centers='centroid',
                                )
    else:
        raise ValueError("Unsupported clustering method")

    ## Train >>
    if model is None:
        raise ValueError("Model is not defined")
    else:
        model.fit(df)
        
    ## Predict >>
    result_df = pd.DataFrame()
    if cluster_method == 'hdbscan':
        cluster_labels = model.fit_predict(df)
    else:
        cluster_labels = model.predict(df)
    if append_to_df:  # Append to original dataframe
        df[f'{cluster_method}_clusters'] = cluster_labels
        result_df = df
    else:  # Create a DataFrame with cluster labels indexed by the original DataFrame's index
        result_df = pd.DataFrame(cluster_labels, index=df.index, columns=[f'{cluster_method}_clusters'])
        
    ## Calculate the scores >>
    scores = {}
    # Silhouette score calculation
    if cluster_method != 'hdbscan':  # Silhouette score is not meaningful for hdbscan with variable cluster sizes
        scores['silhouette_score'] = silhouette_score(df, cluster_labels, metric=distance_metric)
        print("The average silhouette_score is :", scores['silhouette_score'])
        
        # Calinski-Harabasz score calculation
        scores['calinski_harabasz_score'] = calinski_harabasz_score(df, cluster_labels)
        print("The Calinski-Harabasz score is :", scores['calinski_harabasz_score'])
        
        # Davies-Bouldin score calculation
        scores['davies_bouldin_score'] = davies_bouldin_score(df, cluster_labels)
        print("The Davies-Bouldin score is :", scores['davies_bouldin_score'])
    else:
        noise_ratio = np.sum(cluster_labels == -1) / len(cluster_labels)
        scores['noise_ratio'] = noise_ratio
        print("Noise point ratio for HDBSCAN is:", noise_ratio)
        # Generate and plot the persistence diagram
        model.condensed_tree_.plot(select_clusters=True,
                                   selection_palette=sns.color_palette())

    ## Save the trained model >>
    print("Saving the trained model ", end="")
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    if save_model_name is None:
        save_model_name = cluster_method
    model_filename = os.path.join(save_model_path, f"{save_model_name}_model.joblib")
    print("to ", model_filename)
    joblib.dump(model, model_filename)
    print("completed with success!")
    print("=="*20)

    return result_df, scores, model



def plot_cluster(df: pd.DataFrame, 
                 cluster_column: str, 
                 title: Optional[str] = None,
                 save_path: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Visualize the clustering results using t-SNE for dimensionality reduction.
    
    Parameters:
    - features: DataFrame containing the features for clustering.
    - cluster_labels: Labels of clusters for each data point.
    """
    ## t-SNE for dimensionality reduction >>
    tsne = TSNE(n_components=2, random_state=0)
    features_2d = tsne.fit_transform(df.drop(cluster_column, axis=1))

    with plt.style.context(big_plt_type):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        sns.scatterplot(x=features_2d[:, 0], 
                        y=features_2d[:, 1], 
                        hue=df[cluster_column], 
                        palette='viridis',
                        ax=ax)
        if title:
            ax.title(title)
            
        # Set the color of ticks, tick labels, and legend text to black
        ax.tick_params(axis='x', colors='black')
        ax.tick_params(axis='y', colors='black')
        ax.xaxis.label.set_color('black')
        ax.yaxis.label.set_color('black')
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0., frameon=False)
        
        # Save the plot if save_path is provided
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', transparent=True)
        
        return fig, ax


def flatten_trajectory_data(all_features, flatten_mode='fine_grained_behavior'):
    """
    Transforms trajectory data into a format suitable for clustering based on the specified mode.
    
    Parameters:
    - all_features: A numpy array with shape (n_samples, n_timesteps, n_features).
    - flatten_mode: String specifying the mode of data flattening. Options are 'aggregate_behavior' or 'fine_grained_behavior'.
    
    Returns:
    - flattened_data: A 2D array whose shape depends on the chosen mode.
    """
    
    if flatten_mode == 'aggregate_behavior':
        # Aggregate features across all time steps
        flattened_data = all_features.reshape(all_features.shape[0], -1)  # Changes shape from (n_samples, n_timesteps, n_features) to (n_samples, n_timesteps*n_features)
    elif flatten_mode == 'fine_grained_behavior':
        # Maintain feature dimensions, but combine samples and time steps
        flattened_data = all_features.reshape(-1, all_features.shape[-1])  # Changes shape from (n_samples, n_timesteps, n_features) to (n_samples*n_timesteps, n_features)
    else:
        raise ValueError("Unsupported flatten_mode. Choose 'aggregate_behavior' or 'fine_grained_behavior'.")

    return flattened_data



class SHAPAnalysis:
    def __init__(self, model, input_data, output_data, input_features, output_features, device):
        self.model = model.to(device)
        self.input_data_reshaped = input_data.reshape(-1, len(input_features))
        self.output_data_reshaped = output_data.reshape(-1, output_features)
        self.input_features = input_features
        self.output_feature_columns = [f'Latent Feature {i+1}' for i in range(output_features)]
        self.device = device
        np.random.seed(42)  # Ensure reproducibility

    def encoder_model_function(self, x):
        x_tensor = torch.tensor(x, dtype=torch.float).to(self.device)
        self.model.eval()
        with torch.no_grad():
            # Transform input dimension from 20 to d_model=4 using the input_fc layer
            x_transformed = self.model.input_fc(x_tensor)
            # Then pass the transformed input to the transformer encoder
            encoded = self.model.transformer_encoder(x_transformed)
        return encoded.cpu().numpy()

    def prepare_background_data(self, sample_size=1000):
        self.background_data_input = shap.sample(self.input_data_reshaped, sample_size)
        # self.background_data_output = shap.sample(self.output_data_reshaped, sample_size)

    def initialize_explainers(self):
        self.input_explainer = shap.KernelExplainer(self.encoder_model_function, self.background_data_input)

    def calculate_shap_values(self, nsamples=100):
        self.input_shap_values = self.input_explainer.shap_values(self.background_data_input, nsamples=nsamples)

    def visualize_input_shap_values(self):
        # Visualize the SHAP values for the input features to understand how each input feature
        # contributes to the encoded representations in the model.
        shap.summary_plot(self.input_shap_values, features=self.background_data_input, feature_names=self.input_features, class_names=self.output_feature_columns)



def plot_bic_aic(features, n_clusters_range=range(1, 11), random_state=42):
    """
    Fit Gaussian Mixture Models with a range of cluster numbers and plot the BIC and AIC,
    with annotations for their minimum values.

    Parameters:
    - features: Original features or features from the autoencoder model.
    - n_clusters_range: A range of cluster numbers to evaluate.
    - random_state: Random state for reproducibility.
    """
    bics = []
    aics = []
    for n_clusters in n_clusters_range:
        gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=random_state)
        gmm.fit(features)
        bics.append(gmm.bic(features))
        aics.append(gmm.aic(features))

    plt.figure(figsize=(8, 4))
    plt.plot(n_clusters_range, bics, "bo-", label="BIC")
    plt.plot(n_clusters_range, aics, "go--", label="AIC")
    plt.xlabel('Number of clusters')
    plt.ylabel('Information Criterion')
    plt.title('BIC and AIC Scores by Number of Clusters')
    plt.legend()

    # Highlight the minimum values for BIC and AIC
    min_bic = np.min(bics)
    min_aic = np.min(aics)
    min_bic_k = n_clusters_range[bics.index(min_bic)]
    min_aic_k = n_clusters_range[aics.index(min_aic)]
    plt.annotate('Minimum BIC', xy=(min_bic_k, min_bic), xytext=(min_bic_k, min_bic + 10),
                 arrowprops=dict(facecolor='blue', shrink=0.05), horizontalalignment='center')
    plt.annotate('Minimum AIC', xy=(min_aic_k, min_aic), xytext=(min_aic_k, min_aic + 10),
                 arrowprops=dict(facecolor='green', shrink=0.05), horizontalalignment='center')
    plt.grid(True)
    plt.show()


"""
Use AIC and BIC to find the optimal number of clusters
"""
def plot_bic_aic_with_cv(features: pd.DataFrame,
                         algorithm: str,
                         n_clusters_range: list = range(1, 11),
                         n_init: int = 10,
                         random_state: int = None,
                         cv: Optional[int] = None,
                         save_path:Optional[str]=None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Fit clustering models with a range of cluster numbers and plot the BIC and AIC,
    with annotations for their minimum values.
    If cv is not None, the BIC and AIC are averaged over the number of folds.

    Parameters:
    - features: Original features or features from the autoencoder model.
    - algorithm: The clustering algorithm to use ('KMeans' or 'GMM').
    - n_clusters_range: A range of cluster numbers to evaluate.
    - n_init: Number of initializations for the clustering algorithm.
    - random_state: Random state for reproducibility.
    - cv: Number of folds for cross-validation. If None, no cross-validation is performed.
    - save_path: Path to save the plot.

    Returns:
    - fig: The matplotlib Figure object containing the plot.
    - ax: The matplotlib Axes object representing the plot.

    Example usage:
    features = pd.DataFrame(...)
    algorithm = 'KMeans'
    n_clusters_range = range(1, 11)
    n_init = 10
    random_state = 42
    cv = 5
    fig, ax = plot_bic_aic_with_cv_2(features, algorithm, n_clusters_range, n_init, random_state, cv)
    plt.show()
    """
    if random_state is None:
        random_state = np.random.RandomState()

    aics = []
    bics = []

    for n_clusters in n_clusters_range:
        if algorithm == 'KMeans':
            model = KMeans(n_clusters=n_clusters, 
                           n_init=n_init, 
                           random_state=random_state)
            
        elif algorithm == 'GMM':
            model = GaussianMixture(n_components=n_clusters, 
                                    covariance_type='full',
                                    random_state=random_state, 
                                    n_init=n_init)
        else:
            raise ValueError("Invalid algorithm. Choose between 'KMeans' and 'GMM'.")

        if cv is not None:
            scores_aic = -cross_val_score(model, features, cv=cv, scoring='aic')
            scores_bic = -cross_val_score(model, features, cv=cv, scoring='bic')
            aics.append(np.mean(scores_aic))
            bics.append(np.mean(scores_bic))
        else:
            model.fit(features)
            bics.append(model.bic(features))
            aics.append(model.aic(features))

    with plt.style.context(big_plt_type):
        fig, ax = plt.subplots(nrows=1, ncols=1)  #, figsize=(8, 4))

        if cv is not None:
            ax.plot(n_clusters_range, bics, "bo-", label="Average BIC")
            ax.plot(n_clusters_range, aics, "go--", label="Average AIC")
        else:
            ax.plot(n_clusters_range, bics, "bo-", label="BIC")
            ax.plot(n_clusters_range, aics, "go--", label="AIC")
        ax.set_xlabel('Number of clusters')
        ax.set_ylabel('Information Criterion')
        ax.set_title('BIC and AIC Scores by Number of Clusters')
        ax.legend()

        # Highlight the minimum values for BIC and AIC
        min_bic = np.min(bics)
        min_aic = np.min(aics)
        min_bic_k = n_clusters_range[bics.index(min_bic)]
        min_aic_k = n_clusters_range[aics.index(min_aic)]
        ax.annotate('Minimum BIC', xy=(min_bic_k, min_bic), xytext=(min_bic_k, min_bic + 10),
                    arrowprops=dict(facecolor='blue', shrink=0.05), horizontalalignment='center')
        ax.annotate('Minimum AIC', xy=(min_aic_k, min_aic), xytext=(min_aic_k, min_aic + 10),
                    arrowprops=dict(facecolor='green', shrink=0.05), horizontalalignment='center')
        
        if save_path is not None:
            save_dir = os.path.dirname(save_path)
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(save_path, transparent=True)  #, bbox_inches='tight')
            
        return fig, ax
    
    
"""
Find the optimal number of features for clustering using:
 - AIC and BIC scores given a range of selected features: for GMM
 - Silhouette, Davies-Bouldin, and Calinski-Harabasz scores given a range of selected features: for KMeans
"""    
def incremental_feature_clustering(data_df: pd.DataFrame,
                                   feature_selection_results: pd.DataFrame, 
                                   algorithm: str, 
                                   n_clusters: int = 1,
                                   n_init: int = 10,
                                   random_state: int = None,
                                   cv: int = None,
                                   distance_metric='euclidean',
                                   save_path: str = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Perform incremental clustering with selected features. Plot the resutls for finding the best elbow
    point representing the index of best featues in sequence. For scoring, the following scores are calculated:
     - AIC and BIC scores given a range of selected features: for GMM
     - Silhouette, Davies-Bouldin, and Calinski-Harabasz scores given a range of selected features: for KMeans

    Parameters:
    - feature_selection_results: DataFrame containing selected features and their importance scores.
    - algorithm: The clustering algorithm to use ('KMeans' or 'GMM').
    - n_clusters_range: A range of cluster numbers to evaluate.
    - n_init: Number of initializations for the clustering algorithm.
    - random_state: Random state for reproducibility.
    - cv: Number of folds for cross-validation. If None, no cross-validation is performed.
    - distance_metric: Distance metric used for silhoutte score. Defaults to 'euclidean',
    - save_path: Path to save the plot.

    Returns:
    - fig: The matplotlib Figure object containing the plot.
    - ax: The matplotlib Axes object representing the plot.
    """
    if random_state is None:
        random_state = np.random.RandomState()

    aics = []
    bics = []
    sils = []  # silhouette_score 
    dfs = []  # davies_bouldin_score  
    cms = []  # calinski_harabasz_score

    # all features columns
    all_features = data_df.columns

    ## Define a custom scoring function using make_scorer >>
    def calinski_harabasz_scorer(estimator, X, y=None):
        labels = estimator.fit_predict(X)
        return calinski_harabasz_score(X, labels)
    
    def silhouette_scorer(estimator, X, y=None, metric=distance_metric):
        labels = estimator.fit_predict(X)
        return silhouette_score(X, labels, metric=metric)

    def davies_bouldin_scorer(estimator, X, y=None):
        labels = estimator.fit_predict(X)
        return davies_bouldin_score(X, labels)
    
    # Iterate through selected features incrementally
    for i in range(1, len(feature_selection_results) + 1):
        selected_features = feature_selection_results['selected_features'][:i]

        # drop the columns not included in the selected features
        selected_features = selected_features[selected_features.isin(all_features)]
        
        # if after the drop, the selected features are empty, skip the iteration
        if selected_features.empty:
            continue
        
        # from data_df: drop the columns not included in the selected features
        df = data_df[selected_features]

        # Perform clustering with selected features
        if algorithm == 'KMeans':
            model = KMeans(n_clusters=n_clusters, 
                           n_init=n_init, 
                           random_state=random_state)
        elif algorithm == 'GMM':
            model = GaussianMixture(n_components=n_clusters, 
                                    covariance_type='full',
                                    random_state=random_state, 
                                    n_init=n_init)
        else:
            raise ValueError("Invalid algorithm. Choose between 'KMeans' and 'GMM'.")

        if cv is not None:
            if algorithm == 'GMM':
                scores_aic = -cross_val_score(model, df, cv=cv, scoring='aic')
                scores_bic = -cross_val_score(model, df, cv=cv, scoring='bic')
                aics.append(np.mean(scores_aic))
                bics.append(np.mean(scores_bic))
            elif algorithm == 'KMeans':
                scores_sils = -cross_val_score(model, df, cv=cv, scoring=silhouette_scorer)
                scores_dfs = -cross_val_score(model, df, cv=cv, scoring=davies_bouldin_scorer)
                scores_cms = -cross_val_score(model, df, cv=cv, scoring=calinski_harabasz_scorer)
                sils.append(np.mean(scores_sils))
                dfs.append(np.mean(scores_dfs))
                cms.append(np.mean(scores_cms))
        else:
            model.fit(df)
            y_pred = model.predict(df)
            if algorithm == 'GMM':
                bics.append(model.bic(df))
                aics.append(model.aic(df))
            elif algorithm == 'KMeans':
                sils.append(silhouette_score(df, y_pred, metric=distance_metric))
                cms.append(calinski_harabasz_score(df, y_pred))
                dfs.append(davies_bouldin_score(df, y_pred))
                

    # Plot AIC and BIC scores
    with plt.style.context(big_plt_type):
        fig, ax = plt.subplots(nrows=1, ncols=1)

        if cv is not None:
            if algorithm == 'GMM':
                ax.plot(range(1, len(feature_selection_results) + 1), bics, "bo-", label="Average BIC")
                ax.plot(range(1, len(feature_selection_results) + 1), aics, "go--", label="Average AIC")
            elif algorithm == 'KMeans':
                ax.plot(range(1, len(feature_selection_results) + 1), sils, "ro-", label="Average Silhouette Score")
                ax.plot(range(1, len(feature_selection_results) + 1), dfs, "yo--", label="Average Davies-Bouldin Score")
                ax.plot(range(1, len(feature_selection_results) + 1), cms, "mo--", label="Average Calinski-Harabasz Score")
        else:
            if algorithm == 'GMM':
                ax.plot(range(1, len(feature_selection_results) + 1), bics, "bo-", label="BIC")
                ax.plot(range(1, len(feature_selection_results) + 1), aics, "go--", label="AIC")
            elif algorithm == 'KMeans':
                ax.plot(range(1, len(feature_selection_results) + 1), sils, "ro-", label="Silhouette Score")
                ax.plot(range(1, len(feature_selection_results) + 1), dfs, "yo--", label="Davies-Bouldin Score")
                ax.plot(range(1, len(feature_selection_results) + 1), cms, "mo--", label="Calinski-Harabasz Score")
        ax.set_xlabel('Number of selected features in order of importance')
        ax.set_ylabel('Information Criterion')
        ax.set_title('Scores by Best Selected Features')
        ax.legend()

        # Highlight the minimum values
        if algorithm == 'GMM':
            min_bic = np.min(bics)
            min_aic = np.min(aics)
            min_bic_k = np.argmin(bics) + 1
            min_aic_k = np.argmin(aics) + 1
            ax.annotate('Minimum BIC', xy=(min_bic_k, min_bic), xytext=(min_bic_k, min_bic + 10),
                        arrowprops=dict(facecolor='blue', shrink=0.05), horizontalalignment='center')
            ax.annotate('Minimum AIC', xy=(min_aic_k, min_aic), xytext=(min_aic_k, min_aic + 10),
                        arrowprops=dict(facecolor='green', shrink=0.05), horizontalalignment='center')
        elif algorithm == 'KMeans':
            min_sil = np.min(sils)
            min_df = np.min(dfs)
            min_cm = np.min(cms)
            min_sil_k = np.argmin(sils) + 1
            min_df_k = np.argmin(dfs) + 1
            min_cm_k = np.argmin(cms) + 1
            ax.annotate('Maximum Silhouette Score', xy=(min_sil_k, min_sil), xytext=(min_sil_k, min_sil + 0.1),
                        arrowprops=dict(facecolor='red', shrink=0.05), horizontalalignment='center')
            ax.annotate('Minimum Davies-Bouldin Score', xy=(min_df_k, min_df), xytext=(min_df_k, min_df + 0.1),
                        arrowprops=dict(facecolor='yellow', shrink=0.05), horizontalalignment='center')
            ax.annotate('Maximum Calinski-Harabasz Score', xy=(min_cm_k, min_cm), xytext=(min_cm_k, min_cm + 0.1),
                        arrowprops=dict(facecolor='magenta', shrink=0.05), horizontalalignment='center')

        if save_path is not None:
            save_dir = os.path.dirname(save_path)
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(save_path, transparent=True)

        return fig, ax


def describe_clusters(dataframe, group_columns, describe_features):
    """
    Group the data by specified columns and provide descriptive statistics for selected features.

    Parameters:
        dataframe (DataFrame): The original DataFrame with the data.
        group_columns (list of str): List of column names to group by.
        describe_features (list of str): List of features to describe in each group.

    Returns:
        DataFrame: A DataFrame containing the descriptive statistics for the selected features across different groups.
    """
    grouped = dataframe.groupby(group_columns)
    description = grouped[describe_features].describe().transpose()
    return description