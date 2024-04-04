#########################
## IMPORT DEPENDENCIES >>
#########################
import pandas as pd
from typing import Optional


#########################
## TODO >>
#########################
# - [ ] Move all feature selection functions to this file


"""
Given a dictionary with feature selectors and their selected features as dfs: combine in one df
"""
def merge_feature_importance(feature_selectors: dict, 
                             all_feature_names: Optional[list],
                             save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Merge dataframes containing feature selection results based on selected feature names.
    
    Parameters:
    - feature_selectors (dict): Dictionary containing feature selector names as keys and dataframes 
                                of selected features and their importance as values.
    - all_feature_names (list): List of all feature names before selection.
    - save_path (str, optional): Path to save the merged DataFrame as a CSV file. Default is None.
    
    Returns:
    - Merged DataFrame with feature names and importance for each feature selector.
    
    Example usage:
        # Assuming feature_selectors is a dictionary containing dataframes for each feature selector
        # all_feature_names is the list of all feature names
        merged_df = merge_feature_importance(feature_selectors, all_feature_names)
    """
    # Initialize an empty DataFrame to store merged results
    merged_df = pd.DataFrame(columns=['selected_features'])
    
    # Merge dataframes for each feature selector
    for selector_name, selector_df in feature_selectors.items():
        if selector_df is None:
            continue
        # Rename the feature importance column to include the feature selector name
        selector_df = selector_df.rename(columns={'feature_importance': f'feature_importance_{selector_name}'})
       
        # Merge with previously merged DataFrame or initialize if it's the first iteration
        if merged_df.empty:
            merged_df = selector_df
        else:
            merged_df = pd.merge(merged_df, selector_df, how='outer', on='selected_features')    

    # Add missing features with importance value of 0
    if all_feature_names is not None:
        missing_features = set(all_feature_names) - set(merged_df['selected_features'])
        missing_data = {f'feature_importance_{selector_name}': 0 for selector_name in feature_selectors}
        missing_data['selected_features'] = list(missing_features)
        missing_df = pd.DataFrame(missing_data)
        
        # Concatenate merged dataframe and missing dataframe
        merged_df = pd.concat([merged_df, missing_df], ignore_index=True)
    
    # Fill NaNs with 0
    merged_df = merged_df.fillna(0)
    
    # Find the mean and variance of feature importance for each feature
    merged_df['mean'] = merged_df.drop('selected_features', axis=1).filter(regex='^feature_importance_').mean(axis=1)
    merged_df['variance'] = merged_df.drop('selected_features', axis=1).filter(regex='^feature_importance_').var(axis=1)

    # Sort by mean in descending order
    all_features_scores = merged_df.sort_values(by='mean', ascending=False)

    # Save
    if save_path is not None:
        all_features_scores.to_csv(save_path, index=False)
    
    return all_features_scores