import numpy as np
import pandas as pd
from pandas import DataFrame

from sklearn.preprocessing import MinMaxScaler


class CustomMinMaxScaler(MinMaxScaler):
    """
    CustomMinMaxScaler extends the functionality of the MinMaxScaler class from scikit-learn.
    It allows for custom minimum and maximum values to be specified during scaling.
    """

    def __init__(self, feature_range=(0, 1), min=None, max=None, copy=True):
        """
        Initialize the CustomMinMaxScaler object.

        Parameters:
        - feature_range (tuple): The desired range of the transformed data. Default is (0, 1).
        - min (float or array-like): The minimum value(s) to use for scaling. If None, the minimum value(s) will be computed from the data. Default is None.
        - max (float or array-like): The maximum value(s) to use for scaling. If None, the maximum value(s) will be computed from the data. Default is None.
        - copy (bool): Whether to create a copy of the input data. Default is True.
        """
        super().__init__(feature_range=feature_range, copy=copy)
        if isinstance(min, (int, float)):
            self.min = [min]
        else: 
            self.min = min
            
        if isinstance(max, (int, float)):
            self.max = [max]
        else:
            self.max = max

    def fit(self, X, y=None):
        """
        Compute the minimum and maximum values to use for scaling.

        Parameters:
        - X (array-like): The input data to be scaled.
        - y (array-like): Ignored. Present for compatibility.

        Raises:
        - ValueError: If the length of min is not equal to the number of columns in X, or if the length of max is not equal to the number of columns in X.
        """
        if self.min is None:
            self.data_min_ = np.min(X, axis=0)
        else:
            if len(self.min) == 1:
                self.data_min_ = np.ones((X.shape[1],)) * self.min
            elif len(self.min) != X.shape[1]:
                raise ValueError("Length of min should be equal to the number of columns in X")
            else:
                raise Exception("Unknown error by min")
        
        if self.max is None:
            self.data_max_ = np.max(X, axis=0)
        else:
            if len(self.max) == 1:
                self.data_max_ = np.ones((X.shape[1],)) * self.max
            elif len(self.max) != X.shape[1]:
                raise ValueError("Length of max should be equal to the number of columns in X")
            else:
                raise Exception("Unknown error by max")
        return self
    
    def transform(self, X):
        """
        Scale the input data.

        Parameters:
        - X (array-like): The input data to be scaled.

        Returns:
        - X_scaled (array-like): The scaled data.
        """
        X_scaled = (X - self.data_min_) / (self.data_max_ - self.data_min_)
        return X_scaled

    
    
def reduce_resolution(df: DataFrame, 
                      date_time_col: str = 'datetime', 
                      ship_id_col: str = 'obj_id', 
                      trip_id_col: str = 'traj_id', 
                      resolution_minutes: int = 1) -> DataFrame:
    """
    Reduce the resolution of a DataFrame containing time-series data.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame containing time-series data.
    - date_time_col (str): The name of the column in the DataFrame that represents the date and time. Defaults to 'datetime'.
    - ship_id_col (str): The name of the column in the DataFrame that represents the ship ID. Defaults to 'obj_id'.
    - trip_id_col (str,): The name of the column in the DataFrame that represents the trip ID. Defaults to 'traj_id'.
    - resolution_minutes (int): The desired resolution in minutes for reducing the data.
    
    Returns:
    - pd.DataFrame: The reduced resolution DataFrame.
    """
    
    df_columns = df.columns
    if date_time_col not in df_columns:
        raise ValueError(f"{date_time_col} column does not exist in the DataFrame.")
    if ship_id_col not in df_columns:
        raise ValueError(f"{ship_id_col} column does not exist in the DataFrame.")
    if trip_id_col not in df_columns:
        raise ValueError(f"{trip_id_col} column does not exist in the DataFrame.")
    
    if not isinstance(resolution_minutes, int):
        raise TypeError(f"Resolution must be an integer. Not {type(resolution_minutes)}")
    
    if resolution_minutes < 1:
        raise ValueError("Resolution must be greater than or equal to 1 minute.")
    
    # Group by day
    df_grouped_day = df.groupby(pd.Grouper(key=date_time_col, freq='D'), group_keys=False)
    
    # Initialize an empty list to store the reduced resolution data
    reduced_df = pd.DataFrame()
    
    # Iterate over each day group
    for day, day_group in df_grouped_day:
        # Group unique ships in each day
        day_grouped_ship = day_group.groupby(ship_id_col, group_keys=False)
        
        # Iterate over each ship group in a day
        for ship, ship_group in day_grouped_ship:
            # Group each trip for the ship
            trip_grouped = ship_group.groupby(trip_id_col, group_keys=False)
            
            # Iterate over each trip group
            for trip, trip_group in trip_grouped:
                # Check if the trip group has more than 3 measurements
                if len(trip_group) > 3:            
                    # Reduce the measurement resolution to one measurement per new minute
                    reduced_group = trip_group.resample(f'{resolution_minutes}min', on='datetime').first()
                    
                    # Concat the reduced group to the list
                    reduced_df = pd.concat([reduced_df, reduced_group])
    
    # Concatenate the reduced groups into a single dataframe
    
    return reduced_df
