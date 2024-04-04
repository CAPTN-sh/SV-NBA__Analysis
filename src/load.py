import os
from pathlib import Path
from glob import glob
from typing import List, Union, Dict, Optional
from io import StringIO
from typing import List, Dict, Union

import numpy as np
import pandas as pd
from pandas import concat, DataFrame, read_json, read_pickle, to_datetime
import dask.dataframe as dd
from movingpandas import Trajectory, TrajectoryCollection

from sklearn.model_selection import train_test_split

import json
import pickle

import multiprocessing

from torch.utils.data import Dataset



COLUMNS_DTYPES = {
    "accuracy": bool,               # check bool type
    "course": np.float16,
    "draught": np.float64,
    "epoch": np.float64,            # double precision to avoid rounding errors from datetime.timestamp()
    "heading": np.float16,
    "imo": np.int32,
    "lat": np.float64,
    "length": np.float16,
    "lon": np.float64,
    "maneuver": np.int8,
    "mmsi": np.int32,
    "msg_type": np.int8,
    "name": object,
    "ship_type": np.int8,
    "speed": np.float16,
    "status": np.int8,
    "to_bow": np.float16,
    "to_stern": np.float16,
    "to_port": np.float16,     
    "to_starboard": np.float16,
    "tonnage": np.float32,
    "turn": np.float16,
    "type_str": object,
    "volume": np.float32,
    "width": np.float16,
}
# orientation for json file dumps
ORIENT = 'records'

### IO for ais data files ###
def ls_files(src: str) -> List[str]:
    """Mimics $ ls /path/to/src/* functionality."""

    fullpaths=os.path.join(src, "*")
    ls=list(glob(fullpaths))

    return sorted(ls)

def ls_files_by_pattern(src: str, 
                        pattern: str) -> List[str]:
    """Mimics $ ls /path/to/src/*[pattern] functionality."""

    fullpaths=os.path.join(src, "*"+pattern)
    ls=list(glob(fullpaths))

    return sorted(ls)

def load_file_data(file: str) -> List[str] | None:
    """Load lines from file to string buffer."""

    with open(file, encoding='utf-8') as f:
        try:
            lines = f.readlines()
        except UnicodeDecodeError as err:
            print(f"UnicodeDecodeError when reading file: {f}")
            return None
        
    return lines

### IO for trajectories ###
# json dump
def dumps_trajectory(t):

    # get meta
    obj_id = t.obj_id
    traj_id = t.id

    # prepare output dataframe
    if not 'epoch' in t.df.columns:
        raise NotImplemented

    # use epoch float value as index keys
    t_out = t.df.set_index(keys='epoch',
                           drop=False,
                           inplace=False)
    
    # extract shapely.Point(x,y) geometry -> stick to pure python data formats
    t_out['lon'] = t_out['geometry'].x
    t_out['lat'] = t_out['geometry'].y

    t_out = DataFrame(t_out.drop(columns='geometry', inplace=False))

    # display(type(t_out))
    # display(t_out)

    jstr = t_out.to_json(orient=ORIENT,
                         date_format='epoch')
    
    return obj_id, traj_id, jstr
        
def dumps_trajectoryCollection(tc: TrajectoryCollection,
                              key_col: str = 'epoch',
                              x_col: str = 'lon',
                              y_col: str = 'lat',
                              crs: str = 'epsg:4326'):
        
    tc_data = {
        'key_col': key_col, 
        'x_col': x_col,
        'y_col': y_col,
        'crs': crs,
        'trajectories': [
            {'obj_id':  str(d[0]),
             'traj_id': d[1], 
             'data':    d[2]} 
            for d in [dumps_trajectory(t) for t in tc.trajectories]
            ]
    }

    return json.dumps(tc_data)

def dump_multiple_trajectoryCollection(tcs: List[TrajectoryCollection],
                                       file: str
                                       ):
    
    lines = [dumps_trajectoryCollection(tc)+'\n' for tc in tcs]

    with open(file=file, mode="w+") as f:
        f.writelines(lines)

# pickle file
def pickle_multiple_trajectoryCollection(tcs: List[TrajectoryCollection],
                                       file: str
                                       ):

    data = [get_df_trajectoryCollection(tc) for tc in tcs]

    df_tcs = concat(data)

    df_tcs.to_pickle(file)

    return 0

def get_df_trajectoryCollection(tc: TrajectoryCollection,
                              key_col: str = 'epoch',
                              x_col: str = 'lon',
                              y_col: str = 'lat',
                              crs: str = 'epsg:4326'):
    
    data = [get_df_trajectory(t) for t in tc.trajectories ]

    if len(data)<1:
        return None
    
    else:
        # add traj id for reconstruction
        for i, df in enumerate(data):
            df['traj_id'] = i
            df['traj_id'] = df['traj_id'].astype(object)

        df_tc = concat(data)

        return df_tc

def get_df_trajectory(t):

    # get meta
    obj_id = t.obj_id
    traj_id = t.id

    # prepare output dataframe
    if not 'epoch' in t.df.columns:
        raise NotImplemented

    # use epoch float value as index keys
    df_traj = t.df.set_index(keys='epoch',
                           drop=False,
                           inplace=False)
    
    # extract shapely.Point(x,y) geometry -> stick to pure python data formats
    df_traj['lon'] = df_traj['geometry'].x
    df_traj['lat'] = df_traj['geometry'].y

    df_traj['obj_id'] = obj_id
    df_traj['obj_id'] = df_traj['obj_id'].astype(object)

    df_traj = DataFrame(df_traj.drop(columns='geometry', inplace=False))

    return df_traj

# load json
def load_multiple_trajectoryCollection(file: str):

    with open(file) as f:
        lines = f.readlines()

    return [loads_trajectoryCollection(l) for l in lines]

def loads_trajectoryCollection(jstr: str):

    data = json.loads(jstr)

    key_col = data.get('key_col')
    x_col = data.get('x_col')
    y_col = data.get('y_col')
    crs = data.get('crs')

    trajs_str = data.get('trajectories')

    data = [loads_trajectory(t=t, 
                             key_col=key_col,
                             x_col=x_col, 
                             y_col=y_col, 
                             crs=crs) for t in trajs_str]
    
    return TrajectoryCollection(data)

def loads_trajectory(t,
                     key_col,
                     x_col,
                     y_col,
                     crs):
 
    obj_id = t.get('obj_id')
    traj_id = t.get('traj_id')
    traj_data = t.get('data')

    # create proper DataFrame
    df = read_json(
        StringIO(traj_data), 
        orient=ORIENT,
        dtype=COLUMNS_DTYPES)
    
    df['datetime'] = to_datetime(df[key_col], unit='s')

    traj = Trajectory(df=df, 
                    traj_id=traj_id,
                    obj_id=obj_id,
                    x=x_col,
                    y=y_col,
                    t='datetime',
                    crs=crs
                    )
        
    return traj

# load pickle
def load_multiple_trajectoryCollection_pickle(file: str):
    """
    ADD:
        datetime column

    CHECK:
        dtypes
    """
    df = read_pickle(file).convert_dtypes()
    df['obj_id'] = df['obj_id'].astype(int)
    df['datetime'] = to_datetime(df['epoch'], unit='s')
    df.reset_index(drop=True, inplace=True)

    return df
# parallel data loading
def load_multiple_trajectoryCollection_parallel(files: List[str],
                                                num_cores : int = (multiprocessing.cpu_count() - 4)):
    
    loaded = []

    # prepare args for the proper loading task 
    params = [(f, ) for f in files]

    # setup process pool
    with multiprocessing.Pool(processes=num_cores) as pool:

        # execute the task workers
        for exec_result in pool.starmap(load_multiple_trajectoryCollection,
                                        params):

            # report the value to analyse post
            loaded.append(exec_result)

    return loaded

def load_multiple_trajectoryCollection_parallel_pickle(files: List[str],
                                                       num_cores : int = (multiprocessing.cpu_count() - 4)):
    
    loaded = []

    # prepare args for the proper loading task 
    params = [(f, ) for f in files]

    # setup process pool
    with multiprocessing.Pool(processes=num_cores) as pool:

        # execute the task workers
        for exec_result in pool.starmap(load_multiple_trajectoryCollection_pickle, params):

            # report the value to analyse post
            loaded.append(exec_result)

    return loaded

def load_datasets(train_path: str, validate_path: str, test_path: str) -> Dict[str, Dataset]:
    """
    Load datasets from pickle files.

    Args:
        train_path (str): Path to the train pickle file.
        validate_path (str): Path to the validate pickle file.
        test_path (str): Path to the test pickle file.

    Returns:
        dict: A dictionary containing the loaded datasets, with keys 'train', 'validate', and 'test'.

    Raises:
        FileNotFoundError: If any of the specified paths do not exist.
        
    Example:
        import dotsi
        
        # Define the paths to the pickle files
        train_pickle_path = tvt_data_import_assets_dir / 'train_dataset.pickle'
        validate_pickle_path = tvt_data_import_assets_dir / 'validate_dataset.pickle'
        test_pickle_path = tvt_data_import_assets_dir / 'test_dataset.pickle'

        datasets = load_datasets(train_pickle_path, validate_pickle_path, test_pickle_path)
        datasets = dotsi.Dict(datasets)

        train_df = datasets.train.data
        validate_df = datasets.validate.data
        test_df = datasets.test.data
                
        # Check that all works
        print(f'# Train sample: {len(train_df)}, {len(datasets.train)}')
        print(f'# Validate sample: {len(validate_df)}, {len(datasets.validate)}')
        print(f'# Test sample: {len(test_df)}, {len(datasets.test)}')

    """
    
    # Define the paths to the pickle files
    train_pickle_path = Path(train_path)
    validate_pickle_path = Path(validate_path)
    test_pickle_path = Path(test_path)
    
    # Put in one dict
    data_path = {'train': train_pickle_path, 'validate': validate_pickle_path, 'test': test_pickle_path}

    # Check if the paths exist, otherwise raise an exception
    for path in data_path.values():
        if path is not None and not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

    # Load the train Dataset object
    train_Dataset = None
    validate_Dataset = None
    test_Dataset = None

    for path in data_path:
        dataset = Dataset()
        if data_path[path] is not None:
            with open(data_path[path], 'rb') as f:
                dataset = pickle.load(f)

        if path == 'train':         
            train_Dataset = dataset
        elif path == 'validate':    
            validate_Dataset = dataset
        elif path == 'test':        
            test_Dataset = dataset

    return {'train': train_Dataset, 'validate': validate_Dataset, 'test': test_Dataset}


def load_df_to_dataset(data_path: Union[str, Path], 
                       chunk_size: Optional[int] = None, 
                       use_dask: Optional[bool] = False) -> Dataset:
    """
    Load a DataFrame from a given data path and return it as a Dataset object.
    This function can load data of type .csv, .json, .pkl, .parquet, .feather, .hdf, and .pickle.
    
    Args:
        data_path (str): The path to the data file.
        
    Returns:
        Dataset: The loaded dataset.
        
    Raises:
        ValueError: If data_path is not a non-empty string.
        FileNotFoundError: If the specified path does not exist.
    """
    # Validate the input type >
    if data_path is None:
        raise ValueError("data_path must be a non-empty string")
    elif isinstance(data_path, (str, Path)):
        target_path = Path(data_path)
    else:
        raise ValueError("data_path must be a string or pathlib path. Got: {type(data_path)}")
    
    # Validate that the path exists >
    target_path = Path(data_path) 
    if not target_path.exists():
        raise FileNotFoundError(f"Path '{target_path}' does not exist.")
    
    # Initialise >
    dataset = Dataset()
    
    # Load the dataset >
    if target_path.suffix == '.csv':
        if use_dask:
            dataset.data = dd.read_csv(target_path, blocksize=chunk_size)
        else:
            dataset.data = pd.read_csv(target_path, chunksize=chunk_size)
    elif target_path.suffix == '.json':
        if use_dask:
            dataset.data = dd.read_json(target_path, blocksize=chunk_size)
        else:
            dataset.data = pd.read_json(target_path, chunksize=chunk_size)
    elif target_path.suffix == '.pkl':
        if use_dask:
            raise ValueError("Pickle format is not supported by Dask.")
        else:
            if chunk_size:
                raise ValueError("Pandas does not support reading .pkl files in chunks.")
            else:
                dataset.data = pd.read_pickle(target_path)
    elif target_path.suffix == '.parquet':
        if use_dask:
            dataset.data = dd.read_parquet(target_path, engine='pyarrow', blocksize=chunk_size)
        else:
            if chunk_size:
                raise ValueError("Pandas does not support reading parquet files in chunks.")
            else:
                dataset.data = pd.read_parquet(target_path)
    elif target_path.suffix == '.feather':
        if use_dask:
            raise ValueError("Feather format is not supported by Dask.")
        else:
            if chunk_size:
                raise ValueError("Pandas does not support reading feather files in chunks.")
            else:
                dataset.data = pd.read_feather(target_path)
    elif target_path.suffix == '.hdf':
        if use_dask:
            dataset.data = dd.read_hdf(target_path, blocksize=chunk_size)
        else:
            dataset.data = pd.read_hdf(target_path, chunksize=chunk_size)
    elif target_path.suffix == '.pickle':
        if use_dask:
            raise ValueError("Pickle format is not supported by Dask.")
        else: 
            if chunk_size:
                raise ValueError("Pandas does not support reading .pickle files in chunks.")
            else:                    
                data = pd.DataFrame()
                with open(target_path, 'rb') as f:
                    data = pickle.load(f) 
                dataset.data = data
    else:
        raise ValueError(f"Unsupported file format: {target_path.suffix}")
    
    return dataset


"""
Create a stratified sample from a DataFrame based on the specified columns.
"""
def stratified_sample_df(df: pd.DataFrame, 
                         sample_size: int, 
                         stratify_cols: list, 
                         random_seed: int = 42) -> pd.DataFrame:
    """
    Generates a stratified sample from a DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame to sample from.
    - sample_size (int): The desired sample size.
    - stratify_cols (list): List of column names to consider for stratification.
    - random_seed (int): The random seed for reproducibility. Default is 42.

    Returns:
    - A stratified sample DataFrame.

    Example usage:
        df = pd.DataFrame({
            'season': ['spring', 'summer', 'fall', 'winter'] * 3,
            'part_of_day': ['morning', 'afternoon', 'evening'] * 4,
            'value': range(12),
            'loc': [1, 1, 1, 1, 3, 2, 3, 2, 3, 2, 4, 4]
        })

        sample_size = 12  # Adjust this value to meet the minimum requirement or your specific needs
        stratify_cols = ['season', 'loc']
        random_seed = 42

        try:
            sample_df = stratified_sample_df(df, sample_size, stratify_cols, random_seed)
            print(sample_df)
        except ValueError as e:
            print(e)
    """
    # Create a deep copy of the input DataFrame
    df = df.copy(deep=True)
    
    # Calculate the number of unique combinations
    n_unique_combinations = df[stratify_cols].drop_duplicates().shape[0]

    if sample_size < n_unique_combinations:
        raise ValueError(f"sample_size = {sample_size} is less than the number of unique combinations = {n_unique_combinations}. Increase sample_size.")

    # Create a key for stratification
    df['stratify_key'] = df[stratify_cols].astype(str).agg('-'.join, axis=1)

    # Calculate stratify sample sizes
    stratify_sample_frac = min(sample_size / len(df), 1)
    
    # Perform stratified sampling
    stratified_df = df.groupby('stratify_key', group_keys=False).apply(lambda x: x.sample(frac=stratify_sample_frac, random_state=random_seed))

    # Optionally, you could make sure the result is exactly `sample_size` by randomly dropping or adding samples
    if len(stratified_df) > sample_size:
        stratified_df = stratified_df.sample(n=sample_size, random_state=random_seed)
    elif len(stratified_df) < sample_size:
        # Handle case where the resulting sample is less than desired due to rounding in `sample`
        additional_samples = df.sample(n=sample_size - len(stratified_df), random_state=random_seed)
        stratified_df = pd.concat([stratified_df, additional_samples], ignore_index=True)

    # Drop the temporary stratify key
    stratified_df = stratified_df.drop(columns=['stratify_key'])

    return stratified_df
