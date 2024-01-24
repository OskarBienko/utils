import pandas as pd
import numpy as np


def reduce_mem_usage(df: pd.DataFrame=None) -> pd.DataFrame:
    
    '''
    The function modifies the input DataFrame by changing the data types of numeric columns to reduce memory usage.
    '''
    
    start_mem = df.memory_usage().sum() / 1024**2
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    timedelta_cols = df.select_dtypes(include=[np.timedelta64]).columns
    
    # Create a list of valid numeric columns by excluding timedelta columns
    valid_cols = list(set(numeric_cols) - set(timedelta_cols))

    for col in valid_cols:        
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            # For integer columns:
                # Check if the column values fit within the range of a smaller integer type (e.g., int8, int16, int32, int64)
                # Downcast the column to the smallest integer type that can safely store the values
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            
            # For float columns:
                # Check if the column values fit within the range of a smaller float type (e.g., float16, float32)
                # Downcast the column to the smallest float type that can safely store the values
            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float32)
    
    end_mem = df.memory_usage().sum() / 1024**2
    decrease = 100 * (start_mem - end_mem) / start_mem

    print(f'Memory usage of dataframe is {start_mem:.2f} MB')
    print(f'Memory usage after optimization is: {end_mem:.2f} MB')
    print(f'Decreased by {decrease:.2f}%')

    return df