import os
from pathlib import Path
import json
import typer
import pandas as pd
from loguru import logger
from rapidfuzz.distance import Levenshtein
# from tqdm import tqdm
from ydata_profiling import ProfileReport

from belly_rubb_analysis.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, \
    DATATYPES_DIR, INTERIM_DATA_DIR, PROFILE_REPORTS_DIR

app = typer.Typer()

def load_col_types(file_path: str) -> dict:
    """Load column types from JSON file
    
    Params:
        file_path (str): Path to JSON file
        
    Returns:
        dict: A dictionary mapping col names to data types.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_data(file_path: str) -> pd.DataFrame:
    """Loads and returns raw data from given file_path
    
    Params:
        file_path (str): Path to csv file
    
    Returns:
        DataFrame: Loaded DataFrame"""
    return pd.read_csv(file_path)

def convert_data_types(df: pd.DataFrame, col_types: dict) -> pd.DataFrame:
    """Converts datatypes in df to those specified in JSON file

    Params:
        df (pd.DataFrame): Input DataFrame.
        col_types (dict): Dictionary mapping columns to datatypes

    Returns:
        df (pd.DataFrame): Output DataFrame with correct datatypes
    """
    for col, dtype in col_types.items():
        if col in df.columns:
            if dtype == 'string':
                df[col] = df[col].astype('string')
            elif isinstance(dtype, dict):
                if dtype['type'] == 'datetime':
                    df[col] = pd.to_datetime(df[col], errors='raise', format=dtype['format'])
                else:
                    categories = dtype['categories']
                    ordered = dtype['ordered']
                    df[col] = pd.Categorical(df[col], categories=categories, ordered=ordered)
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

def drop_const_col(df: pd.DataFrame) -> pd.DataFrame:
    """Drops columns with a single constant value.

    In addition to dropping columns with a single constant value, it will 
    also drop columns with entirely missing data.
    
    Params:
        df (pd.DataFrame): Input DataFrame
    
    Ouutput:
        df (pd.DataFrame): DataFrame with constant columns dropped
    """
    for col in df.columns:
        if len(df[col].dropna().unique()) < 2:
            df = df.drop(labels=col, axis=1)

    return df

def drop_high_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Drops columns with more than 70% entries missing.
    
    Params:
        df (pd.DataFrame): Input DataFrame
    
    Returns:
        df (pd.DataFrame): DataFrame with columns with more than 70% missing data dropped."""
    mask = df.isna().sum() / df.shape[0] > 0.7
    df = df.drop(columns=mask[mask].index, axis=1)

    return df

def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drops duplicate rows from data.
    
    Params:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        df (pd.DataFrame): DataFrame with no duplicate rows.
    """
    return df.drop_duplicates()

def autocorrect_col_values(df: pd.DataFrame, col: str, valid_values: list) -> pd.DataFrame:
    """Standardizes column values.
    
    Params:
        df (pd.DataFrame): Original DataFrame
        col (str): Column to be standardized
        valid_values (list): List of valid values to standardize to
    
    Returns:
        df (pd.DataFrame): DataFrame with standardized column
    """
    def standardize_value(value: str) -> str:
        """Returns valid match based on similarity
        
        Params:
            value (str): Value to convert
            
        Returns:
            closest_match (str): Closest match from valid_values
        """
        closest_match = max(valid_values, \
                            key=lambda ref: Levenshtein.similarity(value, ref.lower()))

        return closest_match

    df[col] = df[col].apply(standardize_value)

    return df

def calculate_upper_bound(df: pd.DataFrame, col: str) -> int:
    """Calculates upper bound of boxplot.
    
    Can be used for detecting outliers.
    
    Params:
        df (pd.DataFrame): Input DataFrame
        col (str): Column to calculate upper bound for
    
    Returns:
        upper bound (int): Integer representing upper bound
    """
    q3 = df[col].quantile(q=0.75)
    q1 = df[col].quantile(q=0.25)

    iqr = q3 - q1

    return q3 + 1.5 * iqr

def generate_profile_report(
        df: pd.DataFrame,
        output_file_path: str,
        title: str = 'Profile Report') -> None:
    """Generate Profile Report
    
    Params:
        df (pd.DataFrame): DataFrame to make report from
        output_file_path (str): Output filepath
        title (str): Title of report
    """
    profile = ProfileReport(df, title)
    profile.to_file(output_file_path)

def find_similar_csv(table: str, data_dir: str = RAW_DATA_DIR) -> list:
    """Finds csv files from same table
    
    Params:
        table (str): Name of table
        data_dir (str): Directory to look in
    
    Returns:
        similar_files (list): List of all .csv files from that table"""
    # List all files in @dir
    files = os.listdir(path=data_dir)

    similar_files = []

    # Loop through files
    for file in files:
        # Check if file is a .csv file
        if file.endswith('csv'):
            # Get table name
            table_name = file.split('-')[0]

            if table_name.lower() == table.lower():
                similar_files.append(file)

    return similar_files

def combine_csv_files(table: str, data_dir: str = RAW_DATA_DIR.as_posix()) -> pd.DataFrame:
    """Combines table data from multiple csv's into one
    
    Params:
        table (str): Table name to combine
        data_dir (str): Path to data
    
    Returns:
        dfs (pd.DataFrame): Combined DataFrame"""
    # Get all csv files of that table
    all_files = find_similar_csv(table=table, data_dir=data_dir)

    # Combine data from tables into one DataFrame
    dfs = pd.concat([pd.read_csv(data_dir + file) for file in all_files])

    return dfs

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "orders-2023-12-22-2024-12-20.csv",
    output_path: Path = PROCESSED_DATA_DIR / "orders_processed.csv",
    datatype_path: Path = DATATYPES_DIR / "orders_data_types.json"
    # ----------------------------------------------
):
    """Performs data cleaning.
    
    Params:
        input_path (str): Path to raw data
        output_path (str): Path to export processed data
        datatype_path (str): Path to JSON containing datatypes
    """
    logger.info(f"Generating profile report of {input_path.name}")
    generate_profile_report(input_path, PROFILE_REPORTS_DIR)

    logger.info(f"Loading dataset {input_path.name}")
    df = load_data(input_path)
    logger.success("Loaded dataset")

    logger.info(f"Loading datatypes from {datatype_path.name}")
    data_types = load_col_types(datatype_path)
    logger.success("Loaded datatypes")

    logger.info(f"Converting datatypes in {input_path.name}")
    df = convert_data_types(df, data_types)
    print(df['Order Date'].head())
    logger.success("Converted datatypes")

    logger.info(f"Dropping constant columns from {input_path.name}")
    df = drop_const_col(df)
    logger.success("Dropped constant columns")

    logger.info(f"Dropping columns with more than 70% data missing from {input_path.name}")
    df = drop_high_missing(df)
    logger.success("Dropped columns with high missing values")

    logger.info(f"Dropping duplicate rows from {input_path.name}")
    df = drop_duplicates(df)
    logger.success("Dropped duplicate rows")

    logger.info(f"Standardizing values in 'Channels' column from {input_path.name}")
    valid_channel_values = ['Postmates Delivery', \
                            'BELLY RUBB | BBQ Catering | Barbecue To Go and Delivery', \
                            'DoorDash', 'Payment Links']
    df = autocorrect_col_values(df, 'Channels', valid_channel_values)
    logger.success("Standardized Channels values")

    output_filename = INTERIM_DATA_DIR / input_path.name
    logger.info(f"Outputting processed file to {output_filename}")
    df.to_csv(output_filename, index=False)
    logger.success("Saved processed file")

    """ ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing dataset...")
    for i in tqdm(range(10), total=10):
        
    logger.success("Processing dataset complete.")
    # -----------------------------------------
    """

if __name__ == "__main__":
    #app()
    print(combine_csv_files('transactions', data_dir='data/raw/'))
