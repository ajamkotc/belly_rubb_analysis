from pathlib import Path
import json
import typer
import pandas as pd
from loguru import logger
# from tqdm import tqdm

from belly_rubb_analysis.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, \
    DATATYPES_DIR, INTERIM_DATA_DIR

app = typer.Typer()

def load_col_types(file_path: str) -> dict:
    """
    Load column types from JSON file
    
    Params:
        file_path (str): Path to JSON file
        
    Returns:
        dict: A dictionary mapping col names to data types.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads and returns raw data from given file_path
    
    Params:
        file_path (str): Path to csv file
    
    Returns:
        DataFrame: Loaded DataFrame"""
    return pd.read_csv(file_path)

def convert_data_types(df: pd.DataFrame, col_types: dict) -> pd.DataFrame:
    """
    Converts datatypes in df to those specified in JSON file

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
    """
    Drops columns with a single constant value.

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
    """
    Drops columns with more than 70% entries missing.
    
    Params:
        df (pd.DataFrame): Input DataFrame
    
    Returns:
        df (pd.DataFrame): DataFrame with columns with more than 70% missing data dropped."""
    mask = df.isna().sum() / df.shape[0] > 0.7
    df = df.drop(columns=mask[mask].index, axis=1)

    return df

def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops duplicate rows from data.
    
    Params:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        df (pd.DataFrame): DataFrame with no duplicate rows.
    """
    return df.drop_duplicates()

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "orders.csv",
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

    output_filename = INTERIM_DATA_DIR / f"{input_path.stem}_processed.csv"
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
    app()
