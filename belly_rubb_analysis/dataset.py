"""This module provides functions for loading, processing, and analyzing datasets.
Functions:
    load_col_types(file_path: str) -> dict:
        Load column types from a JSON file.
    load_data(file_path: str) -> pd.DataFrame:
        Load and return raw data from a given file path.
    convert_data_types(df: pd.DataFrame, col_types: dict) -> pd.DataFrame:
        Convert data types in a DataFrame to those specified in a JSON file.
    file_exists(file_name: str) -> bool:
        Check if a file exists in the project directory.
    drop_const_col(df: pd.DataFrame) -> pd.DataFrame:
        Drop columns with a single constant value or entirely missing data.
    drop_high_missing(df: pd.DataFrame, missing_ratio: float = 0.70) -> pd.DataFrame:
        Drop columns with a particular ratio of entries missing.
    drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        Drop duplicate rows from data.
    validate_cols(df: pd.DataFrame, col_data: dict) -> pd.DataFrame:
        Validate values in columns based on provided column data.
    autocorrect_col_values(df: pd.DataFrame, col: str, valid_values: list) -> pd.DataFrame:
        Standardize column values based on valid values.
    calculate_upper_bound(df: pd.DataFrame, col: str) -> int:
        Calculate the upper bound of a boxplot for detecting outliers.
    generate_profile_report(df: pd.DataFrame, filename: str, output_directory: Path = PROFILE_REPORTS_DIR) -> None:
        Generate a profile report from a DataFrame.
    find_similar_csv(table: str, data_dir: str = RAW_DATA_DIR.as_posix()) -> list:
        Find CSV files from the same table.
    find_types(table_name: str, directory: Path = DATATYPES_DIR):
        Find the JSON file associated with a table.
    combine_csv_files(table: str, data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
        Combine table data from multiple CSV files into one DataFrame.
    detect_date_col(df: pd.DataFrame, date_keywords=None) -> str:
        Return the name of the date column in a DataFrame.
    main(input_path: Path = RAW_DATA_DIR / "orders", output_path: Path = PROCESSED_DATA_DIR / "orders_processed.csv"):
        Perform data cleaning and processing."""
import os
import pdb
from pathlib import Path
from difflib import get_close_matches
import json
import typer
import pandas as pd
from loguru import logger
from rapidfuzz.distance import Levenshtein
from rapidfuzz.fuzz import ratio
# from tqdm import tqdm
from ydata_profiling import ProfileReport
from belly_rubb_analysis.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, \
    DATATYPES_DIR, INTERIM_DATA_DIR, PROFILE_REPORTS_DIR, PROJ_ROOT

DATE_KEYWORDS = ["date", "order_date", "orderDate", "Order Date", "timestamp"]

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

def convert_dollars(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Removes $ sign from dollar amounts and converts to numeric
    
    Params:
        df (pd.DataFrame): Original dataframe
        col (str): Dictionary mapping columns to datatypes
        
    Returns:
        df (pd.DataFrame): Processed dataframe
    """
    # Remove $ sign
    df[col] = df[col].replace(r'[\$,]', '', regex=True)
    df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

def convert_category(df: pd.DataFrame, col: str, categories: list, ordered: bool) -> pd.DataFrame:
    """Converts column to category type
    
    Params:
        df (pd.DataFrame): Original dataframe
        col (str): Name of column to convert
        categories (list): List of categories in column
        ordered (bool): Whether categories are ordinal or not
        
    returns:
        df (pd.DataFrame): Dataframe with converted column
    """
    df[col] = pd.Categorical(values=df[col], categories=categories, ordered=ordered)

    return df

def convert_data_types(df: pd.DataFrame, col_types: dict) -> pd.DataFrame:
    """Converts datatypes in df to those specified in JSON file

    Params:
        df (pd.DataFrame): Input DataFrame.
        col_types (dict): Dictionary mapping columns to datatypes

    Returns:
        df (pd.DataFrame): Output DataFrame with correct datatypes
    """
    # breakpoint()
    # Loop through dtype dictionary
    for col, dtype in col_types.items():
        # Verify column exists in df
        if col in df.columns:
            # String conversion
            if dtype['type'] == 'string':
                df[col] = df[col].astype('string')
            # Datetime conversion
            elif dtype['type'] == 'datetime':
                df[col] = pd.to_datetime(df[col], errors='coerce', format=dtype['format'])

                # Return only time if specified
                if dtype.get('only_time', False):
                    df[col] = df[col].dt.time
            # Category conversion
            elif dtype['type'] == 'category':
                df = convert_category(df,
                                      col=col,
                                      categories=dtype['categories'],
                                      ordered=dtype['ordered'])
            # Dollar conversion
            elif dtype['type'] == 'dollar':
                df = convert_dollars(df, col)
            # Numeric conversion
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

def file_exists(file_name: str) -> bool:
    """Checks if a file exists in the project directory.
    
    Params:
        file_name (str): Name of file to search for
        
    Returns:
        bool: True if found, False otherwise
    """
    for root, _, files in os.walk(PROJ_ROOT):
        if file_name in files:
            return True

    return False

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

def drop_high_missing(df: pd.DataFrame, missing_ratio: float = 0.70) -> pd.DataFrame:
    """Drops columns with a particular ratio of entries missing.
    
    Params:
        df (pd.DataFrame): Input DataFrame
        ratio (float): Ratio of missing to non-missing values
    
    Returns:
        df (pd.DataFrame): DataFrame with columns missing data dropped."""
    mask = df.isna().sum() / df.shape[0] > missing_ratio
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

def validate_cols(df: pd.DataFrame, col_data: dict) -> pd.DataFrame:
    """Validate values in column.
    
    Params:
        df (pd.DataFrame): Original dataframe
        col_data (dict): Column information
        
    Returns:
        df (pd.DataFrame): Processed dataframe
    """
    validated_df = df

    # Loop through column data
    for col, dtype in col_data.items():
        # Validate that column exists in df
        if col in df.columns:
            # Col has list of valid values
            if isinstance(dtype, dict) and ('valid_values' in dtype):
                # Correct values in column
                validated_df = autocorrect_col_values(
                    df=validated_df,
                    col=col,
                    valid_values=dtype['valid_values']
                )

    return validated_df

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
        filename: str,
        output_directory: Path = PROFILE_REPORTS_DIR) -> None:
    """Generate Profile Report
    
    Params:
        df (pd.DataFrame): DataFrame to make report from
        filename (str): Name of file to save to
        output_directory (Path): Output directory
    """
    profile = ProfileReport(df, title='Pandas Profiling Report')
    profile.to_file(output_directory / filename)

def find_similar_csv(table: str, data_dir: str = RAW_DATA_DIR.as_posix()) -> list:
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
            similarity = ratio(table, file)
            threshold = 90

            if (similarity > threshold) or (table in file):
                similar_files.append(file)
    return similar_files

def find_types(table_name: str, directory: Path = DATATYPES_DIR):
    """Find .json file associated with table
    
    Params:
        table_name (str): Name of table
        directory (Path): Directory to look in
    
    Returns:
        .json Path (Path): Path to matching .json file
    """
    # Load all .json files
    json_files = [file.name for file in directory.glob('*.json')]

    # Find files with table_name in file name or close to it
    matching_file = get_close_matches(table_name, json_files, n=1, cutoff=0.3)[0]

    return directory / matching_file

def combine_csv_files(table: str, data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Combines table data from multiple csv's into one
    
    Params:
        table (str): Table name to combine
        data_dir (str): Path to data
    
    Returns:
        dfs (pd.DataFrame): Combined DataFrame"""
    # Get all csv files of that table
    all_files = find_similar_csv(table=table, data_dir=data_dir)

    print(all_files)
    # Combine data from tables into one DataFrame
    dfs = pd.concat([load_data(data_dir / file) for file in all_files])

    # Check if there is a date column
    date_col = detect_date_col(dfs)

    if date_col:
        # If there is a date column, sort dataframe by those values
        dfs = dfs.sort_values(by=date_col).reset_index(drop=True)

    return dfs

def detect_date_col(df: pd.DataFrame, date_keywords=None) -> str:
    """Returns name of date column in dataframe
    
    Params:
        df (pd.DataFrame): Original dataframe
        date_keywords (list): List of date keywords to search for in df columns
        
    Returns:
        col (str): Name of first column matching any of the date keywords
    """
    if date_keywords is None:
        date_keywords = DATE_KEYWORDS

    for col in df.columns:
        for keyword in date_keywords:
            if keyword.lower() == col.lower():
                return col

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "transactions",
    output_path: Path = PROCESSED_DATA_DIR / "orders_processed.csv"
    # ----------------------------------------------
):
    """Performs data cleaning.
    
    Params:
        input_path (str): Path to raw data
        output_path (str): Path to export processed data
        datatype_path (str): Path to JSON containing datatypes
    """
    breakpoint()
    # Get table name
    table_name = input_path.stem
    logger.info(f"Loaded table {table_name}")

    # Combine with other csv files regarding same table
    logger.info("Combining with other related csv files")
    df = combine_csv_files(table=table_name)

    # Generate profile report if doesn't exist already
    report_path = input_path.stem + '.html'
    logger.info(f"Checking if profile report for {table_name} exists")
    if file_exists(report_path):
        logger.info("Report exists")

        # Prompt user if they'd like to overwrite current report
        user_input = input("Would you like to overwrite the current report? (yes/no)")

        if user_input.lower() in ('yes', 'y'):
            logger.info('Generating profile report for {table_name}')
            generate_profile_report(df=df,filename=input_path.name)
        else:
            logger.info('Skipping profile report generation')
    else:
        logger.info(f"Report does not exist, generating report for {table_name}")
        generate_profile_report(df=df, filename=input_path.name)

    # Find .json file associated with input dataset
    json_file = find_types(table_name=input_path.name)

    # Load data from json file
    logger.info(f"Loading datatypes from {json_file.name}")
    data_types = load_col_types(json_file)

    # Convert columns to datatypes specified in json
    logger.info(f"Converting datatypes in {input_path.name}")
    df = convert_data_types(df, data_types)

    # Drop columns with constant values
    logger.info(f"Dropping constant columns from {input_path.name}")
    df = drop_const_col(df)

    # Drop columns with set amount of data missing
    logger.info(f"Dropping columns with more than 72% data missing from {input_path.name}")
    df = drop_high_missing(df, missing_ratio=0.72)

    # Drop duplicate rows
    logger.info(f"Dropping duplicate rows from {input_path.name}")
    df = drop_duplicates(df)

    # Validate values in columns
    logger.info(f"Standardizing values in {input_path.name}")
    df = validate_cols(df=df, col_data=data_types)

    # Save data
    output_filename = INTERIM_DATA_DIR / (input_path.stem + '_interim.csv')
    logger.info(f"Outputting processed file to {output_filename}")
    df.to_csv(output_filename, index=False)

    logger.success('Processing dataset complete.')

if __name__ == "__main__":
    app()
