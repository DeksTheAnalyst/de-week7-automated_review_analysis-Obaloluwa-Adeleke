"""
ETL (Extract, Transform, Load) functions for the review analysis pipeline.
"""
import pandas as pd
import gspread
from typing import Optional
from src.utils import (
    get_spreadsheet,
    get_worksheet,
    create_worksheet_if_not_exists,
    read_worksheet_to_dataframe,
    write_dataframe_to_worksheet,
    clean_text,
    is_worksheet_protected
)
from src.llm_analysis import batch_analyze_reviews


def extract_raw_data(
    sheet_id: Optional[str] = None,
    raw_worksheet_name: str = "raw_data",
    credentials_file: str = "credentials.json"
) -> pd.DataFrame:
    """
    Extract data from the raw_data worksheet.
    
    Args:
        sheet_id: Google Sheets ID
        raw_worksheet_name: Name of the raw data worksheet
        credentials_file: Path to credentials file
        
    Returns:
        pd.DataFrame: Raw data from the worksheet
    """
    print(f"Extracting data from '{raw_worksheet_name}' worksheet...")
    
    spreadsheet = get_spreadsheet(sheet_id, credentials_file)
    raw_worksheet = get_worksheet(spreadsheet, raw_worksheet_name)
    
    # Verify the worksheet is protected
    if not is_worksheet_protected(raw_worksheet):
        print(f"Warning: '{raw_worksheet_name}' worksheet is not protected!")
    
    df = read_worksheet_to_dataframe(raw_worksheet)
    print(f"Extracted {len(df)} rows from raw data")
    
    return df


def transform_staging_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform and clean data for staging.
    
    Args:
        df: Raw DataFrame to transform
        
    Returns:
        pd.DataFrame: Cleaned and transformed DataFrame
    """
    print("Transforming data for staging...")
    
    # Create a copy to avoid modifying original
    df_transformed = df.copy()
    
    # Get initial shape
    initial_rows = len(df_transformed)
    
    # Clean text columns (assuming common column names from the dataset)
    text_columns = ['Title', 'Review Text']
    for col in text_columns:
        if col in df_transformed.columns:
            df_transformed[col] = df_transformed[col].apply(clean_text)
    
    # Clean other text columns that might exist
    for col in df_transformed.columns:
        if df_transformed[col].dtype == 'object':
            df_transformed[col] = df_transformed[col].apply(clean_text)
    
    # Remove completely empty rows (all values are empty strings or NaN)
    df_transformed = df_transformed.replace('', pd.NA)
    df_transformed = df_transformed.dropna(how='all')
    df_transformed = df_transformed.fillna('')
    
    rows_removed = initial_rows - len(df_transformed)
    if rows_removed > 0:
        print(f"Removed {rows_removed} completely empty rows")
    
    print(f"Transformed {len(df_transformed)} rows for staging")
    
    return df_transformed


def load_staging_data(
    df: pd.DataFrame,
    sheet_id: Optional[str] = None,
    staging_worksheet_name: str = "staging",
    credentials_file: str = "credentials.json"
) -> None:
    """
    Load transformed data into the staging worksheet.
    
    Args:
        df: DataFrame to load
        sheet_id: Google Sheets ID
        staging_worksheet_name: Name of the staging worksheet
        credentials_file: Path to credentials file
        
    Returns:
        None
    """
    print(f"Loading data to '{staging_worksheet_name}' worksheet...")
    
    spreadsheet = get_spreadsheet(sheet_id, credentials_file)
    staging_worksheet = create_worksheet_if_not_exists(
        spreadsheet,
        staging_worksheet_name,
        rows=len(df) + 100,
        cols=len(df.columns) + 10
    )
    
    write_dataframe_to_worksheet(staging_worksheet, df, clear_first=True)
    print(f"Loaded {len(df)} rows to staging")


def extract_staging_data(
    sheet_id: Optional[str] = None,
    staging_worksheet_name: str = "staging",
    credentials_file: str = "credentials.json"
) -> pd.DataFrame:
    """
    Extract data from the staging worksheet.
    
    Args:
        sheet_id: Google Sheets ID
        staging_worksheet_name: Name of the staging worksheet
        credentials_file: Path to credentials file
        
    Returns:
        pd.DataFrame: Data from staging worksheet
    """
    print(f"Extracting data from '{staging_worksheet_name}' worksheet...")
    
    spreadsheet = get_spreadsheet(sheet_id, credentials_file)
    staging_worksheet = get_worksheet(spreadsheet, staging_worksheet_name)
    
    df = read_worksheet_to_dataframe(staging_worksheet)
    print(f"Extracted {len(df)} rows from staging")
    
    return df


def prepare_processed_dataframe(df_staging: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the processed DataFrame with additional columns for LLM results.
    
    Args:
        df_staging: Staging DataFrame
        
    Returns:
        pd.DataFrame: Processed DataFrame with new columns
    """
    print("Preparing processed DataFrame structure...")
    
    df_processed = df_staging.copy()
    
    # Add new columns for LLM results
    df_processed['AI Sentiment'] = ''
    df_processed['AI Summary'] = ''
    df_processed['Action Needed?'] = ''
    
    print(f"Prepared processed DataFrame with {len(df_processed.columns)} columns")
    
    return df_processed


def load_processed_data(
    df: pd.DataFrame,
    sheet_id: Optional[str] = None,
    processed_worksheet_name: str = "processed",
    credentials_file: str = "credentials.json"
) -> None:
    """
    Load processed data into the processed worksheet.
    
    Args:
        df: DataFrame to load
        sheet_id: Google Sheets ID
        processed_worksheet_name: Name of the processed worksheet
        credentials_file: Path to credentials file
        
    Returns:
        None
    """
    print(f"Loading data to '{processed_worksheet_name}' worksheet...")
    
    spreadsheet = get_spreadsheet(sheet_id, credentials_file)
    processed_worksheet = create_worksheet_if_not_exists(
        spreadsheet,
        processed_worksheet_name,
        rows=len(df) + 100,
        cols=len(df.columns) + 5
    )
    
    write_dataframe_to_worksheet(processed_worksheet, df, clear_first=True)
    print(f"Loaded {len(df)} rows to processed")


def run_etl_pipeline(
    sheet_id: Optional[str] = None,
    credentials_file: str = "credentials.json",
    run_llm_analysis: bool = True,
    review_column: str = "Review Text"
) -> pd.DataFrame:
    """
    Run the complete ETL pipeline from raw to staging to processed with LLM analysis.
    
    This is idempotent - running multiple times produces the same result.
    
    Args:
        sheet_id: Google Sheets ID
        credentials_file: Path to credentials file
        run_llm_analysis: Whether to run LLM analysis on reviews
        review_column: Name of the column containing review text
        
    Returns:
        pd.DataFrame: Processed DataFrame with LLM analysis results
    """
    print("=" * 60)
    print("Starting ETL Pipeline")
    print("=" * 60)
    
    # Extract from raw_data
    df_raw = extract_raw_data(sheet_id, credentials_file=credentials_file)
    
    # Transform for staging
    df_staging = transform_staging_data(df_raw)
    
    # Load to staging
    load_staging_data(df_staging, sheet_id, credentials_file=credentials_file)
    
    # Prepare processed structure
    df_processed = prepare_processed_dataframe(df_staging)
    
    # Run LLM analysis if requested
    if run_llm_analysis:
        print("\n" + "=" * 60)
        print("Running LLM Analysis")
        print("=" * 60)
        
        # Check if review column exists
        if review_column not in df_processed.columns:
            print(f"Warning!: Column '{review_column}' not found in data")
            print(f"Available columns: {df_processed.columns.tolist()}")
            print("Skipping LLM analysis...")
        else:
            # Get reviews from the specified column
            reviews = df_processed[review_column].fillna('').tolist()
            
            print(f"Analyzing {len(reviews)} reviews...")
            
            # Batch analyze all reviews
            results = batch_analyze_reviews(reviews)
            
            # Add results to dataframe
            df_processed['AI Sentiment'] = [r['sentiment'] for r in results]
            df_processed['AI Summary'] = [r['summary'] for r in results]
            df_processed['Action Needed?'] = [r['action_needed'] for r in results]
            
            print(f" LLM analysis completed")
            print(f"  - Positive: {sum(1 for r in results if r['sentiment'] == 'Positive')}")
            print(f"  - Negative: {sum(1 for r in results if r['sentiment'] == 'Negative')}")
            print(f"  - Neutral: {sum(1 for r in results if r['sentiment'] == 'Neutral')}")
    
    # Load processed data with LLM results
    load_processed_data(df_processed, sheet_id, credentials_file=credentials_file)
    
    print("\n" + "=" * 60)
    print("ETL Pipeline Completed Successfully")
    print("=" * 60)
    
    return df_processed