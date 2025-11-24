"""
Utility functions for Google Sheets connection and basic operations.
"""
import os
import gspread
from google.oauth2.service_account import Credentials
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv()


def get_google_sheets_client(credentials_file: str = "credentials.json") -> gspread.Client:
    """
    Create and return an authenticated Google Sheets client.
    
    Args:
        credentials_file: Path to the Google service account credentials JSON file
        
    Returns:
        gspread.Client: Authenticated client object
        
    Raises:
        FileNotFoundError: If credentials file doesn't exist
        Exception: If authentication fails
    """
    if not os.path.exists(credentials_file):
        raise FileNotFoundError(f"Credentials file '{credentials_file}' not found.")
    
    # Define the required scopes
    scopes = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]
    
    # Authenticate and create client
    creds = Credentials.from_service_account_file(credentials_file, scopes=scopes)
    client = gspread.authorize(creds)
    
    return client


def get_spreadsheet(sheet_id: Optional[str] = None, credentials_file: str = "credentials.json") -> gspread.Spreadsheet:
    """
    Get a specific Google Spreadsheet by ID.
    
    Args:
        sheet_id: Google Sheets ID (from URL). If None, reads from env variable
        credentials_file: Path to credentials file
        
    Returns:
        gspread.Spreadsheet: The spreadsheet object
        
    Raises:
        ValueError: If sheet_id is not provided and not in environment
        Exception: If spreadsheet cannot be accessed
    """
    if sheet_id is None:
        sheet_id = os.getenv('GOOGLE_SHEET_ID')
        
    if not sheet_id:
        raise ValueError("Sheet ID must be provided or set in GOOGLE_SHEET_ID environment variable")
    
    client = get_google_sheets_client(credentials_file)
    spreadsheet = client.open_by_key(sheet_id)
    
    return spreadsheet


def get_worksheet(spreadsheet: gspread.Spreadsheet, worksheet_name: str) -> gspread.Worksheet:
    """
    Get a specific worksheet from a spreadsheet.
    
    Args:
        spreadsheet: The spreadsheet object
        worksheet_name: Name of the worksheet to retrieve
        
    Returns:
        gspread.Worksheet: The worksheet object
        
    Raises:
        gspread.exceptions.WorksheetNotFound: If worksheet doesn't exist
    """
    return spreadsheet.worksheet(worksheet_name)


def create_worksheet_if_not_exists(
    spreadsheet: gspread.Spreadsheet,
    worksheet_name: str,
    rows: int = 1000,
    cols: int = 20
) -> gspread.Worksheet:
    """
    Create a worksheet if it doesn't already exist.
    
    Args:
        spreadsheet: The spreadsheet object
        worksheet_name: Name of the worksheet to create
        rows: Number of rows in the new worksheet
        cols: Number of columns in the new worksheet
        
    Returns:
        gspread.Worksheet: The worksheet object (existing or newly created)
    """
    try:
        worksheet = spreadsheet.worksheet(worksheet_name)
        print(f"Worksheet '{worksheet_name}' already exists.")
        return worksheet
    except gspread.exceptions.WorksheetNotFound:
        worksheet = spreadsheet.add_worksheet(title=worksheet_name, rows=rows, cols=cols)
        print(f"Worksheet '{worksheet_name}' created successfully.")
        return worksheet


def read_worksheet_to_dataframe(worksheet: gspread.Worksheet) -> pd.DataFrame:
    """
    Read all data from a worksheet into a pandas DataFrame.
    
    Args:
        worksheet: The worksheet to read from
        
    Returns:
        pd.DataFrame: DataFrame containing the worksheet data
    """
    data = worksheet.get_all_records()
    df = pd.DataFrame(data)
    return df


def write_dataframe_to_worksheet(
    worksheet: gspread.Worksheet,
    df: pd.DataFrame,
    clear_first: bool = True
) -> None:
    """
    Write a pandas DataFrame to a worksheet.
    
    Args:
        worksheet: The worksheet to write to
        df: DataFrame to write
        clear_first: Whether to clear existing content first
        
    Returns:
        None
    """
    if clear_first:
        worksheet.clear()
    
    # Convert DataFrame to list of lists (including headers)
    values = [df.columns.tolist()] + df.fillna('').values.tolist()
    
    # Update the worksheet
    worksheet.update(values, 'A1')
    print(f"✓ Written {len(df)} rows to worksheet '{worksheet.title}'")


def is_worksheet_protected(worksheet: gspread.Worksheet) -> bool:
    """
    Check if a worksheet is protected.
    
    Args:
        worksheet: The worksheet to check
        
    Returns:
        bool: True if protected, False otherwise
    """
    try:
        # Try to get protected ranges
        spreadsheet = worksheet.spreadsheet
        protected_ranges = spreadsheet.list_protected_ranges(worksheet.id)
        return len(protected_ranges) > 0
    except Exception:
        return False


def clean_text(text: Any) -> str:
    """
    Clean and standardize text data.
    
    Args:
        text: Input text to clean
        
    Returns:
        str: Cleaned text
    """
    if pd.isna(text) or text == '' or text is None:
        return ''
    
    # Convert to string and strip whitespace
    text = str(text).strip()
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text


def validate_dataframe_columns(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate that a DataFrame contains all required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of column names that must be present
        
    Returns:
        bool: True if all columns present, False otherwise
    """
    missing_columns = set(required_columns) - set(df.columns)
    
    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
        return False
    
    return True