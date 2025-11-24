"""
Unit tests for src/utils.py
"""
import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from src.utils import (
    clean_text,
    validate_dataframe_columns,
    get_google_sheets_client,
    get_spreadsheet
)


class TestCleanText:
    """Tests for the clean_text function."""
    
    def test_clean_text_with_normal_string(self):
        """Test cleaning a normal string."""
        result = clean_text("  Hello World  ")
        assert result == "Hello World"
    
    def test_clean_text_with_extra_whitespace(self):
        """Test cleaning string with extra whitespace."""
        result = clean_text("Hello    World   Test")
        assert result == "Hello World Test"
    
    def test_clean_text_with_empty_string(self):
        """Test cleaning empty string."""
        result = clean_text("")
        assert result == ""
    
    def test_clean_text_with_none(self):
        """Test cleaning None value."""
        result = clean_text(None)
        assert result == ""
    
    def test_clean_text_with_nan(self):
        """Test cleaning NaN value."""
        result = clean_text(pd.NA)
        assert result == ""
    
    def test_clean_text_with_numeric(self):
        """Test cleaning numeric value."""
        result = clean_text(123)
        assert result == "123"
    
    def test_clean_text_with_newlines(self):
        """Test cleaning string with newlines."""
        result = clean_text("Hello\nWorld\n\nTest")
        assert result == "Hello World Test"


class TestValidateDataframeColumns:
    """Tests for the validate_dataframe_columns function."""
    
    def test_validate_with_all_columns_present(self):
        """Test validation when all required columns are present."""
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6],
            'C': [7, 8, 9]
        })
        result = validate_dataframe_columns(df, ['A', 'B'])
        assert result is True
    
    def test_validate_with_missing_columns(self, capsys):
        """Test validation when some columns are missing."""
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        result = validate_dataframe_columns(df, ['A', 'B', 'C'])
        assert result is False
        
        # Check that error message was printed
        captured = capsys.readouterr()
        assert "Missing required columns" in captured.out
        assert "'C'" in captured.out
    
    def test_validate_with_empty_required_list(self):
        """Test validation with empty required columns list."""
        df = pd.DataFrame({'A': [1, 2, 3]})
        result = validate_dataframe_columns(df, [])
        assert result is True
    
    def test_validate_with_empty_dataframe(self):
        """Test validation with empty dataframe."""
        df = pd.DataFrame()
        result = validate_dataframe_columns(df, ['A'])
        assert result is False


class TestGetGoogleSheetsClient:
    """Tests for Google Sheets client creation."""
    
    @patch('src.utils.Credentials.from_service_account_file')
    @patch('src.utils.gspread.authorize')
    @patch('os.path.exists')
    def test_get_client_success(self, mock_exists, mock_authorize, mock_creds):
        """Test successful client creation."""
        mock_exists.return_value = True
        mock_creds.return_value = Mock()
        mock_authorize.return_value = Mock()
        
        client = get_google_sheets_client('test_creds.json')
        
        assert client is not None
        mock_exists.assert_called_once_with('test_creds.json')
        mock_creds.assert_called_once()
        mock_authorize.assert_called_once()
    
    @patch('os.path.exists')
    def test_get_client_missing_credentials(self, mock_exists):
        """Test client creation with missing credentials file."""
        mock_exists.return_value = False
        
        with pytest.raises(FileNotFoundError):
            get_google_sheets_client('missing_creds.json')


class TestGetSpreadsheet:
    """Tests for get_spreadsheet function."""
    
    @patch('src.utils.get_google_sheets_client')
    @patch.dict('os.environ', {'GOOGLE_SHEET_ID': 'test_sheet_id'})
    def test_get_spreadsheet_with_env_variable(self, mock_get_client):
        """Test getting spreadsheet using environment variable."""
        mock_client = Mock()
        mock_spreadsheet = Mock()
        mock_client.open_by_key.return_value = mock_spreadsheet
        mock_get_client.return_value = mock_client
        
        result = get_spreadsheet()
        
        assert result == mock_spreadsheet
        mock_client.open_by_key.assert_called_once_with('test_sheet_id')
    
    @patch('src.utils.get_google_sheets_client')
    def test_get_spreadsheet_with_provided_id(self, mock_get_client):
        """Test getting spreadsheet with provided ID."""
        mock_client = Mock()
        mock_spreadsheet = Mock()
        mock_client.open_by_key.return_value = mock_spreadsheet
        mock_get_client.return_value = mock_client
        
        result = get_spreadsheet('provided_sheet_id')
        
        assert result == mock_spreadsheet
        mock_client.open_by_key.assert_called_once_with('provided_sheet_id')
    
    @patch('src.utils.get_google_sheets_client')
    @patch.dict('os.environ', {}, clear=True)
    def test_get_spreadsheet_no_id(self, mock_get_client):
        """Test getting spreadsheet without ID raises error."""
        mock_get_client.return_value = Mock()
        
        with pytest.raises(ValueError):
            get_spreadsheet()


class TestReadWriteWorksheet:
    """Tests for worksheet read/write operations."""
    
    @patch('src.utils.pd.DataFrame')
    def test_read_worksheet_to_dataframe(self, mock_df):
        """Test reading worksheet to dataframe."""
        from src.utils import read_worksheet_to_dataframe
        
        mock_worksheet = Mock()
        mock_worksheet.get_all_records.return_value = [
            {'A': 1, 'B': 2},
            {'A': 3, 'B': 4}
        ]
        
        result = read_worksheet_to_dataframe(mock_worksheet)
        
        mock_worksheet.get_all_records.assert_called_once()
    
    def test_write_dataframe_to_worksheet(self):
        """Test writing dataframe to worksheet."""
        from src.utils import write_dataframe_to_worksheet
        
        mock_worksheet = Mock()
        mock_worksheet.title = "test_sheet"
        
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        
        write_dataframe_to_worksheet(mock_worksheet, df, clear_first=True)
        
        mock_worksheet.clear.assert_called_once()
        mock_worksheet.update.assert_called_once()


class TestWorksheetOperations:
    """Tests for worksheet operations."""
    
    def test_get_worksheet(self):
        """Test getting a worksheet from spreadsheet."""
        from src.utils import get_worksheet
        
        mock_spreadsheet = Mock()
        mock_worksheet = Mock()
        mock_spreadsheet.worksheet.return_value = mock_worksheet
        
        result = get_worksheet(mock_spreadsheet, 'test_sheet')
        
        assert result == mock_worksheet
        mock_spreadsheet.worksheet.assert_called_once_with('test_sheet')
    
    def test_create_worksheet_if_not_exists_new(self, capsys):
        """Test creating a new worksheet."""
        from src.utils import create_worksheet_if_not_exists
        import gspread
        
        mock_spreadsheet = Mock()
        mock_worksheet = Mock()
        mock_spreadsheet.worksheet.side_effect = gspread.exceptions.WorksheetNotFound
        mock_spreadsheet.add_worksheet.return_value = mock_worksheet
        
        result = create_worksheet_if_not_exists(mock_spreadsheet, 'new_sheet')
        
        assert result == mock_worksheet
        mock_spreadsheet.add_worksheet.assert_called_once()
        
        captured = capsys.readouterr()
        assert "created successfully" in captured.out
    
    def test_create_worksheet_if_not_exists_existing(self, capsys):
        """Test with existing worksheet."""
        from src.utils import create_worksheet_if_not_exists
        
        mock_spreadsheet = Mock()
        mock_worksheet = Mock()
        mock_spreadsheet.worksheet.return_value = mock_worksheet
        
        result = create_worksheet_if_not_exists(mock_spreadsheet, 'existing_sheet')
        
        assert result == mock_worksheet
        
        captured = capsys.readouterr()
        assert "already exists" in captured.out