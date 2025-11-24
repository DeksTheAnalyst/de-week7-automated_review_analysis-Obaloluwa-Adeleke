"""
Unit tests for src/etl.py
"""
import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from src.etl import (
    transform_staging_data,
    prepare_processed_dataframe,
    extract_raw_data,
    load_staging_data,
    extract_staging_data,
    load_processed_data
)


class TestTransformStagingData:
    """Tests for transform_staging_data function."""
    
    def test_transform_with_clean_data(self):
        """Test transforming clean data."""
        df = pd.DataFrame({
            'Title': ['  Great Product  ', 'Nice Item'],
            'Review Text': ['Love it!  ', '  Amazing  '],
            'Rating': [5, 4]
        })
        
        result = transform_staging_data(df)
        
        assert len(result) == 2
        assert result['Title'].iloc[0] == 'Great Product'
        assert result['Review Text'].iloc[1] == 'Amazing'
    
    def test_transform_removes_empty_rows(self):
        """Test that completely empty rows are removed."""
        df = pd.DataFrame({
            'A': ['data', '', None],
            'B': ['data', '', None],
            'C': ['data', '', None]
        })
        
        result = transform_staging_data(df)
        
        # Should only have 1 row (the first one with data)
        assert len(result) == 1
        assert result['A'].iloc[0] == 'data'
    
    def test_transform_cleans_whitespace(self):
        """Test that extra whitespace is cleaned."""
        df = pd.DataFrame({
            'Text': ['  Multiple   Spaces   Here  ']
        })
        
        result = transform_staging_data(df)
        
        assert result['Text'].iloc[0] == 'Multiple Spaces Here'
    
    def test_transform_handles_mixed_types(self):
        """Test transformation with mixed data types."""
        df = pd.DataFrame({
            'Text': ['  Hello  ', 'World'],
            'Number': [123, 456],
            'Float': [1.5, 2.5]
        })
        
        result = transform_staging_data(df)
        
        assert len(result) == 2
        assert result['Number'].iloc[0] == 123
        assert result['Float'].iloc[1] == 2.5


class TestPrepareProcessedDataframe:
    """Tests for prepare_processed_dataframe function."""
    
    def test_prepare_adds_new_columns(self):
        """Test that new columns are added."""
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        
        result = prepare_processed_dataframe(df)
        
        assert 'AI Sentiment' in result.columns
        assert 'AI Summary' in result.columns
        assert 'Action Needed?' in result.columns
    
    def test_prepare_preserves_original_data(self):
        """Test that original data is preserved."""
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        
        result = prepare_processed_dataframe(df)
        
        assert result['A'].tolist() == [1, 2, 3]
        assert result['B'].tolist() == [4, 5, 6]
    
    def test_prepare_initializes_empty_columns(self):
        """Test that new columns are initialized empty."""
        df = pd.DataFrame({
            'A': [1, 2]
        })
        
        result = prepare_processed_dataframe(df)
        
        assert result['AI Sentiment'].iloc[0] == ''
        assert result['AI Summary'].iloc[0] == ''
        assert result['Action Needed?'].iloc[0] == ''
    
    def test_prepare_does_not_modify_original(self):
        """Test that original DataFrame is not modified."""
        df = pd.DataFrame({
            'A': [1, 2, 3]
        })
        original_columns = df.columns.tolist()
        
        result = prepare_processed_dataframe(df)
        
        # Original should still have same columns
        assert df.columns.tolist() == original_columns
        # Result should have more columns
        assert len(result.columns) > len(df.columns)


class TestExtractRawData:
    """Tests for extract_raw_data function."""
    
    @patch('src.etl.get_spreadsheet')
    @patch('src.etl.get_worksheet')
    @patch('src.etl.read_worksheet_to_dataframe')
    @patch('src.etl.is_worksheet_protected')
    def test_extract_raw_data_success(self, mock_protected, mock_read, mock_get_ws, mock_get_ss):
        """Test successful raw data extraction."""
        # Setup mocks
        mock_spreadsheet = Mock()
        mock_worksheet = Mock()
        mock_df = pd.DataFrame({'A': [1, 2, 3]})
        
        mock_get_ss.return_value = mock_spreadsheet
        mock_get_ws.return_value = mock_worksheet
        mock_read.return_value = mock_df
        mock_protected.return_value = True
        
        # Execute
        result = extract_raw_data('test_sheet_id')
        
        # Verify
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        mock_get_ss.assert_called_once()
        mock_get_ws.assert_called_once()
        mock_read.assert_called_once()
    
    @patch('src.etl.get_spreadsheet')
    @patch('src.etl.get_worksheet')
    @patch('src.etl.read_worksheet_to_dataframe')
    @patch('src.etl.is_worksheet_protected')
    def test_extract_warns_unprotected_sheet(self, mock_protected, mock_read, mock_get_ws, mock_get_ss, capsys):
        """Test warning when sheet is not protected."""
        mock_spreadsheet = Mock()
        mock_worksheet = Mock()
        mock_df = pd.DataFrame({'A': [1]})
        
        mock_get_ss.return_value = mock_spreadsheet
        mock_get_ws.return_value = mock_worksheet
        mock_read.return_value = mock_df
        mock_protected.return_value = False
        
        result = extract_raw_data('test_sheet_id')
        
        captured = capsys.readouterr()
        assert "not protected" in captured.out


class TestLoadStagingData:
    """Tests for load_staging_data function."""
    
    @patch('src.etl.get_spreadsheet')
    @patch('src.etl.create_worksheet_if_not_exists')
    @patch('src.etl.write_dataframe_to_worksheet')
    def test_load_staging_success(self, mock_write, mock_create_ws, mock_get_ss):
        """Test successful staging data load."""
        mock_spreadsheet = Mock()
        mock_worksheet = Mock()
        mock_get_ss.return_value = mock_spreadsheet
        mock_create_ws.return_value = mock_worksheet
        
        df = pd.DataFrame({'A': [1, 2, 3]})
        
        load_staging_data(df, 'test_sheet_id')
        
        mock_get_ss.assert_called_once()
        mock_create_ws.assert_called_once()
        mock_write.assert_called_once()


class TestExtractStagingData:
    """Tests for extract_staging_data function."""
    
    @patch('src.etl.get_spreadsheet')
    @patch('src.etl.get_worksheet')
    @patch('src.etl.read_worksheet_to_dataframe')
    def test_extract_staging_success(self, mock_read, mock_get_ws, mock_get_ss):
        """Test successful staging data extraction."""
        mock_spreadsheet = Mock()
        mock_worksheet = Mock()
        mock_df = pd.DataFrame({'A': [1, 2, 3]})
        
        mock_get_ss.return_value = mock_spreadsheet
        mock_get_ws.return_value = mock_worksheet
        mock_read.return_value = mock_df
        
        result = extract_staging_data('test_sheet_id')
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3


class TestLoadProcessedData:
    """Tests for load_processed_data function."""
    
    @patch('src.etl.get_spreadsheet')
    @patch('src.etl.create_worksheet_if_not_exists')
    @patch('src.etl.write_dataframe_to_worksheet')
    def test_load_processed_success(self, mock_write, mock_create_ws, mock_get_ss):
        """Test successful processed data load."""
        mock_spreadsheet = Mock()
        mock_worksheet = Mock()
        mock_get_ss.return_value = mock_spreadsheet
        mock_create_ws.return_value = mock_worksheet
        
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'AI Sentiment': ['Positive', 'Negative', 'Neutral']
        })
        
        load_processed_data(df, 'test_sheet_id')
        
        mock_get_ss.assert_called_once()
        mock_create_ws.assert_called_once()
        mock_write.assert_called_once()


class TestRunETLPipeline:
    """Tests for the complete ETL pipeline."""
    
    @patch('src.etl.extract_raw_data')
    @patch('src.etl.transform_staging_data')
    @patch('src.etl.load_staging_data')
    @patch('src.etl.prepare_processed_dataframe')
    @patch('src.etl.load_processed_data')
    @patch('src.etl.batch_analyze_reviews')
    def test_run_etl_without_llm(self, mock_batch, mock_load_proc, mock_prep, 
                                  mock_load_stage, mock_transform, mock_extract):
        """Test running ETL pipeline without LLM analysis."""
        from src.etl import run_etl_pipeline
        
        # Setup mocks
        mock_df = pd.DataFrame({'Review Text': ['Great!', 'Bad!']})
        mock_extract.return_value = mock_df
        mock_transform.return_value = mock_df
        mock_prep.return_value = mock_df
        
        # Execute
        result = run_etl_pipeline(run_llm_analysis=False)
        
        # Verify
        assert isinstance(result, pd.DataFrame)
        mock_batch.assert_not_called()
    
    @patch('src.etl.extract_raw_data')
    @patch('src.etl.transform_staging_data')
    @patch('src.etl.load_staging_data')
    @patch('src.etl.prepare_processed_dataframe')
    @patch('src.etl.load_processed_data')
    @patch('src.etl.batch_analyze_reviews')
    def test_run_etl_with_llm(self, mock_batch, mock_load_proc, mock_prep,
                              mock_load_stage, mock_transform, mock_extract):
        """Test running ETL pipeline with LLM analysis."""
        from src.etl import run_etl_pipeline
        
        # Setup mocks
        mock_df = pd.DataFrame({'Review Text': ['Great!', 'Bad!']})
        mock_extract.return_value = mock_df
        mock_transform.return_value = mock_df
        mock_prep.return_value = mock_df
        mock_batch.return_value = [
            {'sentiment': 'Positive', 'summary': 'Great', 'action_needed': 'No'},
            {'sentiment': 'Negative', 'summary': 'Bad', 'action_needed': 'Yes'}
        ]
        
        # Execute
        result = run_etl_pipeline(run_llm_analysis=True)
        
        # Verify
        assert isinstance(result, pd.DataFrame)
        mock_batch.assert_called_once()