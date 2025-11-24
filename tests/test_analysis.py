"""
Unit tests for src/analysis.py
"""
import pytest
import pandas as pd
from unittest.mock import Mock, patch
from src.analysis import (
    calculate_sentiment_by_class,
    get_top_sentiment_classes,
    generate_sentiment_analysis_report,
    create_sentiment_visualizations,
    export_analysis_to_csv
)


class TestCalculateSentimentByClass:
    """Tests for calculate_sentiment_by_class function."""
    
    def test_calculate_with_valid_data(self):
        """Test calculating sentiment distribution by class."""
        df = pd.DataFrame({
            'Class Name': ['Dress', 'Dress', 'Pants', 'Pants', 'Pants'],
            'AI Sentiment': ['Positive', 'Negative', 'Positive', 'Positive', 'Neutral']
        })
        
        result = calculate_sentiment_by_class(df)
        
        assert isinstance(result, pd.DataFrame)
        assert 'Class Name' in result.columns
        assert 'AI Sentiment' in result.columns
        assert 'count' in result.columns
        assert 'percentage' in result.columns
        assert len(result) > 0
    
    def test_calculate_filters_empty_sentiments(self):
        """Test that empty sentiments are filtered out."""
        df = pd.DataFrame({
            'Class Name': ['Dress', 'Dress', 'Pants'],
            'AI Sentiment': ['Positive', '', 'Negative']
        })
        
        result = calculate_sentiment_by_class(df)
        
        # Should only have 2 records (empty sentiment filtered out)
        assert len(result) == 2
    
    def test_calculate_percentages_correct(self):
        """Test that percentages are calculated correctly."""
        df = pd.DataFrame({
            'Class Name': ['Dress', 'Dress', 'Dress', 'Dress'],
            'AI Sentiment': ['Positive', 'Positive', 'Negative', 'Neutral']
        })
        
        result = calculate_sentiment_by_class(df)
        
        # Get the positive sentiment for Dress
        positive_row = result[(result['Class Name'] == 'Dress') & 
                             (result['AI Sentiment'] == 'Positive')]
        
        assert len(positive_row) == 1
        assert positive_row['percentage'].iloc[0] == 50.0  # 2 out of 4
    
    def test_calculate_with_single_class(self):
        """Test calculation with single class."""
        df = pd.DataFrame({
            'Class Name': ['Dress', 'Dress', 'Dress'],
            'AI Sentiment': ['Positive', 'Positive', 'Negative']
        })
        
        result = calculate_sentiment_by_class(df)
        
        assert 'Dress' in result['Class Name'].values
        assert len(result) > 0


class TestGetTopSentimentClasses:
    """Tests for get_top_sentiment_classes function."""
    
    def test_get_top_positive_class(self):
        """Test getting class with highest positive sentiment."""
        sentiment_stats = pd.DataFrame({
            'Class Name': ['Dress', 'Pants', 'Shirt'],
            'AI Sentiment': ['Positive', 'Positive', 'Positive'],
            'percentage': [80.0, 60.0, 70.0]
        })
        
        class_name, percentage = get_top_sentiment_classes(sentiment_stats, 'Positive')
        
        assert class_name == 'Dress'
        assert percentage == 80.0
    
    def test_get_top_with_empty_dataframe(self):
        """Test getting top class with empty data."""
        sentiment_stats = pd.DataFrame({
            'Class Name': [],
            'AI Sentiment': [],
            'percentage': []
        })
        
        class_name, percentage = get_top_sentiment_classes(sentiment_stats, 'Positive')
        
        assert class_name == 'None'
        assert percentage == 0.0
    
    def test_get_top_with_no_matching_sentiment(self):
        """Test when no rows match the sentiment."""
        sentiment_stats = pd.DataFrame({
            'Class Name': ['Dress'],
            'AI Sentiment': ['Positive'],
            'percentage': [80.0]
        })
        
        class_name, percentage = get_top_sentiment_classes(sentiment_stats, 'Negative')
        
        assert class_name == 'None'
        assert percentage == 0.0
    
    def test_get_top_negative_class(self):
        """Test getting class with highest negative sentiment."""
        sentiment_stats = pd.DataFrame({
            'Class Name': ['Dress', 'Pants'],
            'AI Sentiment': ['Negative', 'Negative'],
            'percentage': [25.0, 40.0]
        })
        
        class_name, percentage = get_top_sentiment_classes(sentiment_stats, 'Negative')
        
        assert class_name == 'Pants'
        assert percentage == 40.0


class TestGenerateSentimentAnalysisReport:
    """Tests for generate_sentiment_analysis_report function."""
    
    def test_generate_report_structure(self):
        """Test that report has correct structure."""
        df = pd.DataFrame({
            'Class Name': ['Dress', 'Dress', 'Pants'],
            'AI Sentiment': ['Positive', 'Negative', 'Positive']
        })
        
        report = generate_sentiment_analysis_report(df)
        
        assert 'overall_sentiment' in report
        assert 'by_class' in report
        assert 'top_classes' in report
        assert 'positive' in report['overall_sentiment']
        assert 'negative' in report['overall_sentiment']
        assert 'neutral' in report['overall_sentiment']
        assert 'total_reviews' in report['overall_sentiment']
    
    def test_generate_report_calculations(self):
        """Test report calculations are correct."""
        df = pd.DataFrame({
            'Class Name': ['Dress'] * 10,
            'AI Sentiment': ['Positive'] * 7 + ['Negative'] * 2 + ['Neutral'] * 1
        })
        
        report = generate_sentiment_analysis_report(df)
        
        assert report['overall_sentiment']['total_reviews'] == 10
        assert report['overall_sentiment']['positive'] == 70.0
        assert report['overall_sentiment']['negative'] == 20.0
        assert report['overall_sentiment']['neutral'] == 10.0
    
    def test_generate_report_top_classes(self):
        """Test that top classes are identified correctly."""
        df = pd.DataFrame({
            'Class Name': ['Dress', 'Dress', 'Pants', 'Pants'],
            'AI Sentiment': ['Positive', 'Positive', 'Negative', 'Negative']
        })
        
        report = generate_sentiment_analysis_report(df)
        
        assert 'highest_positive' in report['top_classes']
        assert 'highest_negative' in report['top_classes']
        assert 'highest_neutral' in report['top_classes']
    
    def test_generate_report_handles_empty_data(self):
        """Test report generation with minimal data."""
        df = pd.DataFrame({
            'Class Name': ['Dress'],
            'AI Sentiment': ['Positive']
        })
        
        report = generate_sentiment_analysis_report(df)
        
        assert report['overall_sentiment']['total_reviews'] == 1
        assert isinstance(report['by_class'], pd.DataFrame)


class TestCreateSentimentVisualizations:
    """Tests for create_sentiment_visualizations function."""
    
    @patch('src.analysis.plt.savefig')
    @patch('src.analysis.plt.close')
    @patch('src.analysis.os.makedirs')
    def test_create_visualizations_saves_files(self, mock_makedirs, mock_close, mock_savefig):
        """Test that visualizations are created and saved."""
        report = {
            'overall_sentiment': {
                'positive': 60.0,
                'negative': 25.0,
                'neutral': 15.0,
                'total_reviews': 100
            },
            'by_class': pd.DataFrame({
                'Class Name': ['Dress', 'Pants'],
                'AI Sentiment': ['Positive', 'Positive'],
                'percentage': [70.0, 60.0]
            }),
            'top_classes': {
                'highest_positive': {'class': 'Dress', 'percentage': 70.0},
                'highest_negative': {'class': 'Pants', 'percentage': 30.0},
                'highest_neutral': {'class': 'Shirt', 'percentage': 20.0}
            }
        }
        
        result = create_sentiment_visualizations(report, output_dir='test_viz')
        
        assert len(result) == 3
        assert mock_savefig.call_count == 3
        assert mock_close.call_count == 3
        mock_makedirs.assert_called_once()
    
    @patch('src.analysis.plt.savefig')
    @patch('src.analysis.plt.close')
    @patch('src.analysis.os.makedirs')
    def test_create_visualizations_output_paths(self, mock_makedirs, mock_close, mock_savefig):
        """Test that correct output paths are returned."""
        report = {
            'overall_sentiment': {
                'positive': 60.0,
                'negative': 25.0,
                'neutral': 15.0,
                'total_reviews': 100
            },
            'by_class': pd.DataFrame({
                'Class Name': ['Dress'],
                'AI Sentiment': ['Positive'],
                'percentage': [70.0]
            }),
            'top_classes': {
                'highest_positive': {'class': 'Dress', 'percentage': 70.0},
                'highest_negative': {'class': 'Pants', 'percentage': 30.0},
                'highest_neutral': {'class': 'Shirt', 'percentage': 20.0}
            }
        }
        
        result = create_sentiment_visualizations(report, output_dir='custom_dir')
        
        assert any('overall_sentiment_pie.png' in path for path in result)
        assert any('sentiment_by_class_bar.png' in path for path in result)
        assert any('top_classes_sentiment.png' in path for path in result)


class TestExportAnalysisToCSV:
    """Tests for export_analysis_to_csv function."""
    
    @patch('pandas.DataFrame.to_csv')
    def test_export_to_csv(self, mock_to_csv):
        """Test exporting analysis to CSV."""
        report = {
            'by_class': pd.DataFrame({
                'Class Name': ['Dress', 'Pants'],
                'AI Sentiment': ['Positive', 'Negative'],
                'percentage': [70.0, 30.0]
            })
        }
        
        result = export_analysis_to_csv(report, 'test_output.csv')
        
        assert result == 'test_output.csv'
        mock_to_csv.assert_called_once_with('test_output.csv', index=False)
    
    @patch('pandas.DataFrame.to_csv')
    def test_export_with_default_filename(self, mock_to_csv):
        """Test export with default filename."""
        report = {
            'by_class': pd.DataFrame({
                'Class Name': ['Dress'],
                'AI Sentiment': ['Positive'],
                'percentage': [100.0]
            })
        }
        
        result = export_analysis_to_csv(report)
        
        assert 'sentiment_analysis_report.csv' in result
        mock_to_csv.assert_called_once()