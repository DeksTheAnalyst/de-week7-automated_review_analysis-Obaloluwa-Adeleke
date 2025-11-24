"""
Unit tests for src/llm_analysis.py
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.llm_analysis import (
    get_groq_client,
    parse_llm_response,
    determine_action_needed,
    analyze_review_with_llm,
    batch_analyze_reviews
)


class TestGetGroqClient:
    """Tests for get_groq_client function."""
    
    @patch.dict('os.environ', {'GROQ_API_KEY': 'test_key'})
    @patch('src.llm_analysis.Groq')
    def test_get_client_with_api_key(self, mock_groq):
        """Test getting Groq client with valid API key."""
        mock_client = Mock()
        mock_groq.return_value = mock_client
        
        client = get_groq_client()
        
        assert client == mock_client
        mock_groq.assert_called_once_with(api_key='test_key')
    
    @patch.dict('os.environ', {}, clear=True)
    def test_get_client_without_api_key(self):
        """Test that missing API key raises error."""
        with pytest.raises(ValueError, match="GROQ_API_KEY not found"):
            get_groq_client()


class TestParseLLMResponse:
    """Tests for parse_llm_response function."""
    
    def test_parse_valid_response(self):
        """Test parsing a valid LLM response."""
        response = "Sentiment: Positive\nSummary: Great product, highly recommended."
        original = "I love this product!"
        
        sentiment, summary = parse_llm_response(response, original)
        
        assert sentiment == "Positive"
        assert summary == "Great product, highly recommended."
    
    def test_parse_negative_sentiment(self):
        """Test parsing negative sentiment."""
        response = "Sentiment: Negative\nSummary: Poor quality, not worth it."
        original = "Bad product"
        
        sentiment, summary = parse_llm_response(response, original)
        
        assert sentiment == "Negative"
        assert summary == "Poor quality, not worth it."
    
    def test_parse_neutral_sentiment(self):
        """Test parsing neutral sentiment."""
        response = "Sentiment: Neutral\nSummary: It's okay, nothing special."
        original = "Average product"
        
        sentiment, summary = parse_llm_response(response, original)
        
        assert sentiment == "Neutral"
        assert summary == "It's okay, nothing special."
    
    def test_parse_lowercase_sentiment(self):
        """Test parsing with lowercase sentiment."""
        response = "Sentiment: positive\nSummary: Good."
        original = "Nice"
        
        sentiment, summary = parse_llm_response(response, original)
        
        assert sentiment == "Positive"
    
    def test_parse_missing_sentiment(self):
        """Test parsing when sentiment is missing."""
        response = "Summary: Some summary text."
        original = "Original review"
        
        sentiment, summary = parse_llm_response(response, original)
        
        assert sentiment == "Neutral"  # Default
        assert summary == "Some summary text."
    
    def test_parse_missing_summary(self):
        """Test parsing when summary is missing."""
        response = "Sentiment: Positive"
        original = "Original review text"
        
        sentiment, summary = parse_llm_response(response, original)
        
        assert sentiment == "Positive"
        assert summary == "Original review text"  # Falls back to original
    
    def test_parse_empty_summary(self):
        """Test parsing with empty summary."""
        response = "Sentiment: Positive\nSummary: "
        original = "Original text"
        
        sentiment, summary = parse_llm_response(response, original)
        
        assert summary == "Original text"
    
    def test_parse_invalid_sentiment_value(self):
        """Test parsing with invalid sentiment value."""
        response = "Sentiment: Amazing\nSummary: Great"
        original = "Review"
        
        sentiment, summary = parse_llm_response(response, original)
        
        assert sentiment == "Neutral"  # Default for invalid


class TestDetermineActionNeeded:
    """Tests for determine_action_needed function."""
    
    def test_action_needed_for_negative(self):
        """Test that negative sentiment requires action."""
        result = determine_action_needed("Negative")
        assert result == "Yes"
    
    def test_no_action_for_positive(self):
        """Test that positive sentiment doesn't require action."""
        result = determine_action_needed("Positive")
        assert result == "No"
    
    def test_no_action_for_neutral(self):
        """Test that neutral sentiment doesn't require action."""
        result = determine_action_needed("Neutral")
        assert result == "No"
    
    def test_no_action_for_empty(self):
        """Test handling of empty sentiment."""
        result = determine_action_needed("")
        assert result == "No"


class TestAnalyzeReviewWithLLM:
    """Tests for analyze_review_with_llm function."""
    
    def test_analyze_empty_review(self):
        """Test analyzing empty review returns empty results."""
        result = analyze_review_with_llm("")
        
        assert result['sentiment'] == ''
        assert result['summary'] == ''
    
    def test_analyze_none_review(self):
        """Test analyzing None review returns empty results."""
        result = analyze_review_with_llm(None)
        
        assert result['sentiment'] == ''
        assert result['summary'] == ''
    
    @patch('src.llm_analysis.get_groq_client')
    def test_analyze_valid_review(self, mock_get_client):
        """Test analyzing a valid review."""
        # Setup mock
        mock_client = Mock()
        mock_response = Mock()
        mock_message = Mock()
        mock_choice = Mock()
        
        mock_message.content = "Sentiment: Positive\nSummary: Great product!"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_client
        
        result = analyze_review_with_llm("I love this product!")
        
        assert result['sentiment'] == 'Positive'
        assert result['summary'] == 'Great product!'
    
    @patch('src.llm_analysis.get_groq_client')
    @patch('src.llm_analysis.time.sleep')
    def test_analyze_with_retry(self, mock_sleep, mock_get_client):
        """Test that analysis retries on failure."""
        mock_client = Mock()
        
        # Fail twice, then succeed
        mock_client.chat.completions.create.side_effect = [
            Exception("API Error"),
            Exception("API Error"),
            Mock(choices=[Mock(message=Mock(content="Sentiment: Positive\nSummary: Good"))])
        ]
        
        mock_get_client.return_value = mock_client
        
        result = analyze_review_with_llm("Test review", max_retries=3)
        
        assert result['sentiment'] == 'Positive'
        assert mock_client.chat.completions.create.call_count == 3
    
    @patch('src.llm_analysis.get_groq_client')
    def test_analyze_max_retries_exceeded(self, mock_get_client):
        """Test that analysis returns error after max retries."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_get_client.return_value = mock_client
        
        result = analyze_review_with_llm("Test review", max_retries=2)
        
        assert result['sentiment'] == 'Error'
        assert result['summary'] == 'Failed to analyze'


class TestBatchAnalyzeReviews:
    """Tests for batch_analyze_reviews function."""
    
    @patch('src.llm_analysis.analyze_review_with_llm')
    @patch('src.llm_analysis.time.sleep')
    def test_batch_analyze_multiple_reviews(self, mock_sleep, mock_analyze):
        """Test batch analyzing multiple reviews."""
        mock_analyze.side_effect = [
            {'sentiment': 'Positive', 'summary': 'Great!'},
            {'sentiment': 'Negative', 'summary': 'Bad!'},
            {'sentiment': 'Neutral', 'summary': 'Okay.'}
        ]
        
        reviews = ["Great product", "Terrible", "It's okay"]
        
        results = batch_analyze_reviews(reviews, show_progress=False)
        
        assert len(results) == 3
        assert results[0]['sentiment'] == 'Positive'
        assert results[1]['sentiment'] == 'Negative'
        assert results[2]['sentiment'] == 'Neutral'
        assert results[0]['action_needed'] == 'No'
        assert results[1]['action_needed'] == 'Yes'
    
    @patch('src.llm_analysis.analyze_review_with_llm')
    def test_batch_analyze_empty_list(self, mock_analyze):
        """Test batch analyzing empty list."""
        results = batch_analyze_reviews([], show_progress=False)
        
        assert len(results) == 0
        mock_analyze.assert_not_called()
    
    @patch('src.llm_analysis.analyze_review_with_llm')
    def test_batch_analyze_single_review(self, mock_analyze):
        """Test batch analyzing single review."""
        mock_analyze.return_value = {'sentiment': 'Positive', 'summary': 'Great'}
        
        results = batch_analyze_reviews(["One review"], show_progress=False)
        
        assert len(results) == 1
        assert results[0]['sentiment'] == 'Positive'
    
    @patch('src.llm_analysis.analyze_review_with_llm')
    @patch('src.llm_analysis.time.sleep')
    def test_batch_analyze_with_rate_limiting(self, mock_sleep, mock_analyze):
        """Test that rate limiting sleep is called."""
        mock_analyze.return_value = {'sentiment': 'Positive', 'summary': 'Good'}
        
        # Create 25 reviews to trigger rate limiting (at idx=20)
        reviews = ["Review"] * 25
        
        results = batch_analyze_reviews(reviews, show_progress=False)
        
        assert len(results) == 25
        # Should sleep once after 20 reviews
        assert mock_sleep.call_count >= 1