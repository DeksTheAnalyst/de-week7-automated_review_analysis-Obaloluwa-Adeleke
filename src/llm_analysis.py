"""
LLM analysis functions using Groq API for sentiment analysis and summarization.
"""
import os
from dotenv import load_dotenv
from groq import Groq
from typing import Dict, Tuple, Optional
import time

# Load environment variables
load_dotenv()


def get_groq_client() -> Groq:
    """
    Create and return a Groq client instance.
    
    Returns:
        Groq: Authenticated Groq client
        
    Raises:
        ValueError: If GROQ_API_KEY is not set
    """
    api_key = os.getenv('GROQ_API_KEY')
    
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")
    
    return Groq(api_key=api_key)


def analyze_review_with_llm(
    review_text: str,
    model: str = "llama-3.3-70b-versatile",
    max_retries: int = 3
) -> Dict[str, str]:
    """
    Analyze a review using Groq LLM to extract sentiment and summary.
    
    Args:
        review_text: The review text to analyze
        model: The Groq model to use
        max_retries: Maximum number of retry attempts on failure
        
    Returns:
        Dict with keys 'sentiment' and 'summary'
    """
    # Handle empty reviews
    if not review_text or str(review_text).strip() == '':
        return {
            'sentiment': '',
            'summary': ''
        }
    
    client = get_groq_client()
    
    # Create the prompt for the LLM
    prompt = f"""Analyze the following customer review and provide:
1. Sentiment: Classify as exactly one of: Positive, Negative, or Neutral
2. Summary: Provide a one-sentence summary of the review

Review: "{review_text}"

Respond in the following format:
Sentiment: [Positive/Negative/Neutral]
Summary: [One sentence summary]

If the review is too short to summarize meaningfully, just repeat the original text as the summary."""

    # Retry logic for API calls
    for attempt in range(max_retries):
        try:
            # Call the Groq API
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=model,
                temperature=0.3,  # Lower temperature for more consistent results
                max_tokens=150
            )
            
            # Extract the response
            response_text = chat_completion.choices[0].message.content.strip()
            
            # Parse the response
            sentiment, summary = parse_llm_response(response_text, review_text)
            
            return {
                'sentiment': sentiment,
                'summary': summary
            }
            
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  API call failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                time.sleep(1)  # Wait before retrying
                continue
            else:
                print(f"  Failed to analyze review after {max_retries} attempts: {str(e)}")
                return {
                    'sentiment': 'Error',
                    'summary': 'Failed to analyze'
                }


def parse_llm_response(response_text: str, original_review: str) -> Tuple[str, str]:
    """
    Parse the LLM response to extract sentiment and summary.
    
    Args:
        response_text: The raw response from the LLM
        original_review: The original review text (fallback for summary)
        
    Returns:
        Tuple of (sentiment, summary)
    """
    sentiment = "Neutral"
    summary = original_review
    
    lines = response_text.split('\n')
    
    for line in lines:
        line = line.strip()
        
        # Extract sentiment
        if line.startswith('Sentiment:'):
            sentiment_value = line.replace('Sentiment:', '').strip()
            # Validate sentiment
            if sentiment_value in ['Positive', 'Negative', 'Neutral']:
                sentiment = sentiment_value
            elif 'positive' in sentiment_value.lower():
                sentiment = 'Positive'
            elif 'negative' in sentiment_value.lower():
                sentiment = 'Negative'
            else:
                sentiment = 'Neutral'
        
        # Extract summary
        elif line.startswith('Summary:'):
            summary = line.replace('Summary:', '').strip()
            # If summary is empty, use original review
            if not summary:
                summary = original_review
    
    return sentiment, summary


def determine_action_needed(sentiment: str) -> str:
    """
    Determine if action is needed based on sentiment.
    
    Args:
        sentiment: The sentiment classification (Positive/Negative/Neutral)
        
    Returns:
        str: 'Yes' if action needed, 'No' otherwise
    """
    if sentiment == 'Negative':
        return 'Yes'
    return 'No'


def batch_analyze_reviews(
    reviews: list,
    model: str = "llama-3.3-70b-versatile",
    show_progress: bool = True
) -> list:
    """
    Analyze multiple reviews in batch.
    
    Args:
        reviews: List of review texts
        model: The Groq model to use
        show_progress: Whether to show progress indicators
        
    Returns:
        List of dicts with sentiment, summary, and action_needed for each review
    """
    results = []
    total = len(reviews)
    
    for idx, review in enumerate(reviews, 1):
        if show_progress and idx % 10 == 0:
            print(f"  Processing review {idx}/{total}...")
        
        # Analyze the review
        analysis = analyze_review_with_llm(review, model)
        
        # Determine action needed
        action_needed = determine_action_needed(analysis['sentiment'])
        
        results.append({
            'sentiment': analysis['sentiment'],
            'summary': analysis['summary'],
            'action_needed': action_needed
        })
        
        # Small delay to avoid rate limiting
        if idx % 20 == 0:
            time.sleep(0.5)
    
    if show_progress:
        print(f"Completed analyzing {total} reviews")
    
    return results