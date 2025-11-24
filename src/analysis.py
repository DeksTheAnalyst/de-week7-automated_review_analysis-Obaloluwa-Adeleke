"""
Analysis functions for generating insights from processed review data.
"""
import pandas as pd
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import os


def calculate_sentiment_by_class(
    df: pd.DataFrame,
    class_column: str = "Class Name",
    sentiment_column: str = "AI Sentiment"
) -> pd.DataFrame:
    """
    Calculate sentiment distribution by clothing class.
    
    Args:
        df: Processed DataFrame with sentiment analysis
        class_column: Name of the column containing clothing class
        sentiment_column: Name of the column containing sentiment
        
    Returns:
        pd.DataFrame: Sentiment counts and percentages by class
    """
    # Filter out empty sentiments
    df_filtered = df[df[sentiment_column].isin(['Positive', 'Negative', 'Neutral'])].copy()
    
    # Group by class and sentiment
    sentiment_counts = df_filtered.groupby([class_column, sentiment_column]).size().reset_index(name='count')
    
    # Calculate total reviews per class
    class_totals = df_filtered.groupby(class_column).size().reset_index(name='total')
    
    # Merge to get percentages
    sentiment_stats = sentiment_counts.merge(class_totals, on=class_column)
    sentiment_stats['percentage'] = (sentiment_stats['count'] / sentiment_stats['total'] * 100).round(2)
    
    return sentiment_stats


def get_top_sentiment_classes(
    sentiment_stats: pd.DataFrame,
    sentiment: str,
    class_column: str = "Class Name",
    sentiment_column: str = "AI Sentiment"
) -> Tuple[str, float]:
    """
    Get the clothing class with highest percentage of a given sentiment.
    
    Args:
        sentiment_stats: DataFrame with sentiment statistics
        sentiment: The sentiment to find (Positive/Negative/Neutral)
        class_column: Name of the class column
        sentiment_column: Name of the sentiment column
        
    Returns:
        Tuple of (class_name, percentage)
    """
    filtered = sentiment_stats[sentiment_stats[sentiment_column] == sentiment]
    
    if len(filtered) == 0:
        return ("None", 0.0)
    
    top_row = filtered.loc[filtered['percentage'].idxmax()]
    return (top_row[class_column], top_row['percentage'])


def generate_sentiment_analysis_report(
    df: pd.DataFrame,
    class_column: str = "Class Name",
    sentiment_column: str = "AI Sentiment"
) -> Dict:
    """
    Generate a comprehensive sentiment analysis report.
    
    Args:
        df: Processed DataFrame with sentiment analysis
        class_column: Name of the column containing clothing class
        sentiment_column: Name of the column containing sentiment
        
    Returns:
        Dict containing analysis results and insights
    """
    print("\n" + "=" * 60)
    print("Generating Sentiment Analysis Report")
    print("=" * 60)
    
    # Calculate sentiment statistics
    sentiment_stats = calculate_sentiment_by_class(df, class_column, sentiment_column)
    
    # Overall sentiment distribution
    overall_sentiment = df[df[sentiment_column].isin(['Positive', 'Negative', 'Neutral'])][sentiment_column].value_counts()
    total_reviews = len(df[df[sentiment_column].isin(['Positive', 'Negative', 'Neutral'])])
    
    overall_percentages = (overall_sentiment / total_reviews * 100).round(2)
    
    # Get top classes for each sentiment
    top_positive_class, top_positive_pct = get_top_sentiment_classes(sentiment_stats, 'Positive', class_column, sentiment_column)
    top_negative_class, top_negative_pct = get_top_sentiment_classes(sentiment_stats, 'Negative', class_column, sentiment_column)
    top_neutral_class, top_neutral_pct = get_top_sentiment_classes(sentiment_stats, 'Neutral', class_column, sentiment_column)
    
    # Compile report
    report = {
        'overall_sentiment': {
            'positive': overall_percentages.get('Positive', 0),
            'negative': overall_percentages.get('Negative', 0),
            'neutral': overall_percentages.get('Neutral', 0),
            'total_reviews': total_reviews
        },
        'by_class': sentiment_stats,
        'top_classes': {
            'highest_positive': {
                'class': top_positive_class,
                'percentage': top_positive_pct
            },
            'highest_negative': {
                'class': top_negative_class,
                'percentage': top_negative_pct
            },
            'highest_neutral': {
                'class': top_neutral_class,
                'percentage': top_neutral_pct
            }
        }
    }
    
    # Print report
    print("\n Overall Sentiment Distribution:")
    print(f"  Positive: {report['overall_sentiment']['positive']:.2f}%")
    print(f"  Negative: {report['overall_sentiment']['negative']:.2f}%")
    print(f"  Neutral:  {report['overall_sentiment']['neutral']:.2f}%")
    print(f"  Total Reviews Analyzed: {report['overall_sentiment']['total_reviews']}")
    
    print("\n Top Classes by Sentiment:")
    print(f"  Highest Positive Sentiment: {top_positive_class} ({top_positive_pct:.2f}%)")
    print(f"  Highest Negative Sentiment: {top_negative_class} ({top_negative_pct:.2f}%)")
    print(f"  Highest Neutral Sentiment:  {top_neutral_class} ({top_neutral_pct:.2f}%)")
    
    print("\n📋 Detailed Breakdown by Class:")
    print(sentiment_stats.to_string(index=False))
    
    return report


def create_sentiment_visualizations(
    report: Dict,
    output_dir: str = "visualizations"
) -> List[str]:
    """
    Create visualizations for the sentiment analysis report.
    
    Args:
        report: The analysis report dictionary
        output_dir: Directory to save visualizations
        
    Returns:
        List of file paths to created visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    saved_files = []
    
    # 1. Overall Sentiment Pie Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sentiments = ['Positive', 'Negative', 'Neutral']
    percentages = [
        report['overall_sentiment']['positive'],
        report['overall_sentiment']['negative'],
        report['overall_sentiment']['neutral']
    ]
    
    colors = ['#28a745', '#dc3545', '#6c757d']
    
    ax.pie(percentages, labels=sentiments, autopct='%1.1f%%', colors=colors, startangle=90)
    ax.set_title('Overall Sentiment Distribution', fontsize=16, fontweight='bold')
    
    pie_chart_path = os.path.join(output_dir, 'overall_sentiment_pie.png')
    plt.savefig(pie_chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    saved_files.append(pie_chart_path)
    
    # 2. Sentiment by Class Bar Chart
    sentiment_stats = report['by_class']
    
    # Pivot the data for easier plotting
    pivot_data = sentiment_stats.pivot(index='Class Name', columns='AI Sentiment', values='percentage').fillna(0)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    pivot_data.plot(kind='bar', ax=ax, color=['#28a745', '#dc3545', '#6c757d'], width=0.8)
    
    ax.set_title('Sentiment Distribution by Clothing Class', fontsize=16, fontweight='bold')
    ax.set_xlabel('Clothing Class', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.legend(title='Sentiment', fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    bar_chart_path = os.path.join(output_dir, 'sentiment_by_class_bar.png')
    plt.savefig(bar_chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    saved_files.append(bar_chart_path)
    
    # 3. Top Classes Horizontal Bar Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    top_classes = report['top_classes']
    classes = [
        top_classes['highest_positive']['class'],
        top_classes['highest_negative']['class'],
        top_classes['highest_neutral']['class']
    ]
    percentages = [
        top_classes['highest_positive']['percentage'],
        top_classes['highest_negative']['percentage'],
        top_classes['highest_neutral']['percentage']
    ]
    colors = ['#28a745', '#dc3545', '#6c757d']
    labels = ['Highest Positive', 'Highest Negative', 'Highest Neutral']
    
    bars = ax.barh(labels, percentages, color=colors)
    
    # Add percentage labels on bars
    for i, (bar, pct) in enumerate(zip(bars, percentages)):
        ax.text(pct + 1, i, f'{pct:.1f}%', va='center', fontweight='bold')
    
    ax.set_xlabel('Percentage (%)', fontsize=12)
    ax.set_title('Top Classes by Sentiment Type', fontsize=16, fontweight='bold')
    ax.set_xlim(0, max(percentages) + 10)
    
    # Add class names as secondary labels
    for i, cls in enumerate(classes):
        ax.text(0.5, i, cls, va='center', ha='left', fontsize=9, style='italic')
    
    plt.tight_layout()
    top_classes_path = os.path.join(output_dir, 'top_classes_sentiment.png')
    plt.savefig(top_classes_path, dpi=300, bbox_inches='tight')
    plt.close()
    saved_files.append(top_classes_path)
    
    print(f"\n Created {len(saved_files)} visualizations in '{output_dir}/' directory")
    
    return saved_files


def export_analysis_to_csv(
    report: Dict,
    output_file: str = "sentiment_analysis_report.csv"
) -> str:
    """
    Export the analysis report to a CSV file.
    
    Args:
        report: The analysis report dictionary
        output_file: Path to output CSV file
        
    Returns:
        str: Path to the saved file
    """
    # Export the detailed breakdown
    sentiment_stats = report['by_class']
    sentiment_stats.to_csv(output_file, index=False)
    
    print(f"\n Exported detailed analysis to '{output_file}'")
    
    return output_file