"""
Main script to run the automated review analysis pipeline.
"""
import sys
from src.etl import run_etl_pipeline
from src.analysis import (
    generate_sentiment_analysis_report,
    create_sentiment_visualizations,
    export_analysis_to_csv
)


def main():
    """
    Main entry point for the review analysis pipeline.
    """
    try:
        print("Starting Automated Review Analysis Pipeline")
        print()
        
        # Run ETL pipeline with LLM analysis
        df_processed = run_etl_pipeline(run_llm_analysis=True)
        
        # Generate analysis report
        report = generate_sentiment_analysis_report(df_processed)
        
        # Create visualizations
        viz_files = create_sentiment_visualizations(report)
        
        # Export to CSV
        csv_file = export_analysis_to_csv(report)
        
        print()
        print("=" * 60)
        print("Pipeline Execution Completed Successfully!")
        print("=" * 60)
        print(f"Processed {len(df_processed)} reviews")
        print(f"Created {len(viz_files)} visualizations")
        print(f"Exported analysis to {csv_file}")
        print()
        print("Generated Files:")
        for viz_file in viz_files:
            print(f"  - {viz_file}")
        print(f"  - {csv_file}")
        print()
        print(" Check your Google Sheet for updated 'processed' worksheet")
        
    except Exception as e:
        print(f" Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()