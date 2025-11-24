Introduction
This project implements an automated review-analysis pipeline that processes customer reviews from an e-commerce dataset. The system leverages Python, Google Sheets (via GSpread), and Groq's LLM API to:

Extract and clean customer review data
Perform sentiment analysis using AI (Positive, Negative, Neutral)
Generate concise summaries of reviews
Flag reviews requiring action (negative sentiment)
Produce analytical insights and visualizations
Maintain full idempotency and data integrity

Problem Statement
Manual review analysis is time-consuming and prone to inconsistency. This automated pipeline provides:

Consistent sentiment classification across all reviews
Automated summarization to quickly understand customer feedback
Actionable insights by identifying reviews requiring immediate attention
Data-driven analytics to understand sentiment patterns by product category


Dataset Overview
Dataset: Women's Clothing E-Commerce Reviews
Scope: First 200 rows from the original dataset
Key Fields:

Clothing ID: Unique identifier for clothing item
Age: Customer age
Title: Review title
Review Text: Full review text
Rating: Customer rating (1-5)
Recommended IND: Recommendation indicator
Positive Feedback Count: Number of positive votes
Division Name: Product division
Department Name: Product department
Class Name: Product class/category

Data Privacy: All brand names have been replaced with "retailer" for anonymization.

Project Structure
automated_review_analysis/
│
├── src/
│   ├── __init__.py
│   ├── utils.py              # Google Sheets utilities and helper functions
│   ├── etl.py                # ETL pipeline (Extract, Transform, Load)
│   ├── llm_analysis.py       # Groq LLM integration for sentiment analysis
│   └── analysis.py           # Analytics and visualization generation
│
├── tests/
│   ├── __init__.py
│   ├── test_utils.py         # Tests for utilities
│   ├── test_etl.py           # Tests for ETL pipeline
│   ├── test_llm_analysis.py  # Tests for LLM analysis
│   └── test_analysis.py      # Tests for analytics
│
├── visualizations/           # Generated charts and visualizations
│   ├── overall_sentiment_pie.png
│   ├── sentiment_by_class_bar.png
│   └── top_classes_sentiment.png
│
├── main.py                   # Main pipeline execution script
├── run_tests.py              # Test runner with coverage
├── extract_first_200_rows.py # Data extraction utility
├── setup_project.py          # Project setup script
├── requirements.txt          # Python dependencies
├── pytest.ini                # Pytest configuration
├── .env.example              # Environment variables template
├── .env                      # Environment variables (not in repo)
├── credentials.json          # Google service account (not in repo)
└── README.md                 # This file
Module Responsibilities
src/utils.py

Google Sheets authentication and connection
Worksheet operations (read, write, create)
Data validation and text cleaning utilities

src/etl.py

Extract data from raw_data worksheet
Transform and clean data for staging
Load data to staging and processed worksheets
Orchestrate the complete ETL pipeline

src/llm_analysis.py

Groq API client initialization
Review sentiment analysis (Positive/Negative/Neutral)
Review summarization
Batch processing with retry logic

src/analysis.py

Sentiment distribution calculations by clothing class
Statistical analysis and reporting
Visualization generation (pie charts, bar charts)
CSV export functionality


Setup and Installation- This also shows reproducibility

Prerequisites

Python 3.8 or higher
Google Cloud Platform account
Groq API account
Git (optional)

Step 1: Clone or Download the Project
bashgit clone <repository-url>
cd automated_review_analysis
Step 2: Install Dependencies
bashpip install -r requirements.txt
Step 3: Set Up Google Sheets API

Go to Google Cloud Console
Create a new project or select existing one
Enable Google Sheets API and Google Drive API
Create a Service Account:

Go to "Credentials" → "Create Credentials" → "Service Account"
Download the JSON key file
Save it as credentials.json in the project root


Share your Google Sheet with the service account email (found in credentials.json)

Step 4: Set Up Groq API

Visit GroqCloud Console
Sign up or log in
Generate an API key
Copy the API key

Step 5: Configure Environment Variables
Create a .env file in the project root:
bashGROQ_API_KEY=your_groq_api_key_here
GOOGLE_SHEET_ID=your_google_sheet_id_here
Note: The Google Sheet ID is found in the URL:
https://docs.google.com/spreadsheets/d/{SHEET_ID}/edit
Step 6: Prepare Your Data

Download the Women's Clothing E-Commerce Reviews dataset
Extract the first 200 rows:

bash   python extract_first_200_rows.py your_dataset.csv  or just copy from local file and paste into your google worksheet

Upload first_200_rows.csv to a new Google Sheet
Rename the worksheet to raw_data
Protect the worksheet: Right-click tab → "Protect sheet" → Set permissions



Run the Complete Pipeline
bashpython main.py

This will:

Extract data from raw_data worksheet
Clean and transform data
Create staging worksheet with cleaned data
Analyze each review using Groq LLM
Create processed worksheet with AI results
Generate sentiment analysis report
Create visualizations (saved to visualizations/)
Export detailed analysis to CSV

Run Tests
run tests directly with pytest:
bashpython -m pytest tests/ -v --cov=src --cov-report=term-missing
Test Coverage: ≥70% across all modules

Screenshots:
### Before: Raw Data Sheet
![Raw Data Sheet](screenshots/raw_data_before.png)

### After: Processed Sheet
![Processed Sheet](screenshots/processed_after.png)

Analysis Summary
Based on the analysis of 200 customer reviews:

 Overall Sentiment Distribution:
  Positive: 64.58%
  Negative: 21.88%
  Neutral:  13.54%
  Total Reviews Analyzed: 192

 Top Classes by Sentiment:
  Highest Positive Sentiment: Sleep (100.00%)
  Highest Negative Sentiment: Swim (100.00%)
  Highest Neutral Sentiment:  Dresses (25.00%)

Detailed Breakdown by Class:
Class Name AI Sentiment  count  total  percentage
   Blouses     Negative     11     43       25.58
   Blouses      Neutral      7     43       16.28
   Blouses     Positive     25     43       58.14
   Dresses     Negative     11     36       30.56
   Dresses      Neutral      9     36       25.00
   Dresses     Positive     16     36       44.44
Fine gauge     Negative      3     14       21.43
Fine gauge      Neutral      1     14        7.14
Fine gauge     Positive     10     14       71.43
 Intimates     Negative      2      5       40.00
 Intimates      Neutral      1      5       20.00
 Intimates     Positive      2      5       40.00
   Jackets      Neutral      1      4       25.00
   Jackets     Positive      3      4       75.00
     Knits     Negative      7     40       17.50
     Knits      Neutral      2     40        5.00
     Knits     Positive     31     40       77.50
    Lounge     Negative      1      5       20.00
    Lounge     Positive      4      5       80.00
 Outerwear     Negative      1      6       16.67
 Outerwear      Neutral      1      6       16.67
 Outerwear     Positive      4      6       66.67
     Pants     Negative      1      7       14.29
     Pants      Neutral      1      7       14.29
     Pants     Positive      5      7       71.43
    Skirts     Negative      1     21        4.76
    Skirts      Neutral      3     21       14.29
    Skirts     Positive     17     21       80.95
     Sleep     Positive      4      4      100.00
  Sweaters     Negative      3      5       60.00
  Sweaters     Positive      2      5       40.00
      Swim     Negative      1      1      100.00
     Trend     Positive      1      1      100.00




Security Notes
Files NOT included in repository (for security):

.env - Contains API keys
credentials.json - Contains Google service account credentials
first_200_rows.csv - Contains dataset

Use .env.example as a template for your .env file.

Pipeline Idempotency
The pipeline is designed to be fully idempotent:

Running multiple times with same input produces same output
Protected worksheets (raw_data) are never modified
Staging and processed worksheets are recreated cleanly
No data corruption on re-runs
Graceful handling of failures and retries


Libraries and tech used

Python 3.13: Core programming language
pandas: Data manipulation and analysis
gspread: Google Sheets API integration
Groq API: LLM for sentiment analysis (model: llama-3.3-70b-versatile)
matplotlib: Data visualization
pytest: Testing framework
pytest-cov: Code coverage reporting
python-dotenv: Environment variable management


