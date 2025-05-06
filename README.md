# CSV & OpenAI Integration

A Streamlit-based application for comprehensive data analysis, cleaning, visualization, and modeling with OpenAI integration.

## Overview

This application provides an intuitive interface for data scientists and analysts to work with CSV data files. It leverages OpenAI's API to generate intelligent insights, cleaning suggestions, and visualizations based on the data content.

Key features:
- 📊 Intelligent data analysis with OpenAI integration
- 🧹 Smart data cleaning with automated code generation
- 📈 Advanced data visualization with customizable options
- 🤖 Machine learning model building and evaluation
- 📑 Export capabilities for reports and findings

## Requirements

This application requires Python 3.8+ and the following libraries:

```
streamlit
pandas
matplotlib
seaborn
python-dotenv
openai
reportlab
markdown
beautifulsoup4
numpy
altair
scikit-learn
```

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/csv-openai-integration.git
   cd csv-openai-integration
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Usage

1. Start the application:
   ```
   streamlit run app.py
   ```

2. The application will open in your default web browser.

3. Upload a CSV file using the file uploader.

4. Navigate through the different tabs to:
   - Preview and understand your data
   - Clean and preprocess your data
   - Generate visualizations and insights
   - Build and evaluate machine learning models

## Project Structure

```
.
├── app.py                     # Main application file
├── requirements.txt           # Project dependencies
├── components/                # Streamlit UI components
│   ├── __init__.py
│   ├── data_analysis_ui.py    # Analysis UI components
│   ├── data_cleaning_ui.py    # Cleaning UI components
│   ├── data_modeling_ui.py    # Modeling UI components
│   ├── data_preview.py        # Data preview components
│   └── data_upload.py         # File upload handling
├── services/                  # External service integrations
│   ├── __init__.py
│   └── openai_service.py      # OpenAI API integration
└── utils/                     # Utility functions
    ├── __init__.py
    ├── data_analysis.py       # Data analysis functions
    ├── data_cleaning.py       # Data cleaning functions
    ├── data_modeling.py       # Machine learning functions
    ├── data_visualization.py  # Visualization functions
    └── file_handlers.py       # File handling utilities
```

## Features in Detail

### Data Preview
- Automatically analyzes and displays dataset statistics
- Identifies null values and patterns
- Generates an intelligent description of the dataset

### Data Cleaning
- Suggests cleaning steps based on data quality issues
- Generates ready-to-use Python code for data cleaning
- Provides before/after comparison of cleaning operations

### Data Analysis
- Creates insightful visualizations based on data characteristics
- Generates narrative insights explaining key findings
- Allows custom analysis based on user instructions

### Machine Learning
- Suggests appropriate models based on dataset characteristics
- Builds and evaluates models with proper train/test splits
- Provides model metrics and visualizations of model performance

### Report Generation
- Creates professional PDF reports of analysis findings
- Includes visualizations, insights, and model results
- Provides downloadable artifacts for sharing

## Fallback Functionality

The application includes fallback mechanisms for when the OpenAI API is unavailable or reaches rate limits. Basic functionality will continue to work even without API connectivity.
