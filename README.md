# CSV & OpenAI Integration

A Streamlit-based application for comprehensive data analysis, cleaning, visualization, and modeling with OpenAI integration.

## Overview

This application provides an intuitive interface for data scientists and analysts to work with CSV data files. It leverages OpenAI's API to generate intelligent insights, cleaning suggestions, and visualizations based on the data content.

Key features:
- 📊 Intelligent data analysis with OpenAI integration
- 🧹 Smart data cleaning with automated code generation
- 📈 Advanced data visualization with customizable options
- 🤖 Machine learning model building and evaluation
- 📑 Export capabilities for PDF reports

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

4. The application will guide you through the data analysis process:
   - **Data Preview**: View your data and see automatic analysis of dataset statistics
   - **Data Cleaning**: Get AI-generated suggestions for cleaning your data
   - **Data Analysis**: Generate visualizations and insights about your data
   - **Machine Learning**: Train and evaluate ML models appropriate for your dataset
   - **Export Reports**: Generate professional PDF reports of your findings

## Features in Detail

### Data Preview
- Automatically analyzes and displays dataset statistics
- Identifies null values and patterns
- Generates an intelligent description of the dataset
- Provides comprehensive null value analysis with visualizations

### Data Cleaning
- Suggests data cleaning steps based on dataset quality issues
- Generates ready-to-use Python code for data cleaning operations
- Provides before/after comparison of cleaning operations
- Interactive UI for approving or discarding changes
- Summary of all cleaning steps applied

### Data Analysis
- Creates insightful visualizations based on data characteristics
- Generates narrative insights explaining key findings
- Allows custom analysis based on user instructions
- Supports exporting visualizations and insights as PDF reports

### Machine Learning
- Suggests appropriate models based on dataset characteristics
- Builds and evaluates models with proper train/test splits
- Provides comprehensive model metrics and performance visualizations
- Feature importance analysis for model interpretability
- Model results export as PDF reports

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

## Module Descriptions

### App.py
The main entry point of the application that handles session state management, file uploads, and navigates between different views of the application.

### Components
- **data_upload.py**: Handles the file upload process, including chunking for large files
- **data_preview.py**: Displays dataset previews and null value analysis
- **data_cleaning_ui.py**: UI for data cleaning operations with before/after comparisons
- **data_analysis_ui.py**: UI for data analysis and visualization generation
- **data_modeling_ui.py**: UI for machine learning model training and evaluation

### Services
- **openai_service.py**: Handles communication with OpenAI API for generating data descriptions, cleaning suggestions, and other AI-powered features

### Utils
- **data_analysis.py**: Functions for analyzing datasets and generating statistics
- **data_cleaning.py**: Functions for generating and executing data cleaning code
- **data_modeling.py**: Functions for machine learning model suggestion, generation, and evaluation
- **data_visualization.py**: Functions for creating data visualizations and PDF reports
- **file_handlers.py**: Utilities for file operations and safe DataFrame handling
