import os
import time
import re
from openai import OpenAI
import pandas as pd
import streamlit as st


def call_openai_api(prompt, system_prompt, retry_count=3, retry_delay=2):
    """Make OpenAI API calls with retry logic"""
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return None, "Error: OpenAI API key not found in environment variables"

    client = OpenAI(api_key=api_key)

    for attempt in range(retry_count):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )

            content = response.choices[0].message.content

            # If we need to extract code from the response
            if "ONLY respond with executable Python code" in system_prompt:
                # Remove any Markdown code blocks if present
                content = re.sub(r'```python\s*', '', content)
                content = re.sub(r'```\s*', '', content)

            return content, None  # Return the content and None for error

        except Exception as e:
            if "insufficient_quota" in str(e):
                return None, "OpenAI API quota exceeded. Please check your billing details."
            elif "rate_limit" in str(e) and attempt < retry_count - 1:
                time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                continue
            else:
                return None, f"OpenAI API error: {str(e)}"

    return None, "Maximum retry attempts reached."


def generate_data_description(df, data_stats, null_analysis):
    """Generate data description and cleaning suggestions using OpenAI"""
    system_prompt = '''
    You are a data analysis assistant. Based on the dataset information provided, you will:

    1. Provide a brief description of what the data appears to be about (topic, domain, purpose)
    2. Suggest specific data cleaning steps based on the null value analysis and other data characteristics

    Keep your response concise and focused on actionable insights. Format your response with two sections:

    DATA DESCRIPTION:
    [Your description of what the data is about]

    CLEANING SUGGESTIONS:
    - [First suggestion]
    - [Second suggestion]
    ...

    Limit your response to these two sections only.
    '''

    # Prepare a prompt that includes data information
    sample_data = df.head(10).to_string()

    # Get column information
    column_info = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        sample = str(df[col].iloc[0]) if not df[col].empty and not pd.isna(df[col].iloc[0]) else "N/A"
        column_info.append(f"{col} (type: {dtype}, sample: {sample})")

    # Create a comprehensive prompt with all the collected information
    prompt = f"""
    I've uploaded a dataset with {data_stats['rows']} rows and {data_stats['columns']} columns.

    Column information:
    {', '.join(column_info)}

    Here's a sample of the data (first 10 rows):
    {sample_data}

    Data statistics:
    - Rows: {data_stats['rows']}
    - Columns: {data_stats['columns']}
    - Data types: {data_stats['dtypes']}

    Null value analysis:
    - Total nulls: {null_analysis['total_nulls']} ({null_analysis['percentage_nulls']:.2f}% of all values)
    - Rows with nulls: {null_analysis['rows_with_nulls']} ({null_analysis['rows_with_nulls_percentage']:.2f}% of rows)
    - Completely null columns: {null_analysis['completely_null_columns']}
    - Columns with >50% nulls: {null_analysis['high_null_columns']}

    Top 5 columns with highest null percentages:
    {sorted([(col, info['null_percentage']) for col, info in null_analysis['columns'].items()], key=lambda x: x[1], reverse=True)[:5]}

    Based on this information, please:
    1. Tell me what this dataset is about (topic, domain, purpose)
    2. Suggest specific steps for cleaning this data
    """

    # Call OpenAI API
    response, error = call_openai_api(prompt, system_prompt)

    if error:
        st.session_state.api_error = error
        return "Could not generate data description.", "Could not generate cleaning suggestions."

    if response:
        # Parse response to extract description and suggestions
        parts = response.split("DATA DESCRIPTION:")
        if len(parts) > 1:
            content = parts[1]
            parts = content.split("CLEANING SUGGESTIONS:")
            if len(parts) > 1:
                description = parts[0].strip()
                suggestions = parts[1].strip()
                return description, suggestions

        # If parsing fails, return the whole response
        return response, ""

    return "Could not generate data description.", "Could not generate cleaning suggestions."