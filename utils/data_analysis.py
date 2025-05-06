import pandas as pd
import numpy as np
from services.openai_service import call_openai_api


def get_data_stats(df):
    """Generate detailed statistics about the DataFrame"""
    stats = {}

    # Basic stats
    stats['rows'] = len(df)
    stats['columns'] = len(df.columns)

    # Data types
    dtype_counts = df.dtypes.value_counts().to_dict()
    stats['dtypes'] = {str(k): v for k, v in dtype_counts.items()}

    # Column-specific stats
    stats['column_stats'] = {}
    for col in df.columns:
        col_stats = {}
        col_stats['dtype'] = str(df[col].dtype)

        # Check if column is numeric type
        try:
            # Handle different types including categorical
            if pd.api.types.is_numeric_dtype(df[col]):
                col_stats['min'] = float(df[col].min()) if not pd.isna(df[col].min()) else None
                col_stats['max'] = float(df[col].max()) if not pd.isna(df[col].max()) else None
                col_stats['mean'] = float(df[col].mean()) if not pd.isna(df[col].mean()) else None
                col_stats['median'] = float(df[col].median()) if not pd.isna(df[col].median()) else None
        except:
            # If there's any error in numeric calculations, skip them
            pass

        # Categorical/string stats
        if (pd.api.types.is_object_dtype(df[col]) or
                pd.api.types.is_categorical_dtype(df[col])):
            # For categorical columns, get the number of unique values
            try:
                unique_values = df[col].nunique()
                col_stats['unique_values'] = unique_values

                # Get top 5 most common values if not too many unique values
                if unique_values < 100:  # Avoid expensive operation for high cardinality columns
                    most_common = df[col].value_counts().head(5).to_dict()
                    col_stats['most_common'] = {str(k): int(v) for k, v in most_common.items()}
            except:
                # If there's any error in processing categorical data, skip it
                pass

        stats['column_stats'][col] = col_stats

    return stats


def analyze_null_values(df):
    """Generate detailed analysis of null values in the DataFrame"""
    null_analysis = {}

    # Total null count
    total_nulls = df.isna().sum().sum()
    null_analysis['total_nulls'] = int(total_nulls)
    null_analysis['percentage_nulls'] = float(total_nulls / (df.shape[0] * df.shape[1]) * 100)

    # Nulls per column
    column_nulls = df.isna().sum().to_dict()
    column_null_percentages = (df.isna().mean() * 100).to_dict()

    null_analysis['columns'] = {}
    for col in df.columns:
        null_analysis['columns'][col] = {
            'null_count': int(column_nulls[col]),
            'null_percentage': float(column_null_percentages[col])
        }

    # Rows with nulls
    rows_with_nulls = df.isna().any(axis=1).sum()
    null_analysis['rows_with_nulls'] = int(rows_with_nulls)
    null_analysis['rows_with_nulls_percentage'] = float(rows_with_nulls / df.shape[0] * 100)

    # Distribution of nulls across rows
    null_counts_per_row = df.isna().sum(axis=1).value_counts().sort_index().to_dict()
    null_analysis['null_counts_per_row'] = {str(k): int(v) for k, v in null_counts_per_row.items()}

    # Check for columns that are completely null
    completely_null_columns = [col for col in df.columns if df[col].isna().all()]
    null_analysis['completely_null_columns'] = completely_null_columns

    # Check for columns with high percentage of nulls (>50%)
    high_null_columns = [col for col in df.columns if df[col].isna().mean() > 0.5]
    null_analysis['high_null_columns'] = high_null_columns

    # Check for patterns of missingness between columns
    if len(df.columns) > 1 and rows_with_nulls > 10:
        # Create a DataFrame of boolean indicators for whether values are missing
        missing_indicators = df.isna()

        # Select columns with at least some nulls
        cols_with_nulls = [col for col in df.columns if missing_indicators[col].sum() > 0]

        if len(cols_with_nulls) > 1:
            # Compute correlation of missingness
            missingness_corr = missing_indicators[cols_with_nulls].corr()

            # Find pairs with high correlation (>0.7)
            high_corr_pairs = []
            for i in range(len(cols_with_nulls)):
                for j in range(i + 1, len(cols_with_nulls)):
                    col1 = cols_with_nulls[i]
                    col2 = cols_with_nulls[j]
                    corr = missingness_corr.loc[col1, col2]
                    if abs(corr) > 0.7:
                        high_corr_pairs.append({
                            'column1': col1,
                            'column2': col2,
                            'correlation': float(corr)
                        })

            null_analysis['correlated_nulls'] = high_corr_pairs

    return null_analysis


def generate_analysis_suggestions(df, data_description, null_analysis):
    """Generate data analysis and visualization suggestions using OpenAI"""
    # First check if df is valid
    if df is None or not hasattr(df, 'empty') or df.empty:
        return "Dataframe is not valid, either None or is empty"

    system_prompt = '''
    You are a data analysis assistant. Based on the dataset information provided, you will suggest specific data analysis approaches and visualizations that would provide the most meaningful insights.

    Keep your response concise and focused on actionable analysis. Format your response with bulleted suggestions like:

    - [First analysis suggestion with specific visualization type]
    - [Second analysis suggestion with specific visualization type]
    ...

    Focus on suggesting analyses that:
    1. Reveal relationships between variables
    2. Identify interesting patterns or outliers
    3. Answer important business or domain questions
    4. Are appropriate for the data types present (numeric, categorical, etc.)
    5. Tell a coherent story about the data

    Limit your response to 5-7 specific, high-value analysis suggestions.
    '''

    # Create a summary of the dataset for the prompt
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()

    # Calculate correlations for numeric data
    correlations = None
    high_correlations = {}
    if len(numeric_cols) > 1:
        try:
            corr_matrix = df[numeric_cols].corr().abs()
            # Get the upper triangle of the correlation matrix
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            # Find pairs with high correlation (>0.7)
            high_corr_pairs = [(col1, col2, upper_tri.loc[col1, col2])
                               for col1 in upper_tri.index
                               for col2 in upper_tri.columns
                               if upper_tri.loc[col1, col2] > 0.7]
            # Format for the prompt
            high_correlations = {f"{pair[0]}-{pair[1]}": round(pair[2], 2) for pair in high_corr_pairs[:5]}
        except Exception as e:
            print(f"Error calculating correlations: {str(e)}")
            high_correlations = {}

    # Get basic statistics
    try:
        numeric_stats = df[numeric_cols].describe() if numeric_cols else pd.DataFrame()
        basic_stats = f"Numeric column statistics:\n{numeric_stats.to_string()}"
    except Exception as e:
        print(f"Error generating basic statistics: {str(e)}")
        basic_stats = "Could not generate basic statistics."

    # Prepare the prompt
    prompt = f"""
    Dataset Description:
    {data_description}

    Dataset Information:
    - Rows: {len(df)}
    - Columns: {len(df.columns)}
    - Numeric columns: {", ".join(numeric_cols) if numeric_cols else "None"}
    - Categorical columns: {", ".join(categorical_cols) if categorical_cols else "None"}
    - Datetime columns: {", ".join(datetime_cols) if datetime_cols else "None"}

    Basic statistics:
    {basic_stats}

    High correlation pairs (if any):
    {high_correlations}

    Based on this information, suggest 5-7 specific data analysis and visualization approaches that would provide meaningful insights about this dataset.

    For each suggestion:
    1. Specify what question or relationship to analyze
    2. Recommend specific visualization type(s) to use
    3. Explain what insights this analysis might reveal

    Make your suggestions specific to this dataset, not generic.
    """

    try:
        # Attempt to call OpenAI API
        from services.openai_service import call_openai_api
        response, error = call_openai_api(prompt, system_prompt)

        # Log the error for debugging
        if error:
            print(f"OpenAI API error: {error}")
            return "OpenAI API Error"

        # Check if we got a valid response
        if response and isinstance(response, str) and len(response.strip()) > 0:
            return response
        else:
            print("Empty or invalid response from OpenAI API")
            return "Empty or invalid response from OpenAI API"

    except Exception as e:
        import traceback
        print(f"Error in generate_analysis_suggestions: {str(e)}")
        print(traceback.format_exc())
        return "Error while generating suggestions"
