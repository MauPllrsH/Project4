import pandas as pd
import numpy as np


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
