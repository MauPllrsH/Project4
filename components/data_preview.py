import streamlit as st
import pandas as pd
import altair as alt
from utils.file_handlers import safe_display_dataframe


def display_data_preview(df):
    """Display the data preview section"""
    st.subheader("Data Preview")
    st.write(f"Rows: {len(df)}, Columns: {len(df.columns)}")

    # Show the DataFrame
    safe_display_dataframe(df, 10)


def display_null_analysis(null_analysis, df=None):
    """Display null value analysis"""
    # Use custom CSS to control the font size
    st.markdown("""
    <style>
    .null-analysis-title {
        font-size: 28px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .null-metrics {
        font-size: 20px;
        font-weight: bold;
    }
    .null-value {
        font-size: 32px;  /* Larger font for the values */
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

    # Use custom HTML for the title with the custom class
    st.markdown('<div class="null-analysis-title">Null Values Analysis</div>', unsafe_allow_html=True)

    # Overview metrics with custom styling
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="null-metrics">Total Null Values</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="null-value">{null_analysis["total_nulls"]:,}</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="null-metrics">Rows with Nulls</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="null-value">{null_analysis["rows_with_nulls"]:,} ({null_analysis["rows_with_nulls_percentage"]:.2f}%)</div>',
            unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="null-metrics">Overall % Nulls</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="null-value">{null_analysis["percentage_nulls"]:.2f}%</div>', unsafe_allow_html=True)

    # Null values by column
    st.subheader("Null Values by Column")

    # Create a DataFrame with null statistics
    null_stats = []
    for col, info in null_analysis['columns'].items():
        if info['null_count'] > 0:  # Only show columns with nulls
            null_stats.append({
                "Column": col,
                "Null Count": info['null_count'],
                "Null %": f"{info['null_percentage']:.2f}%"
            })

    # Check if there are any columns with nulls
    if null_stats:
        # Sort by null count (descending)
        null_stats_df = pd.DataFrame(null_stats).sort_values("Null Count", ascending=False)
        st.dataframe(null_stats_df)
    else:
        st.success("No null values found in the dataset! All columns are complete.")

    # Display additional null insights
    if null_analysis['high_null_columns']:
        st.warning(f"Columns with high null % (>50%): {', '.join(null_analysis['high_null_columns'])}")

    if null_analysis['completely_null_columns']:
        st.error(f"Completely null columns: {', '.join(null_analysis['completely_null_columns'])}")

    # Display null patterns if available
    if 'correlated_nulls' in null_analysis and null_analysis['correlated_nulls']:
        st.subheader("Null Value Patterns")
        st.write("These column pairs have highly correlated null patterns:")
        for pair in null_analysis['correlated_nulls']:
            st.write(f"- {pair['column1']} & {pair['column2']}: correlation = {pair['correlation']:.2f}")
    else:
        st.success("No rows contain null values. Your dataset is complete!")