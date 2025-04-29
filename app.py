import streamlit as st
import traceback
import os
from dotenv import load_dotenv

# Import utility modules
from utils.data_analysis import get_data_stats, analyze_null_values
from utils.file_handlers import cleanup_temp_files

# Import components
from components.data_upload import handle_dataset_upload
from components.data_preview import display_data_preview, display_null_analysis
from components.data_cleaning_ui import display_cleaning_ui

# Import services
from services.openai_service import generate_data_description

# Load environment variables
load_dotenv()

# Set up page
st.title("CSV & OpenAI Integration")


def generate_fallback_suggestions(df, null_analysis):
    """Generate basic cleaning suggestions without using OpenAI API"""
    suggestions = []

    # Check for completely null columns
    if null_analysis['completely_null_columns']:
        suggestions.append(f"Drop completely null columns: {', '.join(null_analysis['completely_null_columns'])}")

    # Check for high null percentage columns
    if null_analysis['high_null_columns']:
        suggestions.append(
            f"Consider dropping or imputing columns with high null percentage: {', '.join(null_analysis['high_null_columns'])}")

    # Check for rows with too many nulls
    if null_analysis['rows_with_nulls_percentage'] > 50:
        suggestions.append(
            f"Consider dropping rows with too many null values (currently {null_analysis['rows_with_nulls_percentage']:.2f}% of rows have nulls)")

    # Check for numeric columns that might need imputation
    numeric_cols = [col for col in df.columns if df[col].dtype.kind in 'ifc']
    if numeric_cols:
        suggestions.append(
            f"Consider imputing missing values in numeric columns using mean or median: {', '.join(numeric_cols[:5])}{'...' if len(numeric_cols) > 5 else ''}")

    # Check for categorical columns that might need imputation
    cat_cols = [col for col in df.columns if df[col].dtype == 'object']
    if cat_cols:
        suggestions.append(
            f"Consider imputing missing values in categorical columns using mode or a specific value: {', '.join(cat_cols[:5])}{'...' if len(cat_cols) > 5 else ''}")

    # Basic data quality suggestions
    suggestions.append("Check for and handle outliers in numeric columns")
    suggestions.append("Normalize or standardize numeric features if needed")
    suggestions.append("Convert categorical variables to numeric using one-hot encoding")

    return "\n".join([f"- {suggestion}" for suggestion in suggestions])


# Initialize session state variables
def init_session_state():
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'full_df' not in st.session_state:
        st.session_state.full_df = None
    if 'clean_df' not in st.session_state:
        st.session_state.clean_df = None
    if 'cleaning_code' not in st.session_state:
        st.session_state.cleaning_code = None
    if 'cleaning_output' not in st.session_state:
        st.session_state.cleaning_output = None
    if 'cleaning_summary' not in st.session_state:
        st.session_state.cleaning_summary = {}
    if 'api_error' not in st.session_state:
        st.session_state.api_error = None
    if 'temp_files' not in st.session_state:
        st.session_state.temp_files = []
    if 'temp_csv_files' not in st.session_state:
        st.session_state.temp_csv_files = []
    if 'temp_cleaned_csv' not in st.session_state:
        st.session_state.temp_cleaned_csv = None
    if 'display_clean_options' not in st.session_state:
        st.session_state.display_clean_options = False
    if 'show_cleaning_preview' not in st.session_state:
        st.session_state.show_cleaning_preview = False
    if 'current_view' not in st.session_state:
        st.session_state.current_view = "original"
    if 'data_description' not in st.session_state:
        st.session_state.data_description = None
    if 'cleaning_suggestions' not in st.session_state:
        st.session_state.cleaning_suggestions = None
    if 'null_analysis' not in st.session_state:
        st.session_state.null_analysis = None
    if 'file_processed' not in st.session_state:
        st.session_state.file_processed = False
    if 'changes_applied' not in st.session_state:
        st.session_state.changes_applied = False


# Check for OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.warning("⚠️ OpenAI API key not found. The app will run without AI-powered data analysis.")
    st.info("To enable AI features, please add your OpenAI API key to the .env file as OPENAI_API_KEY=your_key_here")
    st.session_state.api_error = "OpenAI API key not found in environment variables"

# Initialize session state
init_session_state()

# Display any API errors
if st.session_state.api_error:
    st.error(st.session_state.api_error)
    if "quota" in st.session_state.api_error:
        st.info("You can still use the CSV upload functionality without the OpenAI integration.")

# CSV File Uploader
st.subheader("Upload your CSV file")
uploaded_file = st.file_uploader('Choose a CSV file', type=['csv'])

# Process the CSV file when available and not already processed
if uploaded_file is not None and not st.session_state.file_processed:
    full_df = handle_dataset_upload(uploaded_file)

    if full_df is not None:
        # Store the full dataset in session state for analysis
        st.session_state.full_df = full_df

        # Mark file as processed to prevent reprocessing on rerun
        st.session_state.file_processed = True

        # Reset the changes_applied flag when a new file is uploaded
        st.session_state.changes_applied = False

        # Generate data statistics and null analysis on the full dataset
        with st.spinner("Analyzing full dataset..."):
            data_stats = get_data_stats(full_df)
            null_analysis = analyze_null_values(full_df)

            # Store the null analysis in session state
            st.session_state.null_analysis = null_analysis

            # Generate data description and cleaning suggestions
            try:
                if not st.session_state.api_error:
                    # Try to use OpenAI API
                    st.session_state.data_description, st.session_state.cleaning_suggestions = generate_data_description(
                        full_df, data_stats, null_analysis
                    )

                    # If no cleaning suggestions were generated, use fallback
                    if not st.session_state.cleaning_suggestions:
                        st.warning(
                            "Could not generate cleaning suggestions using OpenAI API. Using basic suggestions instead.")
                        data_description = "This appears to be a dataset with the following structure: " + ", ".join(
                            full_df.columns[:5]) + "..."
                        cleaning_suggestions = generate_fallback_suggestions(full_df, null_analysis)
                        st.session_state.data_description = data_description
                        st.session_state.cleaning_suggestions = cleaning_suggestions
                else:
                    # Use fallback if API error
                    data_description = "This appears to be a dataset with the following structure: " + ", ".join(
                        full_df.columns[:5]) + "..."
                    cleaning_suggestions = generate_fallback_suggestions(full_df, null_analysis)
                    st.session_state.data_description = data_description
                    st.session_state.cleaning_suggestions = cleaning_suggestions
            except Exception as e:
                st.error(f"Error generating data description: {str(e)}")
                st.code(traceback.format_exc())
                # Use fallback suggestions
                data_description = "This appears to be a dataset with the following structure: " + ", ".join(
                    full_df.columns[:5]) + "..."
                cleaning_suggestions = generate_fallback_suggestions(full_df, null_analysis)
                st.session_state.data_description = data_description
                st.session_state.cleaning_suggestions = cleaning_suggestions

        st.success(
            f'File loaded and analyzed successfully! Analyzed {len(full_df)} rows and {len(full_df.columns)} columns.')

# Reset file processing if the file uploader is cleared
if uploaded_file is None and st.session_state.file_processed:
    st.session_state.file_processed = False
    st.session_state.changes_applied = False

# Display and interact with the loaded DataFrame
if st.session_state.df is not None:
    df = st.session_state.df

    # Display data preview
    display_data_preview(df)

    # Display data description if available
    if st.session_state.data_description:
        st.subheader("Data Description")
        st.write(st.session_state.data_description)

    # Display null value analysis
    if st.session_state.null_analysis:
        display_null_analysis(st.session_state.null_analysis)

    # Display data cleaning options if description and suggestions are available
    if st.session_state.data_description and st.session_state.cleaning_suggestions:
        # Updated function call to use display_cleaning_ui
        display_cleaning_ui(
            st.session_state.full_df if st.session_state.full_df is not None else df,
            st.session_state.data_description,
            st.session_state.cleaning_suggestions,
            st.session_state.null_analysis
        )

# Register a callback to clean up temp files when the app reruns
if st.session_state.temp_files:
    cleanup_temp_files(st.session_state.temp_files)