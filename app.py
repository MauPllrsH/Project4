# app.py
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
from components.data_analysis_ui import display_analysis_ui
from components.data_modeling_ui import display_modeling_ui

# Import services
from services.openai_service import generate_data_description

# Load environment variables
load_dotenv()

# Set up page
st.title("CSV & OpenAI Integration")


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

    # Data analysis session state variables
    if 'proceed_to_analysis' not in st.session_state:
        st.session_state.proceed_to_analysis = False
    if 'analysis_suggestions' not in st.session_state:
        st.session_state.analysis_suggestions = None
    if 'display_analysis_options' not in st.session_state:
        st.session_state.display_analysis_options = False
    if 'show_analysis_preview' not in st.session_state:
        st.session_state.show_analysis_preview = False
    if 'visualization_code' not in st.session_state:
        st.session_state.visualization_code = None
    if 'visualization_output' not in st.session_state:
        st.session_state.visualization_output = None
    if 'figures' not in st.session_state:
        st.session_state.figures = []
    if 'analysis_insights' not in st.session_state:
        st.session_state.analysis_insights = None

    # Data modeling session state variables
    if 'proceed_to_modeling' not in st.session_state:
        st.session_state.proceed_to_modeling = False
    if 'modeling_suggestions' not in st.session_state:
        st.session_state.modeling_suggestions = None
    if 'display_modeling_options' not in st.session_state:
        st.session_state.display_modeling_options = False
    if 'show_model_preview' not in st.session_state:
        st.session_state.show_model_preview = False
    if 'model_code' not in st.session_state:
        st.session_state.model_code = None
    if 'model_output' not in st.session_state:
        st.session_state.model_output = None
    if 'model_results' not in st.session_state:
        st.session_state.model_results = None
    if 'model_metrics' not in st.session_state:
        st.session_state.model_metrics = None
    if 'model_figures' not in st.session_state:
        st.session_state.model_figures = []


# Check for OpenAI API key
openai_api_key = None
if 'OPENAI_API_KEY' in os.environ:
    openai_api_key = os.environ['OPENAI_API_KEY']
elif st.secrets and "openai" in st.secrets and "OPENAI_API_KEY" in st.secrets["openai"]:
    openai_api_key = st.secrets["openai"]["OPENAI_API_KEY"]
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
                            "Could not generate cleaning suggestions using OpenAI API.")
                        data_description = "This appears to be a dataset with the following structure: " + ", ".join(
                            full_df.columns[:5]) + "..."
                        cleaning_suggestions = "There was an error when generating cleaning suggestions"
                        st.session_state.data_description = data_description
                        st.session_state.cleaning_suggestions = cleaning_suggestions
                else:
                    # Use fallback if API error
                    data_description = "This appears to be a dataset with the following structure: " + ", ".join(
                        full_df.columns[:5]) + "..."
                    cleaning_suggestions = "There was an error when generating cleaning suggestions"
                    st.session_state.data_description = data_description
                    st.session_state.cleaning_suggestions = cleaning_suggestions
            except Exception as e:
                st.error(f"Error generating data description: {str(e)}")
                st.code(traceback.format_exc())
                # Use fallback suggestions
                data_description = "This appears to be a dataset with the following structure: " + ", ".join(
                    full_df.columns[:5]) + "..."
                cleaning_suggestions = "There was an error when generating cleaning suggestions"
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

    # Determine which data frame to use (original or cleaned)
    active_df = st.session_state.full_df if st.session_state.full_df is not None else df

    # Check if we should proceed to modeling
    if st.session_state.proceed_to_modeling:
        # Display data modeling UI
        display_modeling_ui(
            active_df,
            st.session_state.data_description,
            st.session_state.null_analysis
        )

        # Add a button to go back to data analysis
        if st.button("← Back to Data Analysis"):
            st.session_state.proceed_to_modeling = False
            st.session_state.proceed_to_analysis = True
            st.rerun()

    # Check if we should proceed to data analysis
    elif st.session_state.proceed_to_analysis:
        # Display data analysis UI
        display_analysis_ui(
            active_df,
            st.session_state.data_description,
            st.session_state.null_analysis
        )

        # Add buttons for navigation
        col1, col2 = st.columns(2)

        with col1:
            if st.button("← Back to Data Cleaning"):
                st.session_state.proceed_to_analysis = False
                st.rerun()

        with col2:
            if st.button("Continue to Modeling →"):
                st.session_state.proceed_to_modeling = True
                st.session_state.proceed_to_analysis = False
                st.rerun()
    else:
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
                active_df,
                st.session_state.data_description,
                st.session_state.cleaning_suggestions,
                st.session_state.null_analysis
            )

        # Add a "Continue to Data Analysis" button at the main level for datasets that have already been processed
        # or if the user has already applied cleaning changes
        if st.session_state.changes_applied:
            st.markdown("---")
            if st.button("Continue to Data Analysis", key="continue_main_level"):
                st.session_state.proceed_to_analysis = True
                st.rerun()

# Register a callback to clean up temp files when the app reruns
if st.session_state.temp_files:
    cleanup_temp_files(st.session_state.temp_files)