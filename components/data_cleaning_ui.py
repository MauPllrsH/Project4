import streamlit as st
import pandas as pd
from utils.data_cleaning import generate_cleaning_code, execute_cleaning_code
from utils.file_handlers import save_to_temp_csv, safe_display_dataframe
from utils.data_analysis import analyze_null_values


def display_cleaning_ui(original_df, data_description, cleaning_suggestions, null_analysis):
    """Display data cleaning UI section"""
    st.markdown("---")
    st.subheader("Data Cleaning")

    # Display cleaning suggestions
    st.markdown("**Cleaning Suggestions:**")
    st.markdown(cleaning_suggestions)

    # Add a "Continue to Data Analysis" button at the top level to allow skipping cleaning
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Skip to Data Analysis", key="skip_to_analysis"):
            # Set state to proceed to data analysis
            st.session_state.proceed_to_analysis = True
            st.rerun()

    # Ask if user wants to clean the data
    clean_data = st.radio(
        "Would you like to apply the suggested data cleaning steps?",
        ["No", "Yes"],
        index=0,
        key="clean_data_radio"
    )

    if clean_data == "Yes":
        st.session_state.display_clean_options = True
    else:
        st.session_state.display_clean_options = False
        st.session_state.show_cleaning_preview = False

    # Show cleaning options if user selected yes
    if st.session_state.display_clean_options:
        # Let user provide additional cleaning instructions
        user_instructions = st.text_area(
            "You can provide additional cleaning instructions (optional):",
            "",
            height=100,
            key="user_clean_instructions"
        )

        # Generate cleaning code button
        if st.button("Generate & Preview Cleaning Code"):
            with st.spinner("Generating data cleaning code..."):
                # Generate the cleaning code
                cleaning_code = generate_cleaning_code(
                    original_df,
                    null_analysis,
                    cleaning_suggestions,
                    user_instructions
                )

                # Store the generated code
                st.session_state.cleaning_code = cleaning_code

                # Execute the code to preview results
                clean_df, output, summary = execute_cleaning_code(original_df, cleaning_code)

                if clean_df is not None:
                    # Store the results
                    st.session_state.clean_df = clean_df
                    st.session_state.cleaning_output = output
                    st.session_state.cleaning_summary = summary

                    # Save the cleaned DataFrame to a temporary CSV file
                    temp_csv_path = save_to_temp_csv(clean_df)
                    st.session_state.temp_cleaned_csv = temp_csv_path

                    st.session_state.show_cleaning_preview = True
                    st.success("Generated cleaning code and preview!")

                    # Force a rerun to show the preview
                    st.rerun()
                else:
                    st.error("Error executing cleaning code. Please check the output below.")
                    st.code(output)

        # Show cleaning preview if available
        if st.session_state.show_cleaning_preview:
            show_cleaning_preview(original_df, st.session_state.clean_df,
                                  st.session_state.cleaning_code,
                                  st.session_state.cleaning_output,
                                  st.session_state.cleaning_summary,
                                  null_analysis)


def show_cleaning_preview(original_df, clean_df, cleaning_code, cleaning_output, cleaning_summary, null_analysis):
    """Show cleaning preview with before and after comparison"""
    st.subheader("Data Cleaning Preview")

    # Create tabs for different views
    preview_tab, code_tab, output_tab = st.tabs(["Before & After", "Cleaning Code", "Execution Output"])

    with preview_tab:
        # Display summary of changes
        st.subheader("Summary of Changes")

        # Display changes in key metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows",
                      f"{len(clean_df):,}",
                      f"{len(clean_df) - len(original_df):,}")
        with col2:
            original_nulls = original_df.isna().sum().sum()
            clean_nulls = clean_df.isna().sum().sum()
            st.metric("Null Values",
                      f"{clean_nulls:,}",
                      f"{clean_nulls - original_nulls:,}")
        with col3:
            original_cols = len(original_df.columns)
            clean_cols = len(clean_df.columns)
            st.metric("Columns",
                      f"{clean_cols:,}",
                      f"{clean_cols - original_cols:,}")

        # Display cleaning summary if available
        if cleaning_summary:
            st.subheader("Cleaning Steps Applied")
            for step, details in cleaning_summary.items():
                st.write(f"- **{step}**: {details}")

        # Data preview comparison
        st.subheader("Data Preview Comparison")

        # Allow toggling between views
        view_selector = st.radio(
            "Select view:",
            ["Original Data", "Cleaned Data", "Side by Side"],
            horizontal=True
        )

        if view_selector == "Original Data":
            st.subheader("Original Data")
            st.dataframe(safe_display_dataframe(original_df, 10))

            # Also display null analysis for original data
            with st.expander("Original Data Null Analysis"):
                # Original null analysis metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Null Values", f"{null_analysis['total_nulls']:,}")
                with col2:
                    st.metric("Rows with Nulls",
                              f"{null_analysis['rows_with_nulls']:,} ({null_analysis['rows_with_nulls_percentage']:.2f}%)")
                with col3:
                    st.metric("Overall % Nulls", f"{null_analysis['percentage_nulls']:.2f}%")

        elif view_selector == "Cleaned Data":
            st.subheader("Cleaned Data")
            st.dataframe(safe_display_dataframe(clean_df, 10))

            # Also display null analysis for cleaned data
            with st.expander("Cleaned Data Null Analysis"):
                # Calculate null analysis for cleaned data
                clean_null_analysis = analyze_null_values(clean_df)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Null Values", f"{clean_null_analysis['total_nulls']:,}")
                with col2:
                    st.metric("Rows with Nulls",
                              f"{clean_null_analysis['rows_with_nulls']:,} ({clean_null_analysis['rows_with_nulls_percentage']:.2f}%)")
                with col3:
                    st.metric("Overall % Nulls", f"{clean_null_analysis['percentage_nulls']:.2f}%")

        else:  # Side by Side
            st.subheader("Side by Side Comparison")
            col1, col2 = st.columns(2)
            with col1:
                st.write("Original Data")
                st.dataframe(safe_display_dataframe(original_df, 10))
            with col2:
                st.write("Cleaned Data")
                st.dataframe(safe_display_dataframe(clean_df, 10))

        # Apply changes button
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Apply Changes & Replace Dataset"):
                # Replace the original datasets with cleaned versions
                try:
                    # Store the original data temporarily in case we need to revert
                    st.session_state.original_df = original_df.copy()

                    if st.session_state.full_df is not None:
                        st.session_state.full_df = clean_df.copy()
                        # Also update the display DataFrame
                        if len(clean_df) > 10000:
                            st.session_state.df = clean_df.head(10000).copy()
                        else:
                            st.session_state.df = clean_df.copy()
                    else:
                        st.session_state.df = clean_df.copy()

                    # Update null analysis for the new data
                    st.session_state.null_analysis = analyze_null_values(clean_df)

                    # Update data description and cleaning suggestions
                    if not st.session_state.api_error:
                        with st.spinner("Updating data description and suggestions..."):
                            from utils.data_analysis import get_data_stats
                            from services.openai_service import generate_data_description
                            data_stats = get_data_stats(clean_df)
                            st.session_state.data_description, st.session_state.cleaning_suggestions = generate_data_description(
                                clean_df, data_stats, st.session_state.null_analysis
                            )

                    # Reset cleaning state
                    st.session_state.current_view = "original"
                    st.session_state.show_cleaning_preview = False
                    st.session_state.display_clean_options = False

                    # Set flag to indicate changes have been applied
                    st.session_state.changes_applied = True

                    st.success("Dataset replaced with cleaned version!")
                    # Rerun to update the app
                    st.rerun()

                except Exception as e:
                    import traceback
                    st.error(f"Error replacing dataset: {str(e)}")
                    st.code(traceback.format_exc())

        with col2:
            if st.button("Discard Changes"):
                # Reset cleaning state
                st.session_state.show_cleaning_preview = False
                st.session_state.display_clean_options = False
                st.success("Changes discarded!")
                st.rerun()

        # Add button to proceed to data analysis
        if st.button("Continue to Data Analysis"):
            # Set state to proceed to data analysis
            st.session_state.proceed_to_analysis = True
            st.session_state.show_cleaning_preview = False
            st.session_state.display_clean_options = False
            st.rerun()

    with code_tab:
        st.code(cleaning_code, language="python")

    with output_tab:
        st.text(cleaning_output)