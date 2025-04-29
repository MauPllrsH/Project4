import streamlit as st
import pandas as pd
import base64
import io
from utils.data_analysis import generate_analysis_suggestions
from utils.data_visualization import generate_visualization_code, execute_visualization_code, generate_pdf_report


def display_analysis_ui(df, data_description, null_analysis):
    """Display data analysis UI section"""
    st.markdown("---")
    st.subheader("Data Analysis")

    # Check if we already generated analysis suggestions
    if 'analysis_suggestions' not in st.session_state or st.session_state.analysis_suggestions is None:
        with st.spinner("Generating data analysis suggestions..."):
            analysis_suggestions = generate_analysis_suggestions(df, data_description, null_analysis)
            # Ensure we have valid suggestions
            if analysis_suggestions is None or analysis_suggestions.strip() == "":
                # If still None, use a very simple fallback
                analysis_suggestions = """
                - **Distribution Analysis**: Create histograms for key numeric variables
                - **Correlation Analysis**: Generate a correlation heatmap for numeric columns
                - **Categorical Analysis**: Create bar charts for categorical variables
                """
            st.session_state.analysis_suggestions = analysis_suggestions

    # Display analysis suggestions
    st.markdown("**Analysis Suggestions:**")
    st.markdown(st.session_state.analysis_suggestions)

    # Ask if user wants to proceed with the suggested analysis
    analyze_data = st.radio(
        "Would you like to apply the suggested data analysis?",
        ["No", "Yes"],
        index=0,
        key="analyze_data_radio"
    )

    if analyze_data == "Yes":
        st.session_state.display_analysis_options = True
    else:
        st.session_state.display_analysis_options = False
        st.session_state.show_analysis_preview = False

    # Show analysis options if user selected yes
    if st.session_state.display_analysis_options:
        # Let user provide additional analysis instructions
        user_instructions = st.text_area(
            "You can provide additional analysis instructions (optional):",
            "",
            height=100,
            key="user_analysis_instructions"
        )

        # Generate analysis code button
        if st.button("Generate & Preview Analysis"):
            with st.spinner("Generating data visualization and analysis code..."):
                # Generate the visualization code
                visualization_code = generate_visualization_code(
                    df,
                    st.session_state.analysis_suggestions,
                    user_instructions
                )

                # Store the generated code
                st.session_state.visualization_code = visualization_code

                # Execute the code to preview results
                figures, insights, output = execute_visualization_code(df, visualization_code)

                if figures is not None:
                    # Store the results
                    st.session_state.figures = figures
                    st.session_state.analysis_insights = insights
                    st.session_state.visualization_output = output

                    st.session_state.show_analysis_preview = True
                    st.success("Generated analysis and visualizations!")

                    # Force a rerun to show the preview
                    st.rerun()
                else:
                    st.error("Error executing visualization code. Please check the output below.")
                    st.code(output)

        # Show analysis preview if available
        if st.session_state.show_analysis_preview:
            show_analysis_preview(df,
                                  st.session_state.figures,
                                  st.session_state.analysis_insights,
                                  st.session_state.visualization_code,
                                  st.session_state.visualization_output)


def show_analysis_preview(df, figures, insights, visualization_code, visualization_output):
    """Show analysis preview with visualizations and insights"""
    st.subheader("Data Analysis Results")

    # Create tabs for different views
    preview_tab, code_tab, output_tab = st.tabs(["Visualizations & Insights", "Analysis Code", "Execution Output"])

    with preview_tab:
        # Display insights
        st.subheader("Analysis Insights")
        st.markdown(insights)

        # Display visualizations
        st.subheader("Visualizations")
        for i, fig in enumerate(figures):
            st.pyplot(fig)

        # Generate PDF report button
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Generate PDF Report"):
                with st.spinner("Generating PDF report..."):
                    try:
                        pdf_bytes = generate_pdf_report(df, figures, insights)

                        # Create download button for PDF
                        b64_pdf = base64.b64encode(pdf_bytes).decode()
                        href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="data_analysis_report.pdf">Download PDF Report</a>'
                        st.markdown(href, unsafe_allow_html=True)

                        st.success("PDF report generated! Click the link above to download.")
                    except Exception as e:
                        import traceback
                        st.error(f"Error generating PDF report: {str(e)}")
                        st.code(traceback.format_exc())

        with col2:
            if st.button("Discard Analysis"):
                # Reset analysis state
                st.session_state.show_analysis_preview = False
                st.session_state.display_analysis_options = False
                st.success("Analysis discarded!")
                st.rerun()

    with code_tab:
        st.code(visualization_code, language="python")

    with output_tab:
        st.text(visualization_output)