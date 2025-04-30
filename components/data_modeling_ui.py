import streamlit as st
import pandas as pd
import base64
import io
from utils.data_modeling import generate_modeling_suggestions, generate_model_code, execute_model_code
from utils.file_handlers import save_to_temp_csv


def display_modeling_ui(df, data_description, null_analysis):
    """Display data modeling UI section"""
    st.markdown("---")
    st.subheader("Data Modeling")

    # Check if we already generated modeling suggestions
    if 'modeling_suggestions' not in st.session_state or st.session_state.modeling_suggestions is None:
        with st.spinner("Generating modeling suggestions..."):
            modeling_suggestions = generate_modeling_suggestions(df, data_description, null_analysis)
            # Ensure we have valid suggestions
            if modeling_suggestions is None or modeling_suggestions.strip() == "":
                # If still None, use a very simple fallback
                modeling_suggestions = """
                - **Linear Regression**: For numerical target variables
                - **Logistic Regression**: For binary classification problems
                - **Random Forest**: For both regression and classification with high accuracy
                - **Gradient Boosting**: For complex relationships and high performance
                - **K-Means Clustering**: For unsupervised learning to identify groups
                """
            st.session_state.modeling_suggestions = modeling_suggestions

    # Display modeling suggestions
    st.markdown("**Modeling Suggestions:**")
    st.markdown(st.session_state.modeling_suggestions)

    # Ask if user wants to proceed with model creation
    col1, col2 = st.columns([3, 1])

    with col1:
        model_approach = st.selectbox(
            "Select a modeling approach:",
            ["Linear Regression", "Logistic Regression", "Random Forest", "Gradient Boosting",
             "K-Means Clustering", "Support Vector Machine", "Neural Network", "Custom Approach"],
            key="model_approach_select"
        )

    with col2:
        create_model = st.radio(
            "Create model?",
            ["No", "Yes"],
            index=0,
            key="create_model_radio"
        )

    if create_model == "Yes":
        st.session_state.display_modeling_options = True
    else:
        st.session_state.display_modeling_options = False
        st.session_state.show_model_preview = False

    # Show modeling options if user selected yes
    if st.session_state.display_modeling_options:
        # Target variable selection
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        all_cols = df.columns.tolist()

        target_variable = st.selectbox(
            "Select target variable to predict:",
            all_cols,
            key="target_variable_select"
        )

        # Let user provide additional modeling instructions
        user_instructions = st.text_area(
            "You can provide additional modeling instructions (optional):",
            "",
            height=100,
            key="user_modeling_instructions"
        )

        # Generate model code button
        if st.button("Generate & Train Model"):
            with st.spinner("Generating and training model..."):
                # Generate the model code
                model_code = generate_model_code(
                    df,
                    target_variable,
                    model_approach,
                    user_instructions
                )

                # Store the generated code
                st.session_state.model_code = model_code

                # Execute the code to train model and generate results
                model_results, metrics, figures, output = execute_model_code(df, model_code)

                if model_results is not None:
                    # Store the results
                    st.session_state.model_results = model_results
                    st.session_state.model_metrics = metrics
                    st.session_state.model_figures = figures
                    st.session_state.model_output = output

                    st.session_state.show_model_preview = True
                    st.success("Model trained successfully!")

                    # Force a rerun to show the preview
                    st.rerun()
                else:
                    st.error("Error training model. Please check the output below.")
                    st.code(output)

        # Show model preview if available
        if st.session_state.show_model_preview:
            show_model_preview(df,
                               st.session_state.model_results,
                               st.session_state.model_metrics,
                               st.session_state.model_figures,
                               st.session_state.model_code,
                               st.session_state.model_output)


def show_model_preview(df, model_results, metrics, figures, model_code, model_output):
    """Show model preview with results, metrics, and visualizations"""
    st.subheader("Model Results")

    # Create tabs for different views
    preview_tab, code_tab, output_tab = st.tabs(["Results & Metrics", "Model Code", "Execution Output"])

    with preview_tab:
        # Display metrics
        st.subheader("Model Performance Metrics")

        # Display metrics in a nice format
        metrics_df = pd.DataFrame(metrics.items(), columns=['Metric', 'Value'])
        st.table(metrics_df)

        # Display any model-specific results
        if isinstance(model_results, dict) and model_results.get('model_type'):
            st.subheader(f"{model_results['model_type']} Model Results")

            # Display feature importance if available
            if 'feature_importance' in model_results:
                st.write("**Feature Importance:**")
                if isinstance(model_results['feature_importance'], pd.DataFrame):
                    st.dataframe(model_results['feature_importance'])
                else:
                    st.write(model_results['feature_importance'])

        # Display visualizations
        st.subheader("Model Visualizations")
        for i, fig in enumerate(figures):
            st.pyplot(fig)

        # Generate PDF report button
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Generate PDF Report"):
                with st.spinner("Generating PDF report..."):
                    try:
                        # Import here to avoid circular imports
                        from utils.data_visualization import generate_model_pdf_report

                        pdf_bytes = generate_model_pdf_report(df, model_results, metrics, figures)

                        # Create download button for PDF
                        b64_pdf = base64.b64encode(pdf_bytes).decode()
                        href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="model_report.pdf">Download PDF Report</a>'
                        st.markdown(href, unsafe_allow_html=True)

                        st.success("PDF report generated! Click the link above to download.")
                    except Exception as e:
                        import traceback
                        st.error(f"Error generating PDF report: {str(e)}")
                        st.code(traceback.format_exc())

        with col2:
            if st.button("Discard Model"):
                # Reset modeling state
                st.session_state.show_model_preview = False
                st.session_state.display_modeling_options = False
                st.success("Model discarded!")
                st.rerun()

    with code_tab:
        st.code(model_code, language="python")

    with output_tab:
        st.text(model_output)