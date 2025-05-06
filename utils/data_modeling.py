import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import traceback
import sys
from io import StringIO
from services.openai_service import call_openai_api
import matplotlib
import streamlit as st

matplotlib.use('Agg')  # Use non-interactive backend


def generate_modeling_suggestions(df, data_description, null_analysis):
    """Generate modeling suggestions using OpenAI"""
    # First check if df is valid
    if df is None or not hasattr(df, 'empty') or df.empty:
        return "⚠️ OpenAI API connection error. Please check your API key and internet connection. The AI-powered suggestions are unavailable at the moment."

    system_prompt = '''
    You are a data science assistant. Based on the dataset information provided, you will suggest specific machine learning models and approaches that would be most appropriate for this data.

    Keep your response concise and focused on actionable modeling suggestions. Format your response with bulleted suggestions like:

    - [First model suggestion with explanation of why it's appropriate]
    - [Second model suggestion with explanation of why it's appropriate]
    ...

    Focus on suggesting models that:
    1. Are appropriate for the data types and features present
    2. Match the likely prediction or clustering goal
    3. Would perform well given the dataset size and characteristics
    4. Consider interpretability vs performance tradeoffs
    5. Account for any data quality issues mentioned

    Limit your response to 4-6 specific, well-justified model suggestions.
    '''

    # Create a summary of the dataset for the prompt
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()

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

    Null value information:
    - Total nulls: {null_analysis['total_nulls']} ({null_analysis['percentage_nulls']:.2f}% of all values)
    - Rows with nulls: {null_analysis['rows_with_nulls']} ({null_analysis['rows_with_nulls_percentage']:.2f}% of rows)

    Based on this information, suggest 4-6 specific machine learning models or approaches that would be most appropriate for this dataset.

    For each suggestion:
    1. Name the specific model or approach
    2. Explain why it would be appropriate for this dataset
    3. Mention any preprocessing steps that would be necessary
    4. Note any potential challenges or limitations

    Make your suggestions specific to this dataset, not generic.
    """

    try:
        # Attempt to call OpenAI API
        response, error = call_openai_api(prompt, system_prompt)

        # Log the error for debugging
        if error:
            print(f"OpenAI API error: {error}")
            return "⚠️ OpenAI API connection error. Please check your API key and internet connection. The AI-powered suggestions are unavailable at the moment."

        # Check if we got a valid response
        if response and isinstance(response, str) and len(response.strip()) > 0:
            return response
        else:
            print("Empty or invalid response from OpenAI API")
            return "⚠️ OpenAI API returned an empty response. Please try again later."

    except Exception as e:
        import traceback
        print(f"Error in generate_modeling_suggestions: {str(e)}")
        print(traceback.format_exc())
        return "⚠️ An error occurred while generating modeling suggestions. Please try again later."


def generate_model_code(df, target_variable, model_approach, user_instructions=""):
    """Generate model training code using OpenAI"""

    # Get detailed column information
    column_info = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        num_nulls = df[col].isna().sum()
        unique_values = df[col].nunique()
        column_info.append(f"{col} (type: {dtype}, nulls: {num_nulls}, unique values: {unique_values})")

    system_prompt = '''
    You are a Python machine learning code generation assistant. Generate SIMPLE, RELIABLE code that trains a machine learning model and stores results for display in a Streamlit application.

    CRITICALLY IMPORTANT: 
    1. Place ALL imports at the top of the code 
    2. ONLY return executable Python code (no markdown) 
    3. The DataFrame is already available as 'df' 
    4. DO NOT use any Python print() statements 
    5. Store ALL results in the metrics dictionary and figures list 
    6. Handle different categorical columns appropriately - if a column is datetime or object type, convert values into something that can be used for prediction
    7. For classification reports, store the full output in metrics["Classification Report"]
    8. ALWAYS create a feature importance visualization regardless of the model type

    Your code MUST follow this EXACT structure:
    # START CODE
    # Imports
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.inspection import permutation_importance
    # Add only the imports you actually need

    # Initialize result variables
    figures = []
    model_results = {"model_type": "MODEL_NAME_HERE"}
    metrics = {}

    # Main try/except block
    try:
        # Make a copy of the dataset
        df_model = df.copy()

        # Basic preprocessing - handle nulls and convert categoricals

        # Split data 80/20
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a model with appropriate parameters

        # Calculate metrics and store in metrics dictionary
        # For classification metrics, include the full classification_report
        if is_classification:
            report = classification_report(y_test, y_pred, output_dict=True)
            metrics["Classification Report"] = report

        # Generate feature importance using ONE of the following methods (in order of preference):
        # 1. Use model.feature_importances_ if available
        # 2. Use model.coef_ if available (for linear models)
        # 3. Use permutation_importance if neither is available
        # ALWAYS create a feature importance plot regardless of model type

        # Create visualizations and add to figures list
        # Always close figures after appending them

    except Exception as e:
        # Store error in metrics
        metrics["error"] = str(e)
    # END CODE

    Key requirements:
    1. ONLY add imports you actually need
    2. ONLY reference columns that actually exist in the dataset
    3. Use standard train_test_split from sklearn
    4. Store metrics as Python native types (float, list, dict)
    5. Close all figures after appending to the figures list
    6. Handle errors gracefully in the try/except block
    7. For classification models, always include a confusion matrix plot
    8. ALWAYS include a feature importance plot using one of these methods:
       a. For tree-based models (Random Forest, Gradient Boosting): use model.feature_importances_
       b. For linear models (Linear/Logistic Regression): use model.coef_
       c. For models without built-in importance: use permutation_importance from sklearn.inspection
       d. For K-Means: plot feature importance based on distance to cluster centers
    9. Make sure ALL metrics are properly stored
    10. Set max_iter=1000 for models like LogisticRegression that might not converge
    
    FEATURE IMPORTANCE INSTRUCTIONS:
    Always create a feature importance visualization regardless of the model type. Use this approach:

    1. For tree-based models (RandomForest, GradientBoosting): 
   - Use model.feature_importances_ directly

    2. For linear models (LinearRegression, LogisticRegression, Ridge, Lasso):
   - Use the absolute values of model.coef_
   - For multi-class: take the mean absolute value across classes

    3. For models without built-in importance (SVC, KNN, Neural Networks):
   - Use permutation_importance from sklearn.inspection:
     ```
     from sklearn.inspection import permutation_importance
     r = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
     importances = r.importances_mean
     ```

    4. For K-Means clustering:
   - Plot feature importance based on the variance of cluster centers across features

    Always sort features by importance and display the top 10-15 most important ones.
    Always include this visualization in your code no matter what model type is used.
    '''

    # Prepare the prompt
    prompt = f"""
    Dataset Information:
    - Rows: {len(df)}
    - Columns: {len(df.columns)}

    Column information (PLEASE CHECK CAREFULLY and only use these exact columns):
    {', '.join(column_info)}

    DataFrame dtypes:
    {df.dtypes.to_string()}

    User's model selection:
    - Selected model approach: {model_approach}
    - Target variable to predict: '{target_variable}'

    User specific instructions:
    {user_instructions}

    IMPORTANT INSTRUCTION:
    Generate Python code for a {model_approach} model that predicts '{target_variable}'. The code should:
    1. Handle basic preprocessing (just handle nulls and convert categorical variables)
    2. Split data (80% training, 20% testing)
    3. Train a basic {model_approach} model with appropriate parameters
    4. Evaluate with all basic and useful metrics that relate to the model used
    5. IF it's a classification model make sure to store the FULL classification_report in metrics["Classification Report"]
    6. Make sure to use max_iter=1000 for algorithms like LogisticRegression to avoid convergence warnings
    7. Store all results in the provided variables (metrics, model_results, figures)
    8. Create appropriate visualizations (confusion matrix for classification, feature importance if available)
    9. If possible add a feature importance visualization for the top 10 features
    10. Make sure ALL numeric values stored in metrics are Python native types (float), not NumPy types
    """

    # Call OpenAI API
    response, error = call_openai_api(prompt, system_prompt)

    if error:
        return "⚠️ OpenAI API connection error. Please check your API key and internet connection. The AI-powered code generation is unavailable at the moment."

    if response:
        # Fix common indentation issues in the response
        response = fix_indentation(response)
        return response

    return "⚠️ OpenAI API returned an empty response. Please try again later."


def fix_indentation(code_str):
    """Fix common indentation issues in generated code"""
    lines = code_str.split('\n')
    fixed_lines = []
    inside_try = False
    try_indent_level = 0
    current_indent = 0

    for i, line in enumerate(lines):
        # Skip empty lines
        if not line.strip():
            fixed_lines.append(line)
            continue

        # Count leading spaces to determine indentation
        leading_spaces = len(line) - len(line.lstrip())

        # Check for try statement
        if line.strip().startswith('try:'):
            inside_try = True
            try_indent_level = leading_spaces
            current_indent = try_indent_level + 4  # Python standard is 4 spaces
            fixed_lines.append(line)

        # Check if we're inside a try block but not properly indented
        elif inside_try and leading_spaces <= try_indent_level and not line.strip().startswith(
                ('except', 'finally', 'else:')):
            # This line should be indented inside the try block
            fixed_lines.append(' ' * current_indent + line.strip())

        # Check for except/finally that should end the try block
        elif inside_try and line.strip().startswith(('except', 'finally')):
            inside_try = False
            fixed_lines.append(' ' * try_indent_level + line.strip())

        # No special handling needed
        else:
            fixed_lines.append(line)

    return '\n'.join(fixed_lines)


def process_metrics(metrics):
    """Process metrics to make them display-friendly in Streamlit.
    Especially handles the classification_report which has a nested structure.
    """
    processed_metrics = {}

    # Process each metric
    for key, value in metrics.items():
        # Handle classification report specially
        if key == "Classification Report" and isinstance(value, dict):
            # Extract the main metrics for each class
            for class_name, class_metrics in value.items():
                if class_name in ['accuracy', 'macro avg', 'weighted avg']:
                    # Keep these for separate display
                    continue

                if isinstance(class_metrics, dict):
                    # Extract precision, recall, f1-score for each class
                    for metric_name, metric_value in class_metrics.items():
                        if metric_name in ['precision', 'recall', 'f1-score']:
                            new_key = f"Class {class_name} - {metric_name}"
                            processed_metrics[new_key] = float(metric_value)

            # Add overall metrics
            if 'accuracy' in value:
                processed_metrics['Overall Accuracy'] = float(value['accuracy'])

            if 'macro avg' in value and isinstance(value['macro avg'], dict):
                for metric_name, metric_value in value['macro avg'].items():
                    if metric_name in ['precision', 'recall', 'f1-score']:
                        new_key = f"Macro Avg - {metric_name}"
                        processed_metrics[new_key] = float(metric_value)

            if 'weighted avg' in value and isinstance(value['weighted avg'], dict):
                for metric_name, metric_value in value['weighted avg'].items():
                    if metric_name in ['precision', 'recall', 'f1-score']:
                        new_key = f"Weighted Avg - {metric_name}"
                        processed_metrics[new_key] = float(metric_value)

        # For confusion matrix, store as is (it will be displayed as a visualization)
        elif key == "Confusion Matrix":
            processed_metrics[key] = value

        # For other metrics, ensure they are Python native types
        elif isinstance(value, (np.float64, np.float32, np.int64, np.int32)):
            processed_metrics[key] = float(value)
        else:
            processed_metrics[key] = value

    return processed_metrics


def execute_model_code(df, code):
    """Validate and execute model training code with safer error handling"""
    # Create a namespace for execution
    exec_globals = {
        'df': df.copy(),
        'pd': pd,
        'np': np,
        'plt': plt,
        'sns': sns,
        'figures': [],
        'model_results': {},
        'metrics': {}
    }

    # Import scikit-learn modules
    try:
        from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, \
            GradientBoostingClassifier
        from sklearn.svm import SVC, SVR
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        from sklearn.cluster import KMeans
        from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, \
            f1_score, confusion_matrix, classification_report, roc_curve, auc, silhouette_score
        from sklearn.neural_network import MLPClassifier, MLPRegressor

        # Add scikit-learn imports to globals
        for name, obj in locals().items():
            if name not in ['name', 'obj', 'exec_globals']:
                exec_globals[name] = obj
    except ImportError as e:
        print(f"Warning: Could not import scikit-learn module: {e}")

    # Capture print output
    old_stdout = sys.stdout
    mystdout = StringIO()
    sys.stdout = mystdout

    try:
        # Validate Python code before execution
        try:
            # Try to compile the code to check for syntax errors
            compile(code, '<string>', 'exec')
        except SyntaxError as syntax_error:
            # Handle indentation errors specifically
            if 'indentation' in str(syntax_error).lower():
                # Fix indentation issues
                fixed_code = fix_code_indentation(code)
                code = fixed_code
                # Try compiling again
                try:
                    compile(code, '<string>', 'exec')
                except SyntaxError as e:
                    # If still failing, raise the error
                    raise ValueError(f"Failed to fix syntax errors in code: {str(e)}")
            else:
                # For other syntax errors
                raise syntax_error

        # Execute the model code
        exec(code, exec_globals)

        # Get the results
        model_results = exec_globals.get('model_results', {})
        metrics = exec_globals.get('metrics', {})
        figures = exec_globals.get('figures', [])

        # Process metrics for cleaner display
        processed_metrics = process_metrics(metrics)

        # Restore stdout
        sys.stdout = old_stdout
        output = mystdout.getvalue()

        return model_results, processed_metrics, figures, output

    except Exception as e:
        # Restore stdout in case of error
        sys.stdout = old_stdout
        error_output = mystdout.getvalue()

        # Create fallback results with error information
        error_message = f"Error executing model code: {str(e)}\n{traceback.format_exc()}"
        print(error_message)

        # Create an error figure
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', fontsize=12)
        ax.set_axis_off()

        # Return error information
        return {"model_type": "Error", "error": str(e)}, {"error": str(e)}, [fig], error_output + "\n" + error_message


def fix_code_indentation(code):
    """Fix common indentation issues in generated code"""
    lines = code.split('\n')
    fixed_lines = []
    in_block = False
    block_keyword = None
    block_indent = 0
    expected_indent = 0

    # First pass: detect and fix basic indentation issues
    for i, line in enumerate(lines):
        # Skip empty lines
        if not line.strip():
            fixed_lines.append(line)
            continue

        # Count leading spaces for current line
        leading_spaces = len(line) - len(line.lstrip())
        line_content = line.strip()

        # Check for block start
        if line_content.endswith(':'):
            in_block = True
            block_keyword = line_content.split()[0]
            block_indent = leading_spaces
            expected_indent = leading_spaces + 4  # Standard Python indent
            fixed_lines.append(line)
            continue

        # Check if we're inside a block but line is not properly indented
        if in_block:
            # Check if this line should end the block
            if block_keyword == 'try' and line_content.startswith(('except', 'finally')):
                # This should be at the same indent level as the try
                in_block = False  # Will start a new except/finally block
                if leading_spaces != block_indent:
                    fixed_lines.append(' ' * block_indent + line_content)
                    continue
            elif leading_spaces < expected_indent:
                # This line should be indented inside the block
                # But only if it's not another block-starting keyword at the same level
                if leading_spaces <= block_indent and not any(
                        line_content.startswith(kw) for kw in
                        ['if', 'else', 'elif', 'for', 'while', 'try', 'except', 'finally', 'def', 'class']
                ):
                    fixed_lines.append(' ' * expected_indent + line_content)
                    continue

        # No special fixing needed for this line
        fixed_lines.append(line)

    # Second pass: fix try-except blocks specifically
    result = []
    i = 0
    while i < len(fixed_lines):
        line = fixed_lines[i]
        if line.strip().startswith('try:'):
            try_indent = len(line) - len(line.lstrip())
            result.append(line)
            i += 1

            # Check the next non-empty line
            while i < len(fixed_lines) and not fixed_lines[i].strip():
                result.append(fixed_lines[i])
                i += 1

            if i < len(fixed_lines):
                next_line = fixed_lines[i]
                next_indent = len(next_line) - len(next_line.lstrip())

                # If the indent is not sufficient, fix it
                if next_indent <= try_indent and not next_line.strip().startswith(('except', 'finally')):
                    # This line should be indented
                    result.append(' ' * (try_indent + 4) + next_line.strip())
                else:
                    result.append(next_line)
            i += 1
        else:
            result.append(line)
            i += 1

    return '\n'.join(result)


def generate_universal_feature_importance_code():
    """Returns code snippet for generating feature importance for any model type"""
    return """
    # Generate feature importance regardless of model type
    feature_names = X.columns

    # Method 1: Use built-in feature_importances_ if available (tree-based models)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        top_indices = indices[:min(15, len(feature_names))]

        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'Feature': [feature_names[i] for i in top_indices],
            'Importance': [importances[i] for i in top_indices]
        })
        model_results['feature_importance'] = importance_df

    # Method 2: Use coefficients if available (linear models)
    elif hasattr(model, 'coef_'):
        coef = model.coef_

        # Handle multi-class case (shape will be [n_classes, n_features])
        if len(coef.shape) > 1 and coef.shape[0] > 1:
            # Take mean absolute value across classes
            importances = np.mean(np.abs(coef), axis=0)
        else:
            # Handle binary classification or regression (flatten if needed)
            importances = np.abs(coef.flatten())

        indices = np.argsort(importances)[::-1]
        top_indices = indices[:min(15, len(feature_names))]

        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'Feature': [feature_names[i] for i in top_indices],
            'Importance': [importances[i] for i in top_indices]
        })
        model_results['feature_importance'] = importance_df

    # Method 3: Use permutation importance as fallback
    else:
        try:
            # Import permutation_importance if not already imported
            from sklearn.inspection import permutation_importance

            # Calculate permutation importance
            r = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42)
            importances = r.importances_mean
            indices = np.argsort(importances)[::-1]
            top_indices = indices[:min(15, len(feature_names))]

            # Create feature importance DataFrame
            importance_df = pd.DataFrame({
                'Feature': [feature_names[i] for i in top_indices],
                'Importance': [importances[i] for i in top_indices]
            })
            model_results['feature_importance'] = importance_df
        except Exception as e:
            # If permutation importance fails, create dummy importance based on data variance
            variances = X.var().sort_values(ascending=False)
            top_features = variances.index[:min(15, len(feature_names))]

            # Create feature importance DataFrame based on feature variance
            importance_df = pd.DataFrame({
                'Feature': top_features,
                'Importance': variances[top_features].values / variances.max()  # Normalize to 0-1
            })
            model_results['feature_importance'] = importance_df
            metrics["feature_importance_note"] = "Built-in feature importance not available; using feature variance as proxy"

    # Always create a visualization of feature importance
    if 'feature_importance' in model_results:
        fig, ax = plt.subplots(figsize=(10, 6))
        importance_df = model_results['feature_importance'].sort_values('Importance', ascending=False)
        sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
        ax.set_title('Feature Importance')
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        figures.append(fig)
        plt.close(fig)
    """


def show_model_preview(df, model_results, metrics, figures, model_code, model_output):
    """Show model preview with results, metrics, and visualizations"""
    st.subheader("Model Results")

    # Create tabs for different views
    preview_tab, code_tab, output_tab = st.tabs(["Results & Metrics", "Model Code", "Execution Output"])

    with preview_tab:
        # Display metrics
        st.subheader("Model Performance Metrics")

        # Display metrics in a nice format
        metrics_items = list(metrics.items())
        # Sort metrics putting error first if it exists, and confusion matrix last
        metrics_items.sort(key=lambda x: (
            0 if x[0] == "error" else
            2 if x[0] == "Confusion Matrix" else
            2 if x[0] == "feature_importance_note" else
            1
        ))

        # Convert to DataFrame excluding Confusion Matrix which is displayed as a visualization
        metrics_df = pd.DataFrame([
            {"Metric": k, "Value": v} for k, v in metrics_items
            if k not in ["Confusion Matrix", "Classification Report"]
        ])

        # Show the metrics table
        if not metrics_df.empty:
            st.table(metrics_df)

        # Display any model-specific results
        if isinstance(model_results, dict) and model_results.get('model_type'):
            st.subheader(f"{model_results['model_type']} Model Results")

            # Display feature importance as both table and plot for better comprehension
            if 'feature_importance' in model_results:
                st.subheader("Feature Importance")

                if "feature_importance_note" in metrics:
                    st.info(metrics["feature_importance_note"])

                col1, col2 = st.columns([1, 2])

                with col1:
                    st.write("**Importance Values:**")
                    if isinstance(model_results['feature_importance'], pd.DataFrame):
                        st.dataframe(model_results['feature_importance'].set_index('Feature'))

                # Find and display the feature importance plot for better visibility
                feature_importance_found = False
                for i, fig in enumerate(figures):
                    if hasattr(fig, 'axes') and fig.axes and hasattr(fig.axes[0], 'get_title'):
                        if fig.axes[0].get_title() == 'Feature Importance':
                            with col2:
                                st.pyplot(fig)
                                feature_importance_found = True
                                break

                if not feature_importance_found:
                    with col2:
                        st.write("Feature importance visualization not found in figures.")

        # Display all other visualizations
        st.subheader("Model Visualizations")
        for i, fig in enumerate(figures):
            # Skip feature importance plot as it's already shown above
            if hasattr(fig, 'axes') and fig.axes and hasattr(fig.axes[0], 'get_title'):
                if fig.axes[0].get_title() == 'Feature Importance':
                    continue
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