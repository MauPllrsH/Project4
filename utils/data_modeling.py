import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import traceback
import sys
from io import StringIO
from services.openai_service import call_openai_api
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend


def generate_modeling_suggestions(df, data_description, null_analysis):
    """Generate modeling suggestions using OpenAI"""
    # First check if df is valid
    if df is None or not hasattr(df, 'empty') or df.empty:
        return generate_fallback_modeling_suggestions()

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
            return generate_fallback_modeling_suggestions()

        # Check if we got a valid response
        if response and isinstance(response, str) and len(response.strip()) > 0:
            return response
        else:
            print("Empty or invalid response from OpenAI API")
            return generate_fallback_modeling_suggestions()

    except Exception as e:
        import traceback
        print(f"Error in generate_modeling_suggestions: {str(e)}")
        print(traceback.format_exc())
        return generate_fallback_modeling_suggestions()


def generate_fallback_modeling_suggestions():
    """Generate basic modeling suggestions without using OpenAI API"""
    suggestions = """
    - **Linear Regression**: Suitable for predicting numerical outcomes with linear relationships. Simple and interpretable but may underfit complex relationships.

    - **Random Forest**: Versatile ensemble method for both classification and regression problems. Handles non-linear relationships well and provides feature importance metrics.

    - **Gradient Boosting (XGBoost/LightGBM)**: Powerful algorithm that often achieves state-of-the-art results. Good for both classification and regression tasks with complex patterns.

    - **Logistic Regression**: Appropriate for binary or multi-class classification problems. Simple, interpretable, and works well with small to medium datasets.

    - **K-Means Clustering**: Useful for unsupervised learning to identify natural groupings in the data. Consider this if your goal is to segment data without labeled outcomes.

    - **Support Vector Machine**: Effective for classification tasks, especially in high-dimensional spaces. Works well when classes are separable and dataset size is medium.
    """

    return suggestions


def generate_model_code(df, target_variable, model_approach, user_instructions=""):
    """Generate simplified model training code using OpenAI that works reliably with Streamlit"""

    # Get detailed column information
    column_info = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        num_nulls = df[col].isna().sum()
        unique_values = df[col].nunique()
        column_info.append(f"{col} (type: {dtype}, nulls: {num_nulls}, unique values: {unique_values})")

    system_prompt = '''
    You are a Python machine learning code generation assistant. Generate SIMPLE, RELIABLE code that trains a machine learning model and stores results for display in a Streamlit application.

    CRITICALLY IMPORTANT: 1. Place ALL imports at the top of the code 2. ONLY return executable Python code (no 
    markdown) 3. The DataFrame is already available as 'df' 4. DO NOT use any Python print() statements 5. Store ALL 
    results in the metrics dictionary and figures list 6. Assume the data is already clean and there is no need to 
    handle na values 7. Make sure to handle the different categorical columns appropiately, if a column is datetime, 
    or an object, do the necessary steps to turn the values into something that can be used to predict

    Your code MUST follow this EXACT structure:
    # START CODE
    # Imports
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    # Add only the imports you actually need

    # Initialize result variables
    figures = []
    model_results = {"model_type": "MODEL_NAME_HERE"}
    metrics = {}

    # Main try/except block
    try:
        # Make a copy of the dataset
        df_model = df.copy()

        # Basic preprocessing - just handle nulls and convert categoricals
        # No pipeline, no fancy preprocessing

        # Split data 80/20
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a SIMPLE model - no hyperparameter tuning

        # Calculate basic metrics and store in metrics dictionary
        # IMPORTANT: Convert all numpy values to Python native types
        metrics["accuracy"] = float(accuracy)  # Example

        # For classification models ONLY, create a simple confusion matrix visualization
        if is_classification:
            fig, ax = plt.subplots(figsize=(8, 6))
            # Plot confusion matrix
            figures.append(fig)
            plt.close(fig)

    except Exception as e:
        # Store error in metrics
        metrics["error"] = str(e)
    # END CODE
    
    Key requirements:
    1. ONLY add imports you actually need
    2. ONLY reference columns that actually exist in the dataset
    3. Keep preprocessing to minimum: drop na, encode categoricals with pd.get_dummies()
    4. Use standard train_test_split from sklearn
    5. Store ALL metrics as native Python types (float, list, dict)
    6. Close all figures after appending to the figures list
    7. Handle errors gracefully in the try/except block
    8. For classification, just make ONE confusion matrix plot
    9. DO NOT attempt any preprocessing beyond the absolute basics
    10. DO NOT create any complex Pipeline objects
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
    Generate extremely simple Python code for a {model_approach} model that predicts '{target_variable}'. The code should:
    1. Handle basic preprocessing (just handle nulls and convert categorical variables)
    2. Split data (80% training, 20% testing)
    3. Train a basic {model_approach} model with default parameters
    4. Evaluate with simple metrics
    5. Create only ONE visualization IF it's a classification model (just a confusion matrix)
    6. Store all results in the provided variables (metrics, model_results, figures)

    Make the code as simple and reliable as possible. Do not include any complex preprocessing, feature engineering, hyperparameter tuning, or multiple visualizations.
    """

    # Call OpenAI API
    response, error = call_openai_api(prompt, system_prompt)

    if error:
        # Return a reliable fallback code that will work
        return get_fallback_model_code(target_variable, model_approach)

    if response:
        # Fix common indentation issues in the response
        response = fix_indentation(response)

        # Check if the response already contains imports and variables
        has_imports = any(line.strip().startswith('import ') for line in response.split('\n'))
        has_variables = any(line.strip().startswith('figures =') for line in response.split('\n'))

        # Only add missing parts
        code_parts = []

        # Add imports if not present
        if not has_imports:
            code_parts.append("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
""")
            # Add model-specific imports
            if "Linear Regression" in model_approach:
                code_parts.append("from sklearn.linear_model import LinearRegression\n")
            elif "Logistic Regression" in model_approach:
                code_parts.append("from sklearn.linear_model import LogisticRegression\n")
            elif "Random Forest" in model_approach:
                code_parts.append("from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor\n")
            elif "Support Vector" in model_approach:
                code_parts.append("from sklearn.svm import SVC, SVR\n")

        # Add variable initialization if not present
        if not has_variables:
            code_parts.append("""
# Initialize variables
figures = []
model_results = {"model_type": "%s"}
metrics = {}
""" % model_approach)

        # Add the API response
        code_parts.append(response)

        # For more reliability, we'll do one more check to ensure a try-except structure exists
        if 'try:' not in response:
            if 'except' not in response:
                # Wrap the entire response in a try-except if it doesn't have one
                code_parts = code_parts[:-1]  # Remove the last part (response)
                code_parts.append("""
try:
    %s
except Exception as e:
    # Set default values in case of errors
    metrics["error"] = str(e)
    model_results["model_type"] = "Error - Model training failed"
    model_results["error"] = str(e)
""" % response.replace('\n', '\n    '))

        code = '\n'.join(code_parts)

        # Perform a final indentation check
        code = validate_python_code(code)
        return code

    # Return a reliable fallback code that will work if there's an issue with the API
    return get_fallback_model_code(target_variable, model_approach)


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


def validate_python_code(code):
    """Attempt to validate Python code syntax and fix common issues"""
    try:
        # Try to compile the code to check for syntax errors
        compile(code, '<string>', 'exec')
        return code
    except SyntaxError as e:
        # If there's an indentation error, try to fix it
        if 'indentation' in str(e).lower():
            # Simple fix: ensure all lines after 'try:' are indented
            lines = code.split('\n')
            fixed_lines = []
            in_try_block = False
            try_indent = 0

            for line in lines:
                stripped = line.strip()

                # Skip empty lines
                if not stripped:
                    fixed_lines.append(line)
                    continue

                # Check for try statement
                if stripped == 'try:':
                    in_try_block = True
                    try_indent = len(line) - len(line.lstrip())
                    fixed_lines.append(line)

                # Handle except/finally
                elif in_try_block and (stripped.startswith('except') or stripped.startswith('finally:')):
                    in_try_block = False
                    fixed_lines.append(' ' * try_indent + stripped)

                # Handle indentation inside try block
                elif in_try_block:
                    current_indent = len(line) - len(line.lstrip())
                    if current_indent <= try_indent:
                        # Not properly indented
                        fixed_lines.append(' ' * (try_indent + 4) + stripped)
                    else:
                        fixed_lines.append(line)

                else:
                    fixed_lines.append(line)

            return '\n'.join(fixed_lines)

        # For other syntax errors, return the fallback code
        return code  # Return original code, will be handled by error handling in execute function


def get_fallback_model_code(target_variable, model_approach):
    """Return an extremely simple fallback model code that should work in all cases"""

    # Set up the imports based on the model approach
    imports = """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
"""

    if "Linear Regression" in model_approach:
        imports += "from sklearn.linear_model import LinearRegression\n"
        imports += "from sklearn.metrics import mean_squared_error, r2_score\n"
        model_type = "Linear Regression"
        is_classification = False
    elif "Logistic Regression" in model_approach:
        imports += "from sklearn.linear_model import LogisticRegression\n"
        imports += "from sklearn.metrics import accuracy_score, confusion_matrix, classification_report\n"
        model_type = "Logistic Regression"
        is_classification = True
    elif "Random Forest" in model_approach:
        if "Classifier" in model_approach or "Classification" in model_approach:
            imports += "from sklearn.ensemble import RandomForestClassifier\n"
            imports += "from sklearn.metrics import accuracy_score, confusion_matrix, classification_report\n"
            model_type = "Random Forest Classifier"
            is_classification = True
        else:
            imports += "from sklearn.ensemble import RandomForestRegressor\n"
            imports += "from sklearn.metrics import mean_squared_error, r2_score\n"
            model_type = "Random Forest Regressor"
            is_classification = False
    elif "Support Vector" in model_approach:
        if "Classifier" in model_approach or "Classification" in model_approach:
            imports += "from sklearn.svm import SVC\n"
            imports += "from sklearn.metrics import accuracy_score, confusion_matrix, classification_report\n"
            model_type = "Support Vector Classifier"
            is_classification = True
        else:
            imports += "from sklearn.svm import SVR\n"
            imports += "from sklearn.metrics import mean_squared_error, r2_score\n"
            model_type = "Support Vector Regressor"
            is_classification = False
    else:
        # Default to Logistic Regression for unknown model types
        imports += "from sklearn.linear_model import LogisticRegression\n"
        imports += "from sklearn.metrics import accuracy_score, confusion_matrix, classification_report\n"
        model_type = "Logistic Regression"
        is_classification = True

    # Basic model code
    model_code = f"""
# Initialize variables
figures = []
model_results = {{"model_type": "{model_type}"}}
metrics = {{}}

try:
    # Make a copy of the dataset
    df_model = df.copy()

    # Check if target variable exists
    if '{target_variable}' not in df_model.columns:
        raise ValueError("Target variable '{target_variable}' not found in the dataset")

    # Handle missing values in target column
    df_model = df_model.dropna(subset=['{target_variable}'])

    # Basic preprocessing
    # Get categorical columns
    categorical_cols = df_model.select_dtypes(include=['object']).columns.tolist()
    # Remove target from categorical columns if it's there
    if '{target_variable}' in categorical_cols:
        categorical_cols.remove('{target_variable}')

    # Convert categorical columns to dummies
    if categorical_cols:
        df_model = pd.get_dummies(df_model, columns=categorical_cols, drop_first=True)

    # Prepare X and y
    X = df_model.drop(columns=['{target_variable}'])
    y = df_model['{target_variable}']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Get numeric columns for scaling
    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()

    # Scale numeric features
    if numeric_cols:
        scaler = StandardScaler()
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test
"""

    # Add model-specific code
    if is_classification:
        model_code += f"""    
    # Train the model
    model = {'LogisticRegression(max_iter=1000)' if 'Logistic' in model_type else 'SVC(probability=True)' if 'Support Vector' in model_type else 'RandomForestClassifier(n_estimators=100, random_state=42)'}
    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = model.predict(X_test_scaled)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    metrics["Accuracy"] = float(accuracy)

    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    metrics["Classification Report"] = report

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    metrics["Confusion Matrix"] = cm.tolist()

    # Create confusion matrix visualization
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    figures.append(fig)
    plt.close(fig)
"""
    else:
        model_code += f"""
    # Train the model
    model = {'LinearRegression()' if 'Linear' in model_type else 'SVR()' if 'Support Vector' in model_type else 'RandomForestRegressor(n_estimators=100, random_state=42)'}
    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = model.predict(X_test_scaled)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    metrics["Mean Squared Error"] = float(mse)
    metrics["R² Score"] = float(r2)
"""

    # Error handling
    model_code += """
except Exception as e:
    # Store error in metrics
    metrics["error"] = str(e)
    model_results["model_type"] = "Error - Model training failed"
    model_results["error"] = str(e)
"""

    # Combine and return the full code
    return imports + model_code


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

        # Restore stdout
        sys.stdout = old_stdout
        output = mystdout.getvalue()

        return model_results, metrics, figures, output

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