import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import uuid
import tempfile
from openai import OpenAI
from dotenv import load_dotenv
import time
import re
import traceback

# Load environment variables
load_dotenv()

# Set up page
st.title("CSV & OpenAI Integration")

# Initialize session state variables if they don't exist
if 'df' not in st.session_state:
    st.session_state.df = None
if 'api_error' not in st.session_state:
    st.session_state.api_error = None
if 'temp_files' not in st.session_state:
    st.session_state.temp_files = []


# Function to clean up temporary files
def cleanup_temp_files():
    for file_path in st.session_state.temp_files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            st.warning(f"Could not remove temporary file {file_path}: {e}")

    st.session_state.temp_files = []


# Function to make OpenAI API calls with retry logic
def call_openai_api(prompt, retry_count=3, retry_delay=2):
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return "Error: OpenAI API key not found in environment variables"

    client = OpenAI(api_key=api_key)

    # Instructions to get only executable Python code
    system_prompt = """
    You are a Python code generation assistant. When given a question about data, ONLY respond with executable Python code.

    Important guidelines:
    1. ONLY return valid Python code that can be executed in a Streamlit environment
    2. Include all necessary imports at the beginning of the code
    3. When creating visualizations, use the following pattern:
       - Import necessary libraries
       - Create the visualization
       - Save to a temporary file
       - Display with st.image()
       - Clean up the file
    4. Use proper and consistent indentation with 4 spaces
    5. DO NOT use markdown formatting or backticks
    6. The DataFrame is already available as 'df'
    7. For file paths, always use os.path.join() for compatibility
    8. Your code will be executed directly, so ensure it's complete and valid
    """

    for attempt in range(retry_count):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )
            # Extract only the Python code from the response
            code = response.choices[0].message.content
            # Remove any markdown code blocks if present
            code = re.sub(r'```python\s*', '', code)
            code = re.sub(r'```\s*', '', code)
            return code
        except Exception as e:
            if "insufficient_quota" in str(e):
                st.session_state.api_error = "OpenAI API quota exceeded. Please check your billing details."
                return None
            elif "rate_limit" in str(e) and attempt < retry_count - 1:
                time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                continue
            else:
                st.session_state.api_error = f"OpenAI API error: {str(e)}"
                return None


# Display any API errors
if st.session_state.api_error:
    st.error(st.session_state.api_error)
    if "quota" in st.session_state.api_error:
        st.info("You can still use the CSV upload functionality without the OpenAI integration.")

# CSV File Uploader
st.subheader("Upload your CSV file")
uploaded_file = st.file_uploader('Choose a CSV file', type=['csv'])

# Process the CSV file when available
if uploaded_file is not None:
    try:
        # Show file info
        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.write(f"Uploaded file size: {file_size_mb:.2f} MB")

        # For larger files, use chunking
        if file_size_mb > 50:
            with st.spinner("Loading large file in chunks..."):
                chunks = []
                chunk_size = 10000

                for chunk in pd.read_csv(uploaded_file, chunksize=chunk_size):
                    chunks.append(chunk)

                st.session_state.df = pd.concat(chunks)
        else:
            # For smaller files, load directly
            with st.spinner("Loading file..."):
                st.session_state.df = pd.read_csv(uploaded_file)

        st.success('File loaded successfully!')

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

# Display and interact with the loaded DataFrame
if st.session_state.df is not None:
    df = st.session_state.df

    # Display basic info
    st.subheader("Data Preview")
    st.write(f"Rows: {len(df)}, Columns: {len(df.columns)}")

    # Show the DataFrame
    st.dataframe(df.head(10))

    # Cleanup any previous temp files
    cleanup_temp_files()

    # Only show OpenAI integration if no API errors
    if not st.session_state.api_error:
        st.subheader("Ask about your data")
        user_query = st.text_input('What would you like to know about this data?',
                                   placeholder='Example: Create a histogram of column X')

        if user_query:
            # Prepare a prompt that includes data information and column types
            column_info = []
            for col in df.columns:
                dtype = str(df[col].dtype)
                sample = str(df[col].iloc[0]) if not df[col].empty else "N/A"
                column_info.append(f"{col} (type: {dtype}, sample: {sample})")

            prompt = f"""
            I have a dataset with {len(df)} rows and {len(df.columns)} columns.

            Column information:
            {', '.join(column_info)}

            Here's a sample of the data:
            {df.head(5).to_string()}

            My question is: {user_query}

            Return ONLY Python code that I can execute in Streamlit. The code should be complete and ready to run.

            Example pattern for visualization:

            import matplotlib.pyplot as plt
            import streamlit as st
            import os

            # Create visualization
            plt.figure(figsize=(10, 6))
            df['column_name'].value_counts().plot(kind='bar')
            plt.title('Title')
            plt.xlabel('X Label')
            plt.ylabel('Y Label')
            plt.tight_layout()

            # Save to file
            temp_dir = 'temp_images'
            os.makedirs(temp_dir, exist_ok=True)
            temp_file = os.path.join(temp_dir, 'output.png')
            plt.savefig(temp_file)
            plt.close()

            # Show in Streamlit
            st.image(temp_file)

            # Cleanup
            os.remove(temp_file)
            """

            with st.spinner("Generating Python code..."):
                code = call_openai_api(prompt)
                if code:
                    st.code(code, language="python")

                    # Execute button
                    if st.button("Execute Code"):
                        try:
                            with st.spinner("Executing..."):
                                # Create a namespace for execution
                                exec_globals = {
                                    'df': df,
                                    'st': st,
                                    'pd': pd,
                                    'plt': plt,
                                    'sns': sns,
                                    'os': os,
                                    'uuid': uuid,
                                    'np': __import__('numpy'),
                                }

                                # Track temp files created during execution
                                before_files = set()
                                for root, dirs, files in os.walk('.'):
                                    for file in files:
                                        before_files.add(os.path.join(root, file))

                                # Execute the code directly
                                exec(code, exec_globals)

                                # Find new files created
                                after_files = set()
                                for root, dirs, files in os.walk('.'):
                                    for file in files:
                                        after_files.add(os.path.join(root, file))

                                # Add new files to the cleanup list
                                new_files = after_files - before_files
                                for file_path in new_files:
                                    if os.path.exists(file_path) and os.path.isfile(file_path):
                                        st.session_state.temp_files.append(file_path)

                        except Exception as e:
                            st.error(f"Error executing code: {str(e)}")
                            st.code(traceback.format_exc())

# Register a callback to clean up temp files when the app reruns
if st.session_state.temp_files:
    cleanup_temp_files()