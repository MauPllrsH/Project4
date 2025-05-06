import os
import uuid
import tempfile
import pandas as pd


def save_to_temp_csv(df, prefix="cleaned_data_"):
    """Save DataFrame to a temporary CSV file and return the file path"""
    temp_dir = tempfile.gettempdir()
    temp_file = os.path.join(temp_dir, f"{prefix}{uuid.uuid4()}.csv")
    df.to_csv(temp_file, index=False)
    return temp_file


def load_from_csv(file_path):
    """Load DataFrame from CSV file"""
    return pd.read_csv(file_path)


def cleanup_temp_files(file_paths):
    """Remove temporary files"""
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Could not remove temporary file {file_path}: {e}")
    return []


def safe_display_dataframe(df, num_rows=10):
    """Safely display a DataFrame, handling PyArrow conversion issues"""
    try:
        # Try normal display first
        return df.head(num_rows)
    except Exception as e:
        # If there's an error (likely PyArrow conversion), convert problematic columns
        if "pyarrow" in str(e).lower() or "arrow" in str(e).lower():
            # Make a copy to avoid modifying the original
            display_df = df.copy()

            # Identify and convert problematic columns
            for col in display_df.columns:
                # Check if column has mixed types (strings in numeric columns)
                if display_df[col].dtype.name in ['float64', 'float32', 'int64', 'int32']:
                    # If it's numeric but might contain strings, convert to string
                    if display_df[col].apply(lambda x: isinstance(x, str)).any():
                        display_df[col] = display_df[col].astype(str)

                # Convert any complex objects to strings if needed
                if display_df[col].dtype.name == 'object':
                    try:
                        # Try to convert to string
                        display_df[col] = display_df[col].astype(str)
                    except:
                        # If conversion fails, replace with placeholder
                        display_df[col] = "[Complex Object]"

            return display_df.head(num_rows)
        else:
            # For other errors, show a simplified representation
            return df.head(num_rows).to_dict()
