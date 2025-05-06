import streamlit as st
import pandas as pd


def handle_dataset_upload(uploaded_file):
    """
    Process an uploaded CSV file and return the full dataframe.
    """
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

                # Store the full dataset for analysis
                full_df = pd.concat(chunks)

                # Keep only first 10,000 rows for display and other operations
                if len(full_df) > 10000:
                    st.session_state.df = full_df.head(10000)
                    st.info(
                        f"Note: For display purposes, only showing the first 10,000 rows, but analysis is performed on all {len(full_df)} rows.")
                else:
                    st.session_state.df = full_df
        else:
            # For smaller files, load directly
            with st.spinner("Loading file..."):
                full_df = pd.read_csv(uploaded_file)

                # Keep only first 10,000 rows for display and other operations
                if len(full_df) > 10000:
                    st.session_state.df = full_df.head(10000)
                    st.info(
                        f"Note: For display purposes, only showing the first 10,000 rows, but analysis is performed on all {len(full_df)} rows.")
                else:
                    st.session_state.df = full_df

        return full_df

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None