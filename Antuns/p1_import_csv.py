import streamlit as st
import pandas as pd

def upload_csv():
    df = None
    uploaded_file = st.file_uploader(label='Upload *csv file from your drive! Choose a file:', type='csv')
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, na_values=-9999)
        st.success("Loading finished!")
        st.dataframe(df, width=1400, height=300)
        st.write('---')
    return df
    