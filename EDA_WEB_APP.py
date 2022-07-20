
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

st.markdown(''' **Exploratory data analysis web application**

This is developed by Usman Nasir
''')

with st.sidebar.header("Upload your dataset (.csv)"):
    upload_file=st.sidebar.file_uploader("Upload your file",type=['csv'])
    df=sns.load_dataset("titanic")
    st.sidebar.markdown('[Example CSV file](df)')

if upload_file is not None:
    @st.cache
    def load_csv():
        csv=pd.read_csv(upload_file)
        return csv

    df=load_csv()
    pr=ProfileReport(df,explorative=True)
    st.header("Input DataFrame")
    st.write(df)
    st.write('---')
    st.write('**Profiling report with pandas**')
    st_profile_report(pr)

else:
    st.info("Awaiting for csv file: ")
    if st.button("Press to use example data"):
        @st.cache
        def load_data():
            a=pd.DataFrame(np.random.rand(100,5),columns=['A','B','C','D','E'])
            return a
        df=load_data()
        pr=ProfileReport(df,explorative=True)
        st.header("Input DataFrame")
        st.write(df)
        st.write('---')
        st.write('**Profiling report with pandas**')
        st_profile_report(pr)