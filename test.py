import streamlit as st
import seaborn as sns
import pandas as pd
st.header("This is brought to you by stramlit dashboards")
st.text("Welcome to the new data science world of web apps")
st.header("It works in converting ML and Data science projects in making web apps")

df=sns.load_dataset('iris')

st.write(df.head(3))

st.write(df['species'].unique())

st.bar_chart(df['sepal_length'].loc[:20])

st.line_chart(df['petal_length'].loc[:20])