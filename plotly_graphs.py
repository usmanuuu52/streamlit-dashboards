import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px

st.title("Streamlit and Plotly based web app")
df=px.data.gapminder()
st.write(df)
st.write(df.columns)

st.write(df.describe())

year_opt=df['year'].unique().tolist()
year=st.selectbox("Which year should we plot",year_opt,0)
df=df[df['year']==year]

# plotting
fig=px.scatter(df,x='gdpPercap',y='lifeExp',size='pop',color='country',hover_name='country',log_x=True,
size_max=25,range_x=[100,100000],range_y=[20,90])
fig.update_layout(width=1000,height=600)
st.write(fig)



