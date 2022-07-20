from turtle import color
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score


# make container
header=st.container()
dataset=st.container()
features=st.container()
model_train=st.container()

with header:
    st.title("Titanic Dataset dashboard")
    st.text("In this project we will make titanic app")

with dataset:
    st.header("Titanic Dataset")
    st.text("we will work on famous titanic dataset")
    # import data
    df=sns.load_dataset("titanic")
    # dropping nan values
    df=df.dropna()
    st.write(df.head())
    st.header("Titanic male and female values counts chart")
    st.bar_chart(df['sex'].value_counts())
    st.subheader("Class chart")
    st.bar_chart(df['class'].value_counts())
    st.subheader("Age of titanic passengers")
    st.bar_chart(df['age'][:30])

    


with features:
    st.header("This is our App features:")
    st.text("What's the feaures are for this app dashboard")
    st.markdown('1. **Feature 1:** This will tell us about feature 1')
    st.markdown('2. **Feature 2:** This will tell us about feature 2')
    st.markdown('3. **Feature 3:** This will tell us about feature 3')
    st.markdown('4. **Feature 4:** This will tell us about feature 4')
with model_train:
    st.header("Titanic model training (who survived)")
    st.text("We will train our model to predict something")

    # making columns
    input,display=st.columns(2)
    # first column = selection column
    max_depth=input.slider("How many people do you know",min_value=10,max_value=100,value=20,step=5)

# n-estimar
n_estimators=input.selectbox("How many trees should be there in a random forest",options=[50,100,200,300,"No limit"])

# adding list of features
input.write(df.columns)

# input features from display
input_features=input.text_input("Which features we should use?")

# machine learning model
model=RandomForestRegressor(max_depth=max_depth,n_estimators=n_estimators)

if n_estimators=='No limit':
    model=RandomForestRegressor(max_depth=max_depth)
else:
    model=RandomForestRegressor(max_depth=max_depth,n_estimators=n_estimators)


x=df[[input_features]]
y=df[['survived']]

model.fit(x,y)
pred=model.predict(y)


#Display metrices
display.subheader("Mean absolute error of model is: ")
display.write(mean_absolute_error(y,pred))
display.subheader("Mean squared error of model is: ")
display.write(mean_squared_error(y,pred))
display.subheader("R2 score of model is: ")
display.write(r2_score(y,pred))
display.subheader("Prediction is: ")
display.write(pred)





