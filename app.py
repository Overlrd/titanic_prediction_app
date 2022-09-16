import streamlit as st
from utils import PrepProcesor, columns

import numpy as np
import pandas as pd
import joblib

model = joblib.load('xgbpipe.joblib')

st.title('Check if Survive :ship:')

passengerid =st.text_input('Input Passenger Id', '1234')
passengerclass= st.select_slider('Choose Passenger Class',[1,2,3])
name = st.text_input('Passenger Name', 'John Doe')
gender = st.select_slider('Select Passenger Gender', ['male','female'])
age = st.slider('Passenger Age',0,100)
sibsp = st.slider('Passenger Siblings', 0,10)
parch = st.slider('Parents or Childrens', 0,2)
tickedId = st.number_input('Ticket Number', 0,10000)
fare = st.number_input('Fare Amount', 0,1000)
cabin = st.text_input('Enter Cabin', 'CS2')
embarked = st.selectbox('Choose Embarkation Point', ['S','C','Q'])


def predict():
	row = np.array([passengerid, passengerclass, name , gender, age , sibsp , parch , tickedId, fare, cabin, embarked])
	x = pd.DataFrame([row], columns=columns)
	prediction = model.predict(x)[0]
	if prediction==1:
		st.success('Passenger Survived :thumbsup:')
	else:
		st.error('Passenger Did Not Survived :thumbsdown:')


st.button('Predict', on_click=predict)

