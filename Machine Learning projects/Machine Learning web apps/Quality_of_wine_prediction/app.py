import pandas as pd
import numpy as np
import joblib
import streamlit as st
from streamlit_option_menu import option_menu
from matplotlib import  pyplot as plt
import seaborn as sb

RF_model = joblib.load('Random_forest')
data = pd.read_csv('wine_quality')
with st.sidebar:
    selected = option_menu('Wine quality prediction',['Wine_quality'])

if selected == 'Wine_quality':
    st.title('Wine Quality Prediction System')
    st.image('wine_image.jpg',width=700)

    col1,col2 = st.columns(2)
    with col1:
        Show_db = st.checkbox('Show database')
    with col2:
        show_corr = st.checkbox('Heat map')
    if Show_db:
        st.dataframe(data.iloc[:,1:-1].head())

    if show_corr:
        plt.figure(figsize=(10,5))
        sb.heatmap(data.corr(),cbar=True,cmap='Blues',annot_kws={'size':10},annot=True,fmt='.1f')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()


    st.write()

    col3,col4,col5 = st.columns(3)
    with col3:
        type = st.selectbox('Wine Type',['White','Red'])

    with col4:
        fixed_acidity = st.number_input('Fixed acidity')
    with col5:
        volatile_acidity = st.number_input('Volatile acidity')

    col6,col7 = st.columns(2)

    with col6:
        residual_sugar = st.number_input('Residual sugar')
    with col7:
        chlorides = st.number_input('Chlorides')

    free_sulfur_dioxide = st.slider('Sulpherdioxide',1,200)
    total_sulfur_dioxide = st.slider('Total sulpherdioxide',1,500)

    col8,col9,col10,col11 = st.columns(4)
    with col8:
        density = st.number_input('Density')

    with col9:
        pH = st.number_input('pH')

    with col10:
        sulphates = st.number_input('Sulphates')

    with col11:
        alcohol = st.number_input('Alcohol')
    with col4:
        citric_acid = st.number_input('Citric acid')
    Wine_quality = ''
    params = {'type':type, 'fixed_acidity':fixed_acidity, 'volatile_acidity':volatile_acidity, 'citric_acid':citric_acid,
    'residual_sugar':residual_sugar, 'chlorides':chlorides, 'free_sulfur_dioxide':free_sulfur_dioxide,
    'total_sulfur_dioxide':total_sulfur_dioxide, 'density':density, 'pH':pH, 'sulphates':sulphates, 'alcohol':alcohol,}

    df = pd.DataFrame(params,index=[0])
    st.write(df)

    wine_qlty = ''

    if st.button('Test'):
        prediction = RF_model.predict([[
            type, fixed_acidity, volatile_acidity, citric_acid,
            residual_sugar, chlorides, free_sulfur_dioxide,
            total_sulfur_dioxide, density, pH, sulphates, alcohol
        ]])

        if prediction[0] ==1 :
            Wine_qlty = 'Good Quality'

        else:
            wine_qlty = 'Bad'

        st.success(wine_qlty)
