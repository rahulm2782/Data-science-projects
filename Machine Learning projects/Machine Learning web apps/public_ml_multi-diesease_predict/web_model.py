import pickle
import  streamlit as st
from streamlit_option_menu import option_menu
import  numpy as np
import pandas as pd


# loding model
Diabities_model = pickle.load(open('diabetes_model.sav','rb'))
Heart_disease_model = pickle.load(open('heart_disease_model.sav','rb'))
parkinson_model = pickle.load(open('parkinsons_model.sav','rb'))

# side bar for navigation
with st.sidebar:
    selected = option_menu('Multi Disease prediction system',['Diabities Prediction',
                                                              'Heart Disease prediction',
                                                              'Parkinson prediction'],
                                                               icons=['activity','heart','person'],
                           default_index = 0)


# providing condition for models

if selected == 'Diabities Prediction':
    st.title('Diabities Prediction using ML model')

    col1,col2,col3 = st.columns(3)
    with col1:
        Pregnancies = st.text_input('Pregnancies')

    with col2:
        Glucose = st.text_input('Glucose')

    with col3:
        BloodPressure = st.text_input('BloodPressure')

    with col1:
        SkinThickness = st.text_input('SkinThickness')

    with col2:
        Insulin = st.text_input('Insulin')

    with col3:
         BMI = st.text_input('BMI')

    with col1:
        DiabetesPedigreeFunction = st.text_input('DiabetesPedigreeFunction')

    with col2:
         Age = st.text_input('Age')

    Diabities_predict = ''



    if st.button('Test Result'):
        prediction = Diabities_model.predict(
            [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])

        if (prediction[0] == 1):

                Diabities_predict = 'Diabitic'

        else:
                Diabities_predict = 'Non Diabitic'

    st.success(Diabities_predict)


#---------------------------end of part 1 ------------------------------------------

#heart Disease

if selected == 'Heart Disease prediction':

    st.title('Heart Disease prediction using ML model')
    age = st.sidebar.slider('age',25,80,35)
    sex = st.sidebar.slider('sex',0,1,0)
    cp = st.sidebar.slider('cp',0,4,2)
    trestbps = st.sidebar.slider('trestbps',80,220,90)
    chol = st.sidebar.slider('Chol',110,300,130)
    fbs = st.sidebar.slider('fbs',0,1,0)
    restecg = st.sidebar.slider('restecg',0,1,1)
    thalach = st.sidebar.slider('thalach',60,200,130)
    exang = st.sidebar.slider('exang',0,1,1)
    oldpeak = st.sidebar.slider('oldpeak',0,3,1)
    ca = st.sidebar.slider('ca',0,1,1)
    slope = st.sidebar.slider('slope',0,1,1)
    thal = st.sidebar.slider('thal',0,1,1)
    
    params = {'age':age,'sex':sex,'cp':cp,'trestbps':trestbps,'chol':chol,'fbs':fbs,'restecg':restecg, 'thalach':thalach,'exang':exang,'oldpeak':oldpeak,'slope':slope,'ca':ca,'thal':thal}
    df = pd.DataFrame(params,index=[0])
    st.write(df)
    
    

    Heart_disease_pred = ''

    

    if st.button('Test Result'):
        prediction = Heart_disease_model.predict(df)
        if (prediction[0]==1):
            diabities_pred = 'Has Heart Disease'

        else:
            diabities_pred = 'Healthy Heart'

        st.success(diabities_pred)

#-------------------------------end of part 2 --------------------------------------

# Parkinson Disease

if selected == 'Parkinson prediction':

    st.title('Parkinson Prediction using ML model')
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')

    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')

    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')

    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')

    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')

    with col1:
        RAP = st.text_input('MDVP:RAP')

    with col2:
        PPQ = st.text_input('MDVP:PPQ')

    with col3:
        DDP = st.text_input('Jitter:DDP')

    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')

    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')

    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')

    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')

    with col3:
        APQ = st.text_input('MDVP:APQ')

    with col4:
        DDA = st.text_input('Shimmer:DDA')

    with col5:
        NHR = st.text_input('NHR')

    with col1:
        HNR = st.text_input('HNR')

    with col2:
        RPDE = st.text_input('RPDE')

    with col3:
        DFA = st.text_input('DFA')

    with col4:
        spread1 = st.text_input('spread1')

    with col5:
        spread2 = st.text_input('spread2')

    with col1:
        D2 = st.text_input('D2')

    with col2:
        PPE = st.text_input('PPE')

    Parkinson_pred = ''

    if st.button('Test Result'):
        prediction = parkinson_model.predict([[
            fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR,
            HNR, RPDE, DFA, spread1, spread2, D2, PPE
        ]])

        if (prediction[0]==1):
            Parkinson_pred = 'Parkinson Positive'

        else:
            Parkinson_pred = 'Healthy'

        st.success(Parkinson_pred)






