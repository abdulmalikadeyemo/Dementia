import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pickle
from sklearn.preprocessing import StandardScaler

st.write("""
# Dementia Prediction App

Dementia is the 7th leading cause of death among all diseases 
and one of the major causes of disability and dependency among 
older people globally. It has physical, psychological, social 
and economic consequences, not only for people living with dementia, 
but also for their carers, families and society at large. 
There is often a lack of awareness and understanding of dementia, 
resulting in stigmatization and barriers to diagnosis and care.

This app predicts weather a patient has Dementia or not!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    #mr_delay = st.sidebar.number_input('MR Delay', min_value=0, max_value=2639, step=1)
    sex = st.sidebar.selectbox('Select your gender', ['Male', 'Female'])
    age = st.sidebar.number_input('Age', min_value=60, max_value=98, step=1)
    educ = st.sidebar.number_input('EDUC', min_value=6, max_value=23, step=1)
    mmse = st.sidebar.number_input('MMSE', min_value=4, max_value=30, step=1)
    cdr = st.sidebar.slider('CDR', 0.0,2.0)
    e_tiv = st.sidebar.number_input('eTIV', min_value=1106, max_value=2004, step=1)
    n_wbv = st.sidebar.number_input('nWBV', min_value=0.644, max_value=0.837, step=0.001,format="%.3f")
    #asf = st.sidebar.number_input('ASF', min_value=0.876, max_value=1.587, step=0.001, format="%.3f")


    data = {
            'Sex': sex,
            'Age' : age,
            'EDUC' : educ,
            'MMSE': mmse,
            'CDR' : cdr,
            'eTIV': e_tiv,
            'nWBV': n_wbv,
             }
            
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()
df_copy = df.copy()
mapping = {'Male':1,'Female':0}

df_copy['Sex'] = df.Sex.map(mapping)



# Scaling  input parameters
scaler = StandardScaler() #create an object of StandardScaler
X_norm = scaler.fit_transform(df_copy) #transform the X features by calling the 'fit_transform' method 

# iris = datasets.load_iris()
# X = iris.data
# Y = iris.target

# clf = RandomForestClassifier()
# clf.fit(X, Y)
target_names = pd.DataFrame({0:['Nondemented'],1:['Demented'], 2:['Converted']})


# load the model from disk
model = pickle.load(open('dementia_model.pkl', 'rb'))
# result = loaded_model.score(X_test, Y_test)
# print(result)



st.subheader('Class labels and their corresponding index number')
st.write((target_names))

st.subheader('User Input parameters')
st.write(df)
if st.button('Predict'):

    prediction = model.predict(X_norm)
    prediction_proba = model.predict_proba(X_norm)

    st.subheader('Prediction')
    st.write(target_names[prediction])
    #st.write(prediction)

    st.subheader('Prediction Probability')
    st.write(prediction_proba)
else:
    st.write('Click **Predict** to make a new prediction')
