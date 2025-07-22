import streamlit as st
import pandas as pd
import joblib as jb
from sklearn.preprocessing import StandardScaler as scl

model = jb.load('LR_heart.pkl')
Scaler = jb.load('scaler.pkl')
expected_columns = jb.load('columns.pkl')

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️")
st.title("Heart Disease Prediction ❤️")
st.markdown('Provide the following details')

age =st.number_input('Age',0,100,20)
sex = st.selectbox("Sex",["Choose any","M","F"])
chest_pain = st.selectbox("chest Pain Type",['Choose any','ATA','NAP',"TA",'ASY'])
Resting_bp = st.number_input('resting BP(mm Hg)',80,200,120)
Cholesterol = st.number_input('Cholesterol(mg/dL)',100,600,200)
Fasting_bs = st.selectbox('Fasting bs >120 mg/dL',[0,1])
Resting_ecg = st.selectbox('Rsting_Ecg ',['Normal','ST',"LVM"])
Max_hr = st.slider('Max Heart Rate',60,220,150)
exercise_agina = st.selectbox('exercise-induces- Agina ',['Y','N'])
oldpeak = st.slider('Oldpeak (ST depression) ',0.0,6.0,1.0)
St_slope = st.selectbox('ST Slope',['Choose any','Up','Flat','Down'])


if st.button('predict'):
    if (
        sex == "Choose any" or
        chest_pain == "Choose any" or
        Resting_ecg == "Choose any" or
        exercise_agina == "Choose any" or
        St_slope == "Choose any"
    ):
        st.warning("⚠️ Please select all dropdown options before predicting.")
    else:
        raw_input ={
                'Age': age,
        'RestingBP': Resting_bp,
        'Cholesterol': Cholesterol,
        'FastingBS': Fasting_bs,
        'MaxHR': Max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + Resting_ecg: 1,
        'ExerciseAngina_' + exercise_agina: 1,
        'ST_Slope_' + St_slope: 1
        }

        input_df = pd.DataFrame([raw_input])

        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[expected_columns]

        scaled_input = Scaler.transform(input_df)
        
        # st.subheader("🔍 Debug Info")
        # st.write("📄 Input DataFrame (after encoding):")
        # st.dataframe(input_df)

        # scaled_input = Scaler.transform(input_df)

        # st.write("🔧 Scaled Input:")
        # st.write(scaled_input)

        # st.write("🧩 Expected columns:")
        # st.write(expected_columns)

        # st.write("🧩 Input columns:")
        # st.write(input_df.columns)

        prediction = model.predict(scaled_input)[0]

        
        if prediction == 1:
            st.error("⚠️ High on Risk of Heart Disease")
        else:
            st.success("✅ Low risk of Heart disease")


