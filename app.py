import streamlit as st
import pandas as pd
import joblib as jb
from sklearn.preprocessing import StandardScaler as scl

model = jb.load('LR_heart.pkl')
Scaler = jb.load('scaler.pkl')
expected_columns = jb.load('columns.pkl')


st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️")
st.title("Heart Disease Risk Prediction ❤️")
st.markdown("Please provide the following details:")


age = st.number_input('Age', min_value=0, max_value=100, value=20)


sex = st.selectbox("Sex", ["Choose any", "Male", "Female"])
sex_map = {"Male": "M", "Female": "F"}


with st.expander("ℹ️ Chest Pain Type Info"):
    st.markdown("""
    - **ATA**: Atypical Angina  
    - **NAP**: Non-Anginal Pain  
    - **TA**: Typical Angina  
    - **ASY**: Asymptomatic  
    - **None**: No chest pain or not sure
    """)
chest_pain = st.selectbox(
    "Chest Pain Type",
    ['Choose any', 'ATA', 'NAP', "TA", 'ASY', 'None']
)


Resting_bp = st.number_input('Resting BP (mm Hg)', min_value=80, max_value=200, value=120)


Cholesterol = st.number_input('Cholesterol (mg/dL)', min_value=100, max_value=600, value=200)


Fasting_bs_str = st.selectbox('Fasting Blood Sugar > 120 mg/dL', ['Choose any', 'No', 'Yes'])
Fasting_bs = 1 if Fasting_bs_str == 'Yes' else 0 if Fasting_bs_str == 'No' else None


with st.expander("ℹ️ Resting ECG Info"):
    st.markdown("""
    - **Normal**: Normal ECG reading  
    - **ST**: ST-T wave abnormality  
    - **LVM**: Left Ventricular Hypertrophy
    """)
Resting_ecg = st.selectbox('Resting ECG', ['Choose any', 'Normal', 'ST', "LVM"])


Max_hr = st.number_input('Max Heart Rate', min_value=60, max_value=220, value=150)


exercise_agina_str = st.selectbox('Exercise-Induced Angina', ['Choose any', 'No', 'Yes'])
exercise_agina = 'Y' if exercise_agina_str == 'Yes' else 'N' if exercise_agina_str == 'No' else None


oldpeak = st.number_input('Oldpeak (ST Depression)', min_value=0.0, max_value=6.0, value=1.0, step=0.1)

with st.expander("ℹ️ ST Slope Info"):
    st.markdown("""
    - **Up**: Upsloping ST segment  
    - **Flat**: Flat ST segment  
    - **Down**: Downsloping ST segment
    """)
St_slope = st.selectbox('ST Slope', ['Choose any', 'Up', 'Flat', 'Down'])


if st.button('Predict'):
    if (
        sex == "Choose any" or
        chest_pain == "Choose any" or
        Resting_ecg == "Choose any" or
        St_slope == "Choose any" or
        Fasting_bs is None or
        exercise_agina is None
    ):
        st.warning("⚠️ Please select all dropdown options before predicting.")
    else:
        raw_input = {
            'Age': age,
            'RestingBP': Resting_bp,
            'Cholesterol': Cholesterol,
            'FastingBS': Fasting_bs,
            'MaxHR': Max_hr,
            'Oldpeak': oldpeak,
            'Sex_' + sex_map[sex]: 1,
        }

       
        if chest_pain != 'None':
            raw_input['ChestPainType_' + chest_pain] = 1

        raw_input['RestingECG_' + Resting_ecg] = 1
        raw_input['ExerciseAngina_' + exercise_agina] = 1
        raw_input['ST_Slope_' + St_slope] = 1

        
        input_df = pd.DataFrame([raw_input])

        
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[expected_columns]

        
        scaled_input = Scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]

       
        if prediction == 1:
            st.error("⚠️ High Risk of Heart Disease")
        else:
            st.success("✅ Low Risk of Heart Disease")
