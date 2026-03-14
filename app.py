import joblib
import streamlit as st
import pandas as pd


# Load trained files
Expected_column = joblib.load('Columns.pkl')
Model = joblib.load('Logistic_Regression.pkl')
Scaler = joblib.load('scaler.pkl')


# App Title
st.title('💔 Heart Disease Prediction System')
st.markdown('### Enter Patient Details')


# User Inputs
age = st.slider('Age',18,100,40)

sex = st.selectbox('Sex',['Male','Female'])

chestpain = st.selectbox(
    'Chest Pain Type',
    ['ATA','NAP','TA','ASY']
)

RestingBP = st.number_input(
    'Resting Blood Pressure (mm/Hg)',
    80,200,120
)

Cholestrol = st.number_input(
    'Cholesterol (mg/dL)',
    100,600,200
)

Fastingbs = st.selectbox(
    'Fasting Blood Sugar > 120 mg/dL',
    [0,1]
)

restingecg = st.selectbox(
    'Resting ECG',
    ['Normal','ST','LVH']
)

maxhr = st.slider(
    'Maximum Heart Rate',
    60,220,150
)

exang = st.selectbox(
    'Exercise Induced Angina',
    [0,1]
)

oldpeak = st.number_input(
    'Old Peak',
    0.0,10.0,1.0
)

slope = st.selectbox(
    'Slope of Peak Exercise ST Segment',
    ['Up','Flat','Down']
)


# Predict Button
if st.button('Predict Heart Disease Risk'):


    # Create dictionary with all columns
    input_dict = {col:0 for col in Expected_column}


    # Numeric values
    input_dict['Age'] = age
    input_dict['RestingBP'] = RestingBP
    input_dict['Cholesterol'] = Cholestrol
    input_dict['FastingBS'] = Fastingbs
    input_dict['MaxHR'] = maxhr
    input_dict['Oldpeak'] = oldpeak


    # One Hot Encoding

    if sex == 'Male':
        input_dict['Sex_M'] = 1


    if chestpain == 'ATA':
        input_dict['ChestPainType_ATA'] = 1

    elif chestpain == 'NAP':
        input_dict['ChestPainType_NAP'] = 1

    elif chestpain == 'TA':
        input_dict['ChestPainType_TA'] = 1


    if restingecg == 'Normal':
        input_dict['RestingECG_Normal'] = 1

    elif restingecg == 'ST':
        input_dict['RestingECG_ST'] = 1


    if exang == 1:
        input_dict['ExerciseAngina_Y'] = 1


    if slope == 'Flat':
        input_dict['ST_Slope_Flat'] = 1

    elif slope == 'Up':
        input_dict['ST_Slope_Up'] = 1


    # Convert input to dataframe
    df = pd.DataFrame([input_dict])


    # Ensure correct column order
    df = df[Expected_column]


    try:

        # Scale numeric features
        numeric_cols = [
            'Age',
            'RestingBP',
            'Cholesterol',
            'MaxHR',
            'Oldpeak'
        ]

        df[numeric_cols] = Scaler.transform(df[numeric_cols])


        # Model prediction
        prediction = Model.predict(df)[0]


        # Probability prediction
        proba = Model.predict_proba(df)[0]

        low_risk_prob = proba[0] * 100
        high_risk_prob = proba[1] * 100


        # Display Prediction Confidence
        st.subheader("Prediction Confidence")


        col1, col2 = st.columns(2)

        col1.metric(
            "Low Risk Probability",
            f"{low_risk_prob:.2f}%"
        )

        col2.metric(
            "High Risk Probability",
            f"{high_risk_prob:.2f}%"
        )


        # Risk progress bar
        st.progress(int(high_risk_prob))


        # Final result
        if prediction == 1:

            st.error(
                f"⚠️ High Risk of Heart Disease ({high_risk_prob:.2f}%)"
            )

        else:

            st.success(
                f"✅ Low Risk of Heart Disease ({low_risk_prob:.2f}%)"
            )


    except Exception as e:

        st.error(f"Prediction Error: {e}")