import pandas as pd
import numpy as np
import joblib

# Load model files
model = joblib.load("Logistic_Regression.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("Columns.pkl")

# Number of random patients
n = 500

# Create random dataset
data = pd.DataFrame({
    "Age": np.random.randint(18, 80, n),
    "RestingBP": np.random.randint(90, 180, n),
    "Cholesterol": np.random.randint(150, 350, n),
    "FastingBS": np.random.randint(0, 2, n),
    "MaxHR": np.random.randint(90, 200, n),
    "Oldpeak": np.round(np.random.uniform(0, 5, n),2),

    "Sex_M": np.random.randint(0,2,n),

    "ChestPainType_ATA": np.random.randint(0,2,n),
    "ChestPainType_NAP": np.random.randint(0,2,n),
    "ChestPainType_TA": np.random.randint(0,2,n),

    "RestingECG_Normal": np.random.randint(0,2,n),
    "RestingECG_ST": np.random.randint(0,2,n),

    "ExerciseAngina_Y": np.random.randint(0,2,n),

    "ST_Slope_Flat": np.random.randint(0,2,n),
    "ST_Slope_Up": np.random.randint(0,2,n)
})

# Match column order
data = data.reindex(columns=columns, fill_value=0)

# Scale numeric columns
numeric_cols = ['Age','RestingBP','Cholesterol','MaxHR','Oldpeak']
data[numeric_cols] = scaler.transform(data[numeric_cols])

# Predict
predictions = model.predict(data)

# Count predictions
unique, counts = np.unique(predictions, return_counts=True)

print("Prediction Summary")
print("------------------")
print(dict(zip(unique, counts)))


proba = model.predict_proba(data)

print(proba[:10])