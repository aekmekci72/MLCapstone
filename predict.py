import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

df = pd.read_csv('/content/DiseaseAndSymptoms.csv')
df.head()

def predict_disease(user_symptoms, mlb, model):
    user_symptoms = [symptom.lower().strip() for symptom in user_symptoms]
    user_symptoms_str = ','.join(user_symptoms)
    user_symptoms_binarized = mlb.transform([user_symptoms_str.split(',')])

    probabilities = model.predict(user_symptoms_binarized)

    predicted_disease = labels.columns[np.argmax(probabilities)]

    return predicted_disease

user_symptoms = ['yellowish_skin','vomiting','indigestion','loss_of_appetite']
predicted_disease = predict_disease(user_symptoms, mlb, model)
print(f"Predicted Disease: {predicted_disease}")

df_precautions = pd.read_csv('/content/Disease precaution.csv')
precautions = df_precautions[df_precautions['Disease'] == predicted_disease].iloc[:, 1:].values.flatten()
precautions_list = [precaution.strip() for precaution in precautions if pd.notna(precaution)]
print(f"Precautions: {', '.join(precautions)}")
