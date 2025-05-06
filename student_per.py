import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_model():
    with open('student_pmodel.pkl', 'rb') as file:
        model, scaler, le = pickle.load(file)
    return model, scaler, le

def preprocess_data(data, scaler, le):
    data["Extracurricular Activities"]=le.transform([data["Extracurricular Activities"]])
    df=pd.DataFrame([data])
    df_transformed=scaler.transform(df)
    return df_transformed

def predict_performance(data):
    model, scaler, le = load_model()
    data_transformed = preprocess_data(data, scaler, le)
    prediction = model.predict(data_transformed)
    return prediction

def main():
    st.title("Student Performance Prediction App")
    st.write("Enter your data to predict student performance.")
    st.sidebar.title("Input Features")

    hours_studied=st.number_input("Hours of Study", min_value=0, max_value=24, value=5, step=1)
    previous_score=st.number_input("Previous score", min_value=40, max_value=100, value=75, step=1)
    extra_cur_activity=st.selectbox("Extra curricular activity", ["Yes", "No"])
    sleeping_hours=st.number_input("Sleeping hours", min_value=3, max_value=12, value=8, step=1)
    number_question_paper=st.number_input("Number of question paper solved", min_value=1, max_value=24, value=5, step=1)

    if st.button("Predict"):
        data = {
            "Hours Studied": hours_studied,
            "Previous Scores": previous_score,
            "Extracurricular Activities": extra_cur_activity,
            "Sleep Hours": sleeping_hours,
            "Sample Question Papers Practiced": number_question_paper
        }
        
        prediction = predict_performance(data)
        st.success(f"Predicted Performance: {prediction}")

if __name__== "__main__":
    main()