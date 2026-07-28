import streamlit as st
import pandas as pd
import pickle


# Load trained model files
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scalar.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))


# Streamlit Page Configuration
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="centered"
)


# Title
st.title("🚢 Titanic Survival Prediction")
st.write("Enter passenger details to predict survival probability.")


# User Inputs

Pclass = st.selectbox(
    "Passenger Class",
    [1, 2, 3]
)

Sex = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

Age = st.number_input(
    "Age",
    min_value=0,
    max_value=100,
    value=25
)

Fare = st.number_input(
    "Fare",
    min_value=0.0,
    value=30.0
)

SibSp = st.number_input(
    "Siblings / Spouses",
    min_value=0,
    value=0
)

Parch = st.number_input(
    "Parents / Children",
    min_value=0,
    value=0
)

Embarked = st.selectbox(
    "Embarked Location",
    ["C", "Q", "S"]
)


# Convert categorical values

sex_value = 1 if Sex == "Male" else 0

embarked_value = {
    "C": 0,
    "Q": 1,
    "S": 2
}

embarked_value = embarked_value[Embarked]


# Prediction Button

if st.button("Predict Survival"):

    input_data = pd.DataFrame(
        {
            "Age": [Age],
            "Fare": [Fare],
            "Sex": [sex_value],
            "Pclass": [Pclass],
            "SibSp": [SibSp],
            "Parch": [Parch],
            "Embarked": [embarked_value]
        }
    )


    # Match training columns
    input_data = input_data.reindex(
        columns=columns,
        fill_value=0
    )


    # Scaling
    scaled_data = scaler.transform(input_data)


    # Prediction
    prediction = model.predict(scaled_data)[0]


    # Result

    if prediction == 1:
        st.success("🎉 Passenger would have survived!")
    else:
        st.error("😔 Passenger would not have survived.")


    # Probability (if supported)
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(scaled_data)

        survival_probability = probability[0][1] * 100

        st.info(
            f"Survival Probability: {survival_probability:.2f}%"
        )
