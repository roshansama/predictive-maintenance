import pickle
import pandas as pd
import streamlit as st
import numpy as np
import shap

# Load the trained model
with open("decision_tree_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Define the correct feature order
features = ["Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]", "TWF", "HDF", "PWF", "OSF"]

# App title and description
st.title("Predictive Maintenance for Industrial Equipment test 1")
st.write("This application predicts equipment failures based on user-provided inputs.")

# Sidebar inputs (RAW values)
input_data = {
    "Rotational speed [rpm]": st.sidebar.number_input("Rotational speed [rpm]", value=2000.0),
    "Torque [Nm]": st.sidebar.number_input("Torque [Nm]", value=50.0),
    "Tool wear [min]": st.sidebar.number_input("Tool wear [min]", value=120.0),
    "TWF": st.sidebar.number_input("TWF", value=0, min_value=0, max_value=1),
    "HDF": st.sidebar.number_input("HDF", value=0, min_value=0, max_value=1),
    "PWF": st.sidebar.number_input("PWF", value=0, min_value=0, max_value=1),
    "OSF": st.sidebar.number_input("OSF", value=0, min_value=0, max_value=1),
}

# Convert inputs to DataFrame
input_df = pd.DataFrame([input_data])

# Standardize numerical features
numerical_features = ["Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
standardized_data = input_df.copy()
standardized_data[numerical_features] = scaler.transform(input_df[numerical_features])

# Remedies and Problem Causes
feature_details = {
    "Rotational speed [rpm]": {
        "problem": "High rotational speed can cause overheating, wear and tear, and mechanical stress.",
        "remedy": (
            "1. Implement automatic speed regulation to maintain safe operational limits.\n"
            "2. Schedule periodic inspections of the motor and gearbox to ensure proper lubrication.\n"
            "3. Use variable frequency drives (VFDs) to adjust speed dynamically based on load requirements."
        ),
    },
    "Torque [Nm]": {
        "problem": "Excessive torque can strain the motor and other mechanical components, leading to premature failure.",
        "remedy": (
            "1. Install torque limiters to prevent excessive torque buildup.\n"
            "2. Calibrate tools and equipment regularly to ensure accurate torque settings.\n"
            "3. Use real-time torque monitoring sensors to identify abnormalities early."
        ),
    },
    "Tool wear [min]": {
        "problem": "Worn-out tools can reduce efficiency, compromise product quality, and damage equipment.",
        "remedy": (
            "1. Replace tools based on a predictive maintenance schedule rather than after failure.\n"
            "2. Use wear-resistant tool materials (e.g., tungsten carbide or ceramic).\n"
            "3. Implement tool monitoring systems that measure wear in real-time and alert operators."
        ),
    },
    "TWF": {
        "problem": "Tool wear failure occurs when tools are used beyond their operational lifespan, causing damage to the equipment.",
        "remedy": (
            "1. Establish a tool replacement threshold and monitor it rigorously.\n"
            "2. Train operators to identify early signs of wear, such as changes in cutting sound or surface finish quality."
        ),
    },
    "HDF": {
        "problem": "Poor heat dissipation leads to overheating, which can damage motors and bearings.",
        "remedy": (
            "1. Install cooling systems such as heat sinks, fans, or water cooling.\n"
            "2. Clean and maintain ventilation systems to ensure unobstructed airflow.\n"
            "3. Apply thermal monitoring sensors to track and alert for overheating."
        ),
    },
    "PWF": {
        "problem": "Power supply fluctuations can disrupt operations and damage equipment.",
        "remedy": (
            "1. Install surge protectors and voltage regulators.\n"
            "2. Monitor power quality and stabilize fluctuations with uninterruptible power supplies (UPS)."
        ),
    },
    "OSF": {
        "problem": "Operational settings outside safe ranges can lead to system failure.",
        "remedy": (
            "1. Regularly calibrate machines to ensure settings are within safe limits.\n"
            "2. Implement real-time monitoring to alert operators of deviations."
        ),
    },
}

# Perform prediction on standardized data
if st.button("Predict"):
    try:
        prediction = model.predict(standardized_data)
        result = "Failure" if prediction[0] == 1 else "No Failure"

        # Display the raw input values
        st.subheader("Input Values")
        st.write(input_df)

        # Display the prediction result
        st.subheader("Prediction Result")
        st.write(f"Prediction: {result}")

        # Show most important feature dynamically using SHAP
        if result == "Failure":
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(standardized_data)

            # Handle binary classification SHAP values
            if isinstance(shap_values, list):  # Multi-class SHAP values
                shap_values_for_class = shap_values[1]  # Class 1 (Failure)
            else:  # Binary classification SHAP values
                shap_values_for_class = shap_values

            # Get the SHAP values for the specific input
            individual_shap_values = shap_values_for_class[0]

            # Dynamically calculate the most important feature for this prediction
            most_important_idx = np.argmax(np.abs(individual_shap_values))
            most_important_feature = features[most_important_idx]

            # Fetch problem and remedy dynamically
            problem = feature_details[most_important_feature]["problem"]
            remedy = feature_details[most_important_feature]["remedy"]

            # Display the most important feature, problem, and remedy
            st.subheader("Key Insights")
            st.write(f"The most important factor causing failure is **{most_important_feature}**.")

            # Use expander for details
            with st.expander("Details (Click to Expand)"):
                st.write(f"**Problem**: {problem}")
                st.write(f"**Suggested Remedies**:\n{remedy}")
    except Exception as e:
        st.error(f"Error in prediction: {e}")
import pickle
import pandas as pd
import streamlit as st
import numpy as np
import shap

# Load the trained model
with open("decision_tree_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Define the correct feature order
features = ["Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]", "TWF", "HDF", "PWF", "OSF"]

# App title and description
st.title("Predictive Maintenance for Industrial Equipment")
st.write("This application predicts equipment failures based on user-provided inputs.")

# Sidebar inputs (RAW values)
input_data = {
    "Rotational speed [rpm]": st.sidebar.number_input("Rotational speed [rpm]", value=2000.0),
    "Torque [Nm]": st.sidebar.number_input("Torque [Nm]", value=50.0),
    "Tool wear [min]": st.sidebar.number_input("Tool wear [min]", value=120.0),
    "TWF": st.sidebar.number_input("TWF", value=0, min_value=0, max_value=1),
    "HDF": st.sidebar.number_input("HDF", value=0, min_value=0, max_value=1),
    "PWF": st.sidebar.number_input("PWF", value=0, min_value=0, max_value=1),
    "OSF": st.sidebar.number_input("OSF", value=0, min_value=0, max_value=1),
}

# Convert inputs to DataFrame
input_df = pd.DataFrame([input_data])

# Standardize numerical features
numerical_features = ["Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
standardized_data = input_df.copy()
standardized_data[numerical_features] = scaler.transform(input_df[numerical_features])

# Remedies and Problem Causes
feature_details = {
    "Rotational speed [rpm]": {
        "problem": "High rotational speed can cause overheating, wear and tear, and mechanical stress.",
        "remedy": (
            "1. Implement automatic speed regulation to maintain safe operational limits.\n"
            "2. Schedule periodic inspections of the motor and gearbox to ensure proper lubrication.\n"
            "3. Use variable frequency drives (VFDs) to adjust speed dynamically based on load requirements."
        ),
    },
    "Torque [Nm]": {
        "problem": "Excessive torque can strain the motor and other mechanical components, leading to premature failure.",
        "remedy": (
            "1. Install torque limiters to prevent excessive torque buildup.\n"
            "2. Calibrate tools and equipment regularly to ensure accurate torque settings.\n"
            "3. Use real-time torque monitoring sensors to identify abnormalities early."
        ),
    },
    "Tool wear [min]": {
        "problem": "Worn-out tools can reduce efficiency, compromise product quality, and damage equipment.",
        "remedy": (
            "1. Replace tools based on a predictive maintenance schedule rather than after failure.\n"
            "2. Use wear-resistant tool materials (e.g., tungsten carbide or ceramic).\n"
            "3. Implement tool monitoring systems that measure wear in real-time and alert operators."
        ),
    },
    "TWF": {
        "problem": "Tool wear failure occurs when tools are used beyond their operational lifespan, causing damage to the equipment.",
        "remedy": (
            "1. Establish a tool replacement threshold and monitor it rigorously.\n"
            "2. Train operators to identify early signs of wear, such as changes in cutting sound or surface finish quality."
        ),
    },
    "HDF": {
        "problem": "Poor heat dissipation leads to overheating, which can damage motors and bearings.",
        "remedy": (
            "1. Install cooling systems such as heat sinks, fans, or water cooling.\n"
            "2. Clean and maintain ventilation systems to ensure unobstructed airflow.\n"
            "3. Apply thermal monitoring sensors to track and alert for overheating."
        ),
    },
    "PWF": {
        "problem": "Power supply fluctuations can disrupt operations and damage equipment.",
        "remedy": (
            "1. Install surge protectors and voltage regulators.\n"
            "2. Monitor power quality and stabilize fluctuations with uninterruptible power supplies (UPS)."
        ),
    },
    "OSF": {
        "problem": "Operational settings outside safe ranges can lead to system failure.",
        "remedy": (
            "1. Regularly calibrate machines to ensure settings are within safe limits.\n"
            "2. Implement real-time monitoring to alert operators of deviations."
        ),
    },
}

# Perform prediction on standardized data
if st.button("Predict"):
    try:
        prediction = model.predict(standardized_data)
        result = "Failure" if prediction[0] == 1 else "No Failure"

        # Display the raw input values
        st.subheader("Input Values")
        st.write(input_df)

        # Display the prediction result
        st.subheader("Prediction Result")
        st.write(f"Prediction: {result}")

        # Show most important feature dynamically using SHAP
        if result == "Failure":
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(standardized_data)

            # Handle binary classification SHAP values
            if isinstance(shap_values, list):  # Multi-class SHAP values
                shap_values_for_class = shap_values[1]  # Class 1 (Failure)
            else:  # Binary classification SHAP values
                shap_values_for_class = shap_values

            # Get the SHAP values for the specific input
            individual_shap_values = shap_values_for_class[0]

            # Dynamically calculate the most important feature for this prediction
            most_important_idx = np.argmax(np.abs(individual_shap_values))
            most_important_feature = features[most_important_idx]

            # Fetch problem and remedy dynamically
            problem = feature_details[most_important_feature]["problem"]
            remedy = feature_details[most_important_feature]["remedy"]

            # Display the most important feature, problem, and remedy
            st.subheader("Key Insights")
            st.write(f"The most important factor causing failure is **{most_important_feature}**.")

            # Use expander for details
            with st.expander("Details (Click to Expand)"):
                st.write(f"**Problem**: {problem}")
                st.write(f"**Suggested Remedies**:\n{remedy}")
    except Exception as e:
        st.error(f"Error in prediction: {e}")
