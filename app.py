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

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8501))
    st._webbrowser.open_new = lambda *args, **kwargs: None
    st.run(port=port)
