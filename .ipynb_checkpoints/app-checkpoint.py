import streamlit as st
import pickle
import numpy as np

# Load the Naive Bayes model
with open("final_model.sav", "rb") as f:
    gnb_model = pickle.load(f)

with open("./styles.css") as f:
    css = f.read()

st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Define the input features
feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']



# Split the main content into two columns
col1, col2 = st.columns([1, 2])  # Adjust the width ratio as needed

# Add an image to the first column
with col1:
    st.image("heart.png", use_column_width=True)  # Replace "your_image_path_here.jpg" with the actual path to your image

# Add the title and description to the second column
with col2:
    # Define the app title and description
    st.title('Heart Disease Prediction')
    st.write('This software predicts if someone has heart disease by examining various input features.')

    # Arrange input fields into three columns
    col3, col4, col5 = st.columns(3)

    # Create input fields for user input
    inputs = []
    for i, feature_name in enumerate(feature_names):
        if i % 3 == 0:
            with col3:
                inputs.append(st.number_input(f'Enter {feature_name.capitalize()}', value=0.0, step=1.0))
        elif i % 3 == 1:
            with col4:
                inputs.append(st.number_input(f'Enter {feature_name.capitalize()}', value=0.0, step=1.0))
        else:
            with col5:
                inputs.append(st.number_input(f'Enter {feature_name.capitalize()}', value=0.0, step=1.0))

    # Predict function
    def predict(features):
        # Convert input features to numpy array
        features_array = np.array(features).reshape(1, -1)
        # Predict probability
        prob = gnb_model.predict_proba(features_array)[0][1]
        return prob

    # Create predict button
    if st.button('Predict'):
        # Make prediction
        prediction = predict(inputs)
        # Display prediction
        if prediction > 0.5:
            st.error(f'There is a high probability of heart disease. Please consult a doctor. Probability: {prediction:.2f}')
        else:
            st.success(f'No heart disease detected. Probability: {prediction:.2f}')
