import streamlit as st
import pandas as pd
import joblib
import time

# Load the pre-trained model and preprocessor
model = joblib.load("house_price_predictor_model.sav")
preprocessor = joblib.load("preprocessor.sav")

# Title
st.title("House Price Prediction Malaysia")

# Sidebar for user input
st.sidebar.header("Input Features")


# Helper function to get user input
def user_input_features():
    rooms = st.sidebar.number_input(
        "Number of Rooms", min_value=1, max_value=10, value=3
    )
    bathrooms = st.sidebar.number_input(
        "Number of Bathrooms", min_value=1, max_value=10, value=2
    )
    size = st.sidebar.number_input(
        "Size (sq ft)", min_value=100, max_value=10000, value=1500, step=100
    )
    car_parks = st.sidebar.number_input(
        "Number of Car Parks", min_value=0, max_value=5, value=1
    )

    location = st.sidebar.selectbox("Location", options=[
        'klcc,_kuala_lumpur', 'ampang,_kuala_lumpur', 'cheras,_kuala_lumpur', 
        'mont_kiara,_kuala_lumpur', 'bangsar,_kuala_lumpur'
    ])
    property_type = st.sidebar.selectbox("Property Type", options=[
        'condominium', 'terrace', 'bungalow', 
        'semi-detached_house', 'apartment', 'flat'
    ])
    furnishing = st.sidebar.selectbox("Furnishing", options=[
        'fully_furnished', 'partly_furnished', 'unfurnished'
    ])

    data = {
        "rooms": rooms,
        "bathrooms": bathrooms,
        "size": size,
        "car_parks": car_parks,
        "location": location,
        "property_type": property_type,
        "furnishing": furnishing,
    }
    return pd.DataFrame(data, index=[0])


# Function to simulate a loading effect
def loading_message(step):
    with st.spinner(f"{step}..."):
        time.sleep(1)


# Main function
if __name__ == "__main__":
    st.markdown("---")
    st.markdown(
        '<h6>Made with &nbsp<img src="https://streamlit.io/images/brand/streamlit-mark-color.png" alt="Streamlit logo" height="16">&nbsp by Ahmad, Nisya, Hendrick, Qaim</h6>',
        unsafe_allow_html=True,
    )
    # Adding custom CSS styling for the prediction text
    st.markdown(
        """
        <style>
        .predicted-price {
            font-size: 24px;
            font-weight: bold;
            color: white;
            background-color: #4CAF50;
            text-align: center;
            padding: 10px;
            border-radius: 10px;
            margin-top: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("User Input Features")
    input_df = user_input_features()

    st.write(input_df)

    # Prediction
    with st.sidebar:
        if st.button("Predict House Price"):
            loading_message("Getting input")
            loading_message("Running Model")

            # Apply the same preprocessing steps as training data
            features_transformed = preprocessor.transform(input_df)

            # Get the prediction
            predicted_price = model.predict(features_transformed)[0]

    # Display the prediction with enhanced styling
    st.markdown(
        f'<div class="predicted-price">RM {predicted_price:,.2f}</div>',
        unsafe_allow_html=True,
    )

    st.write("*Please use the sidebar to input features and predict house price.")

    # Display saved plots
    st.subheader("Exploratory Data Analysis (EDA) Plots")

    st.image("price_distribution_histplot.png", caption="Distribution of House Prices")
    st.image("correlation_heatmap.png", caption="Correlation")
    st.image("pairplot.png", caption="Pairlot")

    # with st.sidebar:
        
