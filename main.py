import streamlit as st
import pandas as pd
from model import HousePricePredictor

# Load the model
predictor = HousePricePredictor('mas_housing.csv')

# Title
st.title("House Price Prediction")

# Sidebar for user input
st.sidebar.header("Input Features")

# Helper function to get user input
def user_input_features():
    rooms = st.sidebar.number_input("Number of Rooms", min_value=1, max_value=10, value=3)
    bathrooms = st.sidebar.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2)
    size = st.sidebar.number_input("Size (sq ft)", min_value=100, max_value=10000, value=1500)
    car_parks = st.sidebar.number_input("Number of Car Parks", min_value=0, max_value=5, value=1)
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
        'rooms': rooms,
        'bathrooms': bathrooms,
        'size': size,
        'car_parks': car_parks,
        'location': location,
        'property_type': property_type,
        'furnishing': furnishing
    }
    return pd.DataFrame(data, index=[0])

# Get user input
input_df = user_input_features()

# Display user input
st.subheader("User Input Features")
st.write(input_df)

# Prediction
if st.button("Predict House Price"):
    predicted_price = predictor.predict_price(input_df.iloc[0].tolist())
    st.subheader("Predicted House Price")
    st.write(f"RM {predicted_price:,.2f}")

# Main function
if __name__ == "__main__":
    st.write("Please use the sidebar to input features and predict house price.")
