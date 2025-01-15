import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from PIL import Image

# Load data
gold_data = pd.read_csv('./view/gld_price_data.csv')

# Split into X and Y
X = gold_data.drop(['Date', 'GLD'], axis=1)
Y = gold_data['GLD']

# Split into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=2)

# Train the model
reg = RandomForestRegressor()
reg.fit(X_train, Y_train)
pred = reg.predict(X_test)
score = r2_score(Y_test, pred)

# Website layout
st.set_page_config(page_title='Gold Price Model', layout='wide')

# Sidebar
st.sidebar.title('Navigation')

# Navigation bar
nav = st.sidebar.selectbox('Select Page:', ['Home', 'Dataset Overview', 'Model Performance', 'Feature Importance', 'Predictions vs Actual'])

# Main page
st.title('Gold Price Prediction Model')
img = Image.open('./view/th.jpeg')
st.image(img, use_container_width=True)

st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

if nav == 'Home':
    st.write("Welcome to the Gold Price Prediction Model app. Use the navigation bar to explore different sections.")
elif nav == 'Dataset Overview':
    st.subheader('Dataset Overview')
    st.write(gold_data)
elif nav == 'Model Performance':
    st.subheader('Model Performance')
    st.write(f'R² Score: {score:.2f}')
elif nav == 'Feature Importance':
    st.subheader('Feature Importance')
    feature_importance = pd.Series(reg.feature_importances_, index=X.columns).sort_values(ascending=False)
    st.bar_chart(feature_importance)
elif nav == 'Predictions vs Actual':
    st.subheader('Predictions vs Actual')
    plt.figure(figsize=(10, 5))
    plt.scatter(Y_test, pred, alpha=0.5)
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title('Actual vs Predicted Gold Prices')
    st.pyplot(plt)

#how to run streamlit app
#streamlit run app.py