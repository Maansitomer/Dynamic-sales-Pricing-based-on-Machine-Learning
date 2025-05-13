import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Page config
st.set_page_config(page_title="Sales Trends Dashboard", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f9f9f9; }
    h1, h2, h3 { color: #1f77b4; }
    .stButton>button { background-color: #4CAF50; color: white; }
    </style>
""", unsafe_allow_html=True)

# Header
st.title("🛍️ AI Sales Trends Intelligence Dashboard")

# Load model and data
model = joblib.load("Sales_Trends_prediction_model_gb1.pkl")
df = pd.read_csv("Multiclass Clothing Sales Dataset.csv")

# ----------------------- Prediction Section -----------------------
with st.expander("Enter product and customer details"):

    col1, col2, col3 = st.columns(3)

    with col1:
        product_category = st.selectbox("Product Category", list(range(10)))  # or real categories
        brand = st.selectbox("Brand", ['Forever 21','Ralph Lauren','Nike','Zara','Wrangler','Balenciaga',
                                       'Uniqlo','Reebok','Gucci','Louis Vuitton','Tommy Hilfiger','Under Armour',
                                       'Calvin Klein','Puma','Diesel','Versace','Levi’s','Adidas','H&M','GAP'])
        gender = st.selectbox("Gender", ['Male', 'Female', 'Unisex'])
        size = st.selectbox("Size", ['S', 'M', 'L', 'XL', 'XXL'])  # map to int later
        color = st.selectbox("Color", ['Red', 'Blue', 'Green', 'Black', 'White'])  # map to int
        season = st.selectbox("Season", ['Winter', 'All-Season', 'Summer'])

    with col2:
        payment_method = st.selectbox("Payment Method", ['Card', 'Cash', 'UPI', 'NetBanking'])
        customer_type = st.selectbox("Customer Type", ['New', 'Returning'])  # encode
        cost_price = st.number_input("Cost Price", value=0.0)
        discount_percentage = st.number_input("Discount Percentage", value=0.0)
        quantity_sold = st.number_input("Quantity Sold", value=1)
        total_sales = st.number_input("Total Sales", value=0.0)
        stock_availability = st.number_input("Stock Availability", value=1)

    with col3:
        customer_age = st.number_input("Customer Age", value=25)
        purchase_frequency = st.number_input("Purchase Frequency", value=0.0)
        store_rating = st.number_input("Store Rating", value=0.0)
        return_rate = st.number_input("Return Rate", value=0.0)
        sales_category = st.selectbox("Sales Category", list(range(10)))  # example values
        demand_index = st.number_input("Demand Index", value=0.0)
        price_elasticity = st.number_input("Price Elasticity", value=0.0)
        profit_margin = st.number_input("Profit Margin", value=0.0)

    # ---------- Manual Encoding ----------
    brand_map = {b: i for i, b in enumerate(['Forever 21','Ralph Lauren','Nike','Zara','Wrangler','Balenciaga',
                                             'Uniqlo','Reebok','Gucci','Louis Vuitton','Tommy Hilfiger','Under Armour',
                                             'Calvin Klein','Puma','Diesel','Versace','Levi’s','Adidas','H&M','GAP'])}
    gender_map = {'Male': 0, 'Female': 1, 'Unisex': 2}
    size_map = {'S': 0, 'M': 1, 'L': 2, 'XL': 3, 'XXL': 4}
    color_map = {'Red': 0, 'Blue': 1, 'Green': 2, 'Black': 3, 'White': 4}
    season_map = {'Winter': 0, 'All-Season': 1, 'Summer': 2}
    payment_map = {'Card': 0, 'Cash': 1, 'UPI': 2, 'NetBanking': 3}
    customer_type_map = {'New': 0, 'Returning': 1}

    # Encoded values
    brand_enc = brand_map[brand]
    gender_enc = gender_map[gender]
    size_enc = size_map[size]
    color_enc = color_map[color]
    season_enc = season_map[season]
    payment_enc = payment_map[payment_method]
    customer_type_enc = customer_type_map[customer_type]

    # ---------- Final Feature Vector ----------
    features = np.array([[product_category, brand_enc, gender_enc, size_enc, color_enc,
                          season_enc, payment_enc, customer_type_enc,
                          cost_price, discount_percentage, quantity_sold, total_sales,
                          stock_availability, customer_age, purchase_frequency,
                          store_rating, return_rate, sales_category,
                          demand_index, price_elasticity, profit_margin]])

    if st.button("💡 Predict Selling Price"):
        prediction = model.predict(features)
        st.success(f"🎯 Predicted Selling Price: ₹{prediction[0]:,.2f}")


# ----------------------- Dashboard Section -----------------------
st.markdown("---")
st.header("📊 Interactive Sales Data Insights")

with st.expander("📌 Key Metric Distributions"):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    for i, col in enumerate(["Selling_Price", "Discount_Percentage", "Quantity_Sold"]):
        sns.histplot(df[col], bins=30, kde=True, ax=axs[i], color="#1f77b4")
        axs[i].set_title(f"Distribution of {col}")
    st.pyplot(fig)

with st.expander("🛒 Sales vs Product Category"):
    colA, colB = st.columns(2)

    with colA:
        st.markdown("#### Quantity Sold by Product Category")
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        sns.boxplot(x="Product_Category", y="Quantity_Sold", data=df, ax=ax2, palette="pastel")
        plt.xticks(rotation=45)
        st.pyplot(fig2)

with st.expander("🕒 Seasonal Trends"):
    st.markdown("#### Selling Price Across Seasons")
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    sns.boxplot(x="Season", y="Selling_Price", data=df, ax=ax4, palette="Set2")
    st.pyplot(fig4)

st.markdown("---")
st.markdown("""
    <div style="text-align:center; font-size: 14px;">
    Made by <strong>Maansi Tomer</strong> | 👗 AI Clothing Sales Intelligence | Streamlit Dashboard
    </div>
""", unsafe_allow_html=True)
