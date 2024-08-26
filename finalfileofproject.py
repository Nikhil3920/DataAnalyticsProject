import streamlit as st
import numpy as np
import pickle
import pyodbc
import pandas as pd

# Load models and scalers from pickle files for regression
with open('model.pkl', 'rb') as file:
    regressor_model = pickle.load(file)
with open('scaler.pkl', 'rb') as file:
    scaler_regressor = pickle.load(file)
with open('t.pkl', 'rb') as file:
    ohe_regressor = pickle.load(file)

# Load models and scalers from pickle files for classification
with open('cmodel.pkl', 'rb') as file:
    classifier_model = pickle.load(file)
with open('cscaler.pkl', 'rb') as file:
    scaler_classifier = pickle.load(file)
with open('ct.pkl', 'rb') as file:
    ohe_classifier = pickle.load(file)
with open('s.pkl', 'rb') as file:
    ohe_status_classifier = pickle.load(file)

# Function to predict selling price
def predict_selling_price(quantity_tons, item_type, application, thickness, width, country, customer, product_ref):
    try:
        item_type_encoded = ohe_regressor.transform([[item_type]])  # Use the loaded encoder directly
        new_sample = np.array([[np.log(float(quantity_tons)), application, np.log(float(thickness)), float(width), country, float(customer), int(product_ref)]])
        new_sample = np.concatenate((new_sample, item_type_encoded.toarray()), axis=1)
        new_sample_scaled = scaler_regressor.transform(new_sample)
        prediction = regressor_model.predict(new_sample_scaled)
        return np.exp(prediction[0])
    except Exception as e:
        st.error(f"Error predicting selling price: {e}")
        return None

# Function to predict status
def predict_status(quantity_tons, application, thickness, width, country, customer, product_ref, item_type):
    try:
        item_type_encoded = ohe_classifier.transform([[item_type]])  # Use the loaded encoder directly
        new_sample = np.array([[np.log(float(quantity_tons)), application, np.log(float(thickness)), float(width), country, float(customer), int(product_ref)]])
        new_sample = np.concatenate((new_sample, item_type_encoded.toarray()), axis=1)
        new_sample_scaled = scaler_classifier.transform(new_sample)
        prediction = classifier_model.predict(new_sample_scaled)
        return prediction[0]
    except Exception as e:
        st.error(f"Error predicting status: {e}")
        return None

# Fetch data from SQL database
def fetch_data(query):
    try:
        # Establish a connection to the SQL Server
        conn = pyodbc.connect(connection_string)
        cursor = conn.cursor()
        cursor.execute(query)
        data = cursor.fetchall()
        conn.close()
        return data
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# Load connection parameters from pickle file
with open('sql_data.pkl', 'rb') as f:
    connection_string, _ = pickle.load(f)

# Load all .pkl files for displaying data
def load_pkl_data(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

country_sales_data = load_pkl_data('country_sales.pkl')
app_avg_qty_data = load_pkl_data('application_avg_qty.pkl')
item_type_win_count_data = load_pkl_data('item_type_win_count.pkl')
top_bottom_customers_data = load_pkl_data('top_bottom_customers.pkl')
country_sales_2_data = load_pkl_data('country_sales_2.pkl')

# Adding custom CSS for animations using animate.css
st.markdown(
    """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css"/>
    <style>
    body {
        background-color: #f0f2f6;
        font-family: 'Arial', sans-serif;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        border-radius: 8px;
        padding: 10px 20px;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .stButton button:active {
        background-color: #3e8e41;
        transform: translateY(4px);
    }
    .css-1d391kg p {
        font-size: 18px;
        font-weight: 600;
        color: #333;
    }
    .header {
        text-align: center;
        padding: 20px;
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
    }
    .stTextInput input {
        border-radius: 8px;
        padding: 10px;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit App
st.title('Copper Modeling Application')

tabs = st.sidebar.radio("Navigation", ["Predict Selling Price", "Predict Status", "SQL Query", "Data Analysis"])

if tabs == "Predict Selling Price":
    st.markdown('<div class="header"><h2>Predict Selling Price</h2></div>', unsafe_allow_html=True)
    st.markdown('Enter the details below:')
    
    quantity_tons = st.text_input('Quantity Tons (Log) (Min: 611728, Max: 1722207579)')
    item_type = st.selectbox('Item Type', ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR'])
    application = st.selectbox('Application',[10,15,41,42,38,56,59,29,27,25,40,69,20,65,28,66,22,26,79,58,39,3,67,68,4,19,5,70,99,2], key='application_price')
    thickness = st.text_input('Thickness (Log) (Min: 0.18, Max: 400)')
    width = st.text_input('Width (Min: 1, Max: 2990)')
    country = st.selectbox('Country',['78','26','27','32','30','39','28','25','77','84','38','79','40','80','113','89','107'],key='country_price')

    customer = st.text_input('Customer ID (Min: 12458, Max: 30408185)')
    product_ref = st.text_input('Product Reference (Min: 611728, Max: 1722207579)')
    
    if st.button('Predict', key='predict_button'):
        selling_price = predict_selling_price(quantity_tons, item_type, application, thickness, width, country, customer, product_ref)
        if selling_price is not None:
            st.success(f'Predicted Selling Price: {selling_price:.2f}')

elif tabs == "Predict Status":
    st.markdown('<div class="header"><h2>Predict Status</h2></div>', unsafe_allow_html=True)
    st.markdown('Enter the details below:')
    
    quantity_tons = st.text_input('Quantity Tons (Log) (Min: 611728, Max: 1722207579)')
    application = st.text_input('Application (Min: 2, Max: 99)')
    thickness = st.text_input('Thickness (Log) (Min: 0.18, Max: 400)')
    width = st.text_input('Width (Min: 1, Max: 2990)')
    country = st.text_input('Country (Min: 25, Max: 113)')
    customer = st.text_input('Customer ID (Min: 12458, Max: 30408185)')
    product_ref = st.text_input('Product Reference (Min: 611728, Max: 1722207579)')
    item_type = st.selectbox('Item Type', ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR'])
    
    if st.button('Predict', key='status_button'):
        status_prediction = predict_status(quantity_tons, application, thickness, width, country, customer, product_ref, item_type)
        if status_prediction is not None:
            if status_prediction == 1:
                st.success('Predicted Status: Won')
            else:
                st.error('Predicted Status: Lost')

elif tabs == "SQL Query":
    st.markdown('<div class="header"><h2>SQL Query</h2></div>', unsafe_allow_html=True)
    query = st.text_area('Enter SQL Query:')
    
    if st.button('Execute', key='sql_button'):
        data = fetch_data(query)
        if data:
            st.write(pd.DataFrame(data))
        else:
            st.error('No data fetched. Check your query or connection.')

elif tabs == "Data Analysis":
    st.subheader('Data Analysis')
    st.markdown('Choose an option for data analysis:')
    
    analysis_option = st.selectbox('Select Analysis Option', [
        'The maximum valid quantity-tons ordered in each item-type category.',
        'For which application, the average order quantity is the highest and the lowest?',
        'Which item type has the max number of ‘Win’ status?',
        'N number of customers who are top and bottom contributors in Revenue. User should have an option to choose the value of n. Return order_id and customer_id',
        'Country-wise sum of sales.'
    ])
    
    if analysis_option == 'The maximum valid quantity-tons ordered in each item-type category.':
        st.subheader('The maximum valid quantity-tons ordered in each item-type category.')
        if isinstance(country_sales_data, pd.DataFrame) and not country_sales_data.empty:
            st.write(country_sales_data)
        else:
            st.write("No data found.")
    
    elif analysis_option == 'For which application, the average order quantity is the highest and the lowest?':
        st.subheader('For which application, the average order quantity is the highest and the lowest?')
        if isinstance(app_avg_qty_data, pd.DataFrame) and not app_avg_qty_data.empty:
            st.write(app_avg_qty_data)
        else:
            st.write("No data found.")
    
    elif analysis_option == 'Which item type has the max number of ‘Win’ status?':
        st.subheader('Which item type has the max number of ‘Win’ status?')
        if isinstance(item_type_win_count_data, pd.DataFrame) and not item_type_win_count_data.empty:
            st.write(item_type_win_count_data)
        else:
            st.write("No data found.")
    
    elif analysis_option == 'N number of customers who are top and bottom contributors in Revenue. User should have an option to choose the value of n. Return order_id and customer_id':
        st.subheader('N number of customers who are top and bottom contributors in Revenue. User should have an option to choose the value of n. Return order_id and customer_id')
        if isinstance(top_bottom_customers_data, pd.DataFrame) and not top_bottom_customers_data.empty:
            st.write(top_bottom_customers_data)
        else:
            st.write("No data found.")
    
    elif analysis_option == 'Country-wise sum of sales.':
        st.subheader('Country-wise sum of sales.')
        if isinstance(country_sales_2_data, pd.DataFrame) and not country_sales_2_data.empty:
            st.write(country_sales_2_data)
        else:
            st.write("No data found.")

